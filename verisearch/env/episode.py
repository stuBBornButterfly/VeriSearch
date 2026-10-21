"""The episode loop: think, search, read, repeat, answer.

Owner: Dhrubo. PROJECT.md section 11, P1 exit: an untrained base model runs 100
questions with real retrieval and produces typed trajectory JSONL.

Depends on protocols, never implementations (section 3, dependency inversion):

- `Retriever` from `verisearch.types` - Ayesha's FAISS client, or `FakeRetriever`
- `Policy` below - vLLM, Transformers, or a scripted fake in a test
- `Tokenizer` below - optional, and only needed for training

Three decisions worth knowing before reading the loop:

**Token ids are optional here and mandatory in training.** A `Policy` may return
them; a `Tokenizer` lets the environment encode the `<info>` block it injects.
With both, every turn carries its ids and `train/masking.py` can build the loss
mask as a pure function of the trajectory. With neither - the P1 case, reading
trajectories on CPU - turns carry text only and nothing downstream breaks.

**The environment splices token ids, never re-tokenised text.** `<info>` ids come
from the tokenizer once, at injection, and are stored on the turn. Re-tokenising
a decoded transcript later would shift token boundaries and silently misalign the
mask, which is the failure KICKOFF.md warns about.

**A search always costs budget.** Empty query, duplicate query, zero hits: the
call happened, so it is counted. The reward charges for it (section 6); the loop
does not forgive it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from verisearch.config import Config
from verisearch.env.parser import Action, parse_turn
from verisearch.env.protocol import (
    STOP_SEQUENCES,
    check_stop_sequences,
    render_info,
    render_prompt,
)
from verisearch.types import (
    ParseFailure,
    Passage,
    Retriever,
    Termination,
    Trajectory,
    Turn,
    TurnKind,
)

__all__ = [
    "EpisodeConfig",
    "Generation",
    "Policy",
    "Question",
    "Tokenizer",
    "run_episode",
]


class Generation(BaseModel):
    """One policy continuation.

    `token_ids` is empty unless the policy supplies it, and `logprobs` - the
    sampling log-probability of each of those tokens - is empty unless the
    backend returns them. vLLM does; a scripted test policy does not. GRPO needs
    them for the importance-sampling ratio, so `train/rollout.py` requires them
    and the episode loop does not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float | None, ...] = ()


class Policy(Protocol):
    """Generates one turn. Implemented by vLLM at P3 and by fakes in tests."""

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation: ...


class Tokenizer(Protocol):
    """Encodes the text the environment injects, so its ids match the policy's."""

    def encode(self, text: str) -> list[int]: ...


class Question(BaseModel):
    """The minimum an episode needs. `data.QAExample` satisfies it via `from_example`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answers: tuple[str, ...] = Field(min_length=1)


class EpisodeConfig(BaseModel):
    """Episode limits. Built from `Config` so the prompt and the loop agree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    search_budget: int = Field(ge=1)
    max_turns: int = Field(ge=2)
    max_new_tokens: int = Field(ge=1)
    top_k: int = Field(ge=1)
    stop_sequences: tuple[str, ...] = STOP_SEQUENCES

    @classmethod
    def from_config(cls, config: Config) -> EpisodeConfig:
        """Take every limit from the YAML, including `retrieval.top_k`."""
        return cls(
            search_budget=config.env.search_budget,
            max_turns=config.env.max_turns,
            max_new_tokens=config.env.max_new_tokens,
            top_k=config.retrieval.top_k,
            stop_sequences=config.env.stop_sequences,
        )


def run_episode(
    question: Question,
    policy: Policy,
    retriever: Retriever,
    config: EpisodeConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> Trajectory:
    """Run one episode to termination and return its trajectory.

    Terminates for exactly one of four reasons, which is what
    `Termination` records and what section 9's failure breakdown counts:

    - `ANSWERED` - the policy emitted a non-empty `<answer>`
    - `PARSE_ERROR` - a fatal malformed turn, with the category in `parse_failure`
    - `SEARCH_BUDGET_EXHAUSTED` - it asked for a search it could not have
    - `MAX_TURNS` - it ran out of turns without answering
    """
    check_stop_sequences(config.stop_sequences)

    prompt = render_prompt(
        question.question, search_budget=config.search_budget, top_k=config.top_k
    )
    transcript = prompt
    turns: list[Turn] = []
    answer: str | None = None
    termination = Termination.MAX_TURNS
    parse_failure: ParseFailure | None = None

    for _ in range(config.max_turns):
        generation = policy.generate(
            transcript,
            stop=config.stop_sequences,
            max_new_tokens=config.max_new_tokens,
        )
        parsed = parse_turn(generation.text)
        # Policy ids cover the whole generation, trailing text included; the
        # turns carry text only. Attaching the ids to the acting turn keeps the
        # completion reconstructable in order without claiming a per-turn split
        # the policy never gave us.
        turns.extend(_with_ids(parsed.turns, generation.token_ids))

        if parsed.fatal:
            termination = Termination.PARSE_ERROR
            parse_failure = parsed.failure
            break

        if parsed.action is Action.ANSWER:
            answer = parsed.answer
            termination = Termination.ANSWERED
            break

        # Counted off the turns rather than a counter, for the same reason
        # `Trajectory.n_searches` is a property: a second source of truth for
        # "how many searches" is a second thing that can disagree with the
        # trajectory everything downstream reads. The acting SEARCH turn is
        # already appended, so the budget is exceeded at `>`, not `>=`.
        if sum(1 for turn in turns if turn.kind is TurnKind.SEARCH) > config.search_budget:
            # It asked for a search it cannot have. The unserved SEARCH turn
            # stays in the trajectory - it is the honest record of what the
            # policy did - and section 6 prices the episode through the
            # no-answer penalty.
            termination = Termination.SEARCH_BUDGET_EXHAUSTED
            break

        query = parsed.query or ""
        passages = retriever.search(query, config.top_k) if query else []
        info_turn = _info_turn(passages, tokenizer)
        turns.append(info_turn)
        # The policy's own text is appended *without* its trailing overshoot.
        # Feeding back a fabricated <info> the model wrote after its search tag
        # would teach it that forging evidence works.
        acted = generation.text[: len(generation.text) - len(parsed.trailing)]
        transcript = f"{transcript}{acted}\n{info_turn.text}\n"

    return Trajectory(
        question_id=question.question_id,
        question=question.question,
        gold_answers=question.gold_answers,
        turns=tuple(turns),
        answer=answer,
        termination=termination,
        parse_failure=parse_failure,
    )


def _with_ids(turns: Sequence[Turn], token_ids: tuple[int, ...]) -> list[Turn]:
    """Attach the generation's ids to its last turn, when the policy gave any."""
    if not token_ids or not turns:
        return list(turns)
    *earlier, acting = turns
    return [*earlier, acting.model_copy(update={"token_ids": token_ids})]


def _info_turn(passages: Sequence[Passage], tokenizer: Tokenizer | None) -> Turn:
    """The environment's turn. Its ids must come from the tokenizer, once."""
    text = render_info(passages)
    token_ids = tuple(tokenizer.encode(text)) if tokenizer is not None else ()
    return Turn(kind=TurnKind.INFO, text=text, token_ids=token_ids, passages=tuple(passages))
