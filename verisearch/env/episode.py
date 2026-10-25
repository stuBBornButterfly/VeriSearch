"""The episode loop: think, search, read, repeat, answer.

Owner: Dhrubo. PROJECT.md section 11, P1 exit: an untrained base model runs 100
questions with real retrieval and produces typed trajectory JSONL.

Depends on protocols, never implementations (section 3, dependency inversion):

- `Retriever` from `verisearch.types` - Ayesha's FAISS client, or `FakeRetriever`
- `Policy` or `TokenPolicy` below - vLLM, Transformers, or a scripted fake
- `Tokenizer` below - only for the token path, which is training's

Three decisions worth knowing before reading the loop:

**Two entry points, one loop.** `run_episode` is the text path: the policy gets
the transcript as a string and returns text, which is all eval and serving need.
`run_token_episode` is the token path: the policy gets the context as token ids
and returns ids, and every turn of the trajectory carries the ids it added to
that context. Training uses the token path and nothing else does.

**On the token path the ids are the source of truth.** The context is a running
list of token ids - the prompt, then each generation's ids, then each injected
block's ids - and the policy is handed that list, never a re-tokenised string.
So the context a token was sampled from is, by construction, the prefix the
trainer replays in front of it. Ayesha's finding A on #31: the first version
re-encoded the decoded transcript every turn, and from turn 2 on the trainer
replayed a context the model never sampled from, so the importance-sampling
ratio compared log-probabilities across two different contexts.

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
    render_injection,
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
    "EpisodeError",
    "Generation",
    "Policy",
    "Question",
    "TokenEpisode",
    "TokenPolicy",
    "Tokenizer",
    "run_episode",
    "run_token_episode",
]


class EpisodeError(Exception):
    """Raised when a policy's output cannot be spliced into the context safely."""


class Generation(BaseModel):
    """One policy continuation.

    `token_ids` and `logprobs` - the sampling log-probability of each of those
    tokens - are empty unless the backend supplies them. On the token path the
    ids are required and `text` must be their decoding; GRPO also needs the
    logprobs for the importance-sampling ratio, so `train/rollout.py` requires
    them and the episode loop does not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float | None, ...] = ()


class Policy(Protocol):
    """Generates one turn from the transcript as text. Eval's and serving's view."""

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation: ...


class TokenPolicy(Protocol):
    """Generates one turn from the context as token ids. Training's view.

    No string goes in, so nothing between two turns can re-tokenise the context
    and move a token boundary.
    """

    def generate_ids(
        self, context_ids: Sequence[int], *, stop: Sequence[str], max_new_tokens: int
    ) -> Generation: ...


class Tokenizer(Protocol):
    """Encodes the opening prompt and the injected blocks, and decodes generations.

    `transformers` tokenizers satisfy it. The token path uses `decode` to check
    that a generation's text is its ids, and to find where a served search's ids
    end.
    """

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class Question(BaseModel):
    """The minimum an episode needs.

    `gold_answers` may be empty: a question the policy server receives at P5 has
    none, and the episode loop never reads them - they are carried through to
    the trajectory for whoever scores it. Ayesha's question on #38. Training
    refuses a question without one (`train.grpo.load_questions`), because the
    reward cannot score it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answers: tuple[str, ...] = ()


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


class TokenEpisode(BaseModel):
    """A token-path episode: the trajectory, plus what training needs beside it.

    `prompt_ids` is the context the first turn was sampled from. `generations`
    holds, in order, what the trajectory kept of each generation: all of it for
    the turn that ended the episode, and up to the token that closed the search
    for a turn the environment answered. Their ids are exactly the generated
    turns' ids, and their logprobs line up with them, which `train/rollout.py`
    checks rather than assumes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectory: Trajectory
    prompt_ids: tuple[int, ...]
    generations: tuple[Generation, ...]


def run_episode(
    question: Question,
    policy: Policy,
    retriever: Retriever,
    config: EpisodeConfig,
) -> Trajectory:
    """Run one episode on the text path and return its trajectory.

    Turns carry text only - any token ids the policy returns are dropped, since
    nothing here can tokenise the injected blocks to match them. Terminates for
    exactly one of four reasons, which is what `Termination` records and what
    section 9's failure breakdown counts:

    - `ANSWERED` - the policy emitted a non-empty `<answer>`
    - `PARSE_ERROR` - a fatal malformed turn, with the category in `parse_failure`
    - `SEARCH_BUDGET_EXHAUSTED` - it asked for a search it could not have
    - `MAX_TURNS` - it ran out of turns without answering
    """
    context = _TextContext(policy, _prompt(question, config))
    trajectory, _ = _loop(question, context, retriever, config)
    return trajectory


def run_token_episode(
    question: Question,
    policy: TokenPolicy,
    retriever: Retriever,
    config: EpisodeConfig,
    *,
    tokenizer: Tokenizer,
) -> TokenEpisode:
    """Run one episode on the token path. Same loop and terminations as `run_episode`.

    The invariant training depends on, and `tests/test_rollout.py` asserts for
    every turn: the ids the policy is handed on turn k are `prompt_ids` followed
    by every id the trajectory's turns carry before that turn.
    """
    context = _TokenContext(policy, tokenizer, tuple(tokenizer.encode(_prompt(question, config))))
    trajectory, generations = _loop(question, context, retriever, config)
    return TokenEpisode(
        trajectory=trajectory,
        prompt_ids=context.prompt_ids,
        generations=tuple(generations),
    )


def _prompt(question: Question, config: EpisodeConfig) -> str:
    return render_prompt(question.question, search_budget=config.search_budget, top_k=config.top_k)


class _Context(Protocol):
    """What the loop needs from a path: generate, keep, inject."""

    def generate(self, *, stop: Sequence[str], max_new_tokens: int) -> Generation: ...

    def keep(self, generation: Generation, acted: str) -> Generation: ...

    def inject(self, kept: Generation, text: str) -> tuple[int, ...]: ...


class _TextContext:
    """The transcript as a string. The text path."""

    def __init__(self, policy: Policy, prompt: str) -> None:
        self._policy = policy
        self._transcript = prompt

    def generate(self, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        raw = self._policy.generate(self._transcript, stop=stop, max_new_tokens=max_new_tokens)
        return Generation(text=raw.text)

    def keep(self, generation: Generation, acted: str) -> Generation:
        """The text up to the end of the search tag. The overshoot is never fed
        back: a fabricated `<info>` after the tag would teach the policy that
        forging evidence works."""
        del generation
        return Generation(text=acted)

    def inject(self, kept: Generation, text: str) -> tuple[int, ...]:
        self._transcript = f"{self._transcript}{kept.text}{text}"
        return ()


class _TokenContext:
    """The context as token ids. The token path."""

    def __init__(
        self, policy: TokenPolicy, tokenizer: Tokenizer, prompt_ids: tuple[int, ...]
    ) -> None:
        self._policy = policy
        self._tokenizer = tokenizer
        self.prompt_ids = prompt_ids
        self._ids = list(prompt_ids)

    def generate(self, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        generation = self._policy.generate_ids(
            tuple(self._ids), stop=stop, max_new_tokens=max_new_tokens
        )
        if not generation.token_ids:
            msg = "the token policy returned no tokens; a turn needs at least one"
            raise EpisodeError(msg)
        if generation.logprobs and len(generation.logprobs) != len(generation.token_ids):
            msg = (
                f"{len(generation.logprobs)} logprobs for {len(generation.token_ids)} tokens; "
                "they are paired by position"
            )
            raise EpisodeError(msg)
        if generation.text != self._tokenizer.decode(generation.token_ids):
            # The parser reads the text and the trainer reads the ids. If they
            # disagree, the turn boundaries the parser finds describe tokens
            # the trainer never sees.
            msg = "a generation's text is not the decoding of its token ids"
            raise EpisodeError(msg)
        return generation

    def keep(self, generation: Generation, acted: str) -> Generation:
        """The generation up to the token that closed the search.

        Anything the policy wrote after `</search>` is cut from *both* streams,
        so it neither reaches the next turn's context nor carries loss. A
        fabricated `<info>` after the search tag would otherwise be fed back, and
        the policy would learn that forging evidence works. With vLLM stopping
        on `</search>` this is at most the tail of the one token that completed
        the stop string, which is kept whole: a token is the unit both streams
        share.
        """
        if acted == generation.text:
            return generation
        # `generate` checked that the text is the decoding of all the ids, so
        # the full sequence covers `acted`; drop tokens from the end while what
        # is left still does.
        ids = generation.token_ids
        end = len(ids)
        while end > 1 and self._tokenizer.decode(ids[: end - 1]).startswith(acted):
            end -= 1
        return Generation(
            text=self._tokenizer.decode(ids[:end]),
            token_ids=ids[:end],
            logprobs=generation.logprobs[:end],
        )

    def inject(self, kept: Generation, text: str) -> tuple[int, ...]:
        """Encode the injected block once, and splice both into the context."""
        injected = tuple(self._tokenizer.encode(text))
        self._ids.extend(kept.token_ids)
        self._ids.extend(injected)
        return injected


def _loop(
    question: Question,
    context: _Context,
    retriever: Retriever,
    config: EpisodeConfig,
) -> tuple[Trajectory, list[Generation]]:
    check_stop_sequences(config.stop_sequences)

    turns: list[Turn] = []
    kept: list[Generation] = []
    answer: str | None = None
    termination = Termination.MAX_TURNS
    parse_failure: ParseFailure | None = None

    for _ in range(config.max_turns):
        generation = context.generate(
            stop=config.stop_sequences,
            max_new_tokens=config.max_new_tokens,
        )
        parsed = parse_turn(generation.text)

        if parsed.fatal or parsed.action is Action.ANSWER:
            # The episode ends here, so nothing is appended after this
            # generation and all of it is kept.
            turns.extend(_with_ids(parsed.turns, generation))
            kept.append(generation)
            if parsed.fatal:
                termination = Termination.PARSE_ERROR
                parse_failure = parsed.failure
            else:
                answer = parsed.answer
                termination = Termination.ANSWERED
            break

        # Counted off the turns rather than a counter, for the same reason
        # `Trajectory.n_searches` is a property: a second source of truth for
        # "how many searches" is a second thing that can disagree with the
        # trajectory everything downstream reads. This turn's search is not
        # appended yet, hence the `+ 1`.
        if sum(1 for turn in turns if turn.kind is TurnKind.SEARCH) + 1 > config.search_budget:
            # It asked for a search it cannot have. The unserved SEARCH turn
            # stays in the trajectory - it is the honest record of what the
            # policy did - and section 6 prices the episode through the
            # no-answer penalty.
            turns.extend(_with_ids(parsed.turns, generation))
            kept.append(generation)
            termination = Termination.SEARCH_BUDGET_EXHAUSTED
            break

        acted = generation.text[: len(generation.text) - len(parsed.trailing)]
        acting = context.keep(generation, acted)
        turns.extend(_with_ids(parsed.turns, acting))
        kept.append(acting)

        query = parsed.query or ""
        passages = retriever.search(query, config.top_k) if query else []
        injected = render_injection(passages)
        turns.append(_info_turn(injected, context.inject(acting, injected), passages))

    trajectory = Trajectory(
        question_id=question.question_id,
        question=question.question,
        gold_answers=question.gold_answers,
        turns=tuple(turns),
        answer=answer,
        termination=termination,
        parse_failure=parse_failure,
    )
    return trajectory, [generation for generation in kept if generation.token_ids]


def _with_ids(turns: Sequence[Turn], generation: Generation) -> list[Turn]:
    """Attach the generation's ids to its last turn, keeping every token.

    Policy ids cover the whole generation; the parsed turns carry text only.
    Attaching the ids to the acting turn keeps the completion reconstructable in
    order without claiming a per-turn split the policy never gave us.

    When the generation parsed to no turns at all - bare prose, or a tag the
    parser refused before any turn closed - the tokens still exist, the policy
    still wrote them, and they still have to carry loss: a failed attempt is
    what GRPO needs to push down, and dropping it hands the trainer a rollout
    with an empty completion. Found by a real smoke run, where TRL indexes
    `ids[-1]` on every completion (`grpo_trainer.py:2325`) and raised
    `IndexError` on the first 0.5B rollout that answered in prose.

    Such a generation is recorded as `TurnKind.UNPARSED`: `generated` is True, so
    it carries loss, and `Trajectory.parse_failure` says why it went nowhere.
    Asked for on #2 and now in the frozen contract. Recorded as `THINK` - which
    is what this did before the contract gained the kind - a dump would read
    "reasoned and gave up" where the policy ignored the format, and section 15
    has those dumps read by eye.
    """
    if not turns:
        if not generation.text and not generation.token_ids:
            return []
        return [
            Turn(
                kind=TurnKind.UNPARSED,
                text=generation.text,
                token_ids=generation.token_ids,
            )
        ]
    if not generation.token_ids:
        return list(turns)
    *earlier, acting = turns
    return [*earlier, acting.model_copy(update={"token_ids": generation.token_ids})]


def _info_turn(text: str, token_ids: tuple[int, ...], passages: Sequence[Passage]) -> Turn:
    """The environment's turn: exactly the text injected, and the ids it was encoded to."""
    return Turn(kind=TurnKind.INFO, text=text, token_ids=token_ids, passages=tuple(passages))
