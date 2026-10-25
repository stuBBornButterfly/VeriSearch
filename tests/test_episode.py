"""Episode loop tests. CPU, no model, no network - the fakes from #9 plus a
scripted policy, which is the point of the protocols in section 3.

Organised by termination reason, because that is what the loop decides and what
section 9's failure breakdown counts.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from verisearch.config import load_config
from verisearch.env.episode import (
    EpisodeConfig,
    EpisodeError,
    Generation,
    Question,
    run_episode,
    run_token_episode,
)
from verisearch.env.protocol import NO_RESULTS, ProtocolError, render_prompt
from verisearch.testing import FakeRetriever
from verisearch.types import ParseFailure, Passage, Termination, TurnKind

QUESTION = Question(
    question_id="q1",
    question="Who wrote the first notes on the Analytical Engine?",
    gold_answers=("Ada Lovelace",),
)


class ScriptedPolicy:
    """Emits a fixed list of generations, one per turn. Records what it was fed.

    A scripted policy is what makes the loop's control flow testable at all: the
    behaviours that matter here - searching twice, running out of budget,
    emitting a malformed turn - are policy behaviours, and waiting for a real
    model to produce them on cue is not a test.
    """

    def __init__(self, *generations: str, token_ids_per_turn: int = 0) -> None:
        self._generations = list(generations)
        self._tokens = token_ids_per_turn
        self.prompts: list[str] = []
        self.stops: list[tuple[str, ...]] = []

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        self.prompts.append(prompt)
        self.stops.append(tuple(stop))
        assert max_new_tokens > 0
        if not self._generations:
            # A policy asked for more turns than the script has is a test bug,
            # not a loop bug: say so loudly rather than returning empty text
            # that the parser would classify as NO_ANSWER.
            msg = f"script exhausted after {len(self.prompts)} turns"
            raise AssertionError(msg)
        text = self._generations.pop(0)
        base = 100 * len(self.prompts)
        ids = tuple(range(base, base + self._tokens))
        return Generation(text=text, token_ids=ids)


class CharTokenizer:
    """One id per character, so decoding is exact. Records what it encodes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token) for token in token_ids)


class ScriptedTokenPolicy:
    """The token path's scripted policy: ids in, the script's ids out.

    Records every context it is handed, which is what the token-context
    invariant is asserted against.
    """

    def __init__(self, *generations: str, tokenizer: CharTokenizer | None = None) -> None:
        self._generations = list(generations)
        self._tokenizer = tokenizer or CharTokenizer()
        self.contexts: list[tuple[int, ...]] = []

    def generate_ids(
        self, context_ids: Sequence[int], *, stop: Sequence[str], max_new_tokens: int
    ) -> Generation:
        assert stop
        assert max_new_tokens > 0
        self.contexts.append(tuple(context_ids))
        text = self._generations.pop(0)
        ids = tuple(ord(char) for char in text)
        return Generation(text=text, token_ids=ids, logprobs=tuple(-0.5 for _ in ids))


def config(**overrides: int) -> EpisodeConfig:
    base = {"search_budget": 2, "max_turns": 4, "max_new_tokens": 64, "top_k": 2}
    return EpisodeConfig.model_validate(base | overrides)


# ---------------------------------------------------------------------------
# ANSWERED
# ---------------------------------------------------------------------------


def test_a_two_hop_episode(fake_retriever: FakeRetriever) -> None:
    policy = ScriptedPolicy(
        "<think>Find the engine first.</think><search>Analytical Engine</search>",
        "<think>Now who wrote the notes.</think><search>Ada Lovelace notes</search>",
        "<answer>Ada Lovelace</answer>",
    )

    traj = run_episode(QUESTION, policy, fake_retriever, config())

    assert traj.termination is Termination.ANSWERED
    assert traj.answer == "Ada Lovelace"
    assert traj.n_searches == 2
    assert [t.kind for t in traj.turns] == [
        TurnKind.THINK,
        TurnKind.SEARCH,
        TurnKind.INFO,
        TurnKind.THINK,
        TurnKind.SEARCH,
        TurnKind.INFO,
        TurnKind.ANSWER,
    ]
    assert fake_retriever.queries == ("Analytical Engine", "Ada Lovelace notes")
    assert len(traj.retrieved_passages) == 4


def test_answering_without_searching_is_allowed_and_recorded() -> None:
    """The no-search policy has to be representable: section 6 requires checking
    empirically that it scores worse than a searching one."""
    traj = run_episode(
        QUESTION, ScriptedPolicy("<answer>Ada Lovelace</answer>"), FakeRetriever(), config()
    )

    assert traj.termination is Termination.ANSWERED
    assert traj.n_searches == 0
    assert traj.retrieved_passages == ()


def test_passages_hang_off_the_info_turn_that_injected_them(
    fake_retriever: FakeRetriever,
) -> None:
    """Faithfulness scores an answer against what was in context when it was
    written, so the passages belong to the INFO turn, not the SEARCH turn."""
    policy = ScriptedPolicy("<search>Ada Lovelace</search>", "<answer>Ada Lovelace</answer>")

    traj = run_episode(QUESTION, policy, fake_retriever, config())

    search_turn, info_turn = traj.turns[0], traj.turns[1]
    assert search_turn.kind is TurnKind.SEARCH
    assert search_turn.passages == ()
    assert len(info_turn.passages) == 2
    assert all(isinstance(p, Passage) for p in info_turn.passages)


# ---------------------------------------------------------------------------
# PARSE_ERROR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        pytest.param("<think>unterminated", ParseFailure.UNCLOSED_TAG, id="unclosed"),
        pytest.param("<think><search>q</search></think>", ParseFailure.NESTED_TAG, id="nested"),
        pytest.param("<info>I retrieved this myself</info>", ParseFailure.UNKNOWN_TAG, id="forged"),
        pytest.param("<answer></answer>", ParseFailure.NO_ANSWER, id="empty-answer"),
    ],
)
def test_a_fatal_turn_ends_the_episode_and_names_the_category(
    generation: str, expected: ParseFailure
) -> None:
    traj = run_episode(QUESTION, ScriptedPolicy(generation), FakeRetriever(), config())

    assert traj.termination is Termination.PARSE_ERROR
    assert traj.parse_failure is expected
    assert traj.answer is None


# ---------------------------------------------------------------------------
# The empty query: a penalty, not a crash
# ---------------------------------------------------------------------------


def test_an_empty_query_costs_budget_and_the_episode_continues() -> None:
    retriever = FakeRetriever()
    policy = ScriptedPolicy("<search>   </search>", "<answer>Ada Lovelace</answer>")

    traj = run_episode(QUESTION, policy, retriever, config())

    assert traj.termination is Termination.ANSWERED
    assert traj.n_searches == 1
    assert retriever.n_searches == 0, "a blank query is not worth a retrieval call"
    # Exactly what was injected, separators included (finding A on #31).
    assert traj.turns[1].text == f"\n<info>\n{NO_RESULTS}\n</info>\n"


def test_a_search_with_no_hits_still_injects_a_turn() -> None:
    """Zero hits is evidence about the query, and the policy only learns to
    rephrase if it sees that the search came back empty."""
    retriever = FakeRetriever(blind_to=("no such thing",))
    policy = ScriptedPolicy("<search>no such thing</search>", "<answer>unknown</answer>")

    traj = run_episode(QUESTION, policy, retriever, config())

    assert [t.kind for t in traj.turns[:2]] == [TurnKind.SEARCH, TurnKind.INFO]
    assert traj.turns[1].passages == ()
    assert NO_RESULTS in traj.turns[1].text


# ---------------------------------------------------------------------------
# Budget and turn limits
# ---------------------------------------------------------------------------


def test_the_search_after_the_budget_ends_the_episode(fake_retriever: FakeRetriever) -> None:
    policy = ScriptedPolicy(*["<search>again</search>"] * 3)

    traj = run_episode(QUESTION, policy, fake_retriever, config(search_budget=2, max_turns=6))

    assert traj.termination is Termination.SEARCH_BUDGET_EXHAUSTED
    assert fake_retriever.n_searches == 2, "the third search is refused, not served"
    assert traj.n_searches == 3, "the refused search is still what the policy did"
    assert traj.answer is None


def test_duplicate_queries_are_served_and_counted(fake_retriever: FakeRetriever) -> None:
    """The duplicate-query penalty is the reward's job (section 6). The loop
    must not quietly dedupe, or the penalty never sees the behaviour."""
    policy = ScriptedPolicy(
        "<search>Ada Lovelace</search>",
        "<search>Ada Lovelace </search>",
        "<answer>Ada Lovelace</answer>",
    )

    traj = run_episode(QUESTION, policy, fake_retriever, config())

    assert fake_retriever.n_searches == 2
    assert traj.n_duplicate_queries == 1


def test_running_out_of_turns_is_max_turns(fake_retriever: FakeRetriever) -> None:
    policy = ScriptedPolicy(*["<think>thinking</think><search>q</search>"] * 2)

    traj = run_episode(QUESTION, policy, fake_retriever, config(search_budget=4, max_turns=2))

    assert traj.termination is Termination.MAX_TURNS
    assert traj.answer is None
    assert len(policy.prompts) == 2


# ---------------------------------------------------------------------------
# What the policy is fed
# ---------------------------------------------------------------------------


def test_each_turn_sees_the_whole_transcript(fake_retriever: FakeRetriever) -> None:
    policy = ScriptedPolicy("<search>Ada Lovelace</search>", "<answer>Ada Lovelace</answer>")

    run_episode(QUESTION, policy, fake_retriever, config())

    first, second = policy.prompts
    assert QUESTION.question in first
    assert second.startswith(first)
    assert "<search>Ada Lovelace</search>" in second
    assert "<info>" in second


def test_overshoot_after_the_search_tag_is_not_fed_back(fake_retriever: FakeRetriever) -> None:
    """A model that writes its own <info> after `</search>` must not see it
    again. Feeding it back teaches the policy that forging evidence works."""
    policy = ScriptedPolicy(
        "<search>Ada Lovelace</search>\n<info>\nfabricated\n</info>",
        "<answer>Ada Lovelace</answer>",
    )

    run_episode(QUESTION, policy, fake_retriever, config())

    assert "fabricated" not in policy.prompts[1]


def test_the_policy_is_given_the_configured_stop_sequences(fake_retriever: FakeRetriever) -> None:
    policy = ScriptedPolicy("<answer>Ada Lovelace</answer>")

    run_episode(QUESTION, policy, fake_retriever, config())

    assert policy.stops == [("</search>",)]


def test_a_stop_set_that_would_corrupt_the_signal_is_refused() -> None:
    with pytest.raises(ProtocolError, match="truncated"):
        run_episode(
            QUESTION,
            ScriptedPolicy("<answer>x</answer>"),
            FakeRetriever(),
            config().model_copy(update={"stop_sequences": ("</search>", "</answer>")}),
        )


# ---------------------------------------------------------------------------
# Token ids: none on the text path, load-bearing on the token path
# ---------------------------------------------------------------------------


def test_the_text_path_carries_text_only(fake_retriever: FakeRetriever) -> None:
    """Even when the policy returns ids: nothing on the text path can tokenise
    the injected blocks to match them, and half-tokenised trajectories are the
    shape `train/masking.py` refuses."""
    policy = ScriptedPolicy(
        "<search>Ada Lovelace</search>", "<answer>Ada Lovelace</answer>", token_ids_per_turn=5
    )

    traj = run_episode(QUESTION, policy, fake_retriever, config())

    assert all(turn.token_ids == () for turn in traj.turns)


def test_on_the_token_path_every_turn_that_has_text_has_ids(
    fake_retriever: FakeRetriever,
) -> None:
    """This is what makes `train/masking.py` a pure function of a trajectory."""
    tokenizer = CharTokenizer()
    policy = ScriptedTokenPolicy("<search>Ada Lovelace</search>", "<answer>Ada Lovelace</answer>")

    episode = run_token_episode(QUESTION, policy, fake_retriever, config(), tokenizer=tokenizer)
    traj = episode.trajectory

    assert [t.kind for t in traj.turns] == [TurnKind.SEARCH, TurnKind.INFO, TurnKind.ANSWER]
    assert traj.turns[0].token_ids == tuple(ord(c) for c in "<search>Ada Lovelace</search>")
    assert traj.turns[1].token_ids == tuple(ord(c) for c in traj.turns[1].text)
    assert traj.turns[2].token_ids == tuple(ord(c) for c in "<answer>Ada Lovelace</answer>")
    assert [g.token_ids for g in episode.generations] == [
        traj.turns[0].token_ids,
        traj.turns[2].token_ids,
    ]


def test_the_prompt_and_each_injected_block_are_encoded_once_and_exactly(
    fake_retriever: FakeRetriever,
) -> None:
    """The ids stored on the INFO turn must be the ids of the text that was
    injected, and the prompt is encoded once, up front. Nothing re-encodes a
    decoded transcript - the failure KICKOFF.md warns about."""
    tokenizer = CharTokenizer()
    policy = ScriptedTokenPolicy("<search>Ada Lovelace</search>", "<answer>Ada</answer>")

    episode = run_token_episode(QUESTION, policy, fake_retriever, config(), tokenizer=tokenizer)

    prompt = render_prompt(QUESTION.question, search_budget=2, top_k=2)
    info_turns = [t for t in episode.trajectory.turns if t.kind is TurnKind.INFO]
    assert tokenizer.calls == [prompt, *(t.text for t in info_turns)]
    assert episode.prompt_ids == tuple(ord(c) for c in prompt)


def test_each_turn_is_sampled_from_the_prompt_plus_every_earlier_turns_ids(
    fake_retriever: FakeRetriever,
) -> None:
    """Finding A on #31, the invariant itself: the context turn k was sampled
    from is exactly the prefix the trainer replays in front of it."""
    policy = ScriptedTokenPolicy(
        "<think>Engine first.</think><search>Analytical Engine</search>",
        "<search>Ada Lovelace notes</search>",
        "<answer>Ada Lovelace</answer>",
    )

    episode = run_token_episode(
        QUESTION, policy, fake_retriever, config(), tokenizer=CharTokenizer()
    )

    turns = episode.trajectory.turns
    acting = [i for i, turn in enumerate(turns) if turn.generated and turn.token_ids]
    assert len(acting) == len(policy.contexts) == 3
    for index, context in zip(acting, policy.contexts, strict=True):
        before = tuple(token for turn in turns[:index] for token in turn.token_ids)
        assert context == episode.prompt_ids + before


def test_overshoot_is_cut_from_both_streams(fake_retriever: FakeRetriever) -> None:
    """A fabricated `<info>` after `</search>` neither reaches the next turn's
    context nor carries loss: the search turn's ids stop where the search did."""
    policy = ScriptedTokenPolicy(
        "<search>Ada Lovelace</search>\n<info>\nfabricated\n</info>",
        "<answer>Ada Lovelace</answer>",
    )
    tokenizer = CharTokenizer()

    episode = run_token_episode(QUESTION, policy, fake_retriever, config(), tokenizer=tokenizer)

    search = episode.trajectory.turns[0]
    assert tokenizer.decode(search.token_ids) == "<search>Ada Lovelace</search>"
    assert episode.generations[0].token_ids == search.token_ids
    assert len(episode.generations[0].logprobs) == len(search.token_ids)
    assert "fabricated" not in tokenizer.decode(policy.contexts[1])


def test_ids_attach_to_the_acting_turn_when_a_turn_has_several(
    fake_retriever: FakeRetriever,
) -> None:
    """A generation covering `<think>` and `<search>` gives one id sequence. It
    lands on the acting turn rather than being split across both, because the
    policy never told us where the boundary is."""
    policy = ScriptedTokenPolicy(
        "<think>Let me look.</think><search>Ada Lovelace</search>",
        "<answer>Ada Lovelace</answer>",
    )

    traj = run_token_episode(
        QUESTION, policy, fake_retriever, config(), tokenizer=CharTokenizer()
    ).trajectory

    assert traj.turns[0].kind is TurnKind.THINK
    assert traj.turns[0].token_ids == ()
    assert CharTokenizer().decode(traj.turns[1].token_ids) == (
        "<think>Let me look.</think><search>Ada Lovelace</search>"
    )


def test_a_token_policy_that_returns_no_tokens_is_refused() -> None:
    class Silent:
        def generate_ids(
            self, context_ids: Sequence[int], *, stop: Sequence[str], max_new_tokens: int
        ) -> Generation:
            return Generation(text="")

    with pytest.raises(EpisodeError, match="no tokens"):
        run_token_episode(QUESTION, Silent(), FakeRetriever(), config(), tokenizer=CharTokenizer())


def test_text_that_is_not_its_ids_is_refused() -> None:
    """The parser reads the text and the trainer reads the ids; if they disagree
    the turn boundaries describe tokens the trainer never sees."""

    class Liar:
        def generate_ids(
            self, context_ids: Sequence[int], *, stop: Sequence[str], max_new_tokens: int
        ) -> Generation:
            return Generation(text="<answer>Ada</answer>", token_ids=(1, 2, 3))

    with pytest.raises(EpisodeError, match="not the decoding"):
        run_token_episode(QUESTION, Liar(), FakeRetriever(), config(), tokenizer=CharTokenizer())


def test_logprobs_that_do_not_line_up_with_the_ids_are_refused() -> None:
    class Short:
        def generate_ids(
            self, context_ids: Sequence[int], *, stop: Sequence[str], max_new_tokens: int
        ) -> Generation:
            text = "<answer>Ada</answer>"
            return Generation(text=text, token_ids=tuple(map(ord, text)), logprobs=(-0.1,))

    with pytest.raises(EpisodeError, match="paired by position"):
        run_token_episode(QUESTION, Short(), FakeRetriever(), config(), tokenizer=CharTokenizer())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_episode_config_comes_from_the_yaml() -> None:
    """`top_k` lives in `retrieval` and the rest in `env`, so the prompt the
    policy reads and the budget the loop enforces cannot disagree."""
    from_yaml = EpisodeConfig.from_config(load_config())

    assert from_yaml.search_budget == 4
    assert from_yaml.max_turns == 6
    assert from_yaml.max_new_tokens == 512
    assert from_yaml.top_k == 3
    # Deliberately not asserting the content of `stop_sequences`: `base.yaml`
    # still lists `</answer>` on this branch and #20 removes it. The guard that
    # matters lives in `check_stop_sequences`, tested in `test_protocol.py`, and
    # `run_episode` calls it before generating a single token.
    assert from_yaml.stop_sequences == load_config().env.stop_sequences


def test_a_generation_that_parses_to_no_turns_still_keeps_its_tokens() -> None:
    """Found by a real smoke run: a 0.5B policy answered in bare prose, the
    generation parsed to no turns, its ids were attached to nothing, and TRL
    raised IndexError on an empty completion (`grpo_trainer.py:2325`).

    Those tokens are what GRPO has to push down, so they carry loss. The kind is
    `UNPARSED` rather than `THINK` (asked for on #2, now in the contract): a dump
    has to show that the policy ignored the format, not that it reasoned and gave
    up, because section 15 has those dumps read by eye.
    """
    policy = ScriptedTokenPolicy("Ada Lovelace, I think.")

    traj = run_token_episode(
        QUESTION, policy, FakeRetriever(), config(), tokenizer=CharTokenizer()
    ).trajectory

    assert traj.termination is Termination.PARSE_ERROR
    assert traj.parse_failure is ParseFailure.NO_ANSWER
    assert [t.kind for t in traj.turns] == [TurnKind.UNPARSED]
    assert traj.turns[0].text == "Ada Lovelace, I think."
    assert traj.turns[0].token_ids == tuple(ord(c) for c in "Ada Lovelace, I think.")
    assert traj.turns[0].generated is True


def test_a_generation_with_no_text_and_no_tokens_adds_no_turn() -> None:
    """The degenerate case stays empty rather than becoming a blank turn; the
    rollout builder refuses it with its own message."""
    traj = run_episode(QUESTION, ScriptedPolicy(""), FakeRetriever(), config())

    assert traj.turns == ()
    assert traj.termination is Termination.PARSE_ERROR


def test_a_served_question_may_have_no_gold_answers() -> None:
    """The policy server's case (#38): the loop never reads them."""
    served = Question(question_id="s1", question="Who wrote the notes?")

    traj = run_episode(served, ScriptedPolicy("<answer>Ada</answer>"), FakeRetriever(), config())

    assert traj.gold_answers == ()
    assert traj.termination is Termination.ANSWERED
