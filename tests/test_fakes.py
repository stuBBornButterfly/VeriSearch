"""Tests for the contract fakes.

A fake is test infrastructure, which is exactly why it needs tests of its own:
every assertion the RL half writes rests on these behaving as documented, and a
fake that quietly changes its ranking or its labelling rule invalidates results
in a half of the codebase its author never looks at.
"""

from __future__ import annotations

import pytest

from verisearch.testing import (
    DEFAULT_CORPUS,
    FakeFaithfulnessScorer,
    FakeRetriever,
    conformance_witnesses,
    make_trajectory,
)
from verisearch.types import (
    Entailment,
    FaithfulnessScorer,
    Passage,
    Retriever,
    Termination,
    TurnKind,
)

# ---------------------------------------------------------------------------
# FakeRetriever
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, 1),
        (3, 3),
        (5, 5),
        (99, 5),  # Never more than the corpus holds.
    ],
)
def test_search_returns_at_most_k(fake_retriever: FakeRetriever, k: int, expected: int) -> None:
    assert len(fake_retriever.search("Ada Lovelace", k)) == expected


@pytest.mark.parametrize("k", [0, -1, -100])
def test_search_rejects_non_positive_k(fake_retriever: FakeRetriever, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        fake_retriever.search("Ada Lovelace", k)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_blank_query_returns_nothing(fake_retriever: FakeRetriever, query: str) -> None:
    """The parser owns ParseFailure.EMPTY_QUERY. The retriever just has no hits."""
    assert fake_retriever.search(query, 3) == []


def test_blank_query_is_still_recorded(fake_retriever: FakeRetriever) -> None:
    """A wasted search still costs retrieval budget, so it must show up in `calls`."""
    fake_retriever.search("", 3)
    assert fake_retriever.n_searches == 1


def test_blind_to_forces_the_zero_result_path(blind_retriever: FakeRetriever) -> None:
    assert blind_retriever.search("no such thing", 3) == []
    assert blind_retriever.search("Ada Lovelace", 3) != []


def test_blind_to_is_case_and_whitespace_insensitive(blind_retriever: FakeRetriever) -> None:
    assert blind_retriever.search("  NO SUCH THING  ", 3) == []


def test_relevant_query_ranks_the_relevant_passage_first(fake_retriever: FakeRetriever) -> None:
    hits = fake_retriever.search("Ada Lovelace mathematician", 3)
    assert hits[0].doc_id == "d1"


def test_off_topic_query_still_returns_k_results(fake_retriever: FakeRetriever) -> None:
    """A dense index always returns its nearest neighbours, however bad.

    This is the property that keeps the fake honest: an episode loop tested
    against a retriever that returns nothing for a bad query never learns to
    handle irrelevant context, which is the common case in real retrieval.
    """
    hits = fake_retriever.search("quantum chromodynamics lattice", 3)
    assert len(hits) == 3
    assert all(hit.score == 0.0 for hit in hits)


def test_scores_are_descending_and_in_unit_range(fake_retriever: FakeRetriever) -> None:
    hits = fake_retriever.search("English mathematician Analytical Engine", 5)
    scores = [hit.score for hit in hits]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores if score is not None)


def test_search_is_deterministic(fake_retriever: FakeRetriever) -> None:
    first = fake_retriever.search("Charles Babbage computer", 4)
    second = fake_retriever.search("Charles Babbage computer", 4)
    assert first == second


def test_ties_break_on_doc_id_not_corpus_order() -> None:
    """Stable ordering under equal scores, so a reordering is a real signal."""
    corpus = (
        Passage(doc_id="zebra", title="Same", text="identical text"),
        Passage(doc_id="alpha", title="Same", text="identical text"),
    )
    retriever = FakeRetriever(corpus)
    assert [hit.doc_id for hit in retriever.search("identical", 2)] == ["alpha", "zebra"]


def test_calls_are_recorded_in_order_with_duplicates(fake_retriever: FakeRetriever) -> None:
    """`n_duplicate_queries` in Trajectory is a reward penalty. This is how it gets tested."""
    for query in ("one", "two", "one"):
        fake_retriever.search(query, 2)
    assert fake_retriever.queries == ("one", "two", "one")
    assert fake_retriever.calls == [("one", 2), ("two", 2), ("one", 2)]
    assert fake_retriever.n_searches == 3


def test_reset_clears_calls_but_not_the_corpus(fake_retriever: FakeRetriever) -> None:
    fake_retriever.search("Ada Lovelace", 2)
    fake_retriever.reset()
    assert fake_retriever.n_searches == 0
    assert fake_retriever.search("Ada Lovelace", 2) != []


def test_default_corpus_contains_distractors(fake_corpus: tuple[Passage, ...]) -> None:
    """Without these, `unsupported_rate` cannot be non-zero and proves nothing."""
    assert {"d4", "d5"} <= {passage.doc_id for passage in fake_corpus}


def test_custom_corpus_replaces_the_default() -> None:
    corpus = (Passage(doc_id="only", title="Only", text="the only passage"),)
    assert FakeRetriever(corpus).search("passage", 5) == [
        corpus[0].model_copy(update={"score": 1.0})
    ]


# ---------------------------------------------------------------------------
# FakeFaithfulnessScorer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Ada Lovelace was an English mathematician.", Entailment.ENTAILED),
        ("Ada Lovelace was not an English mathematician.", Entailment.CONTRADICTED),
        ("Paris is the capital of France.", Entailment.NEUTRAL),
    ],
)
def test_label_for_a_single_claim(
    fake_scorer: FakeFaithfulnessScorer,
    answer: str,
    expected: Entailment,
) -> None:
    result = fake_scorer.score(answer, list(DEFAULT_CORPUS))
    assert len(result.claims) == 1
    assert result.claims[0].label is expected


def test_supporting_doc_ids_only_on_entailed_claims(fake_scorer: FakeFaithfulnessScorer) -> None:
    passages = list(DEFAULT_CORPUS)
    entailed = fake_scorer.score("Ada Lovelace was an English mathematician.", passages)
    assert entailed.claims[0].supporting_doc_ids == ("d1",)

    contradicted = fake_scorer.score("Ada Lovelace was not an English mathematician.", passages)
    assert contradicted.claims[0].supporting_doc_ids == ()

    neutral = fake_scorer.score("Paris is the capital of France.", passages)
    assert neutral.claims[0].supporting_doc_ids == ()


def test_supporting_doc_ids_are_sorted() -> None:
    """Reproducible, not dict-insertion-ordered."""
    corpus = [
        Passage(doc_id="zz", title="Topic", text="alpha beta gamma"),
        Passage(doc_id="aa", title="Topic", text="alpha beta gamma"),
    ]
    result = FakeFaithfulnessScorer().score("alpha beta gamma", corpus)
    assert result.claims[0].supporting_doc_ids == ("aa", "zz")


def test_multi_sentence_answer_is_scored_per_sentence(
    fake_scorer: FakeFaithfulnessScorer,
) -> None:
    answer = "Ada Lovelace was an English mathematician. Paris is the capital of France."
    result = fake_scorer.score(answer, list(DEFAULT_CORPUS))
    assert [claim.label for claim in result.claims] == [
        Entailment.ENTAILED,
        Entailment.NEUTRAL,
    ]
    assert result.score == pytest.approx(0.5)


def test_no_passages_means_nothing_is_supported(fake_scorer: FakeFaithfulnessScorer) -> None:
    result = fake_scorer.score("Ada Lovelace was an English mathematician.", [])
    assert result.claims[0].label is Entailment.NEUTRAL
    assert result.unsupported_rate == pytest.approx(1.0)


def test_score_is_the_complement_of_unsupported_rate(
    fake_scorer: FakeFaithfulnessScorer,
) -> None:
    """The reward reads `unsupported_rate`; W&B logs `score`. They must agree."""
    answer = (
        "Ada Lovelace was an English mathematician. "
        "The Analytical Engine was designed by Charles Babbage. "
        "Paris is the capital of France."
    )
    result = fake_scorer.score(answer, list(DEFAULT_CORPUS))
    assert result.score == pytest.approx(1.0 - result.unsupported_rate)


@pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
def test_empty_answer_yields_no_claims(fake_scorer: FakeFaithfulnessScorer, answer: str) -> None:
    result = fake_scorer.score(answer, list(DEFAULT_CORPUS))
    assert result.claims == ()
    assert result.score == pytest.approx(0.0)


@pytest.mark.parametrize("empty_score", [0.0, 0.5, 1.0])
def test_empty_claims_score_is_configurable(empty_score: float) -> None:
    """The open DECIDE in types.py, exposed as a knob instead of pre-decided."""
    scorer = FakeFaithfulnessScorer(empty_claims_score=empty_score)
    assert scorer.score("", list(DEFAULT_CORPUS)).score == pytest.approx(empty_score)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"empty_claims_score": -0.1}, "empty_claims_score"),
        ({"empty_claims_score": 1.5}, "empty_claims_score"),
        ({"support_threshold": 0.0}, "support_threshold"),
        ({"support_threshold": 1.1}, "support_threshold"),
    ],
)
def test_scorer_rejects_out_of_range_settings(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FakeFaithfulnessScorer(**kwargs)


def test_lower_threshold_entails_more() -> None:
    answer = "Ada Lovelace published notes about a general-purpose computing machine."
    passages = list(DEFAULT_CORPUS)
    strict = FakeFaithfulnessScorer(support_threshold=1.0).score(answer, passages)
    lenient = FakeFaithfulnessScorer(support_threshold=0.2).score(answer, passages)
    assert strict.unsupported_rate >= lenient.unsupported_rate


# ---------------------------------------------------------------------------
# make_trajectory
# ---------------------------------------------------------------------------


def test_default_trajectory_is_think_then_answer() -> None:
    traj = make_trajectory()
    assert [turn.kind for turn in traj.turns] == [TurnKind.THINK, TurnKind.ANSWER]
    assert traj.termination is Termination.ANSWERED
    assert traj.n_searches == 0


def test_searches_expand_to_search_info_pairs() -> None:
    traj = make_trajectory(searches=("first", "second"))
    assert [turn.kind for turn in traj.turns] == [
        TurnKind.THINK,
        TurnKind.SEARCH,
        TurnKind.INFO,
        TurnKind.SEARCH,
        TurnKind.INFO,
        TurnKind.ANSWER,
    ]
    assert traj.search_queries == ("first", "second")
    assert traj.n_searches == 2


def test_duplicate_queries_are_visible_to_the_penalty() -> None:
    traj = make_trajectory(searches=("same", "same", "other"))
    assert traj.n_duplicate_queries == 1


def test_no_answer_defaults_to_max_turns_termination() -> None:
    traj = make_trajectory(answer=None)
    assert traj.answer is None
    assert traj.termination is Termination.MAX_TURNS
    assert TurnKind.ANSWER not in [turn.kind for turn in traj.turns]


def test_explicit_termination_overrides_the_default() -> None:
    traj = make_trajectory(answer=None, termination=Termination.SEARCH_BUDGET_EXHAUSTED)
    assert traj.termination is Termination.SEARCH_BUDGET_EXHAUSTED


def test_retrieved_passages_flattens_across_info_turns(fake_retriever: FakeRetriever) -> None:
    first = fake_retriever.search("Ada Lovelace", 2)
    second = fake_retriever.search("Charles Babbage", 2)
    traj = make_trajectory(
        searches=("Ada Lovelace", "Charles Babbage"),
        passages_per_search=(first, second),
    )
    assert len(traj.retrieved_passages) == 4


def test_passages_attach_to_info_turns_not_search_turns(fake_retriever: FakeRetriever) -> None:
    hits = fake_retriever.search("Ada Lovelace", 2)
    traj = make_trajectory(searches=("Ada Lovelace",), passages_per_search=(hits,))
    by_kind = {turn.kind: turn for turn in traj.turns}
    assert by_kind[TurnKind.SEARCH].passages == ()
    assert len(by_kind[TurnKind.INFO].passages) == 2


def test_passages_per_search_length_must_match() -> None:
    with pytest.raises(ValueError, match="passages_per_search has 1 entries for 2 searches"):
        make_trajectory(searches=("a", "b"), passages_per_search=([],))


def test_negative_tokens_per_turn_is_rejected() -> None:
    with pytest.raises(ValueError, match="tokens_per_turn must be non-negative"):
        make_trajectory(tokens_per_turn=-1)


def test_token_ids_are_empty_by_default() -> None:
    assert all(turn.token_ids == () for turn in make_trajectory(searches=("q",)).turns)


def test_token_ids_are_distinct_per_turn() -> None:
    traj = make_trajectory(searches=("q",), tokens_per_turn=4)
    all_ids = [token for turn in traj.turns for token in turn.token_ids]
    assert len(all_ids) == len(set(all_ids))
    assert len(all_ids) == 4 * len(traj.turns)


def test_masking_is_a_pure_function_of_the_trajectory() -> None:
    """The argument for `Turn.token_ids`, written out as a test.

    PROJECT.md section 3 makes `tests/test_masking.py` mandatory and blocking.
    This is that test's shape: injected INFO tokens carry zero loss weight,
    everything the policy generated carries one. No tokenizer, no forward pass,
    no GPU - which is the only reason a test this important will still be run in
    six weeks.
    """
    traj = make_trajectory(searches=("Ada Lovelace", "Charles Babbage"), tokens_per_turn=3)

    weights = [
        (token, 1.0 if turn.generated else 0.0) for turn in traj.turns for token in turn.token_ids
    ]
    injected = {
        token for turn in traj.turns if turn.kind is TurnKind.INFO for token in turn.token_ids
    }

    assert injected, "the INFO turns must contribute tokens or this test proves nothing"
    assert all(weight == 0.0 for token, weight in weights if token in injected)
    assert all(weight == 1.0 for token, weight in weights if token not in injected)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fakes_satisfy_the_protocols() -> None:
    """mypy enforces this statically; the runtime call proves the objects work."""
    retriever, scorer = conformance_witnesses()
    passages = retriever.search("Ada Lovelace", 2)
    assert len(passages) == 2
    assert scorer.score("Ada Lovelace was an English mathematician.", passages).claims


def test_fakes_are_usable_through_the_protocol_types(
    fake_retriever: FakeRetriever,
    fake_scorer: FakeFaithfulnessScorer,
) -> None:
    """Assigning to the protocol-typed names is what mypy checks on this branch."""
    retriever: Retriever = fake_retriever
    scorer: FaithfulnessScorer = fake_scorer
    assert retriever.search("mechanical general-purpose computer", 1)[0].doc_id == "d2"
    assert scorer.score("nothing relevant here", []).unsupported_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Documented limits of lexical matching
# ---------------------------------------------------------------------------


def test_mentioning_a_term_scores_the_same_as_being_about_it(
    fake_retriever: FakeRetriever,
) -> None:
    """A caveat for anyone writing a test against this fake.

    "Analytical Engine" appears in full in both the passage *about* the engine
    and the passage about Lovelace that merely mentions it, so token overlap
    scores them identically and `doc_id` decides the order. A dense retriever
    would separate them. Do not write a test that depends on which one comes
    back first - query on terms unique to the passage you mean, as
    `test_fakes_are_usable_through_the_protocol_types` does.
    """
    hits = fake_retriever.search("Analytical Engine", 2)
    assert {hit.doc_id for hit in hits} == {"d1", "d2"}
    assert [hit.score for hit in hits] == [1.0, 1.0]


@pytest.mark.parametrize("query", ["...", "???", "-- -- --"])
def test_query_with_no_word_characters_scores_zero(
    fake_retriever: FakeRetriever, query: str
) -> None:
    """Not blank, so it is a real search that burns budget - and matches nothing."""
    hits = fake_retriever.search(query, 2)
    assert len(hits) == 2
    assert all(hit.score == 0.0 for hit in hits)


def test_punctuation_only_sentences_are_not_claims(fake_scorer: FakeFaithfulnessScorer) -> None:
    """A trailing "..." must not land in the denominator of the headline metric."""
    result = fake_scorer.score(
        "Ada Lovelace was an English mathematician. ...", list(DEFAULT_CORPUS)
    )
    assert len(result.claims) == 1
    assert result.unsupported_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FakeRetriever.search_batch - the protocol grew, so the fake did
# ---------------------------------------------------------------------------


def test_the_fake_batches_in_order_and_agrees_with_search(
    fake_retriever: FakeRetriever,
) -> None:
    """A fake that disagreed with the real retriever on ordering would let the
    episode loop pass its tests and mis-attribute passages against the service."""
    queries = ["Analytical Engine", "Ada Lovelace", "nothing relevant here"]

    batched = fake_retriever.search_batch(queries, 2)
    fake_retriever.calls.clear()
    looped = [fake_retriever.search(query, 2) for query in queries]

    assert [[p.doc_id for p in row] for row in batched] == [
        [p.doc_id for p in row] for row in looped
    ]


def test_the_fake_records_every_query_in_a_batch(fake_retriever: FakeRetriever) -> None:
    """`calls` is how a test asserts the episode loop batched rather than looped."""
    fake_retriever.search_batch(["one", "two"], 3)
    assert fake_retriever.calls == [("one", 3), ("two", 3)]


@pytest.mark.parametrize(
    ("queries", "k"),
    [
        pytest.param(["a"], 0, id="zero-k"),
        pytest.param(["a"], -1, id="negative-k"),
        pytest.param([], 1, id="no-queries"),
    ],
)
def test_the_fake_refuses_bad_batch_input(
    fake_retriever: FakeRetriever, queries: list[str], k: int
) -> None:
    """Loud in the fake, so a caller bug is found in a unit test rather than
    against a live service at group 8."""
    with pytest.raises(ValueError, match=r"k must be positive|at least one query"):
        fake_retriever.search_batch(queries, k)
