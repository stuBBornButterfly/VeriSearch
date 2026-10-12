"""Tests for `verisearch/faithfulness/`.

Core dependencies only. `torch` and `transformers` are in the `[retrieval]`
extra, so `TransformersNliScorer` is not exercised - which is exactly why every
decision that could be silently wrong lives in a pure function.

The label-order tests carry the most weight. An MNLI head emits three logits and
which one means entailment is a property of the checkpoint. Get it backwards and
hallucinated claims score as supported, the faithfulness reward inverts, and the
training signal rewards the behaviour the project exists to reduce - with
nothing in any curve to show for it.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from verisearch.faithfulness.claims import (
    ClaimError,
    extract_claims,
    normalize_claim,
    split_sentences,
    unsupported_rate,
)
from verisearch.faithfulness.nli import (
    ClaimVerdict,
    NliError,
    NliLabel,
    NliSpec,
    TransformersNliScorer,
    best_verdict,
    build_pairs,
    faithfulness_score,
    resolve_label_order,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# The order used by MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.
MORITZ_ORDER = {0: "entailment", 1: "neutral", 2: "contradiction"}
# The order used by facebook/bart-large-mnli and several others. The reverse.
BART_ORDER = {0: "contradiction", 1: "neutral", 2: "entailment"}


def _verdict(claim: str, label: NliLabel, probability: float = 0.9) -> ClaimVerdict:
    return ClaimVerdict(
        claim=claim,
        label=label,
        entailment_probability=probability,
        best_passage_index=0,
    )


# ---------------------------------------------------------------------------
# Label order - the highest-stakes guard in the module
# ---------------------------------------------------------------------------


def test_label_order_is_read_from_the_checkpoint() -> None:
    assert resolve_label_order(MORITZ_ORDER) == (
        NliLabel.ENTAILMENT,
        NliLabel.NEUTRAL,
        NliLabel.CONTRADICTION,
    )


def test_the_opposite_checkpoint_order_is_handled() -> None:
    """Two widely used MNLI checkpoints order their labels in opposite
    directions. A hardcoded order is wrong for one of them and fails silently."""
    assert resolve_label_order(BART_ORDER) == (
        NliLabel.CONTRADICTION,
        NliLabel.NEUTRAL,
        NliLabel.ENTAILMENT,
    )


@pytest.mark.parametrize(
    "id2label",
    [
        pytest.param({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}, id="uppercase"),
        pytest.param({0: "Entailment", 1: " neutral ", 2: "contradiction"}, id="mixed-and-padded"),
        pytest.param({0: "entail", 1: "neutral", 2: "contradict"}, id="short-forms"),
        pytest.param(
            {0: "entailment", 1: "neutral", 2: "not_entailment"}, id="not-entailment-alias"
        ),
    ],
)
def test_label_names_are_matched_tolerantly(id2label: dict[int, str]) -> None:
    assert len(resolve_label_order(id2label)) == 3


def test_opaque_label_names_are_refused_rather_than_guessed() -> None:
    """LABEL_0/1/2 carries no information about which logit is entailment.
    Guessing would reintroduce the exact bug this function prevents."""
    with pytest.raises(NliError, match="unrecognised NLI label"):
        resolve_label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_empty_label_map_is_refused() -> None:
    with pytest.raises(NliError, match="empty id2label"):
        resolve_label_order({})


def test_non_contiguous_indices_are_refused() -> None:
    """Indices must be 0..n-1 because they index into a logit vector."""
    with pytest.raises(NliError, match=r"keys must be 0\.\.2"):
        resolve_label_order({0: "entailment", 1: "neutral", 3: "contradiction"})


def test_duplicate_labels_are_refused() -> None:
    with pytest.raises(NliError, match="two indices to the same label"):
        resolve_label_order({0: "entailment", 1: "entailment", 2: "neutral"})


def test_a_two_class_model_is_refused() -> None:
    """A binary entailment head is not an MNLI head and the aggregation below
    assumes three classes."""
    with pytest.raises(NliError, match="missing"):
        resolve_label_order({0: "entailment", 1: "contradiction"})


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def test_the_passage_is_the_premise_and_the_claim_the_hypothesis() -> None:
    """NLI is directional. Reversed, this asks whether the claim supports the
    evidence - a different question that produces plausible numbers."""
    pairs = build_pairs("Weir was born in Sydney.", ["Peter Weir was born in Sydney in 1944."])
    premise, hypothesis = pairs[0]

    assert premise.startswith("Peter Weir was born")
    assert hypothesis == "Weir was born in Sydney."


def test_one_pair_per_passage() -> None:
    """Each claim is checked against each passage separately rather than against
    one concatenated premise, which would overrun max_length and dilute a claim
    that only passage three supports."""
    pairs = build_pairs("A claim.", ["p1", "p2", "p3"])
    assert [premise for premise, _ in pairs] == ["p1", "p2", "p3"]
    assert {hypothesis for _, hypothesis in pairs} == {"A claim."}


def test_an_empty_claim_is_refused() -> None:
    with pytest.raises(NliError, match="empty claim"):
        build_pairs("   ", ["p1"])


def test_zero_passages_is_refused_rather_than_scored() -> None:
    """An answer with no retrieved evidence is a distinct case - it is the
    answer-from-parametric-memory hack - and must not quietly score 0.0."""
    with pytest.raises(NliError, match="zero passages"):
        build_pairs("A claim.", [])


# ---------------------------------------------------------------------------
# Picking the best passage
# ---------------------------------------------------------------------------


def test_best_verdict_uses_the_models_own_label_order() -> None:
    """Same probabilities, opposite label orders, opposite verdicts. This is the
    silent catastrophe the whole module is arranged around."""
    order_a = resolve_label_order(MORITZ_ORDER)
    order_b = resolve_label_order(BART_ORDER)
    probabilities = [[0.9, 0.05, 0.05]]

    assert best_verdict("c", probabilities, order_a, threshold=0.5).entailed
    assert not best_verdict("c", probabilities, order_b, threshold=0.5).entailed


def test_the_highest_entailment_probability_wins() -> None:
    """A claim supported by any single retrieved passage is supported. Requiring
    agreement across passages would punish a correct answer for retrieving a
    diverse top-k."""
    order = resolve_label_order(MORITZ_ORDER)
    probabilities = [
        [0.10, 0.80, 0.10],
        [0.85, 0.10, 0.05],
        [0.20, 0.70, 0.10],
    ]
    verdict = best_verdict("c", probabilities, order, threshold=0.5)

    assert verdict.entailed
    assert verdict.best_passage_index == 1
    assert verdict.entailment_probability == pytest.approx(0.85)


def test_below_threshold_reports_the_actual_argmax() -> None:
    """Section 9 counts contradiction and mere absence of support both as
    unsupported, but they mean different things when reading trajectories by
    hand - so the distinction is kept."""
    order = resolve_label_order(MORITZ_ORDER)

    contradicted = best_verdict("c", [[0.1, 0.2, 0.7]], order, threshold=0.5)
    unsupported = best_verdict("c", [[0.1, 0.7, 0.2]], order, threshold=0.5)

    assert contradicted.label is NliLabel.CONTRADICTION
    assert unsupported.label is NliLabel.NEUTRAL
    assert not contradicted.entailed
    assert not unsupported.entailed


@pytest.mark.parametrize(
    ("probability", "threshold", "entailed"),
    [
        pytest.param(0.50, 0.5, True, id="exactly-at-threshold-counts"),
        pytest.param(0.49, 0.5, False, id="just-below"),
        pytest.param(0.80, 0.9, False, id="high-but-strict-threshold"),
        pytest.param(0.95, 0.9, True, id="clears-strict-threshold"),
    ],
)
def test_threshold_is_inclusive(probability: float, threshold: float, entailed: bool) -> None:
    order = resolve_label_order(MORITZ_ORDER)
    row = [probability, (1 - probability) / 2, (1 - probability) / 2]
    assert best_verdict("c", [row], order, threshold=threshold).entailed is entailed


def test_best_verdict_rejects_no_probabilities() -> None:
    with pytest.raises(NliError, match="no probabilities"):
        best_verdict("c", [], resolve_label_order(MORITZ_ORDER), threshold=0.5)


def test_best_verdict_rejects_a_wrong_width_row() -> None:
    """A row that is not one probability per class means the model and the label
    order disagree, which is not recoverable."""
    with pytest.raises(NliError, match="expected 3"):
        best_verdict("c", [[0.5, 0.5]], resolve_label_order(MORITZ_ORDER), threshold=0.5)


def test_best_verdict_rejects_an_incomplete_label_order() -> None:
    with pytest.raises(NliError, match="must cover all 3"):
        best_verdict("c", [[0.5, 0.5]], [NliLabel.ENTAILMENT, NliLabel.NEUTRAL], threshold=0.5)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param([NliLabel.ENTAILMENT], 1.0, id="all-entailed"),
        pytest.param([NliLabel.NEUTRAL], 0.0, id="none-entailed"),
        pytest.param([NliLabel.ENTAILMENT, NliLabel.NEUTRAL], 0.5, id="half"),
        pytest.param(
            [NliLabel.ENTAILMENT, NliLabel.NEUTRAL, NliLabel.CONTRADICTION],
            pytest.approx(1 / 3),
            id="contradiction-counts-as-unsupported",
        ),
    ],
)
def test_faithfulness_score(labels: Sequence[NliLabel], expected: float) -> None:
    verdicts = [_verdict(f"c{i}", label) for i, label in enumerate(labels)]
    assert faithfulness_score(verdicts) == expected


def test_faithfulness_score_with_no_claims_is_an_open_decision_not_a_default() -> None:
    """Baking a value in here would quietly settle a reward-design decision that
    belongs to both owners - it is DECIDE #4 on the types proposal."""
    with pytest.raises(NliError, match="undefined with no claims"):
        faithfulness_score([])


def test_unsupported_rate_is_the_complement() -> None:
    assert unsupported_rate([True, False]) == 0.5
    assert unsupported_rate([True, True]) == 0.0
    assert unsupported_rate([False, False]) == 1.0


def test_unsupported_rate_with_no_claims_is_an_error() -> None:
    """ "No claims" and "no unsupported claims" are different states. Conflating
    them reports a model that answers nothing as perfectly faithful."""
    with pytest.raises(ClaimError, match="undefined with no claims"):
        unsupported_rate([])


# ---------------------------------------------------------------------------
# NliSpec fingerprint
# ---------------------------------------------------------------------------


def test_spec_fingerprint_is_stable() -> None:
    assert NliSpec().fingerprint == NliSpec().fingerprint
    assert len(NliSpec().fingerprint) == 12


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"model_name": "other/model"}, id="model"),
        pytest.param({"entailment_threshold": 0.7}, id="threshold"),
        pytest.param({"max_length": 256}, id="max-length"),
    ],
)
def test_spec_fingerprint_changes_with_the_verdicts(change: dict[str, object]) -> None:
    """A faithfulness number is only comparable with another from the same
    scorer at the same threshold."""
    assert NliSpec.model_validate(change).fingerprint != NliSpec().fingerprint


def test_spec_fingerprint_ignores_batch_size() -> None:
    assert NliSpec(batch_size=4).fingerprint == NliSpec(batch_size=64).fingerprint


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def test_a_short_answer_is_one_claim() -> None:
    """Multi-hop answers are usually a name or a date."""
    assert extract_claims("Peter Weir") == ("Peter Weir",)


def test_multiple_sentences_become_multiple_claims() -> None:
    answer = "Peter Weir directed Gallipoli. He was born in Sydney."
    assert extract_claims(answer) == (
        "Peter Weir directed Gallipoli.",
        "He was born in Sydney.",
    )


@pytest.mark.parametrize(
    ("answer", "expected_count"),
    [
        pytest.param("Dr. Weir directed it.", 1, id="title-abbreviation"),
        pytest.param("It grossed 3.5 million dollars.", 1, id="decimal"),
        pytest.param("He worked in the U.S. and in Australia.", 1, id="dotted-initialism"),
        pytest.param("Born in St. Louis, Missouri.", 1, id="place-abbreviation"),
        pytest.param("E. B. White wrote it.", 1, id="initials"),
        pytest.param("Films, e.g. Gallipoli, did well.", 1, id="e-g"),
    ],
)
def test_abbreviations_do_not_split_sentences(answer: str, expected_count: int) -> None:
    assert len(extract_claims(answer)) == expected_count


def test_question_and_exclamation_marks_also_end_sentences() -> None:
    assert len(extract_claims("Who directed it? Peter Weir did!")) == 2


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(None, id="no-answer"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("...", id="punctuation-only"),
        pytest.param("!!", id="short-punctuation"),
    ],
)
def test_nothing_checkable_gives_no_claims(answer: str | None) -> None:
    """An empty tuple is a real and distinct state, not an error."""
    assert extract_claims(answer) == ()


def test_duplicate_sentences_are_dropped() -> None:
    """Section 6 expects "copy passages verbatim to farm faithfulness".
    Repeating one supported sentence would otherwise drive the entailed fraction
    towards 1.0 for free."""
    answer = "Weir directed it. Weir directed it. WEIR DIRECTED IT."
    assert extract_claims(answer) == ("Weir directed it.",)


def test_claims_are_capped() -> None:
    """An answer with hundreds of sentences is degenerate repetition, not an
    answer worth scoring exhaustively - and each claim costs NLI calls across
    thousands of trajectories."""
    answer = " ".join(f"Fact number {i} is true." for i in range(100))
    assert len(extract_claims(answer, max_claims=5)) == 5


def test_max_claims_must_be_positive() -> None:
    with pytest.raises(ClaimError, match="max_claims must be"):
        extract_claims("A sentence.", max_claims=0)


def test_whitespace_inside_a_claim_is_collapsed() -> None:
    assert extract_claims("Peter    Weir\n\ndirected  it.") == ("Peter Weir directed it.",)


def test_normalize_claim_collapses_all_whitespace() -> None:
    assert normalize_claim("  a \t b \n c  ") == "a b c"


def test_split_sentences_is_usable_on_its_own() -> None:
    """Exposed separately because the sentence boundary rules are the part most
    likely to need adjusting against real answers."""
    assert split_sentences("One. Two.") == ["One.", "Two."]


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------


def test_missing_transformers_says_how_to_fix_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    scorer = TransformersNliScorer()
    with pytest.raises(NliError, match=r"uv pip install -e '\.\[retrieval\]'"):
        _ = scorer.label_order


def test_scorer_rejects_an_empty_claim_list() -> None:
    """Checked before the model is touched, so it needs no extra."""
    with pytest.raises(NliError, match="no claims to check"):
        TransformersNliScorer().check_claims([], ["a passage"])


def test_check_claims_reports_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real entry point fails the same way as `label_order`, with an
    actionable message rather than a bare ImportError."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    scorer = TransformersNliScorer()
    with pytest.raises(NliError, match="retrieval extra"):
        scorer.check_claims(["Weir directed it."], ["Peter Weir directed Gallipoli."])


def test_missing_torch_is_also_reported_as_an_nli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transformers present but torch absent is an odd install, and it must not
    surface as a bare ImportError from deep inside a batch loop."""
    monkeypatch.setitem(sys.modules, "torch", None)
    scorer = TransformersNliScorer()
    with pytest.raises(NliError, match="retrieval extra"):
        scorer._probabilities([("a premise", "a claim")])


def test_second_load_is_a_no_op_once_a_model_is_set() -> None:
    """Guards the caching branch without needing a real model."""
    scorer = TransformersNliScorer()
    scorer._model = object()  # type: ignore[assignment]
    scorer._load()
    assert scorer._model is not None
