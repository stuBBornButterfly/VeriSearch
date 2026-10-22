"""Evaluation metrics. Pure functions over primitives.

PROJECT.md section 9. Nothing here takes a `Trajectory`, a model, or a config -
every function takes strings and numbers and returns numbers. Three reasons:

* Dhrubo's correctness reward calls `best_f1` directly. A reward component must
  be pure and deterministic (section 3), so the metric it wraps must be too.
* The same F1 has to serve the reward *and* the reported result. Two
  implementations would drift, and the drift would be invisible.
* It keeps this module independent of the unmerged contract proposal. Assembling
  trajectories into a report belongs in `eval/harness.py`, after `types.py` is
  agreed.

`normalize_answer` follows the official SQuAD v1.1 script exactly, including the
order of operations, and `token_f1` follows the official HotpotQA script
including its yes/no rule. That is not stylistic: a number produced by a
different normalisation is not comparable with published work, and section 2
compares against published baselines.

Owner: Ayesha.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Defaults mirror configs/base.yaml. The harness passes config values through;
# these exist so a function is usable from a notebook without a Config object.
DEFAULT_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95

# The official HotpotQA script refuses partial credit when either side is a
# categorical answer: "yes" overlapping "yes, it was" must not score 0.5.
CATEGORICAL_ANSWERS = frozenset({"yes", "no", "noanswer"})

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
_TITLE_NOISE = re.compile(r"[\s_]+")


class MetricError(Exception):
    """Raised when a metric cannot be computed. PROJECT.md section 3: fail loud."""


class FailureReason(StrEnum):
    """Failure breakdown categories from PROJECT.md section 9.

    Ordered by precedence in `classify_failure`: a trajectory that retrieved
    nothing *and* hit the turn limit is reported as EMPTY_RETRIEVAL, because
    that is the cause and the turn limit is the symptom.
    """

    CORRECT = "correct"
    EMPTY_RETRIEVAL = "empty_retrieval"
    NO_ANSWER = "no_answer"
    TURN_LIMIT = "turn_limit"
    WRONG_ANSWER = "wrong_answer"


# ---------------------------------------------------------------------------
# Answer scoring. SQuAD / HotpotQA convention, deliberately not improved on.
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Official SQuAD v1.1 normalisation.

    Lowercase, strip ASCII punctuation, drop articles, collapse whitespace - in
    that order. The order is part of the definition: removing articles before
    punctuation would leave "the-film" as "the-film" rather than "film".

    Accents are deliberately *not* folded, because the reference script does not
    fold them. `data.datasets.normalize_question` does fold them, because it
    serves duplicate detection where recall matters more than comparability.
    Keep the two apart.
    """
    lowered = text.lower()
    depunctuated = lowered.translate(_PUNCTUATION_TABLE)
    dearticled = _ARTICLES.sub(" ", depunctuated)
    return " ".join(dearticled.split())


def token_f1(prediction: str, gold: str, *, yes_no_strict: bool = True) -> float:
    """Token-level F1 against one gold answer, on normalised tokens.

    `yes_no_strict` implements the official HotpotQA rule: when either side is
    "yes", "no" or "noanswer" and they differ, the score is 0 rather than
    whatever token overlap happens to exist. Without it a policy that always
    answers "yes" farms partial credit on every yes/no question. HotpotQA is the
    primary training set, so this is on by default.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    categorical = pred_norm in CATEGORICAL_ANSWERS or gold_norm in CATEGORICAL_ANSWERS
    if yes_no_strict and categorical and pred_norm != gold_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        # Zero, including when BOTH are empty. The reference script reaches 0
        # here via num_same == 0, and matching it is the point - a differential
        # test against a verbatim transcription caught this returning 1.0.
        # It is also the safe answer: an empty prediction against a gold that
        # normalises away (e.g. "-") would otherwise be free credit.
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def best_f1(prediction: str, golds: Sequence[str], *, yes_no_strict: bool = True) -> float:
    """Max F1 over acceptable answers.

    Max, not mean: the aliases are alternative spellings of one correct answer,
    so matching any of them is correct. Averaging would punish a right answer
    for not being all of its own synonyms at once.
    """
    if not golds:
        raise MetricError("best_f1 needs at least one gold answer")
    return max(token_f1(prediction, gold, yes_no_strict=yes_no_strict) for gold in golds)


def best_exact_match(prediction: str, golds: Sequence[str]) -> float:
    """Max EM over acceptable answers."""
    if not golds:
        raise MetricError("best_exact_match needs at least one gold answer")
    return max(exact_match(prediction, gold) for gold in golds)


class AnswerScore(BaseModel):
    """EM and F1 for one prediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


def score_answer(
    prediction: str | None,
    golds: Sequence[str],
    *,
    yes_no_strict: bool = True,
) -> AnswerScore:
    """Score one prediction. `None` - the model never answered - scores zero.

    An unanswered question is wrong, not missing. Dropping it from the
    denominator would reward a policy for refusing hard questions.
    """
    if prediction is None:
        return AnswerScore(exact_match=0.0, f1=0.0)
    return AnswerScore(
        exact_match=best_exact_match(prediction, golds),
        f1=best_f1(prediction, golds, yes_no_strict=yes_no_strict),
    )


def answer_token_count(prediction: str | None) -> int:
    """Length in normalised tokens. PROJECT.md section 6 lists answer length as
    the canary for the copy-passages-verbatim reward hack."""
    if prediction is None:
        return 0
    return len(normalize_answer(prediction).split())


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


class ConfidenceInterval(BaseModel):
    """A point estimate with a percentile bootstrap interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: float
    low: float
    high: float
    confidence: float = Field(gt=0.0, lt=1.0)
    resamples: int = Field(ge=1)
    n: int = Field(ge=1)

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero.

        For a paired delta this is the question "is this difference real?".
        PROJECT.md section 9: "A 2-point gain on 500 examples is noise." This is
        how that gets checked instead of asserted.
        """
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({pct}% CI, n={self.n})"


def _resampled_means(
    values: NDArray[np.float64],
    resamples: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    return np.asarray(values[indices].mean(axis=1), dtype=np.float64)


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean of per-example scores.

    PROJECT.md section 9: 95% CI, 1000 resamples, on every reported number.

    Seeded, so the same scores give the same interval. An unseeded CI changes
    between two runs of the report and makes a result impossible to check.
    """
    if not values:
        raise MetricError("bootstrap_ci needs at least one value")
    if resamples < 1:
        raise MetricError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise MetricError(f"confidence must be in (0, 1), got {confidence}")

    array = np.asarray(values, dtype=np.float64)
    means = _resampled_means(array, resamples, seed)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])

    return ConfidenceInterval(
        point=float(array.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
        n=array.size,
    )


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """CI for the mean per-example difference `candidate - baseline`.

    Paired, resampling examples rather than the two score lists independently.
    Two systems evaluated on the same questions are correlated; treating them as
    independent inflates the interval and hides real differences.

    This is what the nightly `eval-regression` stage in section 10 needs - "F1
    has not dropped more than 2 points" is only answerable with an interval on
    the *difference*.
    """
    if len(baseline) != len(candidate):
        raise MetricError(
            f"paired bootstrap needs equal-length inputs, got {len(baseline)} and {len(candidate)}"
        )
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_ci(deltas, resamples=resamples, confidence=confidence, seed=seed)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Canonical form for comparing Wikipedia paragraph titles.

    Underscores and spaces are interchangeable in Wikipedia titles, and the
    datasets are not consistent about which they use. Without this, precision@k
    reads as zero on data that is actually correct.
    """
    return _TITLE_NOISE.sub(" ", title).strip().lower()


def precision_at_k(retrieved_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    """Fraction of retrieved paragraphs that are gold. PROJECT.md section 9.

    Duplicates in `retrieved_titles` are kept: retrieving the same paragraph
    three times really is three slots spent, and a policy that does it is
    querying badly. That is what this metric is for.
    """
    if not retrieved_titles:
        raise MetricError("precision_at_k needs at least one retrieved title")
    gold = {normalize_title(t) for t in gold_titles}
    hits = sum(1 for t in retrieved_titles if normalize_title(t) in gold)
    return hits / len(retrieved_titles)


def recall_at_k(retrieved_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    """Fraction of gold paragraphs that were retrieved.

    Not required by section 9, but precision alone cannot distinguish "found one
    of two hops" from "found both" - and multi-hop questions need every hop.
    """
    if not gold_titles:
        raise MetricError("recall_at_k needs at least one gold title")
    gold = {normalize_title(t) for t in gold_titles}
    found = {normalize_title(t) for t in retrieved_titles} & gold
    return len(found) / len(gold)


# ---------------------------------------------------------------------------
# Failure breakdown
# ---------------------------------------------------------------------------


def classify_failure(
    *,
    exact_match_score: float,
    has_answer: bool,
    hit_turn_limit: bool,
    retrieved_count: int,
) -> FailureReason:
    """Bucket one episode. PROJECT.md section 9: wrong / no answer / turn limit /
    empty retrieval.

    Keyword-only, because four positional booleans and numbers at a call site is
    a bug waiting to happen.

    Precedence is cause before symptom: retrieving nothing explains a wrong
    answer, and a wrong answer does not explain retrieving nothing. A correct
    answer is reported as correct even if retrieval was empty - that case is
    real (the model answered from parametric memory) and section 6 lists it as a
    reward hack to watch, so it must not be hidden inside a failure bucket.
    """
    if exact_match_score > 0.0:
        return FailureReason.CORRECT
    if retrieved_count == 0:
        return FailureReason.EMPTY_RETRIEVAL
    if not has_answer:
        return FailureReason.NO_ANSWER
    if hit_turn_limit:
        return FailureReason.TURN_LIMIT
    return FailureReason.WRONG_ANSWER


def failure_breakdown(reasons: Sequence[FailureReason]) -> dict[str, float]:
    """Fractions per category, summing to 1.0. Every category is present, zero
    included - a missing key in a report reads as "not measured"."""
    if not reasons:
        raise MetricError("failure_breakdown needs at least one reason")
    counts = Counter(reasons)
    total = len(reasons)
    return {reason.value: counts[reason] / total for reason in FailureReason}
