"""The evaluation harness: trajectories in, section 9's table out.

PROJECT.md section 9 lists what every run must report and ends with the rule
that governs this module: *"Every reported number carries a CI. A 2-point gain
on 500 examples is noise."* So nothing here returns a bare mean. The per-example
scores are kept, written to JSONL, and aggregated through
`metrics.bootstrap_ci`.

**No generation here.** The harness takes an `EpisodeRunner` - anything that can
turn one example into a `Trajectory`. Dhrubo's episode loop satisfies it, and so
does a fake, which is why every metric in section 9 is testable on this machine
with no GPU and no model. The one thing this module must never do is produce
trajectories itself, because then evaluating the real policy and evaluating a
fake would be two different code paths and only one of them would be tested.

**Metrics that do not apply are `None`, never `0.0`.** The two places it matters:

* retrieval precision for baseline 1, which retrieves nothing by construction.
  Recording `0.0` would put a retrieval-quality number in the table for a
  system with no retrieval, and it would drag the mean of every aggregate that
  column appears in.
* unsupported-claim rate for an episode that never answered. There are no
  claims to check, and `0.0` would make *never answering* the best possible
  hallucination score - which is precisely the degenerate policy section 6
  warns about.

A `ConfidenceInterval` carries its own `n`, so an aggregate computed over a
subset reports the size of that subset rather than hiding it.

Owner: Ayesha.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from verisearch.eval.baselines import (
    BaselineKind,
    EvalContext,
    EvalError,
    JointClaim,
    joint_improvement_claim,
)
from verisearch.eval.metrics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    ConfidenceInterval,
    FailureReason,
    answer_token_count,
    bootstrap_ci,
    classify_failure,
    failure_breakdown,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    score_answer,
)
from verisearch.types import Termination, Trajectory

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from verisearch.data.datasets import QAExample
    from verisearch.types import FaithfulnessScorer

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class EpisodeRunner(Protocol):
    """Turns one example into one trajectory.

    Deliberately the whole of the harness's dependency on generation. Dhrubo's
    episode loop satisfies it; so does a fake built from `make_trajectory`.
    """

    def run(self, example: QAExample) -> Trajectory: ...


class ExampleResult(BaseModel):
    """One row. Written to JSONL, and the unit every aggregate is built from.

    Section 15 requires reading raw trajectories after every significant change,
    and a mean cannot be audited. These rows are what make a reported number
    traceable back to the questions that produced it.
    """

    model_config = _FROZEN

    question_id: str = Field(min_length=1)
    answer: str | None
    exact_match: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    n_searches: int = Field(ge=0)
    n_duplicate_queries: int = Field(ge=0)
    n_retrieved: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    termination: Termination
    failure: FailureReason
    # None when the episode retrieved nothing. See the module docstring.
    precision_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    # None when the dataset ships no supporting titles - Bamboogle does not.
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    # None when the episode never answered, so there are no claims to check.
    unsupported_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class EvalReport(BaseModel):
    """Section 9's evaluation table for one run, every number with an interval."""

    model_config = _FROZEN

    run_name: str = Field(min_length=1)
    kind: BaselineKind
    context: EvalContext

    exact_match: ConfidenceInterval
    f1: ConfidenceInterval
    searches_per_question: ConfidenceInterval
    answer_tokens: ConfidenceInterval
    # Each None when no example in the run had the metric defined.
    unsupported_rate: ConfidenceInterval | None
    precision_at_k: ConfidenceInterval | None
    recall_at_k: ConfidenceInterval | None

    failure_breakdown: dict[str, float]
    # Fraction of episodes that issued at least one repeated query. Feeds the
    # -0.1 duplicate penalty in section 6; a rate that does not fall during
    # training means the penalty is not reaching the policy.
    duplicate_query_episode_rate: float = Field(ge=0.0, le=1.0)
    # The retrieval-cost lambda the evaluated policy was trained with, taken
    # from the checkpoint by whatever ran the policy - never from a flag. None
    # for anything the sweep did not train: the base model and the baselines.
    # `frontier.point_from` refuses a sweep point without it, because a report
    # filed under the wrong lambda draws a smooth frontier that is wrong.
    trained_lambda: float | None = None

    def summary(self) -> str:
        lines = [
            f"{self.run_name} ({self.kind.value}) on {self.context.dataset}/{self.context.split}",
            f"  n={self.context.n_examples}  k={self.context.k}  "
            f"retriever={self.context.retriever_fingerprint}",
        ]
        if self.trained_lambda is not None:
            lines.append(f"  trained at lambda={self.trained_lambda:g}")
        lines += [
            f"  EM                  {self.exact_match}",
            f"  F1                  {self.f1}",
            f"  searches/question   {self.searches_per_question}",
            f"  answer tokens       {self.answer_tokens}",
        ]
        for label, interval in (
            ("unsupported rate   ", self.unsupported_rate),
            ("precision@k        ", self.precision_at_k),
            ("recall@k           ", self.recall_at_k),
        ):
            lines.append(
                f"  {label} {interval}" if interval is not None else f"  {label} n/a for this run"
            )
        lines.append(
            "  failures            "
            + ", ".join(f"{name}={share:.3f}" for name, share in self.failure_breakdown.items())
        )
        return "\n".join(lines)


class EvalRun(BaseModel):
    """A report plus the rows it was computed from. Both, always.

    Keeping them together is what makes a comparison possible: a paired
    bootstrap needs per-example scores, so a report on its own cannot be
    compared with anything.
    """

    model_config = _FROZEN

    report: EvalReport
    results: tuple[ExampleResult, ...]


def score_example(
    example: QAExample,
    trajectory: Trajectory,
    *,
    scorer: FaithfulnessScorer | None = None,
    yes_no_strict: bool = True,
) -> ExampleResult:
    """Score one episode against one example.

    The two identity checks are the reason this takes both. A runner that
    returns a trajectory for a different question - an off-by-one in a batch, a
    cached result, a shuffled queue - produces a row where the prediction is
    scored against another question's gold answers. Every number computes, F1
    comes out near zero, and it reads as a model that cannot answer.
    """
    if trajectory.question_id != example.question_id:
        raise EvalError(
            f"runner returned a trajectory for {trajectory.question_id!r} while evaluating "
            f"{example.question_id!r}. The prediction would be scored against another "
            f"question's gold answers and the result would look like a weak model."
        )
    if tuple(trajectory.gold_answers) != tuple(example.answers):
        raise EvalError(
            f"{example.question_id}: trajectory carries gold answers "
            f"{trajectory.gold_answers!r} but the eval set says {example.answers!r}. "
            f"The runner read a different dataset than this harness is scoring against."
        )

    score = score_answer(trajectory.answer, example.answers, yes_no_strict=yes_no_strict)
    passages = trajectory.retrieved_passages
    titles = [passage.title for passage in passages]

    return ExampleResult(
        question_id=example.question_id,
        answer=trajectory.answer,
        exact_match=score.exact_match,
        f1=score.f1,
        n_searches=trajectory.n_searches,
        n_duplicate_queries=trajectory.n_duplicate_queries,
        n_retrieved=len(passages),
        answer_tokens=answer_token_count(trajectory.answer),
        termination=trajectory.termination,
        failure=classify_failure(
            exact_match_score=score.exact_match,
            has_answer=trajectory.answer is not None,
            hit_turn_limit=trajectory.termination is Termination.MAX_TURNS,
            retrieved_count=len(passages),
        ),
        precision_at_k=precision_at_k(titles, example.supporting_titles) if titles else None,
        recall_at_k=(
            recall_at_k(titles, example.supporting_titles) if example.supporting_titles else None
        ),
        # Scored even when nothing was retrieved: an answer written with no
        # evidence has every claim unsupported, and that is the single most
        # important case this metric exists to catch. Only an episode with no
        # answer at all is skipped.
        unsupported_rate=(
            None
            if trajectory.answer is None or scorer is None
            else scorer.score(trajectory.answer, list(passages)).unsupported_rate
        ),
    )


def _optional_ci(
    values: Sequence[float | None],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> ConfidenceInterval | None:
    """CI over the examples where the metric is defined, or None if none are.

    The subset's size lands in the interval's `n`, so a precision@k computed
    over 180 of 500 examples says 180 rather than looking like a full-set
    number.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    return bootstrap_ci(present, resamples=resamples, confidence=confidence, seed=seed)


def aggregate(
    results: Sequence[ExampleResult],
    *,
    run_name: str,
    kind: BaselineKind,
    context: EvalContext,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> EvalReport:
    """Build section 9's table from scored rows."""
    if not results:
        raise EvalError("cannot aggregate an empty result set")
    if len(results) != context.n_examples:
        raise EvalError(
            f"aggregating {len(results)} results under a context that says "
            f"{context.n_examples} examples - the denominators would be wrong"
        )

    def ci(values: Sequence[float]) -> ConfidenceInterval:
        return bootstrap_ci(values, resamples=resamples, confidence=confidence, seed=seed)

    def optional(values: Sequence[float | None]) -> ConfidenceInterval | None:
        return _optional_ci(values, resamples=resamples, confidence=confidence, seed=seed)

    return EvalReport(
        run_name=run_name,
        kind=kind,
        context=context,
        exact_match=ci([r.exact_match for r in results]),
        f1=ci([r.f1 for r in results]),
        searches_per_question=ci([float(r.n_searches) for r in results]),
        answer_tokens=ci([float(r.answer_tokens) for r in results]),
        unsupported_rate=optional([r.unsupported_rate for r in results]),
        precision_at_k=optional([r.precision_at_k for r in results]),
        recall_at_k=optional([r.recall_at_k for r in results]),
        failure_breakdown=failure_breakdown([r.failure for r in results]),
        duplicate_query_episode_rate=sum(1 for r in results if r.n_duplicate_queries > 0)
        / len(results),
    )


def evaluate(
    runner: EpisodeRunner,
    examples: Sequence[QAExample],
    *,
    run_name: str,
    kind: BaselineKind,
    context: EvalContext,
    scorer: FaithfulnessScorer | None = None,
    yes_no_strict: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> EvalRun:
    """Run every example, score it, and aggregate. The whole eval entry point.

    `context` is passed in rather than derived here, because it has to be built
    once and shared by every baseline in a comparison - deriving it per run
    would let two runs disagree about what they measured and still compare
    cleanly.
    """
    if not examples:
        raise EvalError("cannot evaluate an empty example set")
    if len(examples) != context.n_examples:
        raise EvalError(
            f"{len(examples)} examples but the context says {context.n_examples}; "
            f"one of them is not the eval set this run is supposed to measure"
        )

    results = tuple(
        score_example(example, runner.run(example), scorer=scorer, yes_no_strict=yes_no_strict)
        for example in examples
    )
    report = aggregate(
        results,
        run_name=run_name,
        kind=kind,
        context=context,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return EvalRun(report=report, results=results)


class Comparison(BaseModel):
    """One candidate against one reference, paired per example."""

    model_config = _FROZEN

    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    n_paired: int = Field(ge=1)
    exact_match_delta: ConfidenceInterval
    f1_delta: ConfidenceInterval
    searches_delta: ConfidenceInterval
    # None when the two runs share no example where both answered.
    unsupported_rate_delta: ConfidenceInterval | None
    # The number of examples behind `unsupported_rate_delta`, which is the
    # subset where *both* systems answered - see `compare_runs`.
    n_paired_unsupported: int = Field(ge=0)
    claim: JointClaim | None

    def summary(self) -> str:
        lines = [
            f"{self.candidate} vs {self.reference}, paired on n={self.n_paired}",
            f"  EM delta            {self.exact_match_delta}",
            f"  F1 delta            {self.f1_delta}",
            f"  searches delta      {self.searches_delta}",
        ]
        if self.unsupported_rate_delta is not None:
            lines.append(
                f"  unsupported delta   {self.unsupported_rate_delta} "
                f"(n={self.n_paired_unsupported}, both answered)"
            )
        else:
            lines.append("  unsupported delta   n/a - no example where both answered")
        if self.claim is not None:
            lines.append(self.claim.summary())
        return "\n".join(lines)


def compare_runs(
    reference: EvalRun,
    candidate: EvalRun,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> Comparison:
    """Paired deltas, candidate minus reference, with the headline claim.

    Refuses incomparable runs via `check_comparable`, then checks the rows line
    up by question id. The context fingerprint already covers order, but a
    results file that was sorted after being written would pass that check and
    fail this one.

    **One caveat to carry into the writeup.** The unsupported-rate delta is
    computed over the examples where *both* systems produced an answer. It has
    to be - there is no claim to check against silence - but it means a system
    that answers less often is being compared on the subset it chose to answer.
    `n_paired_unsupported` is reported next to the number so the gap between it
    and `n_paired` is visible rather than implied.
    """
    # Imported here rather than at module scope only to keep the dependency
    # direction obvious: harness depends on baselines, never the reverse.
    from verisearch.eval.baselines import check_comparable

    check_comparable(reference.report.context, candidate.report.context)

    reference_ids = [r.question_id for r in reference.results]
    candidate_ids = [r.question_id for r in candidate.results]
    if reference_ids != candidate_ids:
        mismatch = next(
            (
                i
                for i, (a, b) in enumerate(zip(reference_ids, candidate_ids, strict=True))
                if a != b
            ),
            None,
        )
        raise EvalError(
            f"paired comparison needs the same examples in the same order; they first "
            f"differ at position {mismatch}. Pairing positionally across a reordering "
            f"compares each question against a different one and still returns a "
            f"narrow interval."
        )

    def delta(getter: str) -> ConfidenceInterval:
        return paired_bootstrap_delta(
            [float(getattr(r, getter)) for r in reference.results],
            [float(getattr(r, getter)) for r in candidate.results],
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )

    both_answered = [
        (ref.unsupported_rate, cand.unsupported_rate)
        for ref, cand in zip(reference.results, candidate.results, strict=True)
        if ref.unsupported_rate is not None and cand.unsupported_rate is not None
    ]
    unsupported_delta = (
        paired_bootstrap_delta(
            [ref for ref, _ in both_answered],
            [cand for _, cand in both_answered],
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        if both_answered
        else None
    )

    f1_delta = delta("f1")
    searches_delta = delta("n_searches")
    return Comparison(
        reference=reference.report.run_name,
        candidate=candidate.report.run_name,
        n_paired=len(reference_ids),
        exact_match_delta=delta("exact_match"),
        f1_delta=f1_delta,
        searches_delta=searches_delta,
        unsupported_rate_delta=unsupported_delta,
        n_paired_unsupported=len(both_answered),
        # Without a faithfulness delta there is no claim to make. Returning a
        # two-part claim labelled as the headline result would be the most
        # expensive kind of convenience in this project.
        claim=(
            joint_improvement_claim(
                f1_delta=f1_delta,
                unsupported_rate_delta=unsupported_delta,
                searches_delta=searches_delta,
            )
            if unsupported_delta is not None
            else None
        ),
    )


def write_results(results: Iterable[ExampleResult], path: str | Path) -> int:
    """Write per-example rows to JSONL and return the count.

    One object per line, in run order. Section 15: a reported number has to be
    traceable to the questions behind it, and a mean cannot be audited.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json())
            handle.write("\n")
            count += 1
    if count == 0:
        raise EvalError(f"wrote no results to {target} - an empty eval file is a silent failure")
    return count


def read_results(path: str | Path) -> tuple[ExampleResult, ...]:
    """Read rows back, preserving file order."""
    source = Path(path)
    if not source.is_file():
        raise EvalError(f"no results file at {source}")
    rows: list[ExampleResult] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(ExampleResult.model_validate_json(stripped))
            except ValueError as exc:
                raise EvalError(f"{source}:{number} is not an eval result: {exc}") from exc
    if not rows:
        raise EvalError(f"{source} contains no results")
    return tuple(rows)


def write_report(report: EvalReport, path: str | Path) -> None:
    """Write one report as indented JSON, next to its JSONL rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_run(report_path: str | Path, results_path: str | Path) -> EvalRun:
    """Reassemble a run from disk, re-checking that the two halves belong together.

    Comparing a report against someone else's rows would produce deltas against
    numbers the report never described, so the row count is verified against the
    context rather than assumed.
    """
    source = Path(report_path)
    if not source.is_file():
        raise EvalError(f"no report at {source}")
    try:
        report = EvalReport.model_validate_json(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise EvalError(f"{source} is not an eval report: {exc}") from exc

    results = read_results(results_path)
    if len(results) != report.context.n_examples:
        raise EvalError(
            f"{source} describes {report.context.n_examples} examples but "
            f"{Path(results_path)} holds {len(results)} rows - these two files are not "
            f"from the same run"
        )
    return EvalRun(report=report, results=results)
