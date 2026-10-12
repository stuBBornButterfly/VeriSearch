"""The accuracy-vs-retrieval frontier. The deliverable the lambda sweep exists for.

PROJECT.md section 2: *"Lambda sweep of at least 4 points, plotted as an accuracy
vs retrieval-count frontier."* That plot is the project's one original figure, and
everything in `configs/sweep_lambda.yaml`, every GPU hour of the sweep, and the
whole retrieval-cost term exist to produce it.

Which is why almost none of this file is plotting. The plot is four lines at the
bottom, deferred behind the `[plot]` extra and never run in CI. Everything above
it is the part that can be wrong in a way nobody notices:

* **a missing sweep point.** The manifest asks for four lambdas; three runs
  finished. Three points draw a perfectly plausible curve and the figure caption
  still says "four-point sweep". `load_sweep_results` checks the manifest and
  names the lambdas it could not find.
* **a frontier drawn across two experiments.** One run in the sweep used a
  rebuilt index, or a different eval set. Each report is internally valid, the
  curve is smooth, and the trade-off it shows is between two different things.
  `check_comparable` from `baselines.py` is applied pairwise across every point.
* **a sweep where the penalty never reached the policy.** If searches per
  question do not fall as lambda rises, there is no trade-off to plot - the
  reward term is not working, and that is a bug to fix rather than a frontier to
  publish. `diagnose` says so before the figure gets drawn.
* **dominance read off point estimates.** A point with 0.4 more F1 at the same
  search count is not better if the intervals overlap. `pareto_frontier` marks
  those as contested rather than quietly dropping the dominated one.

Owner: Ayesha.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from verisearch.config import sweep_run_name
from verisearch.eval.baselines import EvalError, check_comparable
from verisearch.eval.harness import EvalReport
from verisearch.eval.metrics import ConfidenceInterval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from verisearch.config import SweepSpec

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# Filenames `evaluate` + `write_report` / `write_results` produce per run. The
# sweep's layout is `<output_dir>/<run_name>/`, one directory per lambda.
REPORT_FILENAME = "report.json"
RESULTS_FILENAME = "results.jsonl"


class FrontierPoint(BaseModel):
    """One lambda's run, reduced to the two axes plus what qualifies them.

    `unsupported_rate` is carried even though it is not an axis: a point that
    buys accuracy at fewer searches while hallucinating more is not on the
    frontier this project is claiming, and the only way to see that is to have
    the number next to the other two.
    """

    model_config = _FROZEN

    lambda_value: float = Field(ge=0.0)
    run_name: str = Field(min_length=1)
    accuracy: ConfidenceInterval
    searches: ConfidenceInterval
    unsupported_rate: ConfidenceInterval | None

    @property
    def label(self) -> str:
        return f"λ={self.lambda_value:g}"


class Dominance(BaseModel):
    """Why one point is not on the frontier, and how confidently.

    `contested` is the field that matters. A point dominated on point estimates
    alone, with overlapping intervals, is not reliably dominated - and dropping
    it from the figure would be presenting a conclusion the data does not carry.
    """

    model_config = _FROZEN

    run_name: str = Field(min_length=1)
    dominated_by: str = Field(min_length=1)
    contested: bool
    note: str = Field(min_length=1)


class Frontier(BaseModel):
    """Every point in lambda order, plus which ones are dominated and why.

    `points` holds the whole sweep, not just the survivors. The figure draws all
    of them and marks the losers, because a frontier plot that silently omits
    the settings that lost is a figure that cannot be argued with - and because
    "which lambdas did you try" is the first question anyone will ask about it.
    """

    model_config = _FROZEN

    points: tuple[FrontierPoint, ...]
    dominated: tuple[Dominance, ...]

    @property
    def excluded_names(self) -> frozenset[str]:
        return frozenset(entry.run_name for entry in self.dominated)

    @property
    def on_frontier(self) -> tuple[FrontierPoint, ...]:
        """The non-dominated points, still in lambda order."""
        excluded = self.excluded_names
        return tuple(point for point in self.points if point.run_name not in excluded)


def _ordered(points: Sequence[FrontierPoint]) -> tuple[FrontierPoint, ...]:
    """Sort by lambda and refuse a duplicate.

    A repeated lambda puts two markers on the same x position with different y
    values. The plot looks like noise in the measurement rather than a mistake
    in the manifest.
    """
    if not points:
        raise EvalError("a frontier needs at least one point")
    lambdas = [point.lambda_value for point in points]
    if len(set(lambdas)) != len(lambdas):
        raise EvalError(
            f"duplicate lambda in the sweep: {sorted(lambdas)}. Two points at one x "
            f"position read as measurement noise rather than a manifest error."
        )
    return tuple(sorted(points, key=lambda point: point.lambda_value))


def point_from(report: EvalReport, lambda_value: float) -> FrontierPoint:
    """Reduce one run's report to a frontier point.

    Accuracy is F1, not EM. Section 9 reports both, but the frontier needs the
    one that moves continuously - EM on a few hundred multi-hop questions steps
    in visible jumps and makes a four-point curve look like a staircase.
    """
    return FrontierPoint(
        lambda_value=lambda_value,
        run_name=report.run_name,
        accuracy=report.f1,
        searches=report.searches_per_question,
        unsupported_rate=report.unsupported_rate,
    )


def _intervals_overlap(first: ConfidenceInterval, second: ConfidenceInterval) -> bool:
    return first.low <= second.high and second.low <= first.high


def pareto_frontier(points: Sequence[FrontierPoint]) -> Frontier:
    """Keep the points nothing else beats on both axes.

    `a` dominates `b` when it is at least as accurate and costs no more
    searches, and is strictly better on one of them. Computed on point
    estimates, because that is what a Pareto frontier is - but every exclusion
    whose intervals overlap on the axis that decided it is marked `contested`.

    The honest reading of a contested exclusion: these two settings are not
    distinguishable at this sample size. Dropping one from the figure would be
    reporting a conclusion the data does not support, so the figure draws it and
    labels it.
    """
    ordered = _ordered(points)
    dominated: list[Dominance] = []

    for candidate in ordered:
        dominator = next(
            (
                other
                for other in ordered
                if other.run_name != candidate.run_name
                and other.accuracy.point >= candidate.accuracy.point
                and other.searches.point <= candidate.searches.point
                and (
                    other.accuracy.point > candidate.accuracy.point
                    or other.searches.point < candidate.searches.point
                )
            ),
            None,
        )
        if dominator is None:
            continue

        accuracy_tie = _intervals_overlap(dominator.accuracy, candidate.accuracy)
        searches_tie = _intervals_overlap(dominator.searches, candidate.searches)
        contested = accuracy_tie and searches_tie
        dominated.append(
            Dominance(
                run_name=candidate.run_name,
                dominated_by=dominator.run_name,
                contested=contested,
                note=(
                    "intervals overlap on both axes - these two settings are not "
                    "distinguishable at this sample size"
                    if contested
                    else "separated on at least one axis"
                ),
            )
        )

    return Frontier(points=ordered, dominated=tuple(dominated))


class SweepDiagnostics(BaseModel):
    """Whether the sweep measured a trade-off at all.

    Read this before the plot. Section 6's whole premise is that charging per
    search makes the policy search less; if that did not happen, there is no
    frontier, and the figure would be presenting a flat scatter as a curve.
    """

    model_config = _FROZEN

    n_points: int = Field(ge=1)
    searches_fall_with_lambda: bool
    accuracy_falls_with_lambda: bool
    searches_span: float
    accuracy_span: float
    warnings: tuple[str, ...]

    @property
    def is_a_frontier(self) -> bool:
        """True when raising lambda actually bought fewer searches."""
        return self.searches_fall_with_lambda and self.searches_span > 0.0

    def summary(self) -> str:
        verdict = "trade-off present" if self.is_a_frontier else "NO TRADE-OFF MEASURED"
        lines = [
            f"Sweep diagnostics ({self.n_points} points): {verdict}",
            f"  searches span {self.searches_span:.3f}, "
            f"monotone down: {self.searches_fall_with_lambda}",
            f"  accuracy span {self.accuracy_span:.3f}, "
            f"monotone down: {self.accuracy_falls_with_lambda}",
        ]
        lines.extend(f"  warning: {text}" for text in self.warnings)
        return "\n".join(lines)


def diagnose(points: Sequence[FrontierPoint]) -> SweepDiagnostics:
    """Check the sweep has the shape the experiment predicts.

    Expected: searches fall as lambda rises, and accuracy falls with them. Both
    monotone, both with real spread. The failures worth naming:

    * searches flat or rising - the penalty is not reaching the policy. Either
      lambda is too small to matter next to a correctness reward of 1.0, or the
      term is not wired into the composed reward at all. Not a frontier.
    * accuracy rising with lambda - the opposite of the predicted trade-off. It
      can be real (fewer, better-chosen searches) and it would be the most
      interesting result in the project, which is exactly why it must be flagged
      rather than plotted without comment.
    * only two points - section 2 asks for at least four.
    """
    ordered = _ordered(points)
    searches = [point.searches.point for point in ordered]
    accuracy = [point.accuracy.point for point in ordered]

    searches_down = all(b <= a for a, b in pairwise(searches))
    accuracy_down = all(b <= a for a, b in pairwise(accuracy))
    searches_span = max(searches) - min(searches)
    accuracy_span = max(accuracy) - min(accuracy)

    warnings: list[str] = []
    if len(ordered) < 4:
        warnings.append(f"only {len(ordered)} points; section 2 requires a sweep of at least 4")
    if not searches_down or searches_span == 0.0:
        warnings.append(
            "searches per question do not fall as lambda rises - the retrieval penalty is "
            "not reaching the policy. Check it is in the composed reward and that lambda "
            "is large enough to matter against a correctness reward of 1.0. There is no "
            "trade-off to plot until this changes."
        )
    if not accuracy_down:
        warnings.append(
            "accuracy does not fall as lambda rises. This may be real - fewer but "
            "better-chosen searches - and would be the most interesting result here, "
            "which is why it needs saying out loud rather than plotting silently."
        )
    if any(point.unsupported_rate is None for point in ordered):
        warnings.append(
            "at least one point has no unsupported-claim rate, so the frontier cannot "
            "show whether accuracy was bought with hallucination"
        )

    return SweepDiagnostics(
        n_points=len(ordered),
        searches_fall_with_lambda=searches_down,
        accuracy_falls_with_lambda=accuracy_down,
        searches_span=searches_span,
        accuracy_span=accuracy_span,
        warnings=tuple(warnings),
    )


def check_one_experiment(points: Sequence[FrontierPoint], reports: Sequence[EvalReport]) -> None:
    """Refuse a frontier whose points were not measured the same way.

    Pairwise `check_comparable` against the first report. A sweep takes hours
    per point, so an index rebuild or a regenerated split between the first run
    and the last is an ordinary thing to happen - and the resulting curve is
    smooth, plausible, and a comparison between two different experiments.
    """
    if len(points) != len(reports):
        raise EvalError(f"{len(points)} points but {len(reports)} reports; they must correspond")
    first, *rest = reports
    for report in rest:
        try:
            check_comparable(first.context, report.context)
        except EvalError as exc:
            raise EvalError(
                f"sweep points {first.run_name!r} and {report.run_name!r} were not "
                f"measured the same way, so no frontier through them means anything. {exc}"
            ) from exc


def frontier_table(frontier: Frontier, diagnostics: SweepDiagnostics | None = None) -> str:
    """Markdown table for `writeup/report.md`.

    Core dependencies only - no plotting library. The numbers behind a figure
    have to be readable without the figure, because section 15 requires a
    reported result to be traceable and a PNG is not.
    """
    excluded = {d.run_name: d for d in frontier.dominated}
    lines = [
        "| λ | run | F1 (95% CI) | searches/q (95% CI) | unsupported (95% CI) | on frontier |",
        "|---|---|---|---|---|---|",
    ]
    for point in frontier.points:
        mark = "yes"
        if point.run_name in excluded:
            dominance = excluded[point.run_name]
            mark = (
                f"contested (vs {dominance.dominated_by})"
                if dominance.contested
                else f"no (dominated by {dominance.dominated_by})"
            )
        unsupported = str(point.unsupported_rate) if point.unsupported_rate is not None else "n/a"
        lines.append(
            f"| {point.lambda_value:g} | {point.run_name} | {point.accuracy} | "
            f"{point.searches} | {unsupported} | {mark} |"
        )
    if diagnostics is not None:
        lines.extend(["", "```", diagnostics.summary(), "```"])
    return "\n".join(lines)


def load_sweep_results(
    spec: SweepSpec,
    *,
    root: str | Path | None = None,
) -> tuple[tuple[FrontierPoint, ...], tuple[EvalReport, ...]]:
    """Read every run the manifest asked for, and refuse a partial sweep.

    This is the guard the module is built around. A sweep of four lambdas where
    one run crashed leaves three reports on disk. Three points plot a smooth,
    believable curve, and nothing downstream knows a fourth was expected - the
    figure caption says "four-point sweep" because the manifest said four.

    So the manifest is the source of truth for which runs must exist, and the
    missing lambdas are named rather than counted.
    """
    base = Path(root) if root is not None else Path()
    output_dir = base / spec.output_dir

    points: list[FrontierPoint] = []
    reports: list[EvalReport] = []
    missing: list[str] = []

    for value in spec.lambdas:
        run_name = sweep_run_name(spec.name, value)
        path = output_dir / run_name / REPORT_FILENAME
        if not path.is_file():
            missing.append(f"λ={value:g} ({path})")
            continue
        try:
            report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise EvalError(f"{path} is not an eval report: {exc}") from exc
        points.append(point_from(report, value))
        reports.append(report)

    if missing:
        raise EvalError(
            f"sweep {spec.name!r} is incomplete - no report for "
            + "; ".join(missing)
            + ". A frontier drawn through the runs that finished looks exactly like a "
            "complete one. Re-run the missing points, or edit the manifest so it asks "
            "for what you intend to plot."
        )

    check_one_experiment(points, reports)
    return tuple(points), tuple(reports)


def plot_frontier(  # pragma: no cover - matplotlib is in the [plot] extra
    frontier: Frontier,
    *,
    title: str = "Accuracy vs retrieval cost",
    baseline: FrontierPoint | None = None,
) -> Figure:
    """Draw the figure. Needs the `[plot]` extra.

    **Unverified.** `matplotlib` is not a core or dev dependency, so this has
    never run and CI cannot run it. Written as a thin shell over `Frontier` for
    exactly that reason: the point selection, the dominance decisions and the
    numbers are all decided above and tested, and a bug traced into this
    function should be fixed by moving logic out of it.

    Every point is drawn, including the dominated ones, with error bars on both
    axes. A frontier figure that silently omits the settings that lost is a
    figure that cannot be argued with.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EvalError(
            "plotting the frontier needs the plot extra: uv pip install -e '.[plot]'"
        ) from exc

    excluded = frontier.excluded_names
    figure, axes = plt.subplots(figsize=(6.0, 4.5))

    for point in frontier.points:
        on_frontier = point.run_name not in excluded
        axes.errorbar(
            point.searches.point,
            point.accuracy.point,
            xerr=[
                [point.searches.point - point.searches.low],
                [point.searches.high - point.searches.point],
            ],
            yerr=[
                [point.accuracy.point - point.accuracy.low],
                [point.accuracy.high - point.accuracy.point],
            ],
            marker="o" if on_frontier else "x",
            capsize=3,
            label=point.label if on_frontier else f"{point.label} (dominated)",
        )

    on_curve = frontier.on_frontier
    if len(on_curve) > 1:
        axes.plot(
            [p.searches.point for p in on_curve],
            [p.accuracy.point for p in on_curve],
            linestyle="--",
            linewidth=1.0,
            color="grey",
        )

    if baseline is not None:
        axes.scatter(
            [baseline.searches.point],
            [baseline.accuracy.point],
            marker="*",
            s=160,
            color="black",
            label=baseline.run_name,
            zorder=5,
        )

    axes.set_xlabel("searches per question")
    axes.set_ylabel("multi-hop F1")
    axes.set_title(title)
    axes.legend(fontsize="small")
    figure.tight_layout()
    return figure


def save_frontier(  # pragma: no cover - matplotlib is in the [plot] extra
    frontier: Frontier,
    path: str | Path,
    *,
    title: str = "Accuracy vs retrieval cost",
    baseline: FrontierPoint | None = None,
) -> Path:
    """Write the figure to disk and return where it landed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_frontier(frontier, title=title, baseline=baseline)
    figure.savefig(target, dpi=200)
    return target
