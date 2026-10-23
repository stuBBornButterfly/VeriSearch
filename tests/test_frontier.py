"""Tests for `eval/frontier.py`.

Everything except the figure. `plot_frontier` and `save_frontier` need the
`[plot]` extra and carry pragmas; what is tested here is every decision they
depend on - which points exist, which are dominated, whether the sweep measured
a trade-off at all, and whether all of its runs measured the same thing.

Frontier points are built from hand-written `EvalReport`s rather than by running
the harness. The reports here are the *output* of a sweep that this machine
cannot run, so stating them directly is the only honest way to assert "a sweep
shaped like this produces a frontier shaped like that".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from verisearch.config import SweepSpec, sweep_run_name
from verisearch.eval.baselines import BaselineKind, EvalError, context_for
from verisearch.eval.frontier import (
    REPORT_FILENAME,
    Dominance,
    Frontier,
    FrontierPoint,
    check_one_experiment,
    diagnose,
    frontier_table,
    load_sweep_results,
    pareto_frontier,
    point_from,
)
from verisearch.eval.harness import EvalReport, write_report
from verisearch.eval.metrics import ConfidenceInterval, FailureReason

if TYPE_CHECKING:
    from pathlib import Path

LAMBDAS = (0.0, 0.02, 0.05, 0.1)


def ci(point: float, half_width: float = 0.02, *, n: int = 500) -> ConfidenceInterval:
    return ConfidenceInterval(
        point=point,
        low=point - half_width,
        high=point + half_width,
        confidence=0.95,
        resamples=1000,
        n=n,
    )


def make_report(
    *,
    run_name: str,
    f1: float,
    searches: float,
    unsupported: float | None = 0.2,
    fingerprint: str = "idx-abc",
    dataset: str = "hotpotqa",
    question_ids: tuple[str, ...] = ("q1", "q2", "q3"),
    trained_lambda: float | None = None,
) -> EvalReport:
    return EvalReport(
        run_name=run_name,
        kind=BaselineKind.TRAINED,
        context=context_for(
            question_ids,
            dataset=dataset,
            split="dev",
            retriever_fingerprint=fingerprint,
            k=3,
        ),
        exact_match=ci(f1 - 0.05),
        f1=ci(f1),
        searches_per_question=ci(searches, 0.1),
        answer_tokens=ci(6.0, 0.5),
        unsupported_rate=None if unsupported is None else ci(unsupported),
        precision_at_k=ci(0.5),
        recall_at_k=ci(0.7),
        failure_breakdown=dict.fromkeys((r.value for r in FailureReason), 0.2),
        duplicate_query_episode_rate=0.1,
        trained_lambda=trained_lambda,
    )


def sweep_reports(
    f1s: tuple[float, ...],
    searches: tuple[float, ...],
    *,
    name: str = "lambda_frontier",
    unsupported: tuple[float | None, ...] | None = None,
    lambdas: tuple[float, ...] = LAMBDAS,
) -> tuple[tuple[FrontierPoint, ...], tuple[EvalReport, ...]]:
    rates = unsupported if unsupported is not None else (0.2,) * len(f1s)
    reports = tuple(
        make_report(
            run_name=sweep_run_name(name, value),
            f1=f1,
            searches=count,
            unsupported=rate,
            trained_lambda=value,
        )
        for value, f1, count, rate in zip(lambdas, f1s, searches, rates, strict=True)
    )
    points = tuple(
        point_from(report, value) for report, value in zip(reports, lambdas, strict=True)
    )
    return points, reports


# The shape the experiment predicts: raising lambda buys fewer searches and
# costs some accuracy.
EXPECTED_F1 = (0.62, 0.60, 0.57, 0.52)
EXPECTED_SEARCHES = (3.4, 2.8, 2.1, 1.3)


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


def test_a_point_carries_both_axes_and_the_hallucination_rate() -> None:
    """Unsupported rate is not an axis, but a point that buys accuracy at fewer
    searches while hallucinating more is not on the frontier this project
    claims, and the only way to see that is to have the number alongside."""
    report = make_report(
        run_name="lambda_frontier-lam0.05", f1=0.57, searches=2.1, trained_lambda=0.05
    )
    point = point_from(report, 0.05)

    assert point.lambda_value == 0.05
    assert point.run_name == "lambda_frontier-lam0.05"
    assert point.accuracy == report.f1
    assert point.searches == report.searches_per_question
    assert point.unsupported_rate is not None
    assert point.label == "λ=0.05"


def test_accuracy_on_the_frontier_is_f1_not_em() -> None:
    """EM on a few hundred multi-hop questions steps in visible jumps and makes
    a four-point curve look like a staircase."""
    report = make_report(run_name="r", f1=0.6, searches=2.0, trained_lambda=0.0)
    assert point_from(report, 0.0).accuracy == report.f1
    assert point_from(report, 0.0).accuracy != report.exact_match


def test_a_duplicate_lambda_is_refused() -> None:
    """Two points at one x position read as measurement noise rather than a
    manifest error."""
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    doubled = (*points, points[0])
    with pytest.raises(EvalError, match="duplicate lambda"):
        pareto_frontier(doubled)


def test_an_empty_sweep_is_refused() -> None:
    with pytest.raises(EvalError, match="at least one point"):
        pareto_frontier([])


def test_points_come_back_in_lambda_order_whatever_order_they_arrive_in() -> None:
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    frontier = pareto_frontier(tuple(reversed(points)))
    assert [p.lambda_value for p in frontier.points] == list(LAMBDAS)


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------


def test_a_clean_trade_off_leaves_every_point_on_the_frontier() -> None:
    """Each setting buys fewer searches at some accuracy cost, so nothing is
    beaten on both axes at once."""
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    frontier = pareto_frontier(points)

    assert frontier.dominated == ()
    assert len(frontier.on_frontier) == 4
    assert frontier.points == frontier.on_frontier


def test_a_point_beaten_on_both_axes_is_marked_dominated() -> None:
    """λ=0.02 here is both less accurate and more expensive than λ=0.05. A
    frontier through it would claim a trade-off that this sweep did not find."""
    points, _ = sweep_reports((0.62, 0.50, 0.57, 0.52), (3.4, 2.8, 2.1, 1.3))
    frontier = pareto_frontier(points)

    assert [d.run_name for d in frontier.dominated] == ["lambda_frontier-lam0.02"]
    assert frontier.dominated[0].dominated_by == "lambda_frontier-lam0.05"
    assert [p.lambda_value for p in frontier.on_frontier] == [0.0, 0.05, 0.1]
    # All four are still kept - the figure marks losers, it does not hide them.
    assert len(frontier.points) == 4


def test_dominance_inside_the_intervals_is_reported_as_contested() -> None:
    """A point is not beaten if the intervals overlap on both axes. Dropping it
    would present a conclusion the data does not carry.

    These intervals are +/-0.02 on F1 and +/-0.1 on searches, so *neither* the
    0.001 gap at lambda=0.02 nor the 0.03 gap at lambda=0.05 is decisive against
    lambda=0. Both are excluded on point estimates and both are flagged, which
    is the whole point: at n=500 this sweep has not separated three of its four
    settings, and the figure has to say so rather than drawing a confident curve
    through two of them.
    """
    points, _ = sweep_reports((0.60, 0.599, 0.57, 0.52), (2.0, 2.01, 2.1, 1.3))
    frontier = pareto_frontier(points)

    contested = [d for d in frontier.dominated if d.contested]
    assert [d.run_name for d in contested] == [
        "lambda_frontier-lam0.02",
        "lambda_frontier-lam0.05",
    ]
    assert all(d.dominated_by == "lambda_frontier-lam0" for d in contested)
    assert "not distinguishable at this sample size" in contested[0].note
    assert [p.lambda_value for p in frontier.on_frontier] == [0.0, 0.1]


def test_a_gap_wider_than_the_intervals_is_decisive() -> None:
    """The same sweep with tighter intervals: once the error bars separate, the
    exclusion is a real result rather than a coin flip."""
    tight = tuple(
        FrontierPoint(
            lambda_value=value,
            run_name=sweep_run_name("lambda_frontier", value),
            accuracy=ci(f1, 0.002),
            searches=ci(count, 0.002),
            unsupported_rate=ci(0.2),
        )
        for value, f1, count in zip(
            LAMBDAS, (0.60, 0.599, 0.57, 0.52), (2.0, 2.01, 2.1, 1.3), strict=True
        )
    )
    frontier = pareto_frontier(tight)
    assert [d.contested for d in frontier.dominated] == [False, False]


def test_dominance_outside_the_intervals_is_not_contested() -> None:
    points, _ = sweep_reports((0.62, 0.30, 0.57, 0.52), (3.4, 3.0, 2.1, 1.3))
    frontier = pareto_frontier(points)
    assert [d.contested for d in frontier.dominated] == [False]
    assert "separated on at least one axis" in frontier.dominated[0].note


def test_an_equal_point_does_not_dominate_its_twin_on_both_axes() -> None:
    """Dominance needs a strict improvement on at least one axis, otherwise two
    identical settings would each exclude the other and the frontier would be
    empty."""
    points, _ = sweep_reports((0.6, 0.6, 0.5, 0.4), (2.0, 2.0, 1.5, 1.0))
    frontier = pareto_frontier(points)
    assert frontier.on_frontier != ()


def test_a_single_point_is_its_own_frontier() -> None:
    points, _ = sweep_reports((0.6,), (2.0,), lambdas=(0.0,))
    frontier = pareto_frontier(points)
    assert frontier.dominated == ()
    assert len(frontier.on_frontier) == 1


# ---------------------------------------------------------------------------
# Diagnostics - did the sweep measure anything
# ---------------------------------------------------------------------------


def test_the_predicted_shape_is_reported_as_a_trade_off() -> None:
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    report = diagnose(points)

    assert report.is_a_frontier
    assert report.searches_fall_with_lambda
    assert report.accuracy_falls_with_lambda
    assert report.searches_span == pytest.approx(2.1)
    assert report.warnings == ()
    assert "trade-off present" in report.summary()


def test_a_flat_search_count_means_the_penalty_never_reached_the_policy() -> None:
    """The most important diagnostic here. If charging per search did not make
    the policy search less, there is no frontier - the figure would be
    presenting a flat scatter as a curve, and the bug is in the reward."""
    points, _ = sweep_reports(EXPECTED_F1, (2.5, 2.5, 2.5, 2.5))
    report = diagnose(points)

    assert not report.is_a_frontier
    assert report.searches_span == 0.0
    assert any("not reaching the policy" in w for w in report.warnings)
    assert "NO TRADE-OFF MEASURED" in report.summary()


def test_searches_rising_with_lambda_is_flagged() -> None:
    points, _ = sweep_reports(EXPECTED_F1, (1.3, 2.1, 2.8, 3.4))
    report = diagnose(points)
    assert not report.searches_fall_with_lambda
    assert not report.is_a_frontier
    assert any("do not fall as lambda rises" in w for w in report.warnings)


def test_accuracy_rising_with_lambda_is_flagged_not_hidden() -> None:
    """It can be real - fewer but better-chosen searches - and it would be the
    most interesting result in the project, which is exactly why it has to be
    said out loud rather than plotted without comment."""
    points, _ = sweep_reports((0.52, 0.57, 0.60, 0.62), EXPECTED_SEARCHES)
    report = diagnose(points)

    assert report.is_a_frontier
    assert not report.accuracy_falls_with_lambda
    assert any("most interesting result" in w for w in report.warnings)


def test_fewer_than_four_points_is_flagged_against_section_2() -> None:
    points, _ = sweep_reports((0.6, 0.5), (2.0, 1.5), lambdas=(0.0, 0.1))
    report = diagnose(points)
    assert any("at least 4" in w for w in report.warnings)


def test_a_missing_hallucination_rate_is_flagged() -> None:
    """Without it the frontier cannot show whether accuracy was bought with
    hallucination, which is half of what the project claims."""
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES, unsupported=(0.2, None, 0.2, 0.2))
    report = diagnose(points)
    assert any("no unsupported-claim rate" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# One experiment, not several
# ---------------------------------------------------------------------------


def test_every_sweep_point_must_have_been_measured_the_same_way() -> None:
    points, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    check_one_experiment(points, reports)


def test_a_rebuilt_index_partway_through_a_sweep_is_refused() -> None:
    """A sweep takes hours per point, so an index rebuild between the first run
    and the last is an ordinary thing to happen - and the resulting curve is
    smooth, plausible, and a comparison between two different experiments."""
    points, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    tampered = (
        *reports[:3],
        reports[3].model_copy(
            update={
                "context": reports[3].context.model_copy(
                    update={"retriever_fingerprint": "idx-rebuilt"}
                )
            }
        ),
    )
    with pytest.raises(EvalError, match="were not measured the same way"):
        check_one_experiment(points, tampered)


def test_a_sweep_point_evaluated_on_a_different_set_is_refused() -> None:
    points, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    tampered = (
        reports[0].model_copy(
            update={
                "context": context_for(
                    ("q1", "q2", "q9"),
                    dataset="hotpotqa",
                    split="dev",
                    retriever_fingerprint="idx-abc",
                    k=3,
                )
            }
        ),
        *reports[1:],
    )
    with pytest.raises(EvalError, match="examples_fingerprint"):
        check_one_experiment(points, tampered)


def test_mismatched_point_and_report_counts_are_refused() -> None:
    points, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    with pytest.raises(EvalError, match="they must correspond"):
        check_one_experiment(points, reports[:3])


# ---------------------------------------------------------------------------
# Loading a sweep from disk
# ---------------------------------------------------------------------------


def write_sweep(
    tmp_path: Path,
    reports: tuple[EvalReport, ...],
    *,
    name: str = "lambda_frontier",
    skip: int | None = None,
) -> SweepSpec:
    for index, report in enumerate(reports):
        if index == skip:
            continue
        write_report(report, tmp_path / "out" / report.run_name / REPORT_FILENAME)
    return SweepSpec(
        name=name,
        base=("configs/base.yaml",),
        lambdas=LAMBDAS,
        output_dir="out",
    )


def test_a_complete_sweep_loads_in_manifest_order(tmp_path: Path) -> None:
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports)

    points, loaded = load_sweep_results(spec, root=tmp_path)
    assert [p.lambda_value for p in points] == list(LAMBDAS)
    assert [p.run_name for p in points] == [sweep_run_name(spec.name, v) for v in LAMBDAS]
    assert [r.f1.point for r in loaded] == list(EXPECTED_F1)


def test_an_incomplete_sweep_names_the_lambda_that_is_missing(tmp_path: Path) -> None:
    """The guard the module exists for. Three of four reports plot a smooth,
    believable curve and nothing downstream knows a fourth was expected."""
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports, skip=2)

    with pytest.raises(EvalError, match=r"lambda=0\.05") as caught:
        load_sweep_results(spec, root=tmp_path)
    assert "looks exactly like a complete one" in str(caught.value)


def test_a_malformed_report_in_the_sweep_names_the_file(tmp_path: Path) -> None:
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports)
    bad = tmp_path / "out" / sweep_run_name(spec.name, 0.02) / REPORT_FILENAME
    bad.write_text('{"run_name":"x"}', encoding="utf-8")

    with pytest.raises(EvalError, match="is not an eval report"):
        load_sweep_results(spec, root=tmp_path)


def test_loading_refuses_a_sweep_whose_runs_disagree_about_the_retriever(
    tmp_path: Path,
) -> None:
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    tampered = (
        *reports[:3],
        reports[3].model_copy(
            update={
                "context": reports[3].context.model_copy(
                    update={"retriever_fingerprint": "idx-rebuilt"}
                )
            }
        ),
    )
    spec = write_sweep(tmp_path, tampered)
    with pytest.raises(EvalError, match="not measured the same way"):
        load_sweep_results(spec, root=tmp_path)


def test_the_run_name_format_has_one_definition(tmp_path: Path) -> None:
    """`sweep_run_name` is shared with `config.sweep_points`. Two copies of that
    format string would diverge, and the symptom would be a frontier silently
    missing a point while reporting it read everything the manifest asked for."""
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports)
    # The directory names on disk came from `sweep_run_name` via `sweep_reports`;
    # the loader looks them up the same way.
    assert (tmp_path / "out" / "lambda_frontier-lam0" / REPORT_FILENAME).is_file()
    assert (tmp_path / "out" / "lambda_frontier-lam0.02" / REPORT_FILENAME).is_file()
    assert len(load_sweep_results(spec, root=tmp_path)[0]) == 4


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_carries_every_number_behind_the_figure() -> None:
    """Section 15: a reported result has to be traceable, and a PNG is not."""
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    table = frontier_table(pareto_frontier(points), diagnose(points))

    assert table.count("\n|") >= 5
    for value in LAMBDAS:
        assert f"| {value:g} |" in table
    assert "0.6200" in table
    assert "95% CI" in table
    assert "trade-off present" in table


def test_the_table_marks_dominated_and_contested_points_differently() -> None:
    dominated, _ = sweep_reports((0.62, 0.30, 0.57, 0.52), (3.4, 3.0, 2.1, 1.3))
    assert "dominated by" in frontier_table(pareto_frontier(dominated))

    contested, _ = sweep_reports((0.60, 0.599, 0.57, 0.52), (2.0, 2.01, 2.1, 1.3))
    assert "contested (vs" in frontier_table(pareto_frontier(contested))


def test_the_table_says_n_a_rather_than_zero_for_a_missing_rate() -> None:
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES, unsupported=(0.2, None, 0.2, 0.2))
    assert "| n/a |" in frontier_table(pareto_frontier(points))


def test_the_table_works_without_diagnostics() -> None:
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    table = frontier_table(pareto_frontier(points))
    assert "trade-off" not in table
    assert "| λ |" in table


def test_a_hand_built_frontier_exposes_the_same_properties() -> None:
    """`Frontier` is a plain model, so a caller can construct one - the
    properties must not depend on having gone through `pareto_frontier`."""
    points, _ = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    manual = Frontier(
        points=points,
        dominated=(
            Dominance(
                run_name=points[1].run_name,
                dominated_by=points[2].run_name,
                contested=False,
                note="by hand",
            ),
        ),
    )
    assert manual.excluded_names == frozenset({points[1].run_name})
    assert [p.lambda_value for p in manual.on_frontier] == [0.0, 0.05, 0.1]


# ---------------------------------------------------------------------------
# A report says which lambda it was trained at - Dhrubo's point on #16
# ---------------------------------------------------------------------------


def test_a_report_without_its_trained_lambda_is_not_a_sweep_point() -> None:
    """A baseline's report, or one from before the field existed."""
    report = make_report(run_name="lambda_frontier-lam0.05", f1=0.57, searches=2.1)
    with pytest.raises(EvalError, match="does not record the lambda"):
        point_from(report, 0.05)


def test_a_report_read_at_the_wrong_lambda_is_refused() -> None:
    report = make_report(
        run_name="lambda_frontier-lam0.05", f1=0.57, searches=2.1, trained_lambda=0.02
    )
    with pytest.raises(EvalError, match=r"trained at lambda=0\.02 but is being read as"):
        point_from(report, 0.05)


def test_loading_refuses_a_report_filed_under_another_lambda(tmp_path: Path) -> None:
    """The 0.1 checkpoint's report in the 0.05 directory: right path, right
    name on disk, wrong policy. Only the lambda the report carries catches it."""
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports)
    target = tmp_path / "out" / sweep_run_name(spec.name, 0.05) / REPORT_FILENAME
    misfiled = reports[2].model_copy(update={"trained_lambda": 0.1})
    write_report(misfiled, target)

    with pytest.raises(EvalError, match=r"trained at lambda=0\.1"):
        load_sweep_results(spec, root=tmp_path)


def test_loading_refuses_a_report_copied_into_another_runs_directory(tmp_path: Path) -> None:
    _, reports = sweep_reports(EXPECTED_F1, EXPECTED_SEARCHES)
    spec = write_sweep(tmp_path, reports)
    target = tmp_path / "out" / sweep_run_name(spec.name, 0.05) / REPORT_FILENAME
    write_report(reports[1], target)

    with pytest.raises(EvalError, match="copied into another run's directory"):
        load_sweep_results(spec, root=tmp_path)


def test_the_summary_says_which_lambda_a_trained_run_used() -> None:
    trained = make_report(run_name="r", f1=0.6, searches=2.0, trained_lambda=0.05)
    baseline = make_report(run_name="b", f1=0.6, searches=2.0)
    assert "trained at lambda=0.05" in trained.summary()
    assert "trained at lambda" not in baseline.summary()


def test_a_report_written_before_the_field_existed_still_loads() -> None:
    report = make_report(run_name="r", f1=0.6, searches=2.0)
    older = report.model_dump(mode="json")
    del older["trained_lambda"]
    assert EvalReport.model_validate(older).trained_lambda is None
