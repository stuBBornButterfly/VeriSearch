"""`writeup/report.md` against the code that fills it.

The report is the P6 deliverable and the only artifact a stranger reads. Its
tables are written now, empty, so that filling them in later is transcription
rather than design - but an empty table drifts silently. A metric added to
`EvalReport` with no row in the report is a number nobody reports; a row whose
metric got renamed is a row nobody can fill.

So the rows are tied to field names by HTML comments that do not render, and
these tests check the two sets match in both directions. PROJECT.md section 15
rule 4 is "never fabricate a number"; this is the structural half of that rule -
the report cannot quietly lose a metric, and it cannot quietly keep a row for
one that no longer exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verisearch.config import load_sweep, sweep_run_name
from verisearch.eval.baselines import BaselineKind
from verisearch.eval.harness import Comparison, EvalReport
from verisearch.eval.metrics import FailureReason

REPORT = Path(__file__).resolve().parent.parent / "writeup" / "report.md"
MANIFEST = Path(__file__).resolve().parent.parent / "configs" / "sweep_lambda.yaml"

# `EvalReport` fields that identify the run rather than measure it. Everything
# else on the model is a section 9 metric and must have a row in the report.
_NOT_METRICS = frozenset({"run_name", "kind", "context"})

# Metrics that belong in both the in-distribution and the OOD table. The
# failure breakdown gets its own table, so it is not expected twice.
_TWICE_EXEMPT = frozenset({"failure_breakdown"})


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text(encoding="utf-8")


def _markers(text: str, kind: str) -> list[str]:
    """Every `<!--kind:value-->` in document order, duplicates kept.

    Duplicates matter: a metric appearing twice is how the OOD table is checked
    to carry the same rows as the in-distribution one.
    """
    return re.findall(rf"<!--{kind}:([^>]+?)-->", text)


def _metric_fields() -> frozenset[str]:
    return frozenset(EvalReport.model_fields) - _NOT_METRICS


def _delta_fields() -> frozenset[str]:
    return frozenset(name for name in Comparison.model_fields if name.endswith("_delta"))


# ---------------------------------------------------------------------------
# The report exists and says what it is.
# ---------------------------------------------------------------------------


def test_the_report_exists() -> None:
    assert REPORT.is_file(), f"{REPORT} is the P6 deliverable and must exist"


def test_a_report_with_placeholders_left_still_warns_that_it_is_a_skeleton(report: str) -> None:
    """The banner and the placeholders have to agree.

    Both directions are failures. A file still full of `{{?}}` that has lost its
    status banner reads as a result; a filled-in report that still calls itself a
    skeleton is a published paper claiming its own numbers are fake. The second
    case is the one that will actually happen - someone pastes the last table in
    and forgets the header.
    """
    unfilled = report.count("{{")
    banner = "Status: skeleton. No number in this file has been measured." in report
    if unfilled:
        assert banner, (
            f"{unfilled} unfilled placeholders but no skeleton banner - "
            "this file reads as a result and is not one"
        )
    else:
        assert not banner, (
            "no placeholders left but the skeleton banner is still there - "
            "remove the banner when the numbers land"
        )


# ---------------------------------------------------------------------------
# Section 9 metrics.
# ---------------------------------------------------------------------------


def test_every_metric_on_the_report_model_has_a_row(report: str) -> None:
    claimed = set(_markers(report, "field"))
    missing = sorted(_metric_fields() - claimed)
    assert not missing, (
        f"{missing} are reported by EvalReport but have no row in writeup/report.md. "
        "A metric with no row is a number nobody reports."
    )


def test_every_comparison_delta_has_a_row(report: str) -> None:
    claimed = set(_markers(report, "field"))
    missing = sorted(_delta_fields() - claimed)
    assert not missing, f"{missing} are computed by compare_runs but have no row in the report"


def test_the_report_has_no_row_for_a_field_that_does_not_exist(report: str) -> None:
    """The other direction: a renamed metric leaves a row nobody can fill."""
    known = _metric_fields() | _delta_fields()
    unknown = sorted(set(_markers(report, "field")) - known)
    assert not unknown, (
        f"writeup/report.md has rows for {unknown}, which are not fields of "
        "EvalReport or Comparison. Either the metric was renamed and the row is "
        "stale, or the row was invented."
    )


def test_the_core_metrics_appear_in_both_the_main_and_the_ood_table(report: str) -> None:
    """Section 2 requires the result to hold out of distribution.

    Which means the OOD table carries the same rows, not a chosen subset - a
    metric that only appears in distribution is a metric that can be quietly
    dropped when it does not transfer.
    """
    counts: dict[str, int] = {}
    for name in _markers(report, "field"):
        counts[name] = counts.get(name, 0) + 1
    expected = _metric_fields() - _TWICE_EXEMPT
    once_only = sorted(name for name in expected if counts.get(name, 0) < 2)
    assert not once_only, (
        f"{once_only} appear in only one results table. Both the in-distribution "
        "and the out-of-distribution table need every metric."
    )


# ---------------------------------------------------------------------------
# Baselines and failure reasons.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(BaselineKind))
def test_every_baseline_is_a_column_in_both_results_tables(report: str, kind: BaselineKind) -> None:
    """Section 8: four baselines plus the system under test, all five reported.

    Columns are labelled with the enum value rather than a prose name, because
    that value is what `--kind` takes on the command line - the column header and
    the command that fills it are the same string.
    """
    header = f"| `{kind.value}` |"
    assert report.count(header) >= 2, (
        f"`{kind.value}` is a BaselineKind but is not a column in both the "
        "in-distribution and the out-of-distribution results table"
    )


@pytest.mark.parametrize("kind", list(BaselineKind))
def test_every_baseline_has_a_reproduction_command(report: str, kind: BaselineKind) -> None:
    assert f"KIND={kind.value}" in report, (
        f"the reproduction section does not say how to produce the `{kind.value}` "
        "column; a result nobody can rerun is not reproducible"
    )


def test_every_failure_reason_has_a_row(report: str) -> None:
    claimed = set(_markers(report, "reason"))
    reasons = {reason.value for reason in FailureReason}
    assert claimed == reasons, (
        "the failure breakdown table and FailureReason disagree: "
        f"missing {sorted(reasons - claimed)}, stale {sorted(claimed - reasons)}"
    )


def test_the_turn_limit_precedence_artefact_is_documented(report: str) -> None:
    """`classify_failure` orders its checks, and the report has to say so.

    `no_answer` is checked before `turn_limit`, and a trajectory that ran out of
    turns has no answer - so the `turn_limit` share is near-empty by
    construction. Reported without that note it reads as "the budget was
    generous", which is the opposite of what it means.
    """
    assert "precedence artefact" in report
    assert "classify_failure" in report


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


def test_every_lambda_in_the_manifest_has_a_row(report: str) -> None:
    """The manifest is the source of truth for which runs must exist.

    A sweep where one point crashed leaves three reports, and three points draw
    a curve that looks exactly like a complete one. The report lists the runs the
    manifest demands, so a missing one is visible in the write-up and not only in
    the sweep script's exit code.
    """
    spec = load_sweep(MANIFEST)
    claimed = set(_markers(report, "lambda"))
    expected = {f"{value:g}" for value in spec.lambdas}
    assert claimed == expected, (
        f"the frontier run table and {MANIFEST.name} disagree: "
        f"missing {sorted(expected - claimed)}, stale {sorted(claimed - expected)}"
    )


def test_the_report_names_each_sweep_run_the_way_the_code_names_it(report: str) -> None:
    """The run name is how `load_sweep_results` finds a report on disk.

    If the report invents its own naming, the names in the write-up do not match
    the directories, and nobody can trace a published number back to a run.
    """
    spec = load_sweep(MANIFEST)
    for value in spec.lambdas:
        name = sweep_run_name(spec.name, value)
        assert f"`{name}`" in report, f"the frontier table does not name the run `{name}`"


def test_lambda_zero_is_marked_as_not_droppable(report: str) -> None:
    """`SweepSpec` refuses a manifest without it, and the reason belongs here.

    Every other point is read relative to the un-penalised run. Someone cutting
    scope for GPU budget reads this file, not the validator.
    """
    assert "lambda=0 cannot be dropped" in report


# ---------------------------------------------------------------------------
# Claims.
# ---------------------------------------------------------------------------


def test_the_headline_claim_is_stated_as_a_conjunction(report: str) -> None:
    """Section 1's claim is three conditions joined by `and`.

    `JointClaim.holds` is `all(...)`. Two out of three is a weaker result, and
    the report has to be written so that reporting it as the headline requires
    editing a sentence that says otherwise.
    """
    assert "all three clauses have to hold at once" in report
    assert "Two out of three is a different and much weaker result" in report


def test_the_reference_baseline_is_named_not_implied(report: str) -> None:
    from verisearch.eval.baselines import REFERENCE_BASELINE

    assert f"`{REFERENCE_BASELINE.value}` (`REFERENCE_BASELINE`)" in report, (
        "the report must name the baseline every headline number is measured "
        "against; 'the baseline' is not a reference"
    )


def test_the_eval_context_fields_are_all_reported(report: str) -> None:
    """Section 8's "identical retriever" is these fields matching.

    Every field of `EvalContext` is one way the claim could be quietly false, so
    every one is printed in the setup section rather than summarised as "same
    setup for all runs".
    """
    from verisearch.eval.baselines import EvalContext

    labels = {
        "dataset": "dataset / split",
        "split": "dataset / split",
        "n_examples": "n examples",
        "examples_fingerprint": "examples fingerprint",
        "retriever_fingerprint": "retriever fingerprint",
        "k": "| k |",
    }
    assert set(labels) == set(EvalContext.model_fields), (
        "EvalContext gained or lost a field; the setup table in writeup/report.md "
        "lists every field on purpose, so update both"
    )
    for field, label in labels.items():
        assert label in report, f"the setup table does not report EvalContext.{field}"


def test_the_serving_table_covers_the_three_required_concurrency_levels(report: str) -> None:
    """Section 9: throughput and p50/p95/p99 at concurrency 1, 8, 32."""
    for level in (1, 8, 32):
        assert f"| {level} | " in report, f"no serving row for concurrency {level}"
    for percentile in ("p50", "p95", "p99"):
        assert f"{percentile} (ms)" in report, f"the serving table has no {percentile} column"


def test_the_serving_table_has_the_header_the_benchmark_prints(report: str) -> None:
    """Paste, never retype - which only works if the columns line up.

    Add a column to `BenchReport.table()` and this fails until the report has
    room for it; rename one and it fails until the report follows.
    """
    from verisearch.serve.benchmark import TABLE_HEADER

    assert TABLE_HEADER in report, (
        "section 8's header is not the one serve/benchmark.py prints, so its table "
        "cannot be pasted in"
    )


def test_the_serving_rows_match_the_configured_levels(report: str) -> None:
    """The rows are the levels `configs/base.yaml` actually runs, no more."""
    from verisearch.config import load_config
    from verisearch.serve.benchmark import TABLE_HEADER

    section = report.split(TABLE_HEADER, 1)[1].split("\n\n", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    levels = tuple(int(row.split("|")[1]) for row in rows)
    assert levels == load_config().serve.concurrency_levels
    width = TABLE_HEADER.count("|")
    for row in rows:
        assert row.count("|") == width, f"serving row has the wrong number of cells: {row}"


def test_the_serving_footer_is_generated_rather_than_typed(report: str) -> None:
    """The hardware line comes from the server's /health. A placeholder for it
    that could be filled in by hand would undo that."""
    assert "{{serve-hardware}}" not in report
    assert "{{serve-model}}" not in report
    assert "BenchReport.table" in report
