"""Tests for `serve/benchmark.py`.

The statistics are tested on hand-built records, where every timestamp is
chosen and every expected number can be worked out by hand. The load generator
is tested for what it promises - how many requests, in what order, with how
many in flight, and what becomes of a failure - and never for how long anything
took, which would make the suite as flaky as the network it runs on.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from verisearch.serve.benchmark import (
    MIN_TAIL_SAMPLES,
    REPORT_FILENAME,
    REQUESTS_FILENAME,
    TABLE_HEADER,
    BenchError,
    BenchReport,
    LevelRun,
    RequestRecord,
    min_samples_for,
    quantile_ci,
    run_benchmark,
    run_level,
    summarize,
    throughput_ci,
    write_bench,
)
from verisearch.serve.server import (
    MAX_QUESTION_CHARS,
    AnswerRequest,
    AnswerResponse,
    ServeError,
    handle_health,
)
from verisearch.types import Termination

if TYPE_CHECKING:
    from pathlib import Path

HEALTH = handle_health(
    policy="Qwen/Qwen2.5-1.5B-Instruct",
    adapter=None,
    hardware="test-cpu",
    config_fingerprint="abc123def456",
)
STATS = {"batches": 4, "resamples": 200, "confidence": 0.95, "seed": 7}


def record(
    index: int,
    start: float,
    end: float,
    *,
    concurrency: int = 1,
    warmup: bool = False,
    ok: bool = True,
    answered: bool = True,
    error_kind: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        concurrency=concurrency,
        index=index,
        warmup=warmup,
        start=start,
        end=end,
        ok=ok,
        answered=answered and ok,
        error_kind=error_kind if not ok else None,
        error="boom" if not ok else None,
    )


def steady_run(n: int, *, concurrency: int = 1, latency: float = 0.1) -> LevelRun:
    """`n` back-to-back requests, each `latency` seconds, no warmup."""
    return LevelRun(
        concurrency=concurrency,
        records=tuple(record(i, i * latency, (i + 1) * latency) for i in range(n)),
        peak_in_flight=concurrency,
    )


# ---------------------------------------------------------------------------
# The tail rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantile", "needed"),
    [(0.50, 10), (0.90, 50), (0.95, 100), (0.99, 500)],
)
def test_a_quantile_needs_five_observations_above_it(quantile: float, needed: int) -> None:
    """50 for p90, not 51: `5 / (1 - 0.9)` lands at 50.000000000000014."""
    assert MIN_TAIL_SAMPLES == 5
    assert min_samples_for(quantile) == needed


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.5, 1.5])
def test_a_quantile_outside_the_open_unit_interval_is_refused(quantile: float) -> None:
    with pytest.raises(BenchError, match="quantile"):
        min_samples_for(quantile)


def test_a_p99_from_too_few_requests_is_not_reported() -> None:
    """The p99 of 499 requests is not a p99 with five observations behind it."""
    assert quantile_ci([1.0] * 499, 0.99, resamples=10, confidence=0.95, seed=0) is None


def test_a_p99_at_the_floor_is_reported_with_an_interval() -> None:
    values = [float(v) for v in range(500)]
    interval = quantile_ci(values, 0.99, resamples=200, confidence=0.95, seed=0)
    assert interval is not None
    assert interval.point == pytest.approx(float(np.quantile(values, 0.99)))
    assert interval.low <= interval.point <= interval.high
    assert interval.n == 500


def test_a_quantile_interval_is_seeded() -> None:
    values = [float(v) for v in np.random.default_rng(1).exponential(100.0, size=200)]
    first = quantile_ci(values, 0.5, resamples=300, confidence=0.95, seed=3)
    again = quantile_ci(values, 0.5, resamples=300, confidence=0.95, seed=3)
    other = quantile_ci(values, 0.5, resamples=300, confidence=0.95, seed=4)
    assert first == again
    assert first != other


@pytest.mark.parametrize(
    ("resamples", "confidence", "message"),
    [(0, 0.95, "resamples"), (100, 1.0, "confidence"), (100, 0.0, "confidence")],
)
def test_a_quantile_interval_refuses_bad_parameters(
    resamples: int, confidence: float, message: str
) -> None:
    with pytest.raises(BenchError, match=message):
        quantile_ci([1.0] * 20, 0.5, resamples=resamples, confidence=confidence, seed=0)


# ---------------------------------------------------------------------------
# Throughput by batch means
# ---------------------------------------------------------------------------


def test_throughput_is_completions_over_the_window() -> None:
    ends = [0.5, 1.5, 2.5, 3.5]
    interval = throughput_ci(ends, window_start=0.0, window_end=4.0, **STATS)
    assert interval is not None
    assert interval.point == pytest.approx(1.0)
    # The interval is over slices, and says so.
    assert interval.n == STATS["batches"]


def test_an_uneven_rate_widens_the_throughput_interval() -> None:
    even = throughput_ci([0.5, 1.5, 2.5, 3.5], window_start=0.0, window_end=4.0, **STATS)
    bursty = throughput_ci([0.1, 0.2, 0.3, 3.9], window_start=0.0, window_end=4.0, **STATS)
    assert even is not None
    assert bursty is not None
    assert even.point == bursty.point
    assert bursty.width > even.width


def test_no_completions_or_no_window_means_no_throughput() -> None:
    assert throughput_ci([], window_start=0.0, window_end=1.0, **STATS) is None
    assert throughput_ci([1.0], window_start=1.0, window_end=1.0, **STATS) is None


def test_a_completion_outside_the_window_is_refused() -> None:
    with pytest.raises(BenchError, match="outside the measured window"):
        throughput_ci([5.0], window_start=0.0, window_end=4.0, **STATS)


def test_batch_means_needs_at_least_two_batches() -> None:
    with pytest.raises(BenchError, match="2 batches"):
        throughput_ci(
            [1.0], window_start=0.0, window_end=2.0, batches=1, resamples=10, confidence=0.9, seed=0
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"start": 2.0, "end": 1.0}, id="ends-before-it-starts"),
        pytest.param({"ok": True, "error_kind": "HTTP 503", "error": "x"}, id="ok-with-error"),
        pytest.param({"ok": False}, id="failed-without-error"),
        pytest.param({"ok": False, "error_kind": "HTTP 503"}, id="kind-without-message"),
        pytest.param({"ok": False, "error": "x"}, id="message-without-kind"),
        pytest.param(
            {"ok": False, "answered": True, "error_kind": "HTTP 503", "error": "x"},
            id="failed-but-answered",
        ),
    ],
)
def test_a_record_that_contradicts_itself_is_refused(fields: dict[str, object]) -> None:
    base: dict[str, object] = {
        "concurrency": 1,
        "index": 0,
        "warmup": False,
        "start": 0.0,
        "end": 1.0,
        "ok": True,
        "answered": False,
    }
    with pytest.raises(ValidationError):
        RequestRecord.model_validate({**base, **fields})


def test_latency_is_in_milliseconds() -> None:
    assert record(0, 1.0, 1.25).latency_ms == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# One level, summarised
# ---------------------------------------------------------------------------


def test_warmup_is_excluded_from_every_number() -> None:
    warm = [record(i, 0.0, 10.0, warmup=True) for i in range(3)]
    measured = [record(3 + i, 10.0 + i, 10.5 + i) for i in range(10)]
    run = LevelRun(concurrency=1, records=(*warm, *measured), peak_in_flight=1)

    result = summarize(run, **STATS)

    assert result.n_measured == 10
    assert result.p50_ms is not None
    assert result.p50_ms.point == pytest.approx(500.0)


def test_failures_are_counted_never_dropped_and_never_timed() -> None:
    """The slowest requests are the ones that time out. Timing only the survivors
    is how a p95 gets flattering; counting them is how it stays honest."""
    fine = [record(i, float(i), i + 0.1) for i in range(12)]
    failed = [
        record(12, 12.0, 42.0, ok=False, error_kind="TimeoutError"),
        record(13, 13.0, 43.0, ok=False, error_kind="TimeoutError"),
        record(14, 14.0, 14.1, ok=False, error_kind="HTTP 503"),
    ]
    run = LevelRun(concurrency=1, records=(*fine, *failed), peak_in_flight=1)

    result = summarize(run, **STATS)

    assert result.n_measured == 15
    assert result.n_ok == 12
    assert result.n_failed == 3
    assert result.errors == {"HTTP 503": 1, "TimeoutError": 2}
    # The interval's own `n` is the check. A median would hide two 30-second
    # timeouts among twelve fast requests, so the value alone proves nothing.
    assert result.p50_ms is not None
    assert result.p50_ms.n == result.n_ok == 12


def test_the_answered_rate_is_over_successful_requests() -> None:
    records = (
        record(0, 0.0, 1.0),
        record(1, 1.0, 2.0, answered=False),
        record(2, 2.0, 3.0, ok=False, error_kind="HTTP 503"),
        record(3, 3.0, 4.0),
    )
    result = summarize(LevelRun(concurrency=1, records=records, peak_in_flight=1), **STATS)
    assert result.n_answered == 2
    assert result.answered_rate == pytest.approx(2 / 3)


def test_a_level_where_everything_failed_reports_nothing_as_a_number() -> None:
    records = tuple(
        record(i, float(i), i + 1.0, ok=False, error_kind="ConnectionRefusedError")
        for i in range(20)
    )
    result = summarize(LevelRun(concurrency=1, records=records, peak_in_flight=1), **STATS)
    assert result.n_ok == 0
    assert result.throughput is None
    assert result.p50_ms is None
    assert result.answered_rate is None


def test_a_level_that_was_all_warmup_is_refused() -> None:
    run = LevelRun(concurrency=1, records=(record(0, 0.0, 1.0, warmup=True),), peak_in_flight=1)
    with pytest.raises(BenchError, match="every request was warmup"):
        summarize(run, **STATS)


def test_warmup_failures_are_kept_but_not_measured() -> None:
    records = (
        record(0, 0.0, 1.0, warmup=True, ok=False, error_kind="TimeoutError"),
        record(1, 1.0, 2.0),
    )
    result = summarize(LevelRun(concurrency=1, records=records, peak_in_flight=1), **STATS)
    assert result.n_warmup_failed == 1
    assert result.n_failed == 0


@pytest.mark.parametrize(
    ("concurrency", "n_records", "peak", "reached"),
    [
        pytest.param(8, 20, 8, True, id="full"),
        pytest.param(8, 20, 1, False, id="serialised"),
        pytest.param(8, 3, 3, True, id="fewer-requests-than-workers"),
    ],
)
def test_a_row_only_counts_as_its_concurrency_if_that_many_were_in_flight(
    concurrency: int, n_records: int, peak: int, reached: bool
) -> None:
    records = tuple(record(i, float(i), i + 1.0, concurrency=concurrency) for i in range(n_records))
    run = LevelRun(concurrency=concurrency, records=records, peak_in_flight=peak)
    assert summarize(run, **STATS).concurrency_reached is reached


# ---------------------------------------------------------------------------
# The report and its table
# ---------------------------------------------------------------------------


def make_report(*runs: LevelRun, reused: bool = False, adapter: str | None = None) -> BenchReport:
    return BenchReport(
        server=HEALTH.model_copy(update={"adapter": adapter}),
        requests_per_level=len(runs[0].records),
        warmup_requests=0,
        request_timeout_s=120.0,
        n_questions=40,
        questions_reused=reused,
        throughput_batches=STATS["batches"],
        resamples=STATS["resamples"],
        confidence=STATS["confidence"],
        levels=tuple(summarize(run, **STATS) for run in runs),
    )


def test_a_clean_run_with_enough_requests_has_no_problems() -> None:
    report = make_report(steady_run(500), steady_run(500, concurrency=8))
    assert report.problems() == ()


def test_every_reason_a_table_is_not_section_9s_is_named() -> None:
    failing = LevelRun(
        concurrency=8,
        records=(
            *(record(i, float(i), i + 0.5, concurrency=8) for i in range(30)),
            record(30, 30.0, 31.0, concurrency=8, ok=False, error_kind="HTTP 503"),
        ),
        peak_in_flight=2,
    )
    problems = make_report(failing).problems()
    assert len(problems) == 3
    assert "1 of 31 requests failed (HTTP 503 x1)" in problems[0]
    assert "at most 2 requests were ever in flight" in problems[1]
    assert "p99 needs 500" in problems[2]


def test_the_table_is_the_header_the_report_expects_then_one_row_per_level() -> None:
    table = make_report(steady_run(20), steady_run(20, concurrency=8)).table()
    lines = table.splitlines()
    assert lines[0] == TABLE_HEADER
    assert lines[1].count("|") == TABLE_HEADER.count("|")
    assert lines[2].startswith("| 1 | ")
    assert lines[3].startswith("| 8 | ")
    for row in lines[2:4]:
        assert row.count("|") == TABLE_HEADER.count("|")


def test_a_cell_without_a_tail_says_n_a_rather_than_printing_a_maximum() -> None:
    row = make_report(steady_run(20)).table().splitlines()[2]
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[0] == "1"
    assert cells[2].startswith("100 [")  # p50 is reportable from 20
    assert cells[3] == "n/a"  # p95 is not
    assert cells[4] == "n/a"  # nor p99
    assert cells[5] == "1.000"
    assert cells[6] == "0 / 20"


def test_an_all_failed_row_prints_n_a_for_the_answered_rate() -> None:
    failed = LevelRun(
        concurrency=1,
        records=tuple(
            record(i, float(i), i + 1.0, ok=False, error_kind="HTTP 503") for i in range(4)
        ),
        peak_in_flight=1,
    )
    cells = make_report(failed).table().splitlines()[2].strip("|").split("|")
    assert [cell.strip() for cell in cells][1:] == ["n/a", "n/a", "n/a", "n/a", "n/a", "4 / 4"]


def test_the_footer_says_what_was_measured_from_what_the_server_reported() -> None:
    footer = make_report(steady_run(20)).footer()
    assert "`test-cpu`" in footer
    assert "`Qwen/Qwen2.5-1.5B-Instruct`" in footer
    assert "no adapter (base model)" in footer
    assert "`abc123def456`" in footer
    assert "95% intervals" in footer
    assert "prefix cache" not in footer


def test_the_footer_names_the_adapter_and_warns_about_reuse() -> None:
    footer = make_report(steady_run(20), reused=True, adapter="ckpt/step-1000").footer()
    assert "adapter `ckpt/step-1000`" in footer
    assert "prefix cache may favour" in footer


# ---------------------------------------------------------------------------
# Sending load
# ---------------------------------------------------------------------------


def reply(request: AnswerRequest, answer: str | None = "yes") -> AnswerResponse:
    assert request.question_id is not None
    return AnswerResponse(
        question_id=request.question_id,
        answer=answer,
        termination=Termination.ANSWERED if answer is not None else Termination.MAX_TURNS,
        search_queries=(),
        sources=(),
    )


class Recorder:
    """Answers everything, and remembers what it was asked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[AnswerRequest] = []

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        with self._lock:
            self.requests.append(request)
        return reply(request)


def test_a_level_sends_warmup_plus_measured_requests_in_order() -> None:
    target = Recorder()
    run = run_level(target, ["a", "b", "c"], concurrency=1, n_requests=5, warmup=2)

    assert [r.index for r in run.records] == list(range(7))
    assert [r.warmup for r in run.records] == [True, True] + [False] * 5
    assert [r.question for r in target.requests] == ["a", "b", "c", "a", "b", "c", "a"]
    assert all(r.ok and r.answered for r in run.records)


def test_every_request_carries_its_own_id() -> None:
    target = Recorder()
    run_level(target, ["a"], concurrency=4, n_requests=12, warmup=0)
    ids = [r.question_id for r in target.requests]
    assert len(set(ids)) == 12
    assert all(i is not None and i.startswith("bench-c4-") for i in ids)


def test_an_offset_moves_a_level_onto_different_questions() -> None:
    target = Recorder()
    run_level(target, ["a", "b", "c", "d"], concurrency=1, n_requests=2, warmup=0, offset=2)
    assert [r.question for r in target.requests] == ["c", "d"]


def test_concurrency_c_means_c_requests_in_flight_at_once() -> None:
    """A barrier for C parties only opens if C requests really are outstanding
    together. A dispatcher that serialised would hang here and time out."""

    class Barrier:
        def __init__(self, parties: int) -> None:
            self._barrier = threading.Barrier(parties, timeout=10)

        def answer(self, request: AnswerRequest) -> AnswerResponse:
            self._barrier.wait()
            return reply(request)

    run = run_level(Barrier(6), ["q"], concurrency=6, n_requests=12, warmup=0)
    assert run.peak_in_flight == 6
    assert summarize(run, **STATS).concurrency_reached


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        pytest.param(ServeError("down", status_code=503), "HTTP 503", id="serve-error"),
        pytest.param(TimeoutError("slow"), "TimeoutError", id="timeout"),
        pytest.param(ConnectionRefusedError("gone"), "ConnectionRefusedError", id="refused"),
    ],
)
def test_an_expected_failure_becomes_a_record_of_its_kind(error: Exception, kind: str) -> None:
    class Failing:
        def answer(self, request: AnswerRequest) -> AnswerResponse:
            raise error

    run = run_level(Failing(), ["q"], concurrency=2, n_requests=4, warmup=0)
    assert all(not r.ok and not r.answered and r.error_kind == kind for r in run.records)


def test_an_unexpected_exception_stops_the_benchmark_instead_of_becoming_a_row() -> None:
    class Buggy:
        def answer(self, request: AnswerRequest) -> AnswerResponse:
            raise KeyError("a bug, not a failed request")

    with pytest.raises(KeyError):
        run_level(Buggy(), ["q"], concurrency=2, n_requests=4, warmup=0)


def test_a_reply_to_another_request_is_a_failure() -> None:
    class Crossed:
        def answer(self, request: AnswerRequest) -> AnswerResponse:
            return reply(request).model_copy(update={"question_id": "someone-else"})

    run = run_level(Crossed(), ["q"], concurrency=1, n_requests=2, warmup=0)
    assert all(r.error_kind == "mismatched response" for r in run.records)


def test_an_episode_with_no_answer_is_a_success_that_did_not_answer() -> None:
    class GivesUp:
        def answer(self, request: AnswerRequest) -> AnswerResponse:
            return reply(request, answer=None)

    run = run_level(GivesUp(), ["q"], concurrency=1, n_requests=3, warmup=0)
    assert all(r.ok and not r.answered for r in run.records)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"concurrency": 0}, "concurrency", id="no-workers"),
        pytest.param({"n_requests": 0}, "n_requests", id="no-requests"),
        pytest.param({"warmup": -1}, "warmup", id="negative-warmup"),
    ],
)
def test_a_level_refuses_nonsense_parameters(kwargs: dict[str, int], message: str) -> None:
    settings = {"concurrency": 1, "n_requests": 1, "warmup": 0, **kwargs}
    with pytest.raises(BenchError, match=message):
        run_level(Recorder(), ["q"], **settings)


def test_a_level_needs_questions() -> None:
    with pytest.raises(BenchError, match="no questions"):
        run_level(Recorder(), [], concurrency=1, n_requests=1, warmup=0)


# ---------------------------------------------------------------------------
# The whole benchmark
# ---------------------------------------------------------------------------

BENCH = {
    "requests_per_level": 3,
    "warmup": 1,
    "request_timeout_s": 5.0,
    "batches": 2,
    "resamples": 100,
    "confidence": 0.95,
    "seed": 0,
}


def test_levels_run_in_order_on_consecutive_slices_of_the_pool() -> None:
    target = Recorder()
    questions = [f"q{i}" for i in range(8)]

    run = run_benchmark(target, HEALTH, questions, levels=(1, 2), **BENCH)

    assert [level.concurrency for level in run.runs] == [1, 2]
    assert [level.concurrency for level in run.report.levels] == [1, 2]
    first = {r.question for r in target.requests if r.question_id.startswith("bench-c1-")}
    second = {r.question for r in target.requests if r.question_id.startswith("bench-c2-")}
    assert first == {"q0", "q1", "q2", "q3"}
    assert second == {"q4", "q5", "q6", "q7"}
    assert not run.report.questions_reused


def test_a_pool_smaller_than_the_load_is_flagged_as_reused() -> None:
    run = run_benchmark(Recorder(), HEALTH, ["a", "a", "b"], levels=(1,), **BENCH)
    assert run.report.n_questions == 2
    assert run.report.questions_reused


def test_a_question_the_server_would_refuse_stops_the_run_before_any_load() -> None:
    target = Recorder()
    questions = ["fine", "x" * (MAX_QUESTION_CHARS + 1)]
    with pytest.raises(BenchError, match="question 1 cannot be sent"):
        run_benchmark(target, HEALTH, questions, levels=(1,), **BENCH)
    assert target.requests == []


@pytest.mark.parametrize(
    ("levels", "questions", "message"),
    [
        pytest.param((), ["q"], "no concurrency levels", id="no-levels"),
        pytest.param((1,), [], "no questions", id="no-questions"),
    ],
)
def test_the_benchmark_refuses_to_run_with_nothing_to_do(
    levels: tuple[int, ...], questions: list[str], message: str
) -> None:
    with pytest.raises(BenchError, match=message):
        run_benchmark(Recorder(), HEALTH, questions, levels=levels, **BENCH)


def test_write_bench_keeps_every_request_alongside_the_report(tmp_path: Path) -> None:
    run = run_benchmark(Recorder(), HEALTH, ["a", "b"], levels=(1, 2), **BENCH)

    report_path, requests_path = write_bench(run, tmp_path / "serve")

    assert report_path.name == REPORT_FILENAME
    assert requests_path.name == REQUESTS_FILENAME
    assert BenchReport.model_validate_json(report_path.read_text(encoding="utf-8")) == run.report
    rows = requests_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 * (BENCH["requests_per_level"] + BENCH["warmup"])
    assert {json.loads(row)["concurrency"] for row in rows} == {1, 2}
