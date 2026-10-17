"""Load benchmark for the policy server.

PROJECT.md section 9, serving row: "throughput req/s, p50 p95 p99, at
concurrency 1, 8, 32." Section 11 makes a table of those, with real numbers, the
P5 exit criterion. This module produces that table and nothing else.

**Closed loop.** At concurrency C there are C workers, and each sends its next
request the moment its previous one returns. That is what "at concurrency 8"
means: eight requests in flight, always. An open-loop benchmark - a fixed
arrival rate - answers a different question, latency at a chosen load, and
needs that rate picked in advance. Section 9 asks for concurrency.

**Every number carries an interval,** because section 9 says so and a latency
table is no exception.

* Latency percentiles: percentile bootstrap over requests, the same method as
  `eval.metrics.bootstrap_ci` applied to a quantile instead of a mean.
* Throughput: batch means. The measured window is cut into equal slices and
  each slice's completion rate is one observation. Requests are not independent
  samples of a *rate*, so bootstrapping them would produce an interval for
  something that was never measured.

**A tail quantile needs a tail.** The p99 of 100 requests is the largest one,
renamed. A quantile q is reported only when at least `MIN_TAIL_SAMPLES`
observations lie above it, so p99 needs 500 requests per level - which is why
`configs/base.yaml` asks for 500. Below that the cell is `n/a`, never a number
that looks like a p99 and is a maximum.

**Failures are counted, never dropped.** Latency is computed over requests that
succeeded, so a server that times out on its slowest 5% has a flattering p95 the
moment those timeouts disappear from the sample. They are counted in the table,
grouped by kind, and any failure makes the command exit non-zero.

**Unanswered is not failed, and is reported.** An episode that ended without
`<answer>` succeeded as HTTP and is a valid latency sample. But a policy that
gives up early is fast, so the answered fraction sits in the table beside the
latency - a quick row with a low answer rate then reads as what it is.

**Warmup is excluded.** The first requests at each level pay for CUDA graph
capture, a cold prefix cache and connection setup. `serve.warmup_requests` is
validated to cover at least one full wave at the highest level.

**Throughput is biased low, never high.** The window opens at the first measured
request, while warmup requests are still in flight and finish uncounted, and it
closes with a drain in which fewer than C requests remain. Both edges cost
completions. A conservative number is the safe direction for a reported one.

Owner: Ayesha.
"""

from __future__ import annotations

import math
import threading
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from verisearch.eval.metrics import ConfidenceInterval, bootstrap_ci
from verisearch.retrieval.client import Transport, UrllibTransport
from verisearch.serve.server import (
    SERVE_PROTOCOL_VERSION,
    AnswerRequest,
    AnswerResponse,
    ServeError,
    ServeHealth,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)

# Observations required above a quantile before it is reported. Five is a floor,
# not a precision target: below it a "p99" is one or two requests and the
# interval around it means nothing.
MIN_TAIL_SAMPLES = 5

REQUESTS_FILENAME = "requests.jsonl"
REPORT_FILENAME = "bench.json"

# The serving table's header, exactly as `writeup/report.md` section 8 prints
# it. `tests/test_report.py` holds the report to this string, so the table this
# module prints is the table the report has room for.
TABLE_HEADER = (
    "| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Answered | Failed |"
)


class BenchError(Exception):
    """A benchmark that cannot be run, or whose input cannot be trusted."""


def min_samples_for(quantile: float) -> int:
    """Requests needed before `quantile` has `MIN_TAIL_SAMPLES` observations above it.

    Rounded before the ceiling, because `1 - q` is inexact: `5 / (1 - 0.9)` is
    50.000000000000014 in floating point, and the honest answer is 50, not 51.
    """
    if not 0.0 < quantile < 1.0:
        raise BenchError(f"quantile must be in (0, 1), got {quantile}")
    return math.ceil(round(MIN_TAIL_SAMPLES / (1.0 - quantile), 9))


def quantile_ci(
    values: Sequence[float],
    quantile: float,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> ConfidenceInterval | None:
    """Percentile bootstrap interval for one quantile, or None without a tail.

    Seeded, for the reason `bootstrap_ci` is: the same latencies must give the
    same interval, or a table cannot be checked against its raw requests.
    """
    if len(values) < min_samples_for(quantile):
        return None
    if resamples < 1:
        raise BenchError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise BenchError(f"confidence must be in (0, 1), got {confidence}")

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    statistics = np.quantile(array[indices], quantile, axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(statistics, [tail, 1.0 - tail])
    return ConfidenceInterval(
        point=float(np.quantile(array, quantile)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
        n=array.size,
    )


def throughput_ci(
    completions: Sequence[float],
    *,
    window_start: float,
    window_end: float,
    batches: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> ConfidenceInterval | None:
    """Completions per second over a window, with a batch-means interval.

    `completions` are the end times of the successful requests. The point
    estimate is exactly `len(completions) / duration`, because every slice has
    the same width and the mean of their rates is the overall rate. The
    interval's `n` is `batches`, not the request count: that is how many
    observations it was computed from.
    """
    if batches < 2:
        raise BenchError(f"a batch-means interval needs at least 2 batches, got {batches}")
    duration = window_end - window_start
    if not completions or duration <= 0.0:
        return None
    if min(completions) < window_start or max(completions) > window_end:
        raise BenchError(
            "a completion falls outside the measured window, so the rate would count "
            "work the window does not cover"
        )
    counts, _ = np.histogram(
        np.asarray(completions, dtype=np.float64),
        bins=batches,
        range=(window_start, window_end),
    )
    rates = counts.astype(np.float64) / (duration / batches)
    return bootstrap_ci(
        [float(rate) for rate in rates],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Raw records. Section 15: a reported number must trace to what produced it.
# ---------------------------------------------------------------------------


class RequestRecord(BaseModel):
    """One request as it happened. Every table cell is computed from these."""

    model_config = _FROZEN

    concurrency: int = Field(ge=1)
    # Issue order within the level. Indices below the level's warmup count are
    # warmup, and `warmup` says so explicitly so a reader of the JSONL need not
    # know the config.
    index: int = Field(ge=0)
    warmup: bool
    start: float
    end: float
    ok: bool
    answered: bool
    # Both set exactly when `ok` is False. `error_kind` is what the table groups
    # by - "HTTP 503", "TimeoutError" - and `error` is the message.
    error_kind: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _a_record_is_internally_consistent(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"request {self.index} ends before it starts")
        has_kind = self.error_kind is not None
        if self.ok == has_kind or has_kind != (self.error is not None):
            raise ValueError(
                f"request {self.index}: a request either succeeded or carries an error kind "
                f"and a message - never both, never neither"
            )
        if self.answered and not self.ok:
            raise ValueError(f"request {self.index} failed, so it cannot have answered")
        return self

    @property
    def latency_ms(self) -> float:
        return (self.end - self.start) * 1000.0


class LevelRun(BaseModel):
    """Every request sent at one concurrency level, warmup included."""

    model_config = _FROZEN

    concurrency: int = Field(ge=1)
    records: tuple[RequestRecord, ...] = Field(min_length=1)
    # The most requests this client ever had in flight at once. The one check
    # that the label on the row is true: a client-side lock, a connection pool of
    # one, or a target that serialises would all report "concurrency 32" while
    # measuring something smaller.
    peak_in_flight: int = Field(ge=1)


class LevelResult(BaseModel):
    """One row of the serving table."""

    model_config = _FROZEN

    concurrency: int = Field(ge=1)
    n_measured: int = Field(ge=1)
    n_ok: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_answered: int = Field(ge=0)
    # Not in the table, because warmup is not measured. Kept because a server
    # that fails its first wave and then recovers is worth knowing about.
    n_warmup_failed: int = Field(ge=0)
    # None when no measured request succeeded.
    throughput: ConfidenceInterval | None
    # Each None when there are too few successful requests for that quantile to
    # have a tail - see `min_samples_for`.
    p50_ms: ConfidenceInterval | None
    p95_ms: ConfidenceInterval | None
    p99_ms: ConfidenceInterval | None
    peak_in_flight: int = Field(ge=1)
    concurrency_reached: bool
    errors: dict[str, int]

    @property
    def answered_rate(self) -> float | None:
        """Over requests that succeeded. None when none did - not 0.0, not 1.0."""
        return self.n_answered / self.n_ok if self.n_ok else None


def summarize(
    run: LevelRun,
    *,
    batches: int,
    resamples: int,
    confidence: float,
    seed: int,
) -> LevelResult:
    """One `LevelRun` in, one table row out."""
    measured = [record for record in run.records if not record.warmup]
    if not measured:
        raise BenchError(
            f"concurrency {run.concurrency}: every request was warmup, so there is "
            f"nothing to report"
        )
    succeeded = [record for record in measured if record.ok]
    latencies = [record.latency_ms for record in succeeded]

    def latency(quantile: float) -> ConfidenceInterval | None:
        return quantile_ci(
            latencies, quantile, resamples=resamples, confidence=confidence, seed=seed
        )

    errors = Counter(record.error_kind for record in measured if record.error_kind is not None)
    return LevelResult(
        concurrency=run.concurrency,
        n_measured=len(measured),
        n_ok=len(succeeded),
        n_failed=len(measured) - len(succeeded),
        n_answered=sum(1 for record in succeeded if record.answered),
        n_warmup_failed=sum(1 for record in run.records if record.warmup and not record.ok),
        throughput=throughput_ci(
            [record.end for record in succeeded],
            window_start=min(record.start for record in measured),
            window_end=max(record.end for record in measured),
            batches=batches,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
        p50_ms=latency(0.50),
        p95_ms=latency(0.95),
        p99_ms=latency(0.99),
        peak_in_flight=run.peak_in_flight,
        # A level sending fewer requests than its concurrency cannot fill every
        # worker, so that is the most it could have reached.
        concurrency_reached=run.peak_in_flight >= min(run.concurrency, len(run.records)),
        errors=dict(sorted(errors.items())),
    )


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


class BenchReport(BaseModel):
    """The serving table, and everything needed to say what it measured."""

    model_config = _FROZEN

    server: ServeHealth
    requests_per_level: int = Field(ge=1)
    warmup_requests: int = Field(ge=0)
    request_timeout_s: float = Field(gt=0.0)
    n_questions: int = Field(ge=1)
    # True when the question pool was smaller than the requests sent across all
    # levels, so some question reached the server more than once. vLLM's prefix
    # cache then favours whichever level asked it second.
    questions_reused: bool
    throughput_batches: int = Field(ge=2)
    resamples: int = Field(ge=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    levels: tuple[LevelResult, ...] = Field(min_length=1)

    def problems(self) -> tuple[str, ...]:
        """Why this table should not be written up as section 9's, if it should not.

        Empty means publishable. The CLI exits 1 on anything here, after
        writing the files, so the numbers survive for diagnosis but a script
        chaining on the exit code does not pass them along as a result.
        """
        found: list[str] = []
        p99_floor = min_samples_for(0.99)
        for level in self.levels:
            label = f"concurrency {level.concurrency}"
            if level.n_failed:
                kinds = ", ".join(f"{kind} x{count}" for kind, count in level.errors.items())
                found.append(
                    f"{label}: {level.n_failed} of {level.n_measured} requests failed ({kinds})"
                )
            if not level.concurrency_reached:
                found.append(
                    f"{label}: at most {level.peak_in_flight} requests were ever in flight, "
                    f"so this row is not a measurement at concurrency {level.concurrency}"
                )
            if level.p99_ms is None:
                found.append(
                    f"{label}: p99 needs {p99_floor} successful requests and this level "
                    f"has {level.n_ok}; section 9 requires it"
                )
        return tuple(found)

    def table(self) -> str:
        """Section 8 of `writeup/report.md`, ready to paste: rows, then the footer."""
        lines = [TABLE_HEADER, "|---|---|---|---|---|---|---|"]
        for level in self.levels:
            answered = level.answered_rate
            cells = [
                str(level.concurrency),
                _cell(level.throughput, digits=2),
                _cell(level.p50_ms, digits=0),
                _cell(level.p95_ms, digits=0),
                _cell(level.p99_ms, digits=0),
                f"{answered:.3f}" if answered is not None else "n/a",
                f"{level.n_failed} / {level.n_measured}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n\n" + self.footer()

    def footer(self) -> str:
        """The line that makes the table a result rather than a set of numbers.

        Every value in it is read from the server's `/health` or from the run.
        Nothing is typed by whoever ran the script.
        """
        adapter = (
            f"adapter `{self.server.adapter}`"
            if self.server.adapter is not None
            else "no adapter (base model)"
        )
        pct = round(self.confidence * 100)
        footer = (
            f"Measured on `{self.server.hardware}`, serving `{self.server.policy}` with "
            f"{adapter}, server config `{self.server.config_fingerprint}`. Closed loop, "
            f"{self.requests_per_level} requests per level after {self.warmup_requests} "
            f"warmup, {self.request_timeout_s:g}s timeout, {self.n_questions} distinct "
            f"questions. {pct}% intervals: percentile bootstrap over requests "
            f"({self.resamples} resamples) for latency, {self.throughput_batches} batch "
            f"means for throughput."
        )
        if self.questions_reused:
            footer += (
                " Some questions were sent more than once, so the prefix cache may favour "
                "levels that ran later."
            )
        return footer


def _cell(interval: ConfidenceInterval | None, *, digits: int) -> str:
    if interval is None:
        return "n/a"
    return f"{interval.point:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]"


class BenchRun(BaseModel):
    """A report plus the requests it was computed from. Both, always."""

    model_config = _FROZEN

    report: BenchReport
    runs: tuple[LevelRun, ...] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Sending load.
# ---------------------------------------------------------------------------


class BenchTarget(Protocol):
    """Anything that answers a question.

    `HttpPolicyClient` over the wire, or a runner wrapped in-process. The gap
    between those two tables is the HTTP overhead, and that is the cheapest way
    to measure it.
    """

    def answer(self, request: AnswerRequest) -> AnswerResponse: ...


class _Dispatcher:
    """Hands out request indices in order and counts what is in flight."""

    def __init__(self, total: int) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self._total = total
        self._in_flight = 0
        self.peak = 0

    def take(self) -> int | None:
        with self._lock:
            if self._next >= self._total:
                return None
            index = self._next
            self._next += 1
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
            return index

    def done(self) -> None:
        with self._lock:
            self._in_flight -= 1


def _send(
    target: BenchTarget,
    request: AnswerRequest,
    *,
    concurrency: int,
    index: int,
    warmup: bool,
    clock: Callable[[], float],
) -> RequestRecord:
    """One request, timed. Expected failures become records; bugs propagate.

    "Expected" is narrow on purpose: the server refusing (`ServeError`) and the
    network failing (`OSError` - refused connections, resets, timeouts).
    Anything else is a defect in this module or in the target, and recording
    it as a failed request would bury it inside a table.
    """
    failure: tuple[str, str] | None = None
    response: AnswerResponse | None = None
    start = clock()
    try:
        response = target.answer(request)
    except ServeError as exc:
        failure = (f"HTTP {exc.status_code}", str(exc))
    except OSError as exc:
        failure = (type(exc).__name__, str(exc))
    end = clock()

    if response is not None and response.question_id != request.question_id:
        failure = (
            "mismatched response",
            f"sent {request.question_id!r}, got the answer to {response.question_id!r}",
        )
    return RequestRecord(
        concurrency=concurrency,
        index=index,
        warmup=warmup,
        start=start,
        end=end,
        ok=failure is None,
        answered=failure is None and response is not None and response.answer is not None,
        error_kind=failure[0] if failure is not None else None,
        error=failure[1] if failure is not None else None,
    )


def run_level(
    target: BenchTarget,
    questions: Sequence[str],
    *,
    concurrency: int,
    n_requests: int,
    warmup: int,
    offset: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> LevelRun:
    """Send `warmup + n_requests` requests with `concurrency` workers, closed loop.

    Request i asks question `(offset + i) % len(questions)`, so a level's
    questions are deterministic, and two levels given different offsets ask
    different questions for as long as the pool lasts.

    Each request carries a distinct id, `bench-c{concurrency}-{i}`, and a reply
    under any other id is recorded as a failure: under load, that is one
    request receiving another's answer.
    """
    if concurrency < 1:
        raise BenchError(f"concurrency must be >= 1, got {concurrency}")
    if n_requests < 1:
        raise BenchError(f"n_requests must be >= 1, got {n_requests}")
    if warmup < 0:
        raise BenchError(f"warmup must be >= 0, got {warmup}")
    if not questions:
        raise BenchError("no questions to send")

    total = warmup + n_requests
    records: list[RequestRecord | None] = [None] * total
    dispatcher = _Dispatcher(total)

    def drain() -> None:
        while (index := dispatcher.take()) is not None:
            try:
                request = AnswerRequest(
                    question=questions[(offset + index) % len(questions)],
                    question_id=f"bench-c{concurrency}-{index}",
                )
                records[index] = _send(
                    target,
                    request,
                    concurrency=concurrency,
                    index=index,
                    warmup=index < warmup,
                    clock=clock,
                )
            finally:
                dispatcher.done()

    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix=f"bench-c{concurrency}"
    ) as pool:
        workers = [pool.submit(drain) for _ in range(concurrency)]
        for worker in workers:
            # Re-raises whatever `_send` let through. That is a bug, and it
            # should stop the benchmark rather than become a row.
            worker.result()

    sent = tuple(record for record in records if record is not None)
    assert len(sent) == total, "every index was taken, so every record was written"
    return LevelRun(concurrency=concurrency, records=sent, peak_in_flight=dispatcher.peak)


def run_benchmark(
    target: BenchTarget,
    server: ServeHealth,
    questions: Sequence[str],
    *,
    levels: Sequence[int],
    requests_per_level: int,
    warmup: int,
    request_timeout_s: float,
    batches: int,
    resamples: int,
    confidence: float,
    seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchRun:
    """Every level, in the order given, then the report.

    Every question is validated before any load is sent. One the server would
    refuse - longer than `MAX_QUESTION_CHARS` - would otherwise surface as a
    crash partway through the second level, an hour after the first began.

    Levels take consecutive slices of the question pool, so the prefix cache
    cannot hand a later level the questions an earlier one just asked. When the
    pool runs out the slices wrap, and the report says so.
    """
    if not levels:
        raise BenchError("no concurrency levels to run")
    if not questions:
        raise BenchError("no questions to send")
    for position, question in enumerate(questions):
        try:
            AnswerRequest(question=question)
        except ValueError as exc:
            raise BenchError(f"question {position} cannot be sent: {exc}") from exc

    per_level = warmup + requests_per_level
    runs = tuple(
        run_level(
            target,
            questions,
            concurrency=concurrency,
            n_requests=requests_per_level,
            warmup=warmup,
            offset=position * per_level,
            clock=clock,
        )
        for position, concurrency in enumerate(levels)
    )
    distinct = len(set(questions))
    report = BenchReport(
        server=server,
        requests_per_level=requests_per_level,
        warmup_requests=warmup,
        request_timeout_s=request_timeout_s,
        n_questions=distinct,
        questions_reused=distinct < per_level * len(levels),
        throughput_batches=batches,
        resamples=resamples,
        confidence=confidence,
        levels=tuple(
            summarize(run, batches=batches, resamples=resamples, confidence=confidence, seed=seed)
            for run in runs
        ),
    )
    return BenchRun(report=report, runs=runs)


def write_bench(run: BenchRun, directory: str | Path) -> tuple[Path, Path]:
    """Write the report as JSON and every request as JSONL. Returns both paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / REPORT_FILENAME
    report_path.write_text(run.report.model_dump_json(indent=2), encoding="utf-8")
    requests_path = target / REQUESTS_FILENAME
    with requests_path.open("w", encoding="utf-8") as handle:
        for level in run.runs:
            for record in level.records:
                handle.write(record.model_dump_json())
                handle.write("\n")
    return report_path, requests_path


# ---------------------------------------------------------------------------
# The wire.
# ---------------------------------------------------------------------------


class HttpPolicyClient:
    """Talks to `serve/server.py`. Implements `BenchTarget`.

    **No retries, unlike `HttpRetriever`.** A retried request reports the
    latency of the attempt that worked and hides the one that failed - and the
    failed one is what the table's last column exists to show. Inside a
    training run a retry is resilience; inside a benchmark it is a thumb on the
    scale.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        transport: Transport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise BenchError(f"the policy server URL must be http(s), got {base_url!r}")
        if timeout <= 0:
            raise BenchError(f"timeout must be positive, got {timeout}")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport: Transport = transport if transport is not None else UrllibTransport()

    @classmethod
    def connect(
        cls,
        base_url: str,
        *,
        timeout: float,
        transport: Transport | None = None,
    ) -> tuple[HttpPolicyClient, ServeHealth]:
        """Build a client and read what the server is serving, before any load."""
        client = cls(base_url, timeout=timeout, transport=transport)
        health = client.health()
        if health.protocol_version != SERVE_PROTOCOL_VERSION:
            raise BenchError(
                f"server speaks protocol {health.protocol_version}, this client speaks "
                f"{SERVE_PROTOCOL_VERSION}; benchmarking across a wire change would time "
                f"a stream of errors"
            )
        if not health.ok:
            raise BenchError(f"{base_url} reports itself unhealthy; not sending load to it")
        return client, health

    def health(self) -> ServeHealth:
        return self._validate(ServeHealth, self._fetch("/health", None), "/health")

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        payload = request.model_dump_json().encode("utf-8")
        return self._validate(AnswerResponse, self._fetch("/answer", payload), "/answer")

    def _fetch(self, path: str, payload: bytes | None) -> bytes:
        url = f"{self._base_url}{path}"
        try:
            return self._transport.request(url, payload, self._timeout)
        except urllib.error.HTTPError as exc:
            # The server puts its reason in `detail`. Every other OSError -
            # refused, reset, timed out - propagates as itself, so the benchmark
            # can group failures by what actually happened.
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ServeError(
                f"{url} returned HTTP {exc.code}: {detail}", status_code=exc.code
            ) from exc

    def _validate(self, model: type[_ResponseT], raw: bytes, path: str) -> _ResponseT:
        try:
            return model.model_validate_json(raw)
        except ValueError as exc:
            # 502: the server answered, with something that is not the contract.
            raise ServeError(
                f"{self._base_url}{path} returned something that is not a {model.__name__}: {exc}",
                status_code=502,
            ) from exc
