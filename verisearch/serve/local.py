"""The policy server's handlers behind the standard library, plus a fake policy.

`create_app` in `server.py` is the FastAPI shell, and FastAPI is not installed
in CI. This is the same wiring behind `http.server.ThreadingHTTPServer`, so the
client, the handlers and the benchmark can be exercised over a real socket on
any machine - the role `make_wire_server` plays in the retrieval tests, moved
into the package so `mypy --strict` checks it and the CLI tests can share it.

With `FakePolicyRunner` behind it, the whole measurement path - load
generation, timing, intervals, the table - runs with no model and no GPU. That
is how the benchmark is tested, and it is how to check a change to it before
spending GPU time on one.

Not a production server, and nothing here is benchmarked as one. Section 9's
numbers come from the FastAPI shell in front of the real runner.

Owner: Ayesha.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from verisearch.serve.server import (
    AnswerRequest,
    PolicyRunner,
    ServeError,
    ServeHealth,
    handle_answer,
)
from verisearch.types import Passage, Termination, Trajectory, Turn, TurnKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from verisearch.serve.benchmark import BenchTarget, HttpPolicyClient

# The listen backlog. `socketserver`'s default is 5, and on Windows a full
# backlog refuses the connection outright - so 32 workers connecting at once
# would record refusals that are an artefact of this test server.
_BACKLOG = 128


class FakePolicyRunner:
    """A deterministic episode: one search, one passage, one answer.

    `delay_s` holds each call for that long, which is what lets a test see
    requests overlap. Safe to call from many threads, as `PolicyRunner` requires.
    """

    def __init__(self, *, delay_s: float = 0.0) -> None:
        if delay_s < 0:
            raise ValueError(f"delay_s must be >= 0, got {delay_s}")
        self._delay_s = delay_s
        self._lock = threading.Lock()
        self.calls = 0

    def run(self, question_id: str, question: str) -> Trajectory:
        with self._lock:
            self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        passage = Passage(doc_id="fake-0", title="Fake", text=f"Evidence about: {question}")
        return Trajectory(
            question_id=question_id,
            question=question,
            gold_answers=(),
            turns=(
                Turn(kind=TurnKind.SEARCH, text=question),
                Turn(kind=TurnKind.INFO, text="<info/>", passages=(passage,)),
                Turn(kind=TurnKind.ANSWER, text="fake answer"),
            ),
            answer="fake answer",
            termination=Termination.ANSWERED,
        )


def make_policy_server(
    runner: PolicyRunner,
    health: ServeHealth,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Bind a server wired exactly as `create_app` wires FastAPI. Port 0 picks one."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            """Silence the per-request stderr line."""

        def _send(self, code: int, body: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, health.model_dump_json())
            else:
                self._send(404, json.dumps({"detail": f"no route {self.path}"}))

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path != "/answer":
                self._send(404, json.dumps({"detail": f"no route {self.path}"}))
                return
            try:
                response = handle_answer(runner, AnswerRequest.model_validate_json(body))
            except ServeError as exc:
                self._send(exc.status_code, json.dumps({"detail": str(exc)}))
                return
            except ValueError as exc:
                # A body that is not an `AnswerRequest`. FastAPI calls this 422.
                self._send(422, json.dumps({"detail": str(exc)}))
                return
            self._send(200, response.model_dump_json())

    class Server(ThreadingHTTPServer):
        request_queue_size = _BACKLOG

    return Server((host, port), Handler)


@contextmanager
def serving(server: ThreadingHTTPServer) -> Iterator[str]:
    """Run `server` on a background thread and yield its base URL."""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host!s}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def conformance_witnesses(
    runner: FakePolicyRunner, client: HttpPolicyClient
) -> tuple[PolicyRunner, BenchTarget]:
    """Return each typed as the protocol it claims, so mypy checks the fit.

    The same mechanism as `retrieval.client.conformance_witnesses`: `tests/` is
    outside mypy's `files`, so a conformance assertion written there can never
    fail. Here the return annotation does the checking, and the test that
    calls this keeps it from being dead code.
    """
    return runner, client
