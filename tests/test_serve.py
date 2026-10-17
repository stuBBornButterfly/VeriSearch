"""Tests for the policy server: `serve/server.py` and its stdlib shell in `serve/local.py`.

Three layers, cheapest first, the same shape as `tests/test_service.py`:

1. the wire models - bounds and `extra="forbid"`
2. `handle_answer`, driven directly with scripted runners, no socket
3. `HttpPolicyClient` against `make_policy_server` over a real socket - the
   same handlers, behind the standard library instead of FastAPI

`create_app` is not tested and cannot be: FastAPI is in the `[retrieval]`
extra. It carries a `pragma: no cover` and holds nothing but route wiring, which
`make_policy_server` duplicates.
"""

from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from verisearch.retrieval.client import RetrievalError, UrllibTransport
from verisearch.serve.benchmark import BenchError, HttpPolicyClient
from verisearch.serve.local import (
    FakePolicyRunner,
    conformance_witnesses,
    make_policy_server,
    serving,
)
from verisearch.serve.server import (
    MAX_QUESTION_CHARS,
    MAX_QUESTION_ID_CHARS,
    SERVE_PROTOCOL_VERSION,
    AnswerRequest,
    AnswerResponse,
    ServeError,
    ServeHealth,
    Source,
    handle_answer,
    handle_health,
    question_id_for,
)
from verisearch.types import Passage, Termination, Trajectory, Turn, TurnKind

if TYPE_CHECKING:
    from collections.abc import Iterator

HEALTH = handle_health(
    policy="Qwen/Qwen2.5-1.5B-Instruct",
    adapter=None,
    hardware="test-cpu",
    config_fingerprint="abc123def456",
)


def episode(
    question_id: str,
    question: str,
    *,
    answer: str | None = "Ada Lovelace",
    passages: tuple[Passage, ...] = (Passage(doc_id="d1", title="Ada", text="Body."),),
) -> Trajectory:
    turns = [
        Turn(kind=TurnKind.SEARCH, text="who was ada"),
        Turn(kind=TurnKind.INFO, text="<info/>", passages=passages),
    ]
    if answer is not None:
        turns.append(Turn(kind=TurnKind.ANSWER, text=answer))
    return Trajectory(
        question_id=question_id,
        question=question,
        gold_answers=(),
        turns=tuple(turns),
        answer=answer,
        termination=Termination.ANSWERED if answer is not None else Termination.MAX_TURNS,
    )


class ScriptedRunner:
    """Returns what it is built to return, or raises what it is built to raise."""

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        answer: str | None = "Ada Lovelace",
        passages: tuple[Passage, ...] | None = None,
        reply_as: tuple[str, str] | None = None,
    ) -> None:
        self._raises = raises
        self._answer = answer
        self._passages = passages
        self._reply_as = reply_as
        self.seen: list[tuple[str, str]] = []

    def run(self, question_id: str, question: str) -> Trajectory:
        self.seen.append((question_id, question))
        if self._raises is not None:
            raise self._raises
        qid, text = self._reply_as if self._reply_as is not None else (question_id, question)
        if self._passages is None:
            return episode(qid, text, answer=self._answer)
        return episode(qid, text, answer=self._answer, passages=self._passages)


# ---------------------------------------------------------------------------
# 1. The wire models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [pytest.param("", id="empty"), pytest.param("x" * (MAX_QUESTION_CHARS + 1), id="too-long")],
)
def test_the_request_bounds_the_question(question: str) -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(question=question)


def test_a_question_at_the_bound_is_accepted() -> None:
    assert len(AnswerRequest(question="x" * MAX_QUESTION_CHARS).question) == MAX_QUESTION_CHARS


@pytest.mark.parametrize(
    "question_id",
    [pytest.param("", id="empty"), pytest.param("q" * (MAX_QUESTION_ID_CHARS + 1), id="too-long")],
)
def test_the_request_bounds_the_question_id(question_id: str) -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(question="Who?", question_id=question_id)


def test_the_wire_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest.model_validate({"question": "Who?", "temperature": 0.7})


def test_a_response_round_trips_through_json_unchanged() -> None:
    response = handle_answer(ScriptedRunner(), AnswerRequest(question="Who?", question_id="q1"))
    assert AnswerResponse.model_validate_json(response.model_dump_json()) == response


# ---------------------------------------------------------------------------
# 2. The handler
# ---------------------------------------------------------------------------


def test_handle_answer_returns_what_the_episode_produced() -> None:
    response = handle_answer(ScriptedRunner(), AnswerRequest(question="Who?", question_id="q1"))
    assert response.question_id == "q1"
    assert response.answer == "Ada Lovelace"
    assert response.termination is Termination.ANSWERED
    assert response.search_queries == ("who was ada",)
    assert response.sources == (Source(doc_id="d1", title="Ada"),)


def test_sources_are_listed_once_each_in_the_order_they_were_first_seen() -> None:
    passages = (
        Passage(doc_id="d2", title="Second", text="."),
        Passage(doc_id="d1", title="First", text="."),
        Passage(doc_id="d2", title="Second", text="."),
    )
    response = handle_answer(
        ScriptedRunner(passages=passages), AnswerRequest(question="Who?", question_id="q1")
    )
    assert [source.doc_id for source in response.sources] == ["d2", "d1"]


def test_an_unanswered_episode_is_a_success_with_no_answer() -> None:
    """The server worked; the policy gave up. Not an error - and not hidden either."""
    response = handle_answer(
        ScriptedRunner(answer=None), AnswerRequest(question="Who?", question_id="q1")
    )
    assert response.answer is None
    assert response.termination is Termination.MAX_TURNS


def test_a_blank_question_is_a_400_and_never_reaches_the_runner() -> None:
    runner = ScriptedRunner()
    with pytest.raises(ServeError, match="blank") as caught:
        handle_answer(runner, AnswerRequest(question="   "))
    assert caught.value.status_code == 400
    assert runner.seen == []


def test_a_request_without_an_id_gets_a_stable_one_from_its_text() -> None:
    runner = ScriptedRunner()
    first = handle_answer(runner, AnswerRequest(question="Who?"))
    second = handle_answer(runner, AnswerRequest(question="Who?"))
    other = handle_answer(runner, AnswerRequest(question="Where?"))
    assert first.question_id == second.question_id == question_id_for("Who?")
    assert first.question_id.startswith("serve-")
    assert other.question_id != first.question_id


def test_retrieval_failing_mid_episode_is_a_503_never_an_answer() -> None:
    """An answer written without the passages it asked for is the exact
    unsupported claim this project measures - with a 200 on it."""
    runner = ScriptedRunner(raises=RetrievalError("service down"))
    with pytest.raises(ServeError, match="retrieval failed mid-episode") as caught:
        handle_answer(runner, AnswerRequest(question="Who?"))
    assert caught.value.status_code == 503


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ConnectionRefusedError("vllm"), id="refused"),
        pytest.param(TimeoutError("vllm"), id="timeout"),
        pytest.param(urllib.error.URLError("vllm"), id="urlerror"),
    ],
)
def test_an_unreachable_dependency_is_a_503(error: OSError) -> None:
    with pytest.raises(ServeError, match="unreachable") as caught:
        handle_answer(ScriptedRunner(raises=error), AnswerRequest(question="Who?"))
    assert caught.value.status_code == 503


def test_any_other_runner_error_propagates_as_the_bug_it_is() -> None:
    with pytest.raises(KeyError):
        handle_answer(ScriptedRunner(raises=KeyError("turn")), AnswerRequest(question="Who?"))


@pytest.mark.parametrize(
    "reply_as",
    [
        pytest.param(("q2", "Who?"), id="another-id"),
        pytest.param(("q1", "Where?"), id="another-question"),
    ],
)
def test_an_episode_for_another_request_is_refused(reply_as: tuple[str, str]) -> None:
    """Under concurrency this is one caller receiving another's answer, and every
    field of it is well-formed."""
    runner = ScriptedRunner(reply_as=reply_as)
    with pytest.raises(ServeError, match="another caller") as caught:
        handle_answer(runner, AnswerRequest(question="Who?", question_id="q1"))
    assert caught.value.status_code == 500


def test_health_reports_the_protocol_version_and_what_is_served() -> None:
    assert HEALTH.ok
    assert HEALTH.protocol_version == SERVE_PROTOCOL_VERSION
    assert HEALTH.adapter is None
    assert HEALTH.hardware == "test-cpu"


# ---------------------------------------------------------------------------
# The fake runner
# ---------------------------------------------------------------------------


def test_the_fake_runner_answers_the_question_it_was_asked() -> None:
    runner = FakePolicyRunner()
    response = handle_answer(runner, AnswerRequest(question="Who?", question_id="q7"))
    assert response.question_id == "q7"
    assert response.answer == "fake answer"
    assert runner.calls == 1


def test_the_fake_runner_counts_calls_from_many_threads() -> None:
    runner = FakePolicyRunner()
    threads = [threading.Thread(target=runner.run, args=(f"q{i}", "Who?")) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert runner.calls == 16


def test_the_fake_runner_refuses_a_negative_delay() -> None:
    with pytest.raises(ValueError, match="delay_s"):
        FakePolicyRunner(delay_s=-1.0)


def test_the_fake_and_the_client_satisfy_the_protocols_they_claim() -> None:
    """The check is mypy's, on `conformance_witnesses`' annotation. This keeps it live."""
    runner = FakePolicyRunner()
    client = HttpPolicyClient("http://127.0.0.1:1", timeout=1.0)
    assert conformance_witnesses(runner, client) == (runner, client)


# ---------------------------------------------------------------------------
# 3. The real thing: client and handlers over a socket
# ---------------------------------------------------------------------------


@pytest.fixture
def live() -> Iterator[tuple[str, FakePolicyRunner]]:
    runner = FakePolicyRunner()
    with serving(make_policy_server(runner, HEALTH)) as url:
        yield url, runner


@pytest.mark.integration
def test_end_to_end_answer_over_a_real_socket(live: tuple[str, FakePolicyRunner]) -> None:
    url, runner = live
    client, health = HttpPolicyClient.connect(url, timeout=5.0)
    assert health == HEALTH

    response = client.answer(AnswerRequest(question="Who wrote it?", question_id="q1"))

    assert response.question_id == "q1"
    assert response.answer == "fake answer"
    assert response.sources == (Source(doc_id="fake-0", title="Fake"),)
    assert runner.calls == 1


@pytest.mark.integration
def test_a_server_side_refusal_arrives_as_a_serve_error_with_its_status() -> None:
    runner = ScriptedRunner(raises=RetrievalError("service down"))
    with serving(make_policy_server(runner, HEALTH)) as url:
        client, _ = HttpPolicyClient.connect(url, timeout=5.0)
        with pytest.raises(ServeError, match="HTTP 503") as caught:
            client.answer(AnswerRequest(question="Who?"))
    assert caught.value.status_code == 503


@pytest.mark.integration
def test_a_blank_question_comes_back_as_a_400(live: tuple[str, FakePolicyRunner]) -> None:
    url, runner = live
    client, _ = HttpPolicyClient.connect(url, timeout=5.0)
    with pytest.raises(ServeError) as caught:
        client.answer(AnswerRequest(question="   "))
    assert caught.value.status_code == 400
    assert runner.calls == 0


@pytest.mark.integration
def test_a_body_that_is_not_a_request_is_a_422(live: tuple[str, FakePolicyRunner]) -> None:
    url, _ = live

    class Raw:
        def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
            return UrllibTransport().request(url, b'{"prompt": "x"}', timeout)

    client = HttpPolicyClient(url, timeout=5.0, transport=Raw())
    with pytest.raises(ServeError) as caught:
        client.answer(AnswerRequest(question="Who?"))
    assert caught.value.status_code == 422


@pytest.mark.integration
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_an_unknown_route_is_a_404(live: tuple[str, FakePolicyRunner], method: str) -> None:
    url, _ = live
    request = urllib.request.Request(
        f"{url}/nowhere", data=b"{}" if method == "POST" else None, method=method
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 404


@pytest.mark.integration
def test_nothing_listening_is_an_oserror_not_a_serve_error() -> None:
    """The benchmark groups failures by what happened, so a refused connection
    must arrive as itself rather than being dressed up as an HTTP status."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    client = HttpPolicyClient(f"http://127.0.0.1:{port}", timeout=2.0)
    with pytest.raises(OSError):  # noqa: PT011 - the type is the assertion
        client.answer(AnswerRequest(question="Who?"))


# ---------------------------------------------------------------------------
# The client's own checks, with a stub transport
# ---------------------------------------------------------------------------


class StubTransport:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        return self._body


def health_bytes(**changes: object) -> bytes:
    return HEALTH.model_copy(update=changes).model_dump_json().encode("utf-8")


def test_connect_refuses_a_server_on_another_protocol_version() -> None:
    transport = StubTransport(health_bytes(protocol_version=SERVE_PROTOCOL_VERSION + 1))
    with pytest.raises(BenchError, match="protocol"):
        HttpPolicyClient.connect("http://server", timeout=1.0, transport=transport)


def test_connect_refuses_a_server_that_reports_itself_unhealthy() -> None:
    with pytest.raises(BenchError, match="unhealthy"):
        HttpPolicyClient.connect(
            "http://server", timeout=1.0, transport=StubTransport(health_bytes(ok=False))
        )


def test_a_reply_that_is_not_the_contract_is_a_502() -> None:
    client = HttpPolicyClient("http://server", timeout=1.0, transport=StubTransport(b"<html>"))
    with pytest.raises(ServeError, match="not a AnswerResponse") as caught:
        client.answer(AnswerRequest(question="Who?"))
    assert caught.value.status_code == 502


@pytest.mark.parametrize(
    ("url", "timeout", "message"),
    [
        pytest.param("file:///etc/passwd", 1.0, "http", id="not-http"),
        pytest.param("http://server", 0.0, "timeout", id="zero-timeout"),
    ],
)
def test_the_client_refuses_a_bad_url_or_timeout_before_any_io(
    url: str, timeout: float, message: str
) -> None:
    with pytest.raises(BenchError, match=message):
        HttpPolicyClient(url, timeout=timeout)


def test_a_trailing_slash_on_the_url_does_not_double_up() -> None:
    seen: list[str] = []

    class Recording:
        def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
            seen.append(url)
            return health_bytes()

    HttpPolicyClient("http://server/", timeout=1.0, transport=Recording()).health()
    assert seen == ["http://server/health"]


def test_serve_health_requires_a_named_policy_and_hardware() -> None:
    with pytest.raises(ValidationError):
        ServeHealth(
            ok=True,
            protocol_version=1,
            policy="",
            adapter=None,
            hardware="",
            config_fingerprint="x",
        )
