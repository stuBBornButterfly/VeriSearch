"""Tests for `eval.policy`: the vLLM-backed policy behind the episode loop.

No GPU and no vLLM. Most tests stub the transport and read the JSON it was
handed; a few send real bytes to a stdlib HTTP server that answers the way
`vllm serve` does. The real server is only exercised on the GPU box, at P1's
exit run.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from verisearch.data.datasets import Distribution, QAExample
from verisearch.env.episode import EpisodeConfig, Question, run_episode
from verisearch.env.protocol import STOP_SEQUENCES
from verisearch.eval.policy import (
    COMPLETIONS_PATH,
    PolicyError,
    Sampling,
    VllmHttpPolicy,
    conformance_witness,
    question_from,
)
from verisearch.testing import FakeRetriever
from verisearch.types import ParseFailure, Termination

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

BASE = "http://policy.test:8000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STOP = ("</search>",)


def completion(text: str) -> bytes:
    """What `vllm serve` sends back for n=1, trimmed to the fields that vary."""
    return json.dumps(
        {
            "id": "cmpl-1",
            "object": "text_completion",
            "model": MODEL,
            "choices": [
                {"index": 0, "text": text, "finish_reason": "stop", "stop_reason": "</search>"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode("utf-8")


def http_error(code: int, detail: str = "nope") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        BASE + COMPLETIONS_PATH, code, "error", Message(), io.BytesIO(detail.encode("utf-8"))
    )


class ScriptedTransport:
    """Answers each request from a script, and records what it was sent.

    An entry is either the bytes to return or an exception to raise. A callable
    entry is called with the request body, for answers that depend on it.
    """

    def __init__(self, *script: bytes | Exception | Callable[[dict[str, object]], bytes]) -> None:
        self.script = list(script)
        self.sent: list[tuple[str, dict[str, object], float]] = []
        self._lock = threading.Lock()

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        assert payload is not None, "a completion is always a POST"
        body = json.loads(payload)
        with self._lock:
            self.sent.append((url, body, timeout))
            step = self.script[0] if len(self.script) == 1 else self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(body)
        return step


def policy(
    transport: ScriptedTransport,
    *,
    sampling: Sampling | None = None,
    timeout: float = 30.0,
    retries: int = 1,
) -> VllmHttpPolicy:
    return VllmHttpPolicy(
        BASE, MODEL, sampling=sampling, transport=transport, timeout=timeout, retries=retries
    )


# ---------------------------------------------------------------------------
# 1. The request
# ---------------------------------------------------------------------------


def test_the_stop_string_is_kept_in_the_output() -> None:
    """The one field that cannot be left to vLLM's default. Without it the
    parser never sees `</search>` and every episode ends on turn one."""
    transport = ScriptedTransport(completion("<search>ada</search>"))
    policy(transport).generate("prompt", stop=STOP, max_new_tokens=64)

    (_, body, _) = transport.sent[0]
    assert body["include_stop_str_in_output"] is True


def test_the_request_carries_the_turn_limits_and_the_model() -> None:
    transport = ScriptedTransport(completion("x"))
    policy(transport).generate("the prompt so far", stop=STOP, max_new_tokens=512)

    (url, body, _) = transport.sent[0]
    assert url == BASE + "/v1/completions"
    assert body["model"] == MODEL
    assert body["prompt"] == "the prompt so far"
    assert body["max_tokens"] == 512
    assert body["stop"] == ["</search>"]
    assert body["n"] == 1


def test_it_uses_completions_not_chat() -> None:
    """Training feeds `render_prompt`'s raw text, with no chat template. The
    chat endpoint would wrap it in one and eval would measure another input."""
    transport = ScriptedTransport(completion("x"))
    policy(transport).generate("p", stop=STOP, max_new_tokens=8)
    (url, body, _) = transport.sent[0]
    assert "chat" not in url
    assert "messages" not in body


def test_the_prompt_is_sent_verbatim() -> None:
    """No stripping, no template: whitespace and tags are part of the input."""
    prompt = "  System.\n<info>\n[1] Ada\n</info>\n\n"
    transport = ScriptedTransport(completion("x"))
    policy(transport).generate(prompt, stop=STOP, max_new_tokens=8)
    assert transport.sent[0][1]["prompt"] == prompt


def test_greedy_by_default() -> None:
    transport = ScriptedTransport(completion("x"))
    policy(transport).generate("p", stop=STOP, max_new_tokens=8)
    body = transport.sent[0][1]
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert "seed" not in body


def test_sampling_settings_reach_the_request() -> None:
    transport = ScriptedTransport(completion("x"))
    sampled = policy(transport, sampling=Sampling(temperature=0.7, top_p=0.9, seed=13))
    sampled.generate("p", stop=STOP, max_new_tokens=8)
    body = transport.sent[0][1]
    assert (body["temperature"], body["top_p"], body["seed"]) == (0.7, 0.9, 13)
    assert sampled.sampling == Sampling(temperature=0.7, top_p=0.9, seed=13)


def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    transport = ScriptedTransport(completion("x"))
    VllmHttpPolicy(BASE + "/", MODEL, transport=transport).generate(
        "p", stop=STOP, max_new_tokens=8
    )
    assert transport.sent[0][0] == BASE + "/v1/completions"


def test_the_timeout_reaches_the_transport() -> None:
    transport = ScriptedTransport(completion("x"))
    policy(transport, timeout=7.5).generate("p", stop=STOP, max_new_tokens=8)
    assert transport.sent[0][2] == 7.5


# ---------------------------------------------------------------------------
# 2. The response
# ---------------------------------------------------------------------------


def test_the_text_comes_back_unchanged_with_its_stop_string() -> None:
    transport = ScriptedTransport(completion("<think>hm</think>\n<search>ada</search>"))
    generation = policy(transport).generate("p", stop=STOP, max_new_tokens=8)
    assert generation.text == "<think>hm</think>\n<search>ada</search>"
    assert generation.token_ids == ()  # eval needs text; training brings its own ids


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b'{"object": "error"}', id="no-choices"),
        pytest.param(b'{"choices": [{"index": 0}]}', id="no-text"),
    ],
)
def test_a_malformed_response_raises(raw: bytes) -> None:
    with pytest.raises(PolicyError, match="not a completion"):
        policy(ScriptedTransport(raw)).generate("p", stop=STOP, max_new_tokens=8)


@pytest.mark.parametrize("n", [0, 2])
def test_anything_but_one_choice_raises(n: int) -> None:
    """Asked for n=1. Zero means nothing was generated; two means a proxy or a
    server default changed, and taking the first would be choosing a sample."""
    raw = json.dumps({"choices": [{"text": "a"}] * n}).encode("utf-8")
    with pytest.raises(PolicyError, match=f"{n} choices for n=1"):
        policy(ScriptedTransport(raw)).generate("p", stop=STOP, max_new_tokens=8)


# ---------------------------------------------------------------------------
# 3. Failures raise; they never become an empty generation
# ---------------------------------------------------------------------------


def test_a_server_error_is_retried_then_succeeds() -> None:
    transport = ScriptedTransport(http_error(503), completion("ok"))
    assert policy(transport, retries=1).generate("p", stop=STOP, max_new_tokens=8).text == "ok"
    assert len(transport.sent) == 2


def test_a_server_that_stays_down_raises_after_every_attempt() -> None:
    """Scored as an empty answer, a dead server would show up in section 9 as
    the model failing to answer."""
    # One error per attempt: a real server sends a fresh response each time.
    transport = ScriptedTransport(*(http_error(500, "engine dead") for _ in range(3)))
    with pytest.raises(PolicyError, match=r"failed after 3 attempt\(s\).*engine dead"):
        policy(transport, retries=2).generate("p", stop=STOP, max_new_tokens=8)
    assert len(transport.sent) == 3


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(urllib.error.URLError("connection refused"), id="refused"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
        pytest.param(ConnectionResetError("reset"), id="reset"),
    ],
)
def test_a_transport_failure_raises_after_its_retries(error: Exception) -> None:
    transport = ScriptedTransport(error)
    with pytest.raises(PolicyError, match="Not returning an empty generation"):
        policy(transport, retries=1).generate("p", stop=STOP, max_new_tokens=8)
    assert len(transport.sent) == 2


def test_a_client_error_is_not_retried() -> None:
    """A 4xx - an unknown model name, a prompt past the context window - fails
    identically on a retry. The server's reason is in the message."""
    transport = ScriptedTransport(http_error(404, "The model `x` does not exist."))
    with pytest.raises(PolicyError, match=r"HTTP 404: The model `x` does not exist\."):
        policy(transport, retries=3).generate("p", stop=STOP, max_new_tokens=8)
    assert len(transport.sent) == 1


def test_no_retries_means_one_attempt() -> None:
    transport = ScriptedTransport(http_error(502))
    with pytest.raises(PolicyError, match=r"after 1 attempt\(s\)"):
        policy(transport, retries=0).generate("p", stop=STOP, max_new_tokens=8)


# ---------------------------------------------------------------------------
# 4. Construction refuses what would only fail later
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        pytest.param(("file:///etc/passwd", MODEL), {}, "non-HTTP policy URL", id="file-url"),
        pytest.param(("localhost:8000", MODEL), {}, "non-HTTP policy URL", id="no-scheme"),
        pytest.param((BASE, "  "), {}, "model must name", id="blank-model"),
        pytest.param((BASE, MODEL), {"retries": -1}, "retries must be", id="negative-retries"),
        pytest.param((BASE, MODEL), {"timeout": 0}, "timeout must be", id="zero-timeout"),
    ],
)
def test_bad_construction_is_refused(
    args: tuple[str, str], kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(PolicyError, match=match):
        VllmHttpPolicy(*args, **kwargs)


def test_a_non_positive_turn_limit_is_refused_before_the_socket() -> None:
    transport = ScriptedTransport(completion("x"))
    with pytest.raises(PolicyError, match="max_new_tokens must be positive"):
        policy(transport).generate("p", stop=STOP, max_new_tokens=0)
    assert transport.sent == []


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.5}],
)
def test_impossible_sampling_settings_are_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Sampling(**kwargs)


def test_the_policy_satisfies_the_protocol() -> None:
    """The real check is mypy on `conformance_witness`; this keeps it alive."""
    real = VllmHttpPolicy(BASE, MODEL, transport=ScriptedTransport(completion("x")))
    assert conformance_witness(real) is real


# ---------------------------------------------------------------------------
# 5. Through the real episode loop
# ---------------------------------------------------------------------------


def episode_config() -> EpisodeConfig:
    return EpisodeConfig(
        search_budget=4, max_turns=6, max_new_tokens=128, top_k=2, stop_sequences=STOP_SEQUENCES
    )


def test_a_search_then_an_answer_runs_end_to_end() -> None:
    transport = ScriptedTransport(
        completion("<think>who</think>\n<search>Ada Lovelace</search>"),
        completion("<think>found it</think>\n<answer>Charles Babbage</answer>"),
    )
    retriever = FakeRetriever()
    question = Question(question_id="q1", question="Who designed it?", gold_answers=("Babbage",))

    trajectory = run_episode(question, policy(transport), retriever, episode_config())

    assert trajectory.termination is Termination.ANSWERED
    assert trajectory.answer == "Charles Babbage"
    assert retriever.queries == ("Ada Lovelace",)
    # The second turn was generated from a prompt holding the first turn and
    # the injected passages - the loop's transcript reached the server.
    second_prompt = str(transport.sent[1][1]["prompt"])
    assert second_prompt.startswith(str(transport.sent[0][1]["prompt"]))
    assert "<search>Ada Lovelace</search>" in second_prompt
    assert "<info>" in second_prompt


def test_without_the_stop_string_every_search_turn_is_unclosed() -> None:
    """What vLLM's default would do, shown through the real parser: the text
    stops right before `</search>`, and the episode dies on turn one."""
    transport = ScriptedTransport(completion("<think>who</think>\n<search>Ada Lovelace"))
    question = Question(question_id="q1", question="Who?", gold_answers=("x",))

    trajectory = run_episode(question, policy(transport), FakeRetriever(), episode_config())

    assert trajectory.termination is Termination.PARSE_ERROR
    assert trajectory.parse_failure is ParseFailure.UNCLOSED_TAG
    assert len(transport.sent) == 1


def test_one_policy_is_safe_across_threads() -> None:
    """`PolicyRunner` calls one policy from many threads. Each episode must get
    the answer to its own prompt, never a neighbour's."""

    def answer_for(body: dict[str, object]) -> bytes:
        prompt = str(body["prompt"])
        marker = prompt[prompt.index("[") + 1 : prompt.index("]")]
        return completion(f"<answer>{marker}</answer>")

    transport = ScriptedTransport(answer_for)
    shared = policy(transport)

    def one(i: int) -> str:
        question = Question(question_id=f"q{i}", question=f"Which? [{i}]", gold_answers=("x",))
        trajectory = run_episode(question, shared, FakeRetriever(), episode_config())
        assert trajectory.question_id == f"q{i}"
        return trajectory.answer or ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(one, range(32)))

    assert answers == [str(i) for i in range(32)]


def test_a_dataset_example_becomes_an_episode_question() -> None:
    example = QAExample(
        question_id="hotpotqa-h1",
        question="Who designed the Analytical Engine?",
        answers=("Charles Babbage", "Babbage"),
        dataset="hotpotqa",
        distribution=Distribution.IN_DISTRIBUTION,
        supporting_titles=("Analytical Engine",),
    )
    assert question_from(example) == Question(
        question_id="hotpotqa-h1",
        question="Who designed the Analytical Engine?",
        gold_answers=("Charles Babbage", "Babbage"),
    )


# ---------------------------------------------------------------------------
# 6. Real bytes over a socket, to a server shaped like `vllm serve`
# ---------------------------------------------------------------------------


@pytest.fixture
def vllm_like() -> Iterator[tuple[str, list[tuple[str, dict[str, object]]]]]:
    """A stdlib HTTP server that answers `/v1/completions` the way vLLM does,
    including keeping the stop string only when asked to."""
    received: list[tuple[str, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            """Silence the per-request stderr line."""

        def _send(self, code: int, raw: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            received.append((self.path, body))
            if self.path != COMPLETIONS_PATH:
                self._send(404, json.dumps({"detail": "Not Found"}).encode("utf-8"))
                return
            if body["model"] != MODEL:
                message = f"The model `{body['model']}` does not exist."
                self._send(404, json.dumps({"message": message}).encode("utf-8"))
                return
            text = "<search>Ada Lovelace"
            if body.get("include_stop_str_in_output"):
                text += "</search>"
            self._send(200, completion(text))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.integration
def test_end_to_end_over_a_real_socket(
    vllm_like: tuple[str, list[tuple[str, dict[str, object]]]],
) -> None:
    base_url, received = vllm_like
    generation = VllmHttpPolicy(base_url, MODEL).generate("p", stop=STOP, max_new_tokens=16)

    assert generation.text == "<search>Ada Lovelace</search>"
    ((path, body),) = received
    assert path == "/v1/completions"
    assert body["include_stop_str_in_output"] is True


@pytest.mark.integration
def test_an_unknown_model_name_is_reported_with_the_servers_reason(
    vllm_like: tuple[str, list[tuple[str, dict[str, object]]]],
) -> None:
    """The likeliest mistake with a LoRA adapter: the served name is not the
    one passed in. One attempt, and the server's own words."""
    base_url, received = vllm_like
    wrong = VllmHttpPolicy(base_url, "lambda-0.05", retries=3)
    with pytest.raises(PolicyError, match=r"HTTP 404.*lambda-0\.05.*does not exist"):
        wrong.generate("p", stop=STOP, max_new_tokens=16)
    assert len(received) == 1
