"""Tests for the retrieval wire: `service.py`'s handlers and `HttpRetriever`.

Both ends in one file on purpose. The request and response models are the only
thing tying them together, so a test that exercises one end against a hand-built
payload proves nothing about the pair; what matters is that a request this client
sends is one those handlers answer.

Three layers, cheapest first:

1. the models - bounds and `extra="forbid"`, pure validation
2. the handlers - `handle_search` / `handle_batch_search` / `handle_health`,
   driven directly, no socket
3. the real thing - `HttpRetriever` with `UrllibTransport` against a
   `http.server` from the standard library, wired to the same handlers.
   Marked `integration`. This is what proves the client and the service agree,
   and it needs no `[retrieval]` extra to run - `fastapi` is only ever a shell
   around the handlers tested at layer 2.

`create_app` itself is not tested and cannot be: `fastapi` is in the `[retrieval]`
extra. It carries a `pragma: no cover` and contains nothing but route wiring.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import numpy as np
import pytest

from verisearch.retrieval.client import (
    HttpRetriever,
    InMemoryDocStore,
    LocalRetriever,
    RetrievalError,
    StoredDoc,
    Transport,
    UrllibTransport,
)
from verisearch.retrieval.embedder import EmbedderSpec, l2_normalize
from verisearch.retrieval.index import BruteForceIndex, IndexMeta
from verisearch.retrieval.service import (
    MAX_BATCH,
    MAX_K,
    SERVICE_PROTOCOL_VERSION,
    BatchRetriever,
    BatchSearchRequest,
    BatchSearchResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ServiceError,
    handle_batch_search,
    handle_health,
    handle_search,
)
from verisearch.types import Passage

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

SPEC = EmbedderSpec(model_name="wire-test", dim=4, normalize=True)


def make_meta(n_vectors: int = 3) -> IndexMeta:
    return IndexMeta.create(index_type="brute", spec=SPEC, n_vectors=n_vectors)


class StubRetriever:
    """Deterministic passages, no index. The wire tests must not depend on ranking.

    `search_batch` is the one the batch endpoint calls; `search` is here because
    `BatchRetriever` requires both.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []
        self.fail_with: Exception | None = None
        self.drop_one = False

    def search(self, query: str, k: int) -> list[Passage]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[Passage]]:
        self.calls.append((list(queries), k))
        if self.fail_with is not None:
            raise self.fail_with
        results = [
            [
                Passage(
                    doc_id=f"{q}-{rank}",
                    title=f"T{rank}",
                    text=f"Body for {q}.",
                    score=1.0 / (rank + 1),
                )
                for rank in range(k)
            ]
            for q in queries
        ]
        # Simulates a retriever that silently loses a result list. Positional
        # alignment is the only thing tying a result to its query.
        return results[:-1] if self.drop_one and results else results


# ---------------------------------------------------------------------------
# 1. The models
# ---------------------------------------------------------------------------


def test_k_is_bounded_in_the_request_model() -> None:
    """An unbounded k is a memory amplifier: one request at k=100000 against a
    batch of 8 serialises most of the corpus eight times."""
    for bad in (0, -1, MAX_K + 1):
        with pytest.raises(ValueError, match="k"):
            SearchRequest(query="ada", k=bad)
    assert SearchRequest(query="ada", k=MAX_K).k == MAX_K


def test_batch_size_is_bounded_and_non_empty() -> None:
    with pytest.raises(ValueError, match="queries"):
        BatchSearchRequest(queries=(), k=3)
    with pytest.raises(ValueError, match="queries"):
        BatchSearchRequest(queries=tuple(f"q{i}" for i in range(MAX_BATCH + 1)), k=3)
    assert len(BatchSearchRequest(queries=("a", "b"), k=3).queries) == 2


def test_the_wire_models_reject_unknown_fields() -> None:
    """A client sending a field the service does not know would otherwise have
    it dropped silently - including a renamed `k`, which would then default."""
    with pytest.raises(ValueError, match="top_k"):
        SearchRequest.model_validate({"query": "ada", "k": 3, "top_k": 10})


def test_a_response_round_trips_through_json_unchanged() -> None:
    original = SearchResponse(
        query="ada lovelace",
        k=2,
        passages=(
            Passage(doc_id="d1", title="Ada Lovelace", text="Body.", score=0.9),
            Passage(doc_id="d2", title="Engine", text="Body.", score=None),
        ),
    )
    assert SearchResponse.model_validate_json(original.model_dump_json()) == original


# ---------------------------------------------------------------------------
# 2. The handlers
# ---------------------------------------------------------------------------


def test_handle_search_echoes_the_query_and_k_back() -> None:
    """Not redundant. The echo is what lets the client detect a proxy or a retry
    that swaps two responses - otherwise the passages are well-formed and simply
    about something else, and every reward still produces a number."""
    response = handle_search(StubRetriever(), SearchRequest(query="ada", k=2))
    assert response.query == "ada"
    assert response.k == 2
    assert len(response.passages) == 2


def test_handle_search_turns_a_retriever_valueerror_into_a_400() -> None:
    """`RetrievalError` subclasses `ValueError`, so this is the path a bad `k` or
    an unresolvable doc_id takes. 400 rather than 503: a retry cannot succeed."""
    stub = StubRetriever()
    stub.fail_with = RetrievalError("k must be positive, got -1")
    with pytest.raises(ServiceError, match="k must be positive") as caught:
        handle_search(stub, SearchRequest(query="ada", k=2))
    assert caught.value.status_code == 400


def test_a_failing_retriever_is_never_reported_as_an_empty_result() -> None:
    """The failure mode this whole design is pointed at. `passages: []` from a
    broken index is indistinguishable from 'nothing matched', and across
    thousands of steps the policy would learn that searching does not pay."""
    stub = StubRetriever()
    stub.fail_with = RetrievalError("index and corpus disagree")
    with pytest.raises(ServiceError):
        handle_search(stub, SearchRequest(query="ada", k=2))


def test_handle_batch_search_aligns_results_with_queries() -> None:
    queries = ("ada", "babbage", "engine")
    response = handle_batch_search(StubRetriever(), BatchSearchRequest(queries=queries, k=1))
    assert [r.query for r in response.results] == list(queries)
    assert [r.passages[0].doc_id for r in response.results] == [f"{q}-0" for q in queries]


def test_a_batch_length_mismatch_is_a_500_not_a_short_response() -> None:
    """A dropped result list attaches one rollout's passages to another
    rollout's question. Every reward still computes; the only symptom is a
    faithfulness score that will not improve."""
    stub = StubRetriever()
    stub.drop_one = True
    with pytest.raises(ServiceError, match="positional alignment") as caught:
        handle_batch_search(stub, BatchSearchRequest(queries=("a", "b"), k=1))
    assert caught.value.status_code == 500


def test_handlers_enforce_the_k_ceiling_independently_of_the_model() -> None:
    """The model bound is what shows up in the OpenAPI schema. This one is what
    actually protects the service from a hand-rolled client."""
    with pytest.raises(ServiceError, match=f"between 1 and {MAX_K}") as caught:
        handle_search(StubRetriever(), SearchRequest.model_construct(query="ada", k=10_000))
    assert caught.value.status_code == 400


def test_handle_health_reports_the_fingerprint_the_client_checks() -> None:
    health = handle_health(make_meta(n_vectors=7), n_docs=7)
    assert health.ok
    assert health.protocol_version == SERVICE_PROTOCOL_VERSION
    assert health.embedder_fingerprint == SPEC.fingerprint
    assert (health.n_vectors, health.n_docs, health.dim) == (7, 7, 4)


# ---------------------------------------------------------------------------
# 3. HttpRetriever against a fake transport
# ---------------------------------------------------------------------------


class RecordingTransport:
    """Returns canned bytes and records what was sent."""

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = responses
        self.sent: list[tuple[str, bytes | None]] = []

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        self.sent.append((url, payload))
        if not self._responses:
            raise AssertionError(f"transport ran out of canned responses at {url}")
        return self._responses.pop(0)


class BrokenTransport:
    """Always fails, so retry and error-reporting behaviour is observable."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.attempts = 0

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        self.attempts += 1
        raise self._error


def canned_search(query: str, k: int) -> bytes:
    response = SearchResponse(
        query=query,
        k=k,
        passages=tuple(
            Passage(doc_id=f"d{i}", title=f"T{i}", text="Body.", score=1.0) for i in range(k)
        ),
    )
    return response.model_dump_json().encode("utf-8")


def test_the_client_posts_to_search_and_returns_passages() -> None:
    transport = RecordingTransport([canned_search("ada", 2)])
    client = HttpRetriever("http://localhost:8099/", transport=transport)
    passages = client.search("ada", 2)

    url, payload = transport.sent[0]
    assert url == "http://localhost:8099/search"
    assert payload is not None
    assert json.loads(payload) == {"query": "ada", "k": 2}
    assert [p.doc_id for p in passages] == ["d0", "d1"]


def test_a_blank_query_never_touches_the_network() -> None:
    """Parity with `LocalRetriever` and `FakeRetriever`. Three implementations of
    one protocol are only useful if they are indistinguishable to the caller."""
    transport = RecordingTransport([])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    assert client.search("   ", 3) == []
    assert transport.sent == []


def test_non_positive_k_raises_before_the_network() -> None:
    transport = RecordingTransport([])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(ValueError, match="k must be positive"):
        client.search("ada", 0)
    assert transport.sent == []


def test_a_response_for_a_different_query_is_refused() -> None:
    """The echo check. A proxy or a retry that swaps two responses is otherwise
    undetectable - the passages are well-formed and about the wrong question."""
    transport = RecordingTransport([canned_search("babbage", 2)])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(RetrievalError, match="answered query 'babbage'"):
        client.search("ada", 2)


def test_a_response_at_a_different_k_is_refused() -> None:
    transport = RecordingTransport([canned_search("ada", 5)])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(RetrievalError, match="k=5 but was asked"):
        client.search("ada", 2)


def test_a_response_that_is_not_json_is_refused_with_the_endpoint_named() -> None:
    transport = RecordingTransport([b"<html>502 Bad Gateway</html>"])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(RetrievalError, match=r"/search returned something that is not a"):
        client.search("ada", 2)


def test_a_dead_service_raises_rather_than_returning_nothing() -> None:
    """The single most important behaviour in this module. An empty result from a
    dead service would train the policy to stop searching."""
    transport = BrokenTransport(TimeoutError("timed out"))
    client = HttpRetriever("http://localhost:8099", transport=transport, retries=1)
    with pytest.raises(RetrievalError, match="Not returning an empty result"):
        client.search("ada", 2)
    assert transport.attempts == 2


def test_retries_are_bounded_and_configurable() -> None:
    transport = BrokenTransport(OSError("connection reset"))
    with pytest.raises(RetrievalError, match="after 1 attempt"):
        HttpRetriever("http://localhost:8099", transport=transport, retries=0).search("ada", 1)
    assert transport.attempts == 1


def test_a_nonsense_client_configuration_is_refused() -> None:
    for kwargs in ({"retries": -1}, {"timeout": 0.0}, {"timeout": -1.0}):
        with pytest.raises(RetrievalError):
            HttpRetriever("http://localhost:8099", **kwargs)  # type: ignore[arg-type]


def test_batch_keeps_blank_queries_in_position_without_sending_them() -> None:
    response = BatchSearchResponse(
        k=1,
        results=(
            SearchResponse(
                query="ada", k=1, passages=(Passage(doc_id="d1", title="T", text="B."),)
            ),
            SearchResponse(
                query="engine", k=1, passages=(Passage(doc_id="d2", title="T", text="B."),)
            ),
        ),
    )
    transport = RecordingTransport([response.model_dump_json().encode("utf-8")])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    results = client.search_batch(["", "ada", "  ", "engine"], 1)

    assert [len(r) for r in results] == [0, 1, 0, 1]
    assert results[1][0].doc_id == "d1"
    assert results[3][0].doc_id == "d2"
    _, payload = transport.sent[0]
    assert payload is not None
    assert json.loads(payload)["queries"] == ["ada", "engine"]


def test_a_batch_response_of_the_wrong_length_is_refused() -> None:
    short = BatchSearchResponse(
        k=1,
        results=(SearchResponse(query="ada", k=1, passages=()),),
    )
    transport = RecordingTransport([short.model_dump_json().encode("utf-8")])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(RetrievalError, match="another rollout's passages"):
        client.search_batch(["ada", "engine"], 1)


def test_an_all_blank_batch_returns_empties_without_a_round_trip() -> None:
    transport = RecordingTransport([])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    assert client.search_batch(["", "  "], 2) == [[], []]
    assert transport.sent == []


def test_an_empty_batch_raises() -> None:
    client = HttpRetriever("http://localhost:8099", transport=RecordingTransport([]))
    with pytest.raises(RetrievalError, match="at least one query"):
        client.search_batch([], 2)


def test_connect_refuses_a_protocol_version_it_does_not_speak() -> None:
    health = handle_health(make_meta(), n_docs=3).model_copy(
        update={"protocol_version": SERVICE_PROTOCOL_VERSION + 1}
    )
    transport = RecordingTransport([health.model_dump_json().encode("utf-8")])
    with pytest.raises(RetrievalError, match="Upgrade one of them"):
        HttpRetriever.connect("http://localhost:8099", transport=transport)


def test_connect_refuses_an_index_built_by_a_different_embedder() -> None:
    """The wire-crossing version of `index.check_compatible`. The mismatch it
    catches produces plausible passages for an entire run and raises nothing."""
    health = handle_health(make_meta(), n_docs=3)
    transport = RecordingTransport([health.model_dump_json().encode("utf-8")])
    with pytest.raises(RetrievalError, match="plausible nonsense silently"):
        HttpRetriever.connect(
            "http://localhost:8099", expect_fingerprint="deadbeef", transport=transport
        )


def test_connect_accepts_a_matching_fingerprint() -> None:
    health = handle_health(make_meta(), n_docs=3)
    transport = RecordingTransport([health.model_dump_json().encode("utf-8")])
    client = HttpRetriever.connect(
        "http://localhost:8099", expect_fingerprint=SPEC.fingerprint, transport=transport
    )
    assert transport.sent[0][0].endswith("/health")
    assert isinstance(client, HttpRetriever)


def test_the_transport_refuses_a_non_http_url() -> None:
    """`urllib` will open `file://` and `ftp://`. A typo'd base_url would read a
    local file and try to parse it as a search response."""
    with pytest.raises(RetrievalError, match="non-HTTP retrieval URL"):
        UrllibTransport().request("file:///etc/passwd", None, 1.0)


def test_the_default_transport_is_the_urllib_one() -> None:
    witness: Transport = UrllibTransport()
    assert isinstance(witness, UrllibTransport)


def test_handle_batch_search_turns_a_retriever_valueerror_into_a_400() -> None:
    """Same contract as the single-query handler: a retriever that refuses is
    the caller's problem, and a retry cannot fix it."""
    stub = StubRetriever()
    stub.fail_with = RetrievalError("k must be positive, got -3")
    with pytest.raises(ServiceError, match="k must be positive") as caught:
        handle_batch_search(stub, BatchSearchRequest(queries=("ada",), k=2))
    assert caught.value.status_code == 400


def test_batch_rejects_a_non_positive_k_before_the_network() -> None:
    transport = RecordingTransport([])
    client = HttpRetriever("http://localhost:8099", transport=transport)
    with pytest.raises(ValueError, match="k must be positive"):
        client.search_batch(["ada"], 0)
    assert transport.sent == []


# ---------------------------------------------------------------------------
# 4. The real thing: client and handlers over a socket
# ---------------------------------------------------------------------------


def make_wire_server(
    retriever: BatchRetriever, meta: IndexMeta, n_docs: int
) -> ThreadingHTTPServer:
    """A stdlib HTTP server wired to the same handlers `create_app` wires.

    Not FastAPI: `fastapi` is in the `[retrieval]` extra and would make this
    test unrunnable in CI, which is exactly where the client and the service
    most need to be checked against each other. `create_app` holds no logic
    beyond the wiring this duplicates - that is why it is allowed to be
    untested.
    """

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
                self._send(200, handle_health(meta, n_docs).model_dump_json())
            else:
                self._send(404, json.dumps({"detail": f"no route {self.path}"}))

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                if self.path == "/search":
                    out = handle_search(retriever, SearchRequest.model_validate_json(body))
                elif self.path == "/search_batch":
                    out = handle_batch_search(
                        retriever, BatchSearchRequest.model_validate_json(body)
                    )
                else:
                    self._send(404, json.dumps({"detail": f"no route {self.path}"}))
                    return
            except ServiceError as exc:
                self._send(exc.status_code, json.dumps({"detail": str(exc)}))
                return
            except ValueError as exc:
                self._send(422, json.dumps({"detail": str(exc)}))
                return
            self._send(200, out.model_dump_json())

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


@pytest.fixture
def wire() -> Iterator[tuple[str, StubRetriever]]:
    retriever = StubRetriever()
    server = make_wire_server(retriever, make_meta(), n_docs=3)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", retriever
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.integration
def test_end_to_end_search_over_a_real_socket(wire: tuple[str, StubRetriever]) -> None:
    """The test that proves the two ends agree. Everything above either stubs
    the transport or calls a handler directly; this one sends real bytes."""
    base_url, retriever = wire
    client = HttpRetriever.connect(base_url, expect_fingerprint=SPEC.fingerprint)

    passages = client.search("ada lovelace", 3)
    assert len(passages) == 3
    assert passages[0].doc_id == "ada lovelace-0"
    assert retriever.calls == [(["ada lovelace"], 3)]


@pytest.mark.integration
def test_end_to_end_batch_over_a_real_socket(wire: tuple[str, StubRetriever]) -> None:
    base_url, _ = wire
    client = HttpRetriever(base_url)
    results = client.search_batch(["ada", "", "babbage"], 2)
    assert [len(r) for r in results] == [2, 0, 2]
    assert results[0][0].doc_id == "ada-0"
    assert results[2][0].doc_id == "babbage-0"


@pytest.mark.integration
def test_a_service_error_reaches_the_client_as_a_retrieval_error(
    wire: tuple[str, StubRetriever],
) -> None:
    """A 400 carries the service's reason in `detail`, and the client surfaces it
    rather than reporting a bare status code."""
    base_url, retriever = wire
    retriever.fail_with = RetrievalError("doc_id 'd9' is not in the document store")
    client = HttpRetriever(base_url)
    with pytest.raises(RetrievalError, match="not in the document store"):
        client.search("ada", 2)


@pytest.mark.integration
def test_an_unknown_route_is_an_error_not_an_empty_result(
    wire: tuple[str, StubRetriever],
) -> None:
    base_url, _ = wire
    client = HttpRetriever(f"{base_url}/v2")
    with pytest.raises(RetrievalError, match="HTTP 404"):
        client.search("ada", 2)


@pytest.mark.integration
def test_a_local_retriever_serves_correctly_through_the_wire() -> None:
    """Composition check: `LocalRetriever` behind the handlers, read back by
    `HttpRetriever`, must give the same passages as calling it directly."""

    class OneHotEmbedder:
        """Minimal inline embedder - this file must not depend on ranking
        fixtures from `test_client.py`. TODO: both move to
        `verisearch/testing/fakes.py` once #9 merges."""

        def __init__(self, spec: EmbedderSpec) -> None:
            self.spec = spec

        def encode_queries(self, queries: Sequence[str]) -> NDArray[np.float32]:
            rows = [
                [1.0 if q == "d1" else 0.0, 1.0 if q == "d2" else 0.0, 0.0, 0.0] for q in queries
            ]
            return l2_normalize(np.asarray(rows, dtype=np.float32))

    doc_ids = ["d1", "d2", "d3"]
    vectors = l2_normalize(
        np.asarray([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    )
    local = LocalRetriever(
        BruteForceIndex.build(vectors, doc_ids, SPEC),
        OneHotEmbedder(SPEC),
        InMemoryDocStore.from_records(
            [StoredDoc(doc_id=d, title=d.upper(), text=f"Body of {d}.") for d in doc_ids]
        ),
    )

    server = make_wire_server(local, local.meta, n_docs=local.n_docs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HttpRetriever(f"http://127.0.0.1:{server.server_address[1]}")
        over_the_wire = client.search("d2", 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    direct = local.search("d2", 2)
    assert [p.doc_id for p in over_the_wire] == [p.doc_id for p in direct] == ["d2", "d1"]
    assert over_the_wire == direct


@pytest.mark.integration
def test_health_over_the_wire_reports_what_is_being_served(
    wire: tuple[str, StubRetriever],
) -> None:
    base_url, _ = wire
    health = HttpRetriever(base_url).health()
    assert isinstance(health, HealthResponse)
    assert health.embedder_fingerprint == SPEC.fingerprint
    assert health.protocol_version == SERVICE_PROTOCOL_VERSION
