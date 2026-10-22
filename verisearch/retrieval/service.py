"""HTTP retrieval service: one index, many rollout workers.

PROJECT.md section 4: "service.py - HTTP retrieval service". Not the same thing
as `serve/server.py`, which serves the trained policy at P5. This serves the
corpus index.

**Why there is a network hop at all.** The index plus the embedder is hundreds of
megabytes of RAM and a model that wants to sit on a GPU. GRPO generates a group
of 8 rollouts, so if each rollout worker owns its own retriever there are eight
copies of the embedder competing with the policy for the same VRAM. One service
and eight thin clients instead. It also fits the hardware this project actually
has: the hardware report rules out training on this machine, but 15.7 GB of RAM
is plenty to serve an index while the GPU box trains against it.

**How this file is split, and why.** `fastapi` lives in the `[retrieval]` extra,
so the PR pipeline never installs it and `create_app` can never run in CI. So
`create_app` is a shell: it holds no logic beyond wiring routes. Everything that
decides *what the service answers* is a pure function over the request models
above it, and those are core-dependency-only and fully tested. Same split as
`embedder.py` - the untestable part is made as small as it can be rather than
being left to a comment.

The response models are also the client's contract: `client.HttpRetriever`
imports them from here, so there is one definition of the wire format and mypy
checks both ends against it.

Owner: Ayesha.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from verisearch.types import Passage, Retriever

if TYPE_CHECKING:
    from fastapi import FastAPI

    from verisearch.retrieval.index import IndexMeta

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# Bump when the wire format changes incompatibly. `/health` reports it and the
# client refuses a mismatch, so a client talking to a stale service fails at
# startup instead of silently misreading passages for a whole training run.
SERVICE_PROTOCOL_VERSION = 1

# Hard ceiling on `k`, independent of what any client asks for. An unbounded `k`
# is a memory amplifier: one request for k=100000 against a batch of 8 queries
# serialises most of the corpus eight times. 100 is far above the 3-5 the
# experiment uses.
MAX_K = 100

# Ceiling on queries per batch request, for the same reason. A GRPO group is 8.
MAX_BATCH = 64


class ServiceError(Exception):
    """A request that cannot be served, with the status code it maps to.

    Carries the code so the pure handlers can decide it and the FastAPI shell
    stays a translation layer. Deciding 400-vs-503 inside a route would put the
    one judgement call in this module in the one part of it that cannot be
    tested here.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class SearchRequest(BaseModel):
    """One query.

    `k` is bounded here as well as in the handler. The handler check is what
    actually protects a hand-rolled client; this one is what makes the bound
    visible in the generated OpenAPI schema.
    """

    model_config = _FROZEN

    query: str
    k: int = Field(ge=1, le=MAX_K)


class BatchSearchRequest(BaseModel):
    """Several queries in one round trip. One GRPO group is 8 of them."""

    model_config = _FROZEN

    queries: tuple[str, ...] = Field(min_length=1, max_length=MAX_BATCH)
    k: int = Field(ge=1, le=MAX_K)


class SearchResponse(BaseModel):
    """Passages for one query.

    Echoes `query` and `k` back. Not redundant: the client logs mismatches, and
    a reverse proxy or a retry that reorders responses is otherwise invisible -
    the passages would simply belong to a different question and the reward
    would still compute a number.
    """

    model_config = _FROZEN

    query: str
    k: int
    passages: tuple[Passage, ...]


class BatchSearchResponse(BaseModel):
    """Results aligned with the request's `queries`, same order, same length."""

    model_config = _FROZEN

    k: int
    results: tuple[SearchResponse, ...]


class HealthResponse(BaseModel):
    """Everything a client needs to decide whether to trust this service.

    The fingerprint is the point. A client holding a corpus or expecting a
    particular embedder can compare it here and refuse at startup, which is the
    wire-crossing version of `index.check_compatible`.
    """

    model_config = _FROZEN

    ok: bool
    protocol_version: int
    index_type: str
    embedder_model: str
    embedder_fingerprint: str
    dim: int
    n_vectors: int
    n_docs: int
    built_at: str


class BatchRetriever(Protocol):
    """What this service needs from a retriever, which is now all of `Retriever`.

    Kept as its own name rather than importing `types.Retriever`, because the
    two are not the same statement: `Retriever` is the contract both halves of
    the project agreed, and this is what one HTTP service happens to call. They
    coincide today. If the shared protocol grows a method the service does not
    use, this one should not grow with it.

    `queries: Sequence[str]`, not `list[str]`: with `list` a `types.Retriever`
    would *not* satisfy this protocol, which would be a surprising thing for the
    retrieval service to say about the project's own retriever.
    """

    def search(self, query: str, k: int) -> list[Passage]: ...

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]: ...


def batch_witness(retriever: Retriever) -> BatchRetriever:
    """Any `types.Retriever` is a `BatchRetriever`. Checked by mypy, not asserted.

    This is the claim that justified widening `search_batch` here from
    `list[str]` to `Sequence[str]`, and it is the kind of claim that quietly
    stops being true. `tests/` is outside mypy's `files` list, so writing it as
    a test assertion would mean writing something that can never fail; the
    return annotation here is what notices if the two protocols drift apart.
    The test that calls this function is what keeps it from being dead code.
    """
    return retriever


def _check_k(k: int) -> None:
    if k < 1 or k > MAX_K:
        raise ServiceError(f"k must be between 1 and {MAX_K}, got {k}", status_code=400)


def handle_search(retriever: BatchRetriever, request: SearchRequest) -> SearchResponse:
    """Answer one query. All of the service's single-query behaviour.

    Retrieval failures are not converted into an empty result. A client that
    receives `passages: []` cannot tell "nothing matched" from "the index is
    broken", and inside a training run the second one would teach the policy
    that searching is useless - which is exactly the degenerate behaviour the
    lambda term is supposed to be the only pressure toward.
    """
    _check_k(request.k)
    try:
        passages = retriever.search(request.query, request.k)
    except ValueError as exc:
        # `client.RetrievalError` subclasses `ValueError`. A bad `k` or a blank
        # query is the caller's fault; an unresolvable doc_id is the server's,
        # but from here both are "this request cannot be answered", and 503
        # would invite a client retry that cannot succeed.
        raise ServiceError(str(exc), status_code=400) from exc
    return SearchResponse(query=request.query, k=request.k, passages=tuple(passages))


def handle_batch_search(
    retriever: BatchRetriever, request: BatchSearchRequest
) -> BatchSearchResponse:
    """Answer several queries, aligned with the request.

    The alignment is the whole risk. Results come back positionally, so a
    dropped or reordered entry attaches one rollout's passages to another
    rollout's question, every reward still computes, and the only symptom is a
    faithfulness score that will not improve. Hence the explicit length check
    rather than trusting the retriever.
    """
    _check_k(request.k)
    queries = list(request.queries)
    try:
        batched = retriever.search_batch(queries, request.k)
    except ValueError as exc:
        raise ServiceError(str(exc), status_code=400) from exc

    if len(batched) != len(queries):
        raise ServiceError(
            f"retriever returned {len(batched)} result lists for {len(queries)} queries; "
            f"positional alignment is the only thing tying a result to its query",
            status_code=500,
        )
    return BatchSearchResponse(
        k=request.k,
        results=tuple(
            SearchResponse(query=query, k=request.k, passages=tuple(passages))
            for query, passages in zip(queries, batched, strict=True)
        ),
    )


def handle_health(meta: IndexMeta, n_docs: int) -> HealthResponse:
    """Report what this service is serving."""
    return HealthResponse(
        ok=True,
        protocol_version=SERVICE_PROTOCOL_VERSION,
        index_type=meta.index_type,
        embedder_model=meta.embedder_model,
        embedder_fingerprint=meta.embedder_fingerprint,
        dim=meta.dim,
        n_vectors=meta.n_vectors,
        n_docs=n_docs,
        built_at=meta.built_at,
    )


def create_app(  # pragma: no cover - fastapi is in the [retrieval] extra
    retriever: BatchRetriever,
    meta: IndexMeta,
    n_docs: int,
) -> FastAPI:
    """Wire the handlers above into a FastAPI app.

    **Unverified.** `fastapi` is in the `[retrieval]` extra, so neither CI nor
    this machine has it and nothing in this function has run. It is written to
    contain no decisions for exactly that reason: every route is one call into a
    tested handler plus one exception translation. If a bug is ever traced into
    this function, the fix is to move the logic up into a handler rather than to
    test the route.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ServiceError(
            "the retrieval service needs the retrieval extra: uv pip install -e '.[retrieval]'",
            status_code=500,
        ) from exc

    def health() -> HealthResponse:
        return handle_health(meta, n_docs)

    def search(request: SearchRequest) -> SearchResponse:
        try:
            return handle_search(retriever, request)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    def search_batch(request: BatchSearchRequest) -> BatchSearchResponse:
        try:
            return handle_batch_search(retriever, request)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    app = FastAPI(title="VeriSearch retrieval", version=str(SERVICE_PROTOCOL_VERSION))
    # `add_api_route` rather than `@app.get` / `@app.post`. Under
    # `ignore_missing_imports` the decorators are untyped, and mypy's
    # `disallow_untyped_decorators` (part of `strict`) then rejects every route.
    # The alternatives were three `# type: ignore` comments or a per-module
    # relaxation of strict, which PROJECT.md section 3 says needs both owners.
    # This is the same wiring by a different spelling.
    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/search", search, methods=["POST"], response_model=SearchResponse)
    app.add_api_route(
        "/search_batch", search_batch, methods=["POST"], response_model=BatchSearchResponse
    )
    return app
