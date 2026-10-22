"""Retriever implementations: the join from index rows to `Passage` objects.

PROJECT.md section 4 names this file "implements Retriever protocol". That
protocol - `search(query, k) -> list[Passage]` - is the only retrieval surface
the RL half ever sees, so everything FAISS-shaped stops here.

`retrieval/index.py` returns `IndexHit(doc_id, score, row)`. It knows about rows,
ids and scores, and nothing about text. This module owns the join, and the join
is where the interesting failure lives: a hit whose `doc_id` is not in the
document store means the index and the corpus were built from different data.
The index still returns ten confident hits; some of them cannot be turned into
passages. Dropping those quietly returns fewer than `k` passages, which reads as
weak retrieval rather than as a build error - so this module refuses instead.

Deliberately *not* importing `data/build_corpus.py`. `DocRecord` below is a
structural protocol over `doc_id`/`title`/`text`, which `CorpusDoc` already
satisfies, so `InMemoryDocStore.from_records(load_corpus(path))` type-checks with
no dependency between the two packages.

`HttpRetriever` is the other end of `retrieval/service.py`. It imports that
module's response models rather than redeclaring them, so there is one definition
of the wire format and mypy checks both ends against it. It uses stdlib `urllib`
and no HTTP library: a client that needs nothing from the `[retrieval]` extra is
a client the GPU box can run without installing faiss or torch.

Owner: Ayesha.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from verisearch.retrieval.index import IndexHit, IndexMeta, VectorIndex, check_compatible
from verisearch.retrieval.service import (
    SERVICE_PROTOCOL_VERSION,
    BatchSearchRequest,
    BatchSearchResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from verisearch.types import Passage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

    from verisearch.retrieval.embedder import EmbedderSpec
    from verisearch.types import Retriever

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# Cache size for `CachingRetriever`. 2048 (query, k) pairs is a few megabytes of
# passage text and covers far more than one GRPO group's worth of queries.
DEFAULT_CACHE_ENTRIES = 2048

# Socket timeout for `HttpRetriever`, in seconds. Retrieval is a vector search
# over a few tens of thousands of documents; if it has not answered in ten
# seconds the service is wedged, not slow.
DEFAULT_TIMEOUT_SECONDS = 10.0

# Retries per request. One, not three: the failures this sees are a restarted
# service or a dropped socket, both of which either clear immediately or do not
# clear at all. More attempts just make a dead service take longer to notice.
DEFAULT_RETRIES = 1

# Any of the service's response models. Lets `_fetch` and `_validate` be written
# once instead of per endpoint, without an explicit `Any` (banned by section 3).
_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


class RetrievalError(ValueError):
    """Raised when a search cannot be served.

    Subclasses `ValueError` on purpose. `verisearch.testing.fakes.FakeRetriever`
    raises a bare `ValueError` for an invalid `k`, and Dhrubo writes the episode
    loop against the fake, so whatever `except` clause he wraps a search in has
    to catch the real retriever too. If these two diverge, the fake stops being
    a usable stand-in and the divergence only shows up on the GPU box.
    """


class DocRecord(Protocol):
    """A corpus document, structurally.

    `data.build_corpus.CorpusDoc` satisfies this without either module
    importing the other. If a fourth field ever becomes load-bearing for
    retrieval, add it here and mypy will point at every producer.
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def text(self) -> str: ...


class StoredDoc(BaseModel):
    """One document as the store holds it."""

    model_config = _FROZEN

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DocStore(Protocol):
    """Where the join gets its text. Swappable for a database later."""

    def get(self, doc_id: str) -> StoredDoc | None: ...

    def __len__(self) -> int: ...


class InMemoryDocStore:
    """The whole corpus in a dict.

    Sized for this project, not for Wikipedia: the corpus is the union of
    HotpotQA's own context paragraphs (see `data/build_corpus.py` for why), which
    is tens of thousands of documents, not millions. If the corpus ever becomes
    the full abstracts dump this is the class that has to go.
    """

    def __init__(self, docs: dict[str, StoredDoc]) -> None:
        self._docs = docs

    @classmethod
    def from_records(cls, records: Iterable[DocRecord]) -> InMemoryDocStore:
        docs: dict[str, StoredDoc] = {}
        for record in records:
            if record.doc_id in docs:
                # Not recoverable by picking one. Two documents under one id
                # means every hit on that id resolves to whichever happened to
                # be read second, so retrieval results depend on file order.
                raise RetrievalError(
                    f"duplicate doc_id {record.doc_id!r} in the corpus - the join "
                    f"would be ambiguous and would depend on read order"
                )
            docs[record.doc_id] = StoredDoc(
                doc_id=record.doc_id, title=record.title, text=record.text
            )
        if not docs:
            raise RetrievalError("cannot build a document store from zero documents")
        return cls(docs)

    def get(self, doc_id: str) -> StoredDoc | None:
        return self._docs.get(doc_id)

    def __len__(self) -> int:
        return len(self._docs)


def load_doc_store(path: str | Path) -> InMemoryDocStore:
    """Read a JSONL corpus into a document store.

    The format is exactly what `data.build_corpus.write_corpus` emits: one
    object per line with `doc_id`, `title` and `text`. Read here rather than
    through `load_corpus` so that serving the index does not pull in the
    dataset-building package; the field names are the contract between them.
    """
    source = Path(path)
    if not source.is_file():
        raise RetrievalError(f"no corpus file at {source}")

    records: list[StoredDoc] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(StoredDoc.model_validate_json(stripped))
            except ValueError as exc:
                raise RetrievalError(f"{source}:{number} is not a corpus document: {exc}") from exc
    return InMemoryDocStore.from_records(records)


class QueryEmbedder(Protocol):
    """What the client needs from an embedder: its spec, and query vectors.

    Narrower than `SentenceTransformerEmbedder` on purpose - `encode_passages`
    belongs to index building, not to serving, and a serving process that cannot
    embed passages cannot accidentally re-embed the corpus with wrong settings.
    """

    @property
    def spec(self) -> EmbedderSpec: ...

    def encode_queries(self, queries: Sequence[str]) -> NDArray[np.float32]: ...


class LocalRetriever:
    """In-process retrieval: embed, search, join. Implements `types.Retriever`.

    Construction checks the embedder against the index's recorded fingerprint
    via `check_compatible`, so a mismatch fails here rather than returning
    well-formed nonsense for a whole training run.
    """

    def __init__(self, index: VectorIndex, embedder: QueryEmbedder, docs: DocStore) -> None:
        check_compatible(index.meta, embedder.spec)
        if len(docs) < index.meta.n_vectors:
            # Fewer documents than vectors guarantees unresolvable hits. Caught
            # at construction because the alternative is finding out partway
            # through an eval run, per question, at random.
            raise RetrievalError(
                f"index has {index.meta.n_vectors} vectors but the document store holds "
                f"{len(docs)} documents - some hits could not be resolved to passages"
            )
        self._index = index
        self._embedder = embedder
        self._docs = docs

    @property
    def meta(self) -> IndexMeta:
        """The index's metadata. What `service.handle_health` reports.

        Public because serving `/health` needs it, and the alternative was the
        service reaching into `_index` - which would make the index an implicit
        part of this class's interface without saying so.
        """
        return self._index.meta

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int) -> list[Passage]:
        """One query. The protocol method; everything else is a convenience."""
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]:
        """Several queries in one index call.

        Part of the `Retriever` protocol as of #2. It existed before that
        because the index is already batched, so this is a loop rather than new
        machinery; the decision to widen the contract was cheap precisely because
        `search_batch` to the protocol becomes a signature change with no
        implementation behind it.
        """
        if k <= 0:
            # Matches `FakeRetriever`, which Dhrubo codes against.
            raise RetrievalError(f"k must be positive, got {k}")
        if not queries:
            raise RetrievalError("search_batch needs at least one query")

        # A blank query returns no passages rather than raising, again matching
        # the fake. A policy emitting `<search></search>` is a format failure the
        # reward is supposed to price, not an exception that aborts the episode
        # and loses the rest of the GRPO group with it.
        real = [(position, q) for position, q in enumerate(queries) if q.strip()]

        hit_rows: list[list[IndexHit]] = []
        if real:
            vectors = self._embedder.encode_queries([q for _, q in real])
            hit_rows = self._index.search(np.asarray(vectors, dtype=np.float32), k)

        results: list[list[Passage]] = [[] for _ in queries]
        for (position, _), hits in zip(real, hit_rows, strict=True):
            results[position] = self._to_passages(hits)
        return results

    def _to_passages(self, hits: Sequence[IndexHit]) -> list[Passage]:
        """Resolve hits to passages, in the order the index returned them.

        Order is preserved rather than re-sorted: `index.py` breaks score ties
        on row number so its ordering is reproducible, and re-sorting here would
        replace that with Python's own tie behaviour for no gain.
        """
        passages: list[Passage] = []
        for hit in hits:
            doc = self._docs.get(hit.doc_id)
            if doc is None:
                raise RetrievalError(
                    f"index returned doc_id {hit.doc_id!r} (row {hit.row}) which is not in "
                    f"the document store. The index and the corpus were built from "
                    f"different data - rebuild the index from this corpus."
                )
            passages.append(
                Passage(doc_id=doc.doc_id, title=doc.title, text=doc.text, score=hit.score)
            )
        return passages


class CachingRetriever:
    """Memoises `(query, k)`. Wraps any `Retriever`; is one itself.

    GRPO samples a group of 8 rollouts from one prompt, so the same early query
    arrives up to eight times within a few seconds, and PROJECT.md section 6
    prices repeated queries at -0.1 precisely because the policy keeps making
    them. Retrieval is deterministic for a fixed index, so serving a repeat from
    memory changes throughput and nothing else.

    Not thread-safe and not shared between processes. A dict is enough here; a
    process-shared cache would need the index to be immutable across its
    lifetime, which is true today and is not worth depending on.
    """

    def __init__(self, inner: Retriever, *, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if max_entries < 1:
            raise RetrievalError(f"max_entries must be >= 1, got {max_entries}")
        self._inner = inner
        self._max_entries = max_entries
        self._cache: dict[tuple[str, int], tuple[Passage, ...]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    def search(self, query: str, k: int) -> list[Passage]:
        key = (query, k)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            # A fresh list each time. `Passage` is frozen, so the elements are
            # safe to share; the list is not.
            return list(cached)

        self.misses += 1
        # Deliberately outside any try/except: a retrieval error must not be
        # cached, and must not be swallowed into an empty result either.
        passages = self._inner.search(query, k)
        # Insertion-ordered dict, so `_store` evicts the oldest entry. FIFO
        # rather than LRU: queries in a training run do not recur in a pattern
        # that LRU would exploit, and FIFO has no bookkeeping.
        self._store(query, k, passages)
        return passages

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]:
        """Serve what is cached, fetch the rest in one call.

        This is the method that makes the cache worth putting in front of a
        batching retriever rather than behind it. A GRPO group opens with eight
        rollouts asking near-identical questions, so a batch typically contains
        a few distinct queries repeated - looping `search` would serve the
        repeats from memory but still issue one round trip per distinct miss,
        and `inner.search_batch` issues one for all of them.

        Misses are de-duplicated before the inner call, so two rollouts that
        independently invent the same new query cost one lookup, not two.
        """
        if k <= 0:
            raise RetrievalError(f"k must be positive, got {k}")
        if not queries:
            raise RetrievalError("search_batch needs at least one query")

        results: list[list[Passage]] = [[] for _ in queries]
        wanted: dict[str, list[int]] = {}
        for position, query in enumerate(queries):
            cached = self._cache.get((query, k))
            if cached is not None:
                self.hits += 1
                results[position] = list(cached)
            else:
                self.misses += 1
                wanted.setdefault(query, []).append(position)

        if wanted:
            # Outside any try/except, as in `search`: an error must not be
            # cached and must not become an empty result.
            fetched = self._inner.search_batch(list(wanted), k)
            for query, passages in zip(wanted, fetched, strict=True):
                self._store(query, k, passages)
                for position in wanted[query]:
                    results[position] = list(passages)
        return results

    def _store(self, query: str, k: int, passages: list[Passage]) -> None:
        """One cache write, with the FIFO eviction `search` documents."""
        if len(self._cache) >= self._max_entries:
            del self._cache[next(iter(self._cache))]
        self._cache[(query, k)] = tuple(passages)


class Transport(Protocol):
    """One HTTP round trip. A seam, so the client is testable without a server.

    Everything interesting about `HttpRetriever` - retries, status handling,
    response validation, the parity rules it shares with the fake - is logic
    above this line. Keeping the socket behind a protocol means all of it is
    unit-testable, and the real transport still gets exercised end to end
    against `http.server` from the standard library.
    """

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes: ...


class UrllibTransport:
    """stdlib `urllib`. No HTTP library, deliberately.

    `httpx` and `requests` are both perfectly good and neither is a core
    dependency. Adding one would mean the GPU box has to install something to
    talk to retrieval, and the whole point of putting the index behind a service
    is that the training host needs nothing from the `[retrieval]` extra.
    """

    def request(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        if not url.startswith(("http://", "https://")):
            # urllib will happily open file:// or ftp:// from a base_url typo,
            # which would read a local file and try to parse it as a response.
            raise RetrievalError(f"refusing a non-HTTP retrieval URL: {url!r}")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        # urllib defaults to GET without data and POST with it, which is what
        # the service expects. Left to that default rather than forced, but
        # stated here because it is not obvious from the call.
        request = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw: bytes = response.read()
        return raw


class HttpRetriever:
    """Talks to `retrieval/service.py`. Implements `types.Retriever`.

    Request validation happens client-side, before the socket, using the same
    rules as `LocalRetriever` and `FakeRetriever`: a blank query returns no
    passages without a round trip, and a non-positive `k` raises. Three
    implementations of one protocol are only useful if they are
    indistinguishable to the caller.

    **A failed search raises. It never returns an empty list.** Inside a
    training run, `[]` from a broken service is indistinguishable from "nothing
    matched", and across thousands of steps the policy would learn that
    searching does not pay - which is precisely the degenerate collapse the
    lambda term is supposed to be the only pressure toward. A crashed run is
    recoverable. A run that silently trained against a dead retriever is not.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        if retries < 0:
            raise RetrievalError(f"retries must be >= 0, got {retries}")
        if timeout <= 0:
            raise RetrievalError(f"timeout must be positive, got {timeout}")
        self._base_url = base_url.rstrip("/")
        self._transport: Transport = transport if transport is not None else UrllibTransport()
        self._timeout = timeout
        self._retries = retries

    @classmethod
    def connect(
        cls,
        base_url: str,
        *,
        expect_fingerprint: str | None = None,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
    ) -> HttpRetriever:
        """Build a client and check the service before handing it back.

        Separate from `__init__` so that constructing an object never does I/O,
        but this is the path to use. It is the wire-crossing version of
        `index.check_compatible`, and the mismatch it catches - a caller whose
        corpus was built by a different embedder than the service is serving -
        produces plausible passages for an entire run and raises nothing.
        """
        client = cls(base_url, transport=transport, timeout=timeout, retries=retries)
        health = client.health()
        if health.protocol_version != SERVICE_PROTOCOL_VERSION:
            raise RetrievalError(
                f"service speaks protocol {health.protocol_version}, this client speaks "
                f"{SERVICE_PROTOCOL_VERSION}. Upgrade one of them rather than hoping the "
                f"wire format did not change."
            )
        if expect_fingerprint is not None and health.embedder_fingerprint != expect_fingerprint:
            raise RetrievalError(
                f"service is serving an index built with embedder fingerprint "
                f"{health.embedder_fingerprint} ({health.embedder_model}), caller expected "
                f"{expect_fingerprint}. Querying across a mismatch returns plausible "
                f"nonsense silently."
            )
        return client

    def health(self) -> HealthResponse:
        return self._validate(HealthResponse, self._fetch("/health", None), "/health")

    def search(self, query: str, k: int) -> list[Passage]:
        if k <= 0:
            raise RetrievalError(f"k must be positive, got {k}")
        if not query.strip():
            return []
        payload = SearchRequest(query=query, k=k).model_dump_json().encode("utf-8")
        response = self._validate(SearchResponse, self._fetch("/search", payload), "/search")
        self._check_echo(response.query, query, response.k, k)
        return list(response.passages)

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]:
        """Several queries in one round trip. See `LocalRetriever.search_batch`."""
        if k <= 0:
            raise RetrievalError(f"k must be positive, got {k}")
        if not queries:
            raise RetrievalError("search_batch needs at least one query")

        real = [(position, q) for position, q in enumerate(queries) if q.strip()]
        results: list[list[Passage]] = [[] for _ in queries]
        if not real:
            return results

        request = BatchSearchRequest(queries=tuple(q for _, q in real), k=k)
        raw = self._fetch("/search_batch", request.model_dump_json().encode("utf-8"))
        response = self._validate(BatchSearchResponse, raw, "/search_batch")

        if len(response.results) != len(real):
            raise RetrievalError(
                f"/search_batch returned {len(response.results)} results for {len(real)} "
                f"queries. Results are positional, so a length mismatch means some rollout "
                f"would be scored against another rollout's passages."
            )
        for (position, query), result in zip(real, response.results, strict=True):
            self._check_echo(result.query, query, result.k, k)
            results[position] = list(result.passages)
        return results

    def _check_echo(self, got_query: str, want_query: str, got_k: int, want_k: int) -> None:
        """Verify the response is answering the question that was asked.

        The service echoes `query` and `k` precisely so this check can exist. A
        proxy, a retry or a batch misalignment that swaps two responses is
        otherwise undetectable: the passages are well-formed, they are simply
        about something else, and every reward still produces a number.
        """
        if got_query != want_query or got_k != want_k:
            raise RetrievalError(
                f"service answered query {got_query!r} at k={got_k} but was asked "
                f"{want_query!r} at k={want_k}. Responses are being mismatched to "
                f"requests, so passages would be scored against the wrong question."
            )

    def _fetch(self, path: str, payload: bytes | None) -> bytes:
        url = f"{self._base_url}{path}"
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                return self._transport.request(url, payload, self._timeout)
            except urllib.error.HTTPError as exc:
                # A 4xx is this client's own fault and fails identically on a
                # retry, so it is not retried. Read the body: the service puts
                # the actual reason in `detail`.
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RetrievalError(f"{url} returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        raise RetrievalError(
            f"{url} failed after {self._retries + 1} attempt(s): {last}. Not returning an "
            f"empty result - a training run cannot tell that apart from 'nothing matched'."
        ) from last

    def _validate(self, model: type[_ResponseT], raw: bytes, path: str) -> _ResponseT:
        try:
            return model.model_validate_json(raw)
        except ValueError as exc:
            raise RetrievalError(
                f"{self._base_url}{path} returned something that is not a {model.__name__}: {exc}"
            ) from exc


def conformance_witnesses(
    local: LocalRetriever,
    caching: CachingRetriever,
    http: HttpRetriever,
) -> tuple[Retriever, Retriever, Retriever]:
    """Return both retrievers typed as `types.Retriever`, so mypy checks the fit.

    The same mechanism `verisearch/testing/fakes.py` uses, for the same reason:
    `tests/` sits outside mypy's `files` list, so a conformance assertion written
    in a test is never type-checked and can never fail. Written here, the return
    annotation is what notices if a signature drifts from the protocol - and the
    test that calls it is what keeps this function from being dead code.
    """
    return local, caching, http
