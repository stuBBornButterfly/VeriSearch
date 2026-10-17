"""Tests for `retrieval/client.py` - the hit-to-Passage join.

The embedder is the one piece that cannot run here (`sentence-transformers` is
in the `[retrieval]` extra and this machine has no torch), so these tests drive
`LocalRetriever` with `TableEmbedder`: a lookup table from text to a vector the
test wrote itself. That is deliberately not a hash-based fake - the point of
most of these tests is "which document came back, in which order", and that is
only assertable when the test chose the similarities.

TODO: move `TableEmbedder` into `verisearch/testing/fakes.py` once #9 merges, so
it is type-checked and reusable by the service tests. `tests/` sits outside
mypy's `files` list, so it is unchecked where it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from verisearch.retrieval.client import (
    CachingRetriever,
    HttpRetriever,
    InMemoryDocStore,
    LocalRetriever,
    RetrievalError,
    StoredDoc,
    conformance_witnesses,
    load_doc_store,
)
from verisearch.retrieval.embedder import EmbedderSpec, l2_normalize
from verisearch.retrieval.index import BruteForceIndex, IndexHit, VectorIndexError
from verisearch.types import Passage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

# Four orthogonal axes, one per document, so "which document wins" is a fact the
# test states rather than a property of some model.
SPEC = EmbedderSpec(model_name="table-test", dim=4, normalize=True)

DOC_VECTORS: dict[str, list[float]] = {
    "d1": [1.0, 0.0, 0.0, 0.0],
    "d2": [0.0, 1.0, 0.0, 0.0],
    "d3": [0.0, 0.0, 1.0, 0.0],
    "d4": [0.0, 0.0, 0.0, 1.0],
}

QUERY_VECTORS: dict[str, list[float]] = {
    # Squarely on d2.
    "analytical engine": [0.0, 1.0, 0.0, 0.0],
    # Mostly d3, a little d1: pins the expected ordering to two documents.
    "first programmer": [0.3, 0.0, 1.0, 0.0],
    # Equidistant from d1 and d4. The index breaks the tie on row number, and
    # `_to_passages` must not reorder that.
    "tie": [1.0, 0.0, 0.0, 1.0],
}


class TableEmbedder:
    """Returns the vector the test wrote for each query. No model, no torch."""

    def __init__(self, spec: EmbedderSpec, table: dict[str, list[float]]) -> None:
        self.spec = spec
        self._table = table
        self.calls: list[list[str]] = []

    def encode_queries(self, queries: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(queries))
        rows = []
        for query in queries:
            if query not in self._table:
                raise AssertionError(f"test asked for an unlisted query: {query!r}")
            rows.append(self._table[query])
        matrix = np.asarray(rows, dtype=np.float32)
        return l2_normalize(matrix) if self.spec.normalize else matrix


def make_docs() -> list[StoredDoc]:
    return [
        StoredDoc(doc_id="d1", title="Ada Lovelace", text="Ada Lovelace wrote the first program."),
        StoredDoc(doc_id="d2", title="Analytical Engine", text="Babbage designed the engine."),
        StoredDoc(doc_id="d3", title="First Programmer", text="She is called the first coder."),
        StoredDoc(doc_id="d4", title="Lovelace, Texas", text="A town in Texas. A distractor."),
    ]


def make_index() -> BruteForceIndex:
    doc_ids = ["d1", "d2", "d3", "d4"]
    vectors = l2_normalize(np.asarray([DOC_VECTORS[d] for d in doc_ids], dtype=np.float32))
    return BruteForceIndex.build(vectors, doc_ids, SPEC)


def make_retriever() -> tuple[LocalRetriever, TableEmbedder]:
    embedder = TableEmbedder(SPEC, QUERY_VECTORS)
    retriever = LocalRetriever(make_index(), embedder, InMemoryDocStore.from_records(make_docs()))
    return retriever, embedder


# ---------------------------------------------------------------------------
# The document store
# ---------------------------------------------------------------------------


def test_store_resolves_known_ids_and_returns_none_for_unknown() -> None:
    store = InMemoryDocStore.from_records(make_docs())
    assert len(store) == 4
    found = store.get("d2")
    assert found is not None
    assert found.title == "Analytical Engine"
    assert store.get("nope") is None


def test_duplicate_doc_id_is_refused_rather_than_resolved_by_read_order() -> None:
    docs = [*make_docs(), StoredDoc(doc_id="d1", title="Ada Lovelace", text="A different text.")]
    with pytest.raises(RetrievalError, match="duplicate doc_id 'd1'"):
        InMemoryDocStore.from_records(docs)


def test_empty_corpus_is_refused() -> None:
    with pytest.raises(RetrievalError, match="zero documents"):
        InMemoryDocStore.from_records([])


def test_corpus_doc_shape_satisfies_the_record_protocol_structurally() -> None:
    """`build_corpus.CorpusDoc` must work here without either module importing
    the other. Stand in for it with a class of the same shape - if `DocRecord`
    ever grows a field, this is what fails."""

    class CorpusDocLike:
        def __init__(self, doc_id: str, title: str, text: str) -> None:
            self.doc_id = doc_id
            self.title = title
            self.text = text

    store = InMemoryDocStore.from_records(
        [CorpusDocLike("x1", "Title", "Body text."), CorpusDocLike("x2", "Other", "More text.")]
    )
    assert len(store) == 2


# ---------------------------------------------------------------------------
# Loading a corpus from disk
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_doc_store_reads_write_corpus_output(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path / "corpus.jsonl",
        [doc.model_dump_json() for doc in make_docs()],
    )
    store = load_doc_store(source)
    assert len(store) == 4
    found = store.get("d4")
    assert found is not None
    assert found.title == "Lovelace, Texas"


def test_load_doc_store_skips_blank_lines(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            '{"doc_id":"a","title":"A","text":"Body."}',
            "",
            "   ",
            '{"doc_id":"b","title":"B","text":"Body."}',
        ],
    )
    assert len(load_doc_store(source)) == 2


def test_load_doc_store_names_the_bad_line(tmp_path: Path) -> None:
    source = write_jsonl(
        tmp_path / "corpus.jsonl",
        ['{"doc_id":"a","title":"A","text":"Body."}', "{not json"],
    )
    with pytest.raises(RetrievalError, match=r"corpus\.jsonl:2"):
        load_doc_store(source)


def test_load_doc_store_rejects_a_document_with_no_text(tmp_path: Path) -> None:
    """An empty `text` would index as a zero vector and retrieve as a passage
    with nothing in it, which faithfulness would then score against."""
    source = write_jsonl(tmp_path / "corpus.jsonl", ['{"doc_id":"a","title":"A","text":""}'])
    with pytest.raises(RetrievalError, match="not a corpus document"):
        load_doc_store(source)


def test_load_doc_store_rejects_extra_fields(tmp_path: Path) -> None:
    """`extra="forbid"`: a renamed field in `write_corpus` must fail loudly here
    rather than being read as a document missing that field."""
    source = write_jsonl(
        tmp_path / "corpus.jsonl",
        ['{"doc_id":"a","title":"A","text":"Body.","paragraph":"A"}'],
    )
    with pytest.raises(RetrievalError, match="not a corpus document"):
        load_doc_store(source)


def test_missing_corpus_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="no corpus file at"):
        load_doc_store(tmp_path / "absent.jsonl")


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_mismatched_embedder_is_refused_at_construction() -> None:
    """The silent-garbage case. An index built with one model and queried with
    another returns plausible, wrong results and raises nothing."""
    other = EmbedderSpec(model_name="a-different-model", dim=4, normalize=True)
    embedder = TableEmbedder(other, QUERY_VECTORS)
    with pytest.raises(VectorIndexError, match="rebuild the index"):
        LocalRetriever(make_index(), embedder, InMemoryDocStore.from_records(make_docs()))


def test_store_smaller_than_the_index_is_refused_at_construction() -> None:
    short = InMemoryDocStore.from_records(make_docs()[:2])
    with pytest.raises(RetrievalError, match="4 vectors but the document store holds 2"):
        LocalRetriever(make_index(), TableEmbedder(SPEC, QUERY_VECTORS), short)


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def test_search_returns_passages_with_titles_text_and_scores() -> None:
    retriever, _ = make_retriever()
    passages = retriever.search("analytical engine", 2)

    assert [p.doc_id for p in passages] == ["d2", "d1"]
    assert passages[0].title == "Analytical Engine"
    assert passages[0].text == "Babbage designed the engine."
    # d2's vector is the query's vector, so the inner product is 1.
    assert passages[0].score == pytest.approx(1.0)
    assert passages[1].score == pytest.approx(0.0)
    assert all(isinstance(p, Passage) for p in passages)


def test_search_preserves_the_index_ordering_including_its_tie_break() -> None:
    """`index.py` breaks score ties on row number so results are reproducible.
    Re-sorting in the join would silently replace that with Python's own."""
    retriever, _ = make_retriever()
    passages = retriever.search("tie", 2)
    scores = [p.score for p in passages]
    assert scores[0] == pytest.approx(scores[1])
    assert [p.doc_id for p in passages] == ["d1", "d4"]


def test_k_larger_than_the_corpus_returns_everything_without_failing() -> None:
    retriever, _ = make_retriever()
    assert len(retriever.search("analytical engine", 99)) == 4


def test_non_positive_k_raises_and_is_catchable_as_a_valueerror() -> None:
    """`FakeRetriever` raises a bare `ValueError` for this, and Dhrubo writes the
    episode loop against the fake. If the real one were not catchable the same
    way, his `except` clause would pass CI and fail on the GPU box."""
    retriever, _ = make_retriever()
    for bad in (0, -1):
        with pytest.raises(ValueError, match="k must be positive"):
            retriever.search("analytical engine", bad)


@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_a_blank_query_returns_no_passages_and_never_reaches_the_embedder(query: str) -> None:
    """Matches the fake. A policy emitting `<search></search>` is a format
    failure the reward prices, not an exception that kills the GRPO group."""
    retriever, embedder = make_retriever()
    assert retriever.search(query, 3) == []
    assert embedder.calls == []


def test_search_batch_agrees_with_search_called_once_per_query() -> None:
    retriever, _ = make_retriever()
    queries = ["analytical engine", "first programmer", "tie"]
    batched = retriever.search_batch(queries, 2)
    singly = [retriever.search(q, 2) for q in queries]
    assert batched == singly


def test_search_batch_keeps_blank_queries_in_position() -> None:
    """The failure this guards: dropping blanks before the index call and
    appending results back shifts every later rollout's passages onto the wrong
    query, and the reward still computes."""
    retriever, embedder = make_retriever()
    results = retriever.search_batch(["", "analytical engine", "  ", "tie"], 1)

    assert [len(r) for r in results] == [0, 1, 0, 1]
    assert results[1][0].doc_id == "d2"
    assert results[3][0].doc_id == "d1"
    # Only the two real queries were embedded.
    assert embedder.calls == [["analytical engine", "tie"]]


def test_search_batch_with_no_queries_raises() -> None:
    retriever, _ = make_retriever()
    with pytest.raises(RetrievalError, match="at least one query"):
        retriever.search_batch([], 2)


def test_a_hit_the_store_cannot_resolve_is_an_error_not_a_short_result() -> None:
    """Index and corpus built from different data. Returning three passages for
    k=4 reads as weak retrieval; the message has to say it is a build error."""

    class HalfStore:
        def get(self, doc_id: str) -> StoredDoc | None:
            return None if doc_id == "d2" else StoredDoc(doc_id=doc_id, title="T", text="Body.")

        def __len__(self) -> int:
            return 4

    retriever = LocalRetriever(make_index(), TableEmbedder(SPEC, QUERY_VECTORS), HalfStore())
    with pytest.raises(RetrievalError, match="not in the document store"):
        retriever.search("analytical engine", 2)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class CountingRetriever:
    """Records every call so the cache's effect is observable.

    `batches` is separate from `calls` on purpose: the point of the caching
    layer's `search_batch` is that N queries cost one inner round trip, and that
    is only observable if batch calls are counted as batches.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.batches: list[tuple[tuple[str, ...], int]] = []

    def search(self, query: str, k: int) -> list[Passage]:
        self.calls.append((query, k))
        return [Passage(doc_id=f"{query}-{k}", title="T", text="Body.", score=1.0)]

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]:
        self.batches.append((tuple(queries), k))
        return [
            [Passage(doc_id=f"{query}-{k}", title="T", text="Body.", score=1.0)]
            for query in queries
        ]


def test_a_repeated_query_is_served_from_memory() -> None:
    """GRPO samples 8 rollouts from one prompt, so the same early query arrives
    up to eight times within seconds."""
    inner = CountingRetriever()
    cached = CachingRetriever(inner)

    first = cached.search("ada lovelace", 4)
    rest = [cached.search("ada lovelace", 4) for _ in range(7)]

    assert inner.calls == [("ada lovelace", 4)]
    assert all(r == first for r in rest)
    assert (cached.hits, cached.misses) == (7, 1)


def test_the_same_query_at_a_different_k_is_a_different_entry() -> None:
    """k is part of the result, so sharing an entry across k would return the
    wrong number of passages - and the reward would still compute."""
    inner = CountingRetriever()
    cached = CachingRetriever(inner)
    cached.search("q", 2)
    cached.search("q", 5)
    assert inner.calls == [("q", 2), ("q", 5)]
    assert cached.size == 2


def test_the_cache_hands_out_a_fresh_list_each_time() -> None:
    inner = CountingRetriever()
    cached = CachingRetriever(inner)
    first = cached.search("q", 1)
    first.clear()
    assert len(cached.search("q", 1)) == 1


def test_the_cache_is_bounded_and_evicts_the_oldest_entry() -> None:
    inner = CountingRetriever()
    cached = CachingRetriever(inner, max_entries=2)
    for query in ("a", "b", "c"):
        cached.search(query, 1)

    assert cached.size == 2
    # "a" was evicted, so it costs another inner call; "c" is still resident.
    cached.search("a", 1)
    cached.search("c", 1)
    assert [q for q, _ in inner.calls] == ["a", "b", "c", "a"]


def test_a_retrieval_error_is_not_cached() -> None:
    """A cached exception would turn one transient failure into a permanent
    one for the rest of the run."""

    class FlakyRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, k: int) -> list[Passage]:
            self.calls += 1
            if self.calls == 1:
                raise RetrievalError("transient")
            return [Passage(doc_id="d1", title="T", text="Body.")]

    inner = FlakyRetriever()
    cached = CachingRetriever(inner)
    with pytest.raises(RetrievalError, match="transient"):
        cached.search("q", 1)
    assert len(cached.search("q", 1)) == 1
    assert inner.calls == 2
    assert cached.size == 1


def test_a_zero_size_cache_is_refused() -> None:
    with pytest.raises(RetrievalError, match="max_entries must be >= 1"):
        CachingRetriever(CountingRetriever(), max_entries=0)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_all_three_retrievers_satisfy_the_shared_protocol() -> None:
    """The assertion that matters is the return annotation inside
    `conformance_witnesses`, which mypy checks. `tests/` is outside mypy's
    `files` list, so asserting it here instead would never fail. This call is
    what keeps that function from being dead code.

    The HTTP one is included because three implementations of one protocol are
    only useful if the caller cannot tell them apart - and that one is the only
    implementation Dhrubo will actually point at a GPU box."""
    retriever, _ = make_retriever()
    local, caching, http = conformance_witnesses(
        retriever, CachingRetriever(retriever), HttpRetriever("http://localhost:8099")
    )

    assert local.search("analytical engine", 1)[0].doc_id == "d2"
    assert caching.search("analytical engine", 1)[0].doc_id == "d2"
    # The HTTP witness is never called - there is no service here. Its reason
    # for existing is the annotation; `tests/test_service.py` exercises it.
    assert http.search("", 1) == []


def test_an_index_hit_carries_no_text_so_the_join_is_the_only_source() -> None:
    """Pins the layering: if `IndexHit` ever grows a `text` field there would be
    two sources of passage text and no rule for which wins."""
    hit = IndexHit(doc_id="d1", score=0.5, row=0)
    assert not hasattr(hit, "text")
    assert not hasattr(hit, "title")


# ---------------------------------------------------------------------------
# CachingRetriever.search_batch - the method the protocol change enables
# ---------------------------------------------------------------------------


def test_a_batch_of_misses_costs_one_inner_round_trip() -> None:
    """This is why `search_batch` belongs on the shared protocol.

    Before it, `CachingRetriever` held a `Retriever` with no batch method, so
    the cache could only ever loop - one round trip per distinct query, against
    an HTTP service, inside the rollout loop.
    """
    inner = CountingRetriever()
    cached = CachingRetriever(inner)

    results = cached.search_batch(["a", "b", "c"], 2)

    assert [p[0].doc_id for p in results] == ["a-2", "b-2", "c-2"]
    assert inner.batches == [(("a", "b", "c"), 2)]
    assert inner.calls == []
    assert cached.misses == 3
    assert cached.hits == 0


def test_a_batch_served_entirely_from_memory_never_touches_the_inner() -> None:
    inner = CountingRetriever()
    cached = CachingRetriever(inner)
    cached.search_batch(["a", "b"], 2)
    inner.batches.clear()

    results = cached.search_batch(["b", "a"], 2)

    assert [p[0].doc_id for p in results] == ["b-2", "a-2"]
    assert inner.batches == []
    assert cached.hits == 2


def test_a_mixed_batch_fetches_only_the_misses_and_keeps_order() -> None:
    inner = CountingRetriever()
    cached = CachingRetriever(inner)
    cached.search("b", 2)  # warm one entry; this is itself a miss
    inner.calls.clear()
    before = cached.misses

    results = cached.search_batch(["a", "b", "c"], 2)

    assert [p[0].doc_id for p in results] == ["a-2", "b-2", "c-2"]
    assert inner.batches == [(("a", "c"), 2)]
    assert cached.hits == 1
    assert cached.misses - before == 2


def test_a_query_repeated_inside_one_batch_is_fetched_once() -> None:
    """The GRPO case exactly: eight rollouts, one prompt, the same opening query.

    A loop over `search` would serve seven of those from memory only because the
    first had already returned - inside a single batch there is nothing cached
    yet, so without de-duplication this is eight identical lookups.
    """
    inner = CountingRetriever()
    cached = CachingRetriever(inner)

    results = cached.search_batch(["a"] * 8, 3)

    assert inner.batches == [(("a",), 3)]
    assert [p[0].doc_id for p in results] == ["a-3"] * 8


def test_batch_results_are_separate_lists() -> None:
    """A de-duplicated query must not hand the same list to two positions."""
    cached = CachingRetriever(CountingRetriever())
    first, second = cached.search_batch(["a", "a"], 1)
    first.clear()
    assert len(second) == 1


def test_a_batch_respects_the_cache_bound() -> None:
    inner = CountingRetriever()
    cached = CachingRetriever(inner, max_entries=2)

    cached.search_batch(["a", "b", "c"], 1)

    assert cached.size == 2
    assert [p[0].doc_id for p in cached.search_batch(["c"], 1)] == ["c-1"]
    assert cached.hits == 1


def test_a_batch_error_is_not_cached() -> None:
    """Same contract as `search`: one transient failure must not become permanent."""

    class FlakyBatch:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, k: int) -> list[Passage]:
            raise AssertionError("search_batch must not fall back to search")

        def search_batch(self, queries: Sequence[str], k: int) -> list[list[Passage]]:
            self.calls += 1
            if self.calls == 1:
                raise RetrievalError("transient")
            return [[Passage(doc_id="d1", title="T", text="Body.")] for _ in queries]

    inner = FlakyBatch()
    cached = CachingRetriever(inner)
    with pytest.raises(RetrievalError, match="transient"):
        cached.search_batch(["q"], 1)
    assert cached.size == 0
    assert len(cached.search_batch(["q"], 1)[0]) == 1
    assert inner.calls == 2


@pytest.mark.parametrize(
    ("queries", "k", "message"),
    [
        pytest.param(["a"], 0, "k must be positive", id="zero-k"),
        pytest.param(["a"], -1, "k must be positive", id="negative-k"),
        pytest.param([], 1, "at least one query", id="no-queries"),
    ],
)
def test_the_cache_refuses_the_same_bad_input_the_real_retriever_does(
    queries: list[str], k: int, message: str
) -> None:
    """A wrapper that validated less than what it wraps would hide caller bugs
    until they reached the live service."""
    cached = CachingRetriever(CountingRetriever())
    with pytest.raises(RetrievalError, match=message):
        cached.search_batch(queries, k)
