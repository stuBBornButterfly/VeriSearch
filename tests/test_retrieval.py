"""Tests for `verisearch/retrieval/`.

Core dependencies only - numpy and pydantic. `faiss`, `torch` and
`sentence-transformers` are in the `[retrieval]` extra, which the PR pipeline
does not install, so `FaissHnswIndex` and `SentenceTransformerEmbedder` are not
exercised here. What *is* exercised is everything that decides what the vectors
mean and whether an index may be used at all, which is where the silent
failures live.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from verisearch.retrieval.embedder import (
    BGE_QUERY_INSTRUCTION,
    EmbedderError,
    EmbedderSpec,
    SentenceTransformerEmbedder,
    check_matrix,
    l2_normalize,
    prepare_queries,
)
from verisearch.retrieval.index import (
    INDEX_SCHEME_VERSION,
    BruteForceIndex,
    IndexMeta,
    VectorIndexError,
    _import_faiss,
    check_compatible,
    load_index,
    load_meta,
    recall_against_oracle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

DIM = 8


def _spec(**overrides: object) -> EmbedderSpec:
    fields: dict[str, object] = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dim": DIM,
        "normalize": True,
    }
    fields.update(overrides)
    return EmbedderSpec.model_validate(fields)


def _vectors(n: int, seed: int = 0, *, normalize: bool = True) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, DIM)).astype(np.float32)
    return l2_normalize(raw) if normalize else raw


def _index(n: int = 20, seed: int = 0, spec: EmbedderSpec | None = None) -> BruteForceIndex:
    resolved = spec or _spec()
    return BruteForceIndex.build(_vectors(n, seed), [f"doc{i}" for i in range(n)], resolved)


# ---------------------------------------------------------------------------
# EmbedderSpec.fingerprint - the thing the whole module hangs off
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable() -> None:
    assert _spec().fingerprint == _spec().fingerprint
    assert len(_spec().fingerprint) == 12


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"model_name": "BAAI/bge-base-en-v1.5"}, id="model"),
        pytest.param({"dim": 384}, id="dim"),
        pytest.param({"normalize": False}, id="normalize"),
        pytest.param({"query_instruction": BGE_QUERY_INSTRUCTION}, id="instruction"),
    ],
)
def test_fingerprint_changes_with_anything_that_changes_the_vectors(
    change: dict[str, object],
) -> None:
    assert _spec(**change).fingerprint != _spec().fingerprint


def test_fingerprint_ignores_batch_size() -> None:
    """Batch size changes throughput, not output. An index built at batch 32
    must stay valid when queried at batch 128."""
    assert _spec(batch_size=32).fingerprint == _spec(batch_size=128).fingerprint


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_l2_normalize_gives_unit_rows() -> None:
    normalized = l2_normalize(_vectors(10, normalize=False))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert normalized.dtype == np.float32


def test_l2_normalize_leaves_zero_rows_as_zeros() -> None:
    """A zero row means the encoder returned nothing useful. It should score 0
    against everything, which is what a zero row does under inner product. NaN
    would propagate into every score in the index."""
    matrix = np.zeros((2, DIM), dtype=np.float32)
    matrix[1, 0] = 3.0
    normalized = l2_normalize(matrix)

    assert not np.isnan(normalized).any()
    assert np.allclose(normalized[0], 0.0)
    assert np.isclose(np.linalg.norm(normalized[1]), 1.0)


def test_l2_normalize_rejects_non_matrices() -> None:
    with pytest.raises(EmbedderError, match="2-D"):
        l2_normalize(np.zeros(DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# Query/passage asymmetry
# ---------------------------------------------------------------------------


def test_queries_are_unchanged_without_an_instruction() -> None:
    assert prepare_queries(["who directed it"], _spec()) == ["who directed it"]


def test_instruction_is_prefixed_to_queries() -> None:
    prepared = prepare_queries(["who directed it"], _spec(query_instruction=BGE_QUERY_INSTRUCTION))
    assert prepared == [f"{BGE_QUERY_INSTRUCTION}who directed it"]


def test_there_is_no_passage_equivalent_of_prepare_queries() -> None:
    """The asymmetry is the point: BGE's instruction goes on queries only.

    Applying it to passages too is a plausible mistake that degrades retrieval
    without failing, so there is deliberately no function that would let you.
    """
    import verisearch.retrieval.embedder as embedder_module

    assert not [n for n in dir(embedder_module) if "prepare_passage" in n]


def test_prepare_queries_rejects_an_empty_batch() -> None:
    with pytest.raises(EmbedderError, match="at least one query"):
        prepare_queries([], _spec())


# ---------------------------------------------------------------------------
# check_matrix
# ---------------------------------------------------------------------------


def test_check_matrix_accepts_valid_input() -> None:
    check_matrix(_vectors(5), _spec(), what="test")


def test_check_matrix_rejects_a_dimension_mismatch() -> None:
    wrong = l2_normalize(np.ones((3, DIM + 1), dtype=np.float32))
    with pytest.raises(EmbedderError, match="expected 8 dimensions"):
        check_matrix(wrong, _spec(), what="test")


def test_check_matrix_rejects_float64() -> None:
    with pytest.raises(EmbedderError, match="expected float32"):
        check_matrix(np.ones((3, DIM), dtype=np.float64), _spec(), what="test")  # type: ignore[arg-type]


def test_check_matrix_catches_unnormalised_vectors() -> None:
    """The silent one. Inner product equals cosine only on unit vectors, so
    unnormalised input reorders results rather than failing."""
    with pytest.raises(EmbedderError, match="not unit length"):
        check_matrix(_vectors(5, normalize=False) * 4.0, _spec(), what="test")


def test_unnormalised_vectors_are_fine_when_the_spec_says_so() -> None:
    check_matrix(_vectors(5, normalize=False), _spec(normalize=False), what="test")


# ---------------------------------------------------------------------------
# Compatibility - refusing to query an index with the wrong embedder
# ---------------------------------------------------------------------------


def test_matching_spec_is_compatible() -> None:
    check_compatible(_index().meta, _spec())


def test_a_different_model_is_refused() -> None:
    """The failure this module exists to prevent: confident, plausible, wrong
    results with no error anywhere."""
    with pytest.raises(VectorIndexError, match="rebuild the index"):
        check_compatible(_index().meta, _spec(model_name="BAAI/bge-base-en-v1.5"))


def test_a_changed_query_instruction_is_refused() -> None:
    """Subtler than a changed model and just as damaging - the index was built
    against passages embedded without an instruction."""
    with pytest.raises(VectorIndexError, match="fingerprint"):
        check_compatible(_index().meta, _spec(query_instruction=BGE_QUERY_INSTRUCTION))


def test_a_dimension_mismatch_is_refused_before_the_fingerprint() -> None:
    with pytest.raises(VectorIndexError, match="dimensions"):
        check_compatible(_index().meta, _spec(dim=384))


def test_an_old_scheme_version_is_refused() -> None:
    meta = _index().meta.model_copy(update={"scheme_version": INDEX_SCHEME_VERSION - 1})
    with pytest.raises(VectorIndexError, match="Rebuild it"):
        check_compatible(meta, _spec())


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def test_build_records_its_provenance() -> None:
    meta = _index(n=7).meta
    assert meta.n_vectors == 7
    assert meta.index_type == "brute"
    assert meta.embedder_model == "BAAI/bge-small-en-v1.5"
    assert meta.embedder_fingerprint == _spec().fingerprint
    assert meta.built_at.endswith("+00:00")


def test_build_rejects_an_empty_corpus() -> None:
    with pytest.raises(VectorIndexError, match="zero vectors"):
        BruteForceIndex.build(np.zeros((0, DIM), dtype=np.float32), [], _spec())


def test_build_rejects_a_length_mismatch() -> None:
    with pytest.raises(VectorIndexError, match="must correspond"):
        BruteForceIndex.build(_vectors(3), ["a", "b"], _spec())


def test_build_rejects_duplicate_doc_ids() -> None:
    """A duplicate makes a hit ambiguous - you cannot tell which passage it is."""
    with pytest.raises(VectorIndexError, match="unique"):
        BruteForceIndex.build(_vectors(2), ["a", "a"], _spec())


def test_build_rejects_unnormalised_vectors() -> None:
    with pytest.raises(EmbedderError, match="not unit length"):
        BruteForceIndex.build(_vectors(3, normalize=False) * 7.0, ["a", "b", "c"], _spec())


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def test_a_vector_retrieves_itself_first() -> None:
    """Sanity: cosine similarity of a unit vector with itself is 1.0, which must
    beat everything else."""
    vectors = _vectors(30)
    index = BruteForceIndex.build(vectors, [f"doc{i}" for i in range(30)], _spec())

    hits = index.search(vectors[5:6], k=3)[0]

    assert hits[0].doc_id == "doc5"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].row == 5


def test_results_are_sorted_by_descending_score() -> None:
    index = _index(40)
    hits = index.search(_vectors(1, seed=99), k=10)[0]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_is_batched() -> None:
    index = _index(25)
    results = index.search(_vectors(4, seed=7), k=3)
    assert len(results) == 4
    assert all(len(r) == 3 for r in results)


def test_k_larger_than_the_corpus_returns_everything() -> None:
    """Not an error - it just means "everything"."""
    index = _index(5)
    hits = index.search(_vectors(1, seed=3), k=100)[0]
    assert len(hits) == 5


def test_ties_break_on_row_order() -> None:
    """Reproducible rather than implementation-defined. Two identical passages
    must come back in the same order on every run."""
    duplicated = np.repeat(_vectors(1, seed=5), 4, axis=0)
    index = BruteForceIndex.build(duplicated, ["a", "b", "c", "d"], _spec())

    hits = index.search(_vectors(1, seed=5), k=4)[0]
    assert [h.doc_id for h in hits] == ["a", "b", "c", "d"]


def test_search_normalises_queries_itself() -> None:
    """A caller passing an unnormalised query would otherwise silently reorder
    its own results."""
    index = _index(20)
    query = _vectors(1, seed=11)
    scaled = (query * 13.0).astype(np.float32)

    assert [h.doc_id for h in index.search(query, k=5)[0]] == [
        h.doc_id for h in index.search(scaled, k=5)[0]
    ]


def test_search_rejects_a_dimension_mismatch() -> None:
    with pytest.raises(VectorIndexError, match="dimensions"):
        _index().search(np.ones((1, DIM + 3), dtype=np.float32), k=3)


def test_search_rejects_a_one_dimensional_query() -> None:
    with pytest.raises(VectorIndexError, match="2-D"):
        _index().search(np.ones(DIM, dtype=np.float32), k=3)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_a_non_positive_k(k: int) -> None:
    with pytest.raises(VectorIndexError, match="k must be"):
        _index().search(_vectors(1), k=k)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = _index(15, seed=4)
    index.save(tmp_path / "idx")

    reloaded = load_index(tmp_path / "idx", _spec())
    query = _vectors(2, seed=21)

    assert reloaded.meta.embedder_fingerprint == index.meta.embedder_fingerprint
    assert [[h.doc_id for h in r] for r in reloaded.search(query, k=4)] == [
        [h.doc_id for h in r] for r in index.search(query, k=4)
    ]


def test_loading_with_the_wrong_embedder_is_refused(tmp_path: Path) -> None:
    """The guard, end to end. `spec` is a required argument of `load_index`
    precisely so this check cannot be skipped."""
    _index(10).save(tmp_path / "idx")
    with pytest.raises(VectorIndexError, match="rebuild the index"):
        load_index(tmp_path / "idx", _spec(model_name="other/model"))


def test_load_meta_reads_without_loading_vectors(tmp_path: Path) -> None:
    _index(6).save(tmp_path / "idx")
    assert load_meta(tmp_path / "idx").n_vectors == 6


def test_missing_metadata_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexError, match="no index metadata"):
        load_meta(tmp_path / "empty")


def test_corrupt_metadata_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "idx"
    target.mkdir()
    (target / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(VectorIndexError, match="not a valid index metadata"):
        load_meta(target)


def test_missing_doc_ids_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "idx"
    _index(5).save(target)
    (target / "doc_ids.json").unlink()
    with pytest.raises(VectorIndexError, match="no doc ids"):
        load_index(target, _spec())


def test_missing_vectors_is_an_error(tmp_path: Path) -> None:
    target = tmp_path / "idx"
    _index(5).save(target)
    (target / "vectors.npy").unlink()
    with pytest.raises(VectorIndexError, match="no vectors"):
        load_index(target, _spec())


def test_doc_ids_must_be_a_json_list(tmp_path: Path) -> None:
    target = tmp_path / "idx"
    _index(5).save(target)
    (target / "doc_ids.json").write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(VectorIndexError, match="expected a JSON list"):
        load_index(target, _spec())


def test_a_truncated_index_is_detected(tmp_path: Path) -> None:
    """Metadata and payload disagreeing means the files were written by
    different runs, or a write was interrupted."""
    target = tmp_path / "idx"
    _index(5).save(target)
    (target / "doc_ids.json").write_text(json.dumps(["doc0", "doc1"]), encoding="utf-8")
    with pytest.raises(VectorIndexError, match="5 vectors but 2 doc ids"):
        load_index(target, _spec())


def test_an_unknown_index_type_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "idx"
    _index(5).save(target)
    meta = load_meta(target).model_copy(update={"index_type": "annoy"})
    (target / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    with pytest.raises(VectorIndexError, match="unknown index_type"):
        load_index(target, _spec())


# ---------------------------------------------------------------------------
# Recall against the oracle
# ---------------------------------------------------------------------------


def test_an_index_has_perfect_recall_against_itself() -> None:
    """Establishes that the measurement is sound before it is pointed at HNSW,
    whose efSearch is a property of the corpus rather than a constant."""
    index = _index(50, seed=2)
    assert recall_against_oracle(index, index, _vectors(5, seed=31), k=5) == 1.0


def test_recall_is_partial_for_a_worse_index() -> None:
    """A deliberately degraded index must not score 1.0, or the measurement
    cannot distinguish a good HNSW build from a bad one."""
    vectors = _vectors(40, seed=6)
    doc_ids = [f"doc{i}" for i in range(40)]
    oracle = BruteForceIndex.build(vectors, doc_ids, _spec())
    # Half the corpus, so at most half of any exact top-k can be found.
    degraded = BruteForceIndex.build(vectors[:20], doc_ids[:20], _spec())

    recall = recall_against_oracle(degraded, oracle, _vectors(8, seed=12), k=6)
    assert 0.0 <= recall < 1.0


def test_recall_rejects_mismatched_batch_sizes() -> None:
    index = _index(10)
    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        recall_against_oracle(_StubIndex(index.meta), index, _vectors(3), k=2)


class _StubIndex:
    """Returns one result row regardless of how many queries it was given."""

    def __init__(self, meta: IndexMeta) -> None:
        self._meta = meta

    @property
    def meta(self) -> IndexMeta:
        return self._meta

    def search(self, queries: NDArray[np.float32], k: int) -> list[list[object]]:
        return [[]]


# ---------------------------------------------------------------------------
# Missing-extra error paths. Monkeypatched rather than relying on this machine
# lacking the packages, so the tests mean the same thing wherever they run.
# ---------------------------------------------------------------------------


def test_faiss_import_failure_says_how_to_fix_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(VectorIndexError, match=r"uv pip install -e '\.\[retrieval\]'"):
        _import_faiss()


def test_sentence_transformers_import_failure_says_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder(_spec())
    with pytest.raises(EmbedderError, match=r"uv pip install -e '\.\[retrieval\]'"):
        embedder.encode_passages(["some passage"])


def test_encode_passages_rejects_an_empty_batch() -> None:
    """Checked before the model is touched, so it needs no extra."""
    with pytest.raises(EmbedderError, match="at least one passage"):
        SentenceTransformerEmbedder(_spec()).encode_passages([])


def test_recall_with_an_empty_oracle_is_an_error() -> None:
    """A recall of 0/0 is not 0.0, it is a broken measurement."""
    meta = _index(3).meta
    with pytest.raises(VectorIndexError, match="oracle returned no results"):
        recall_against_oracle(_StubIndex(meta), _StubIndex(meta), _vectors(1), k=2)


def test_check_matrix_rejects_a_one_dimensional_array() -> None:
    with pytest.raises(EmbedderError, match="expected a 2-D array"):
        check_matrix(np.ones(DIM, dtype=np.float32), _spec(), what="test")


def test_encode_queries_also_reports_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both entry points must fail the same way - the query path applies the
    instruction first, so it reaches the model by a different route."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder(_spec(query_instruction=BGE_QUERY_INSTRUCTION))
    with pytest.raises(EmbedderError, match="retrieval extra"):
        embedder.encode_queries(["who directed it"])
