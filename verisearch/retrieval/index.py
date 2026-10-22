"""Vector index: build, save, load, search.

PROJECT.md section 5: FAISS HNSW. Two implementations behind one protocol.

* `BruteForceIndex` - exact inner-product search in numpy. Core dependency
  only, so it runs in CI and in tests, and it is the oracle the approximate
  index is checked against. It is also a perfectly good index for the
  5k-passage slices used while developing.
* `FaissHnswIndex` - the real one. `faiss` is in the `[retrieval]` extra, so the
  import is deferred. **Not exercised yet** - neither CI nor this machine has
  faiss installed. Treat it as unverified until the first real index is built.

The guard this module is built around: an index is a pile of numbers that means
nothing without the embedder that produced it. Query an index built with one
model using vectors from another and you get confident, plausible, wrong
results, with no error anywhere. So every index carries an `IndexMeta` holding
its embedder's fingerprint, and `load_index` refuses a mismatch.

Owner: Ayesha.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from verisearch.retrieval.embedder import EmbedderSpec, check_matrix, l2_normalize

if TYPE_CHECKING:
    from collections.abc import Sequence

    import faiss
    from numpy.typing import NDArray

# Bump when the on-disk layout or the scoring changes. An index written under an
# older scheme is refused rather than reinterpreted.
INDEX_SCHEME_VERSION = 1

# HNSW graph parameters. M is the node degree; efConstruction and efSearch trade
# recall against time at build and query respectively.
#
# TODO: move into the `retrieval:` section of configs/base.yaml once the config
# PR merges, so they stop being constants in code (section 3). Defaults are the
# usual starting point, not measured - tune efSearch against BruteForceIndex
# recall on a real slice before trusting them.
DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64

_META_FILENAME = "meta.json"
_DOC_IDS_FILENAME = "doc_ids.json"
_VECTORS_FILENAME = "vectors.npy"
_FAISS_FILENAME = "index.faiss"


class VectorIndexError(Exception):
    """Raised when an index cannot be built, loaded or queried.

    Deliberately not named `IndexError`: that is a builtin meaning "subscript
    out of range", and shadowing it would make `except IndexError` catch the
    wrong thing in a way nobody would notice.
    """


class IndexHit(BaseModel):
    """One search result, at the index's level of abstraction.

    Deliberately not `types.Passage`: the index knows about rows, ids and
    scores, and nothing about titles or passage text. Joining a hit to its text
    is `retrieval/client.py`'s job, which is also where the shared `Passage`
    contract enters - and that waits on the contract session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    score: float
    row: int = Field(ge=0)


class IndexMeta(BaseModel):
    """What an index needs to know about itself to be used safely."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheme_version: int
    index_type: str
    embedder_fingerprint: str
    embedder_model: str
    dim: int = Field(ge=1)
    normalize: bool
    n_vectors: int = Field(ge=1)
    built_at: str
    hnsw_m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None

    @classmethod
    def create(
        cls,
        *,
        index_type: str,
        spec: EmbedderSpec,
        n_vectors: int,
        hnsw_m: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ) -> IndexMeta:
        return cls(
            scheme_version=INDEX_SCHEME_VERSION,
            index_type=index_type,
            embedder_fingerprint=spec.fingerprint,
            embedder_model=spec.model_name,
            dim=spec.dim,
            normalize=spec.normalize,
            n_vectors=n_vectors,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )


def check_compatible(meta: IndexMeta, spec: EmbedderSpec) -> None:
    """Refuse to use an index with the wrong embedder.

    The whole point of the module. Every branch here is a case that would
    otherwise return confident nonsense instead of failing.
    """
    if meta.scheme_version != INDEX_SCHEME_VERSION:
        raise VectorIndexError(
            f"index was written under scheme {meta.scheme_version}, this code is "
            f"{INDEX_SCHEME_VERSION}. Rebuild it rather than reinterpreting it."
        )
    if meta.dim != spec.dim:
        raise VectorIndexError(f"index has {meta.dim} dimensions, embedder produces {spec.dim}")
    if meta.embedder_fingerprint != spec.fingerprint:
        raise VectorIndexError(
            f"index was built with embedder {meta.embedder_model} "
            f"(fingerprint {meta.embedder_fingerprint}), but the live embedder is "
            f"{spec.model_name} (fingerprint {spec.fingerprint}). Querying across a "
            f"mismatch returns plausible nonsense silently - rebuild the index."
        )


class VectorIndex(Protocol):
    """What the retrieval client depends on. Not FAISS."""

    @property
    def meta(self) -> IndexMeta: ...

    def search(self, queries: NDArray[np.float32], k: int) -> list[list[IndexHit]]: ...


def _validate_build_inputs(
    vectors: NDArray[np.float32],
    doc_ids: Sequence[str],
    spec: EmbedderSpec,
) -> None:
    if vectors.shape[0] == 0:
        raise VectorIndexError("cannot build an index with zero vectors")
    if vectors.shape[0] != len(doc_ids):
        raise VectorIndexError(
            f"got {vectors.shape[0]} vectors and {len(doc_ids)} doc_ids; they must correspond"
        )
    if len(set(doc_ids)) != len(doc_ids):
        raise VectorIndexError("doc_ids must be unique - a duplicate makes a hit ambiguous")
    check_matrix(vectors, spec, what="index vectors")


def _prepare_queries(
    queries: NDArray[np.float32],
    meta: IndexMeta,
    k: int,
) -> NDArray[np.float32]:
    if k < 1:
        raise VectorIndexError(f"k must be >= 1, got {k}")
    if queries.ndim != 2:
        raise VectorIndexError(f"queries must be 2-D, got {queries.ndim}-D")
    if queries.shape[1] != meta.dim:
        raise VectorIndexError(f"queries have {queries.shape[1]} dimensions, index has {meta.dim}")
    matrix = np.asarray(queries, dtype=np.float32)
    # Normalise queries here rather than trusting the caller: a single
    # unnormalised query silently reorders its own results, and it is cheap.
    return l2_normalize(matrix) if meta.normalize else matrix


class BruteForceIndex:
    """Exact inner-product search in numpy.

    The oracle. `FaissHnswIndex` is approximate by construction, so "is the
    HNSW index any good" is only answerable by comparing its results against
    this one on the same vectors. Also entirely adequate for the few-thousand
    passage slices used while developing.
    """

    def __init__(self, vectors: NDArray[np.float32], doc_ids: Sequence[str], meta: IndexMeta):
        self._vectors = vectors
        self._doc_ids = tuple(doc_ids)
        self._meta = meta

    @property
    def meta(self) -> IndexMeta:
        return self._meta

    @classmethod
    def build(
        cls,
        vectors: NDArray[np.float32],
        doc_ids: Sequence[str],
        spec: EmbedderSpec,
    ) -> BruteForceIndex:
        _validate_build_inputs(vectors, doc_ids, spec)
        meta = IndexMeta.create(index_type="brute", spec=spec, n_vectors=vectors.shape[0])
        return cls(np.asarray(vectors, dtype=np.float32), doc_ids, meta)

    def search(self, queries: NDArray[np.float32], k: int) -> list[list[IndexHit]]:
        matrix = _prepare_queries(queries, self._meta, k)
        # k larger than the corpus is not an error - it just means "everything".
        effective = min(k, len(self._doc_ids))
        scores = matrix @ self._vectors.T

        results: list[list[IndexHit]] = []
        for row_scores in scores:
            # argpartition for the top-k, then sort just those. Ties break on
            # row order so results are reproducible rather than
            # implementation-defined.
            top = np.argpartition(-row_scores, effective - 1)[:effective]
            ordered = sorted(top, key=lambda i: (-float(row_scores[i]), int(i)))
            results.append(
                [
                    IndexHit(doc_id=self._doc_ids[i], score=float(row_scores[i]), row=int(i))
                    for i in ordered
                ]
            )
        return results

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / _VECTORS_FILENAME, self._vectors)
        (target / _DOC_IDS_FILENAME).write_text(json.dumps(list(self._doc_ids)), encoding="utf-8")
        (target / _META_FILENAME).write_text(self._meta.model_dump_json(indent=2), encoding="utf-8")


def _import_faiss() -> faiss:
    """Import faiss or fail with an actionable message.

    The return annotation names the module so mypy resolves attribute access
    through the `ignore_missing_imports` override, instead of us writing an
    explicit `Any` (banned by PROJECT.md section 3).
    """
    try:
        import faiss
    except ImportError as exc:
        raise VectorIndexError(
            "FAISS is in the retrieval extra: uv pip install -e '.[retrieval]'"
        ) from exc
    return faiss  # pragma: no cover - unreachable without the extra


class FaissHnswIndex:  # pragma: no cover - requires the retrieval extra
    """FAISS HNSW with inner-product metric.

    **Unverified.** Neither CI nor this machine has faiss installed, so nothing
    below has run. The structure mirrors `BruteForceIndex` so that the first
    thing done after installing the extra is a recall comparison against it.
    """

    def __init__(self, raw: faiss.Index, doc_ids: Sequence[str], meta: IndexMeta):
        self._raw = raw
        self._doc_ids = tuple(doc_ids)
        self._meta = meta

    @property
    def meta(self) -> IndexMeta:
        return self._meta

    @classmethod
    def build(
        cls,
        vectors: NDArray[np.float32],
        doc_ids: Sequence[str],
        spec: EmbedderSpec,
        *,
        m: int = DEFAULT_HNSW_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> FaissHnswIndex:
        _validate_build_inputs(vectors, doc_ids, spec)
        faiss = _import_faiss()

        index = faiss.IndexHNSWFlat(spec.dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        meta = IndexMeta.create(
            index_type="hnsw",
            spec=spec,
            n_vectors=vectors.shape[0],
            hnsw_m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        return cls(index, doc_ids, meta)

    def search(self, queries: NDArray[np.float32], k: int) -> list[list[IndexHit]]:
        matrix = _prepare_queries(queries, self._meta, k)
        effective = min(k, len(self._doc_ids))
        scores, indices = self._raw.search(np.ascontiguousarray(matrix), effective)

        results: list[list[IndexHit]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            hits: list[IndexHit] = []
            for score, row in zip(row_scores, row_indices, strict=True):
                # FAISS pads with -1 when it finds fewer neighbours than asked
                # for. Those are not results and must not become hits.
                if int(row) < 0:
                    continue
                hits.append(
                    IndexHit(doc_id=self._doc_ids[int(row)], score=float(score), row=int(row))
                )
            results.append(hits)
        return results

    def save(self, directory: str | Path) -> None:
        faiss = _import_faiss()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._raw, str(target / _FAISS_FILENAME))
        (target / _DOC_IDS_FILENAME).write_text(json.dumps(list(self._doc_ids)), encoding="utf-8")
        (target / _META_FILENAME).write_text(self._meta.model_dump_json(indent=2), encoding="utf-8")


def load_meta(directory: str | Path) -> IndexMeta:
    """Read an index's metadata without loading its vectors."""
    path = Path(directory) / _META_FILENAME
    if not path.is_file():
        raise VectorIndexError(f"no index metadata at {path}")
    try:
        return IndexMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise VectorIndexError(f"{path} is not a valid index metadata file: {exc}") from exc


def load_index(directory: str | Path, spec: EmbedderSpec) -> VectorIndex:
    """Load an index and verify it was built with this embedder.

    `spec` is required, not optional. Making it optional would let a caller skip
    the one check that matters, and the failure it prevents is silent.
    """
    target = Path(directory)
    meta = load_meta(target)
    check_compatible(meta, spec)

    doc_ids_path = target / _DOC_IDS_FILENAME
    if not doc_ids_path.is_file():
        raise VectorIndexError(f"no doc ids at {doc_ids_path}")
    raw_ids: object = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw_ids, list):
        raise VectorIndexError(f"{doc_ids_path}: expected a JSON list")
    doc_ids = [str(item) for item in raw_ids]

    if len(doc_ids) != meta.n_vectors:
        raise VectorIndexError(
            f"metadata says {meta.n_vectors} vectors but {len(doc_ids)} doc ids are stored"
        )

    if meta.index_type == "brute":
        vectors_path = target / _VECTORS_FILENAME
        if not vectors_path.is_file():
            raise VectorIndexError(f"no vectors at {vectors_path}")
        vectors = np.asarray(np.load(vectors_path), dtype=np.float32)
        return BruteForceIndex(vectors, doc_ids, meta)

    if meta.index_type == "hnsw":  # pragma: no cover - requires the retrieval extra
        faiss = _import_faiss()
        faiss_path = target / _FAISS_FILENAME
        if not faiss_path.is_file():
            raise VectorIndexError(f"no faiss index at {faiss_path}")
        raw = faiss.read_index(str(faiss_path))
        return FaissHnswIndex(raw, doc_ids, meta)

    raise VectorIndexError(f"unknown index_type {meta.index_type!r}")


def recall_against_oracle(
    approximate: VectorIndex,
    oracle: VectorIndex,
    queries: NDArray[np.float32],
    k: int,
) -> float:
    """Fraction of the exact top-k that the approximate index also returned.

    HNSW is approximate: `efSearch` buys recall with time, and the right value
    is a property of the corpus, not a constant someone picked. This is how that
    gets measured instead of assumed, and it is the first thing to run once the
    real index exists.
    """
    approx_results = approximate.search(queries, k)
    exact_results = oracle.search(queries, k)

    found = 0
    total = 0
    for approx, exact in zip(approx_results, exact_results, strict=True):
        exact_ids = {hit.doc_id for hit in exact}
        approx_ids = {hit.doc_id for hit in approx}
        found += len(exact_ids & approx_ids)
        total += len(exact_ids)
    if total == 0:
        raise VectorIndexError("oracle returned no results, cannot compute recall")
    return found / total
