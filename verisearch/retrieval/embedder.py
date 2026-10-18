"""Passage and query embedding.

PROJECT.md section 5: `BAAI/bge-small-en-v1.5`, 384 dimensions.

The module is split so that everything which decides *what the vectors mean* is
pure and testable, and only the model call is not:

* `EmbedderSpec` - the settings that change the vectors. Its `fingerprint` is
  what an index stores so it can refuse a mismatched embedder later.
* `prepare_queries` / `l2_normalize` - pure, tested.
* `SentenceTransformerEmbedder` - the model call. `sentence-transformers` is in
  the `[retrieval]` extra, so the import is deferred and the PR pipeline never
  needs it.

The failure this module is built around: embedding passages one way and queries
another produces an index that returns plausible-looking nonsense, with no error
anywhere. Nothing crashes, retrieval just quietly gets worse. So the settings
live in one frozen object, that object is fingerprinted, and the fingerprint is
checked at load time by `retrieval/index.py`.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# BGE's documented query instruction for English retrieval. Applied to queries
# only - never to passages, which is the asymmetry that makes it easy to get
# wrong.
#
# Default off: BAAI state that v1.5 "improves its retrieval ability without
# instruction", and recommend measuring both. Treat the default as untested
# rather than correct, measure it once the index exists, and record the result.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 64


class EmbedderError(Exception):
    """Raised when embedding cannot proceed. PROJECT.md section 3: fail loud."""


class EmbedderSpec(BaseModel):
    """Everything that affects the resulting vectors.

    Two specs with the same fingerprint produce comparable vectors; two with
    different fingerprints do not, and mixing them is the silent-garbage case
    this exists to prevent. Add a field here only if it changes the numbers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(min_length=1)
    dim: int = Field(ge=1)
    # Unit-normalised vectors make inner product equal cosine similarity, which
    # is what BGE is trained for and what the index assumes.
    normalize: bool = True
    query_instruction: str | None = None
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)

    @property
    def fingerprint(self) -> str:
        """Stable hash over the fields that change the vectors.

        `batch_size` is excluded on purpose: it changes throughput, not output,
        so an index built at batch 32 is still valid when queried at batch 128.
        """
        payload = {
            "model_name": self.model_name,
            "dim": self.dim,
            "normalize": self.normalize,
            "query_instruction": self.query_instruction,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale each row to unit length.

    Zero rows are left as zeros rather than producing NaN. A zero vector means
    the encoder returned nothing useful for that text; it should score 0 against
    everything, which is what a zero row does under inner product. NaN would
    propagate into every score in the index.
    """
    if matrix.ndim != 2:
        raise EmbedderError(f"expected a 2-D array, got {matrix.ndim}-D")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(matrix / safe, dtype=np.float32)


def prepare_queries(queries: Sequence[str], spec: EmbedderSpec) -> list[str]:
    """Apply the query instruction, if the spec has one.

    Separate from encoding so the asymmetry is visible and testable: this is
    applied to queries, and there is deliberately no passage equivalent.
    """
    if not queries:
        raise EmbedderError("prepare_queries needs at least one query")
    if spec.query_instruction is None:
        return list(queries)
    return [f"{spec.query_instruction}{q}" for q in queries]


def check_matrix(matrix: NDArray[np.float32], spec: EmbedderSpec, *, what: str) -> None:
    """Validate a matrix against the spec: 2-D, right width, float32, unit norm.

    Called at index build and at query time. The normalisation check is the
    important one - inner product only equals cosine on unit vectors, so
    unnormalised input silently changes the ranking rather than failing.
    """
    if matrix.ndim != 2:
        raise EmbedderError(f"{what}: expected a 2-D array, got {matrix.ndim}-D")
    if matrix.shape[1] != spec.dim:
        raise EmbedderError(
            f"{what}: expected {spec.dim} dimensions, got {matrix.shape[1]}. "
            f"Embedder and index disagree about the model."
        )
    if matrix.dtype != np.float32:
        raise EmbedderError(f"{what}: expected float32, got {matrix.dtype}")
    if spec.normalize and matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms[norms > 0.0]
        if nonzero.size and not np.allclose(nonzero, 1.0, atol=1e-3):
            worst = float(np.max(np.abs(nonzero - 1.0)))
            raise EmbedderError(
                f"{what}: spec says normalize=True but rows are not unit length "
                f"(worst deviation {worst:.4f}). Inner product only equals cosine on "
                f"unit vectors, so this would change the ranking without failing."
            )


class SentenceTransformerEmbedder:
    """`sentence-transformers` backend. Needs the `[retrieval]` extra.

    The model load is not exercised in CI - the PR pipeline installs core plus
    dev only. What it does with a loaded model is, against a stand-in with the
    same `encode`. This class is a thin shell over `model.encode` and should
    stay thin for exactly that reason.
    """

    def __init__(self, spec: EmbedderSpec) -> None:
        self.spec = spec
        self._model: object | None = None

    def _load(self) -> object:
        # The ImportError branch is the one that runs without the extra, and it
        # is tested. The success path is what cannot run here, so that is what
        # carries the pragma.
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "SentenceTransformerEmbedder needs the retrieval extra: "
                    "uv pip install -e '.[retrieval]'"
                ) from exc
            self._model = SentenceTransformer(self.spec.model_name)  # pragma: no cover
        return self._model  # pragma: no cover - unreachable without the extra

    def _encode(self, texts: Sequence[str], *, progress: bool) -> NDArray[np.float32]:
        model = self._load()
        encode = getattr(model, "encode", None)
        if encode is None:
            raise EmbedderError(f"{self.spec.model_name} has no encode method")
        raw = encode(
            list(texts),
            batch_size=self.spec.batch_size,
            convert_to_numpy=True,
            show_progress_bar=progress,
            normalize_embeddings=False,
        )
        matrix = np.asarray(raw, dtype=np.float32)
        if self.spec.normalize:
            matrix = l2_normalize(matrix)
        check_matrix(matrix, self.spec, what="encoded batch")
        return matrix

    def encode_passages(self, passages: Sequence[str]) -> NDArray[np.float32]:
        """Encode passages. No instruction prefix, ever.

        Shows a progress bar, with rate and ETA, on stderr. Passages are only
        encoded by an index build, which is one call over the whole corpus and
        is otherwise silent for as long as it takes.
        """
        if not passages:
            raise EmbedderError("encode_passages needs at least one passage")
        return self._encode(passages, progress=True)

    def encode_queries(self, queries: Sequence[str]) -> NDArray[np.float32]:
        """Encode queries, applying the spec's instruction if it has one.

        Never shows a progress bar: queries are encoded on every search, so a
        training run would print one per step.
        """
        return self._encode(prepare_queries(queries, self.spec), progress=False)
