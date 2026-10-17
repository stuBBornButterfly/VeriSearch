"""Test doubles for the shared contract. Imported by `tests/conftest.py`.

Shipped inside the package rather than confined to `tests/` so that `mypy
--strict` checks the fakes against the protocols they claim to satisfy, and so
the RL half can import them from a script or a notebook without pytest. See
`verisearch.testing.fakes` for why that matters.
"""

from __future__ import annotations

from verisearch.testing.fakes import (
    DEFAULT_CORPUS,
    FakeFaithfulnessScorer,
    FakeRetriever,
    conformance_witnesses,
    make_trajectory,
)

__all__ = [
    "DEFAULT_CORPUS",
    "FakeFaithfulnessScorer",
    "FakeRetriever",
    "conformance_witnesses",
    "make_trajectory",
]
