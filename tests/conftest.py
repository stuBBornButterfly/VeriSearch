"""Shared fixtures. PROJECT.md section 4: fake Retriever, fake scorer.

The fakes themselves are defined in `verisearch.testing.fakes`, not here, so
that `mypy --strict` checks them against the protocols in `verisearch.types` -
`tests/` is outside mypy's `files` list. This module only wires them up as
fixtures.

Stateless helpers are not fixtures. Import `make_trajectory` and
`DEFAULT_CORPUS` directly from `verisearch.testing`; a fixture would only add a
layer of indirection between a test and the thing it is building.
"""

from __future__ import annotations

import pytest

from verisearch.testing import DEFAULT_CORPUS, FakeFaithfulnessScorer, FakeRetriever
from verisearch.types import Passage


@pytest.fixture
def fake_corpus() -> tuple[Passage, ...]:
    """The default five-passage corpus: three relevant, two distractors."""
    return DEFAULT_CORPUS


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """A fresh retriever per test, because it records every call it receives."""
    return FakeRetriever()


@pytest.fixture
def blind_retriever() -> FakeRetriever:
    """Returns nothing for the query below. For the zero-passage path."""
    return FakeRetriever(blind_to=("no such thing",))


@pytest.fixture
def fake_scorer() -> FakeFaithfulnessScorer:
    """Default scorer: zero claims scores 0.0, support threshold 0.6."""
    return FakeFaithfulnessScorer()
