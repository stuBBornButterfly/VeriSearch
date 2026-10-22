"""Deterministic train/dev/test assignment.

The whole module exists to avoid one specific, silent disaster: an example that
is in train for one run and test for the next. Results stop being comparable
across runs and the test number is quietly inflated, with nothing in any log to
show for it.

So splitting is **not** shuffle-and-slice. Each example's split is a pure
function of its `question_id` and the seed:

    bucket = sha256(f"{seed}:{question_id}") mapped into [0, 1)

Assignment compares that bucket against cumulative fractions. The consequence
that matters: **adding examples never moves an existing one.** Shuffle-and-slice
reassigns the entire dataset the moment one row is appended, and the leak is
invisible because both runs look fine on their own.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verisearch.data.datasets import DataError, Distribution, QAExample, Split, check_unique_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

# A split assigned under a different hash scheme is not comparable with one
# assigned under this scheme, even at the same seed. Bump this if the scheme
# ever changes, and treat that as invalidating every prior result.
SCHEME_VERSION = 1

_UINT64 = 2**64


class SplitFractions(BaseModel):
    """How much of the in-distribution pool goes to each split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = Field(gt=0.0, lt=1.0)
    dev: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> Self:
        total = self.train + self.dev + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1.0, got {total}")
        return self


# Defaults, not law. PROJECT.md section 7 does not fix these.
#
# Dev is small because it is read often during development and a dev set that
# is too big just costs wall-clock. Test is larger because section 9 reports
# bootstrap confidence intervals, and a narrow interval needs examples.
#
# TODO: move these into a `data:` section of configs/base.yaml once the
# config PR merges, so they stop being constants in code (section 3).
DEFAULT_FRACTIONS = SplitFractions(train=0.8, dev=0.05, test=0.15)


class SplitReport(BaseModel):
    """Counts per split, for the run log and the writeup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    scheme_version: int
    counts: dict[str, int]
    total: int

    def count(self, split: Split) -> int:
        return self.counts.get(split.value, 0)


class LeakageGroup(BaseModel):
    """One normalised question appearing in more than one split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    normalized_question: str
    question_ids: tuple[str, ...]
    splits: tuple[str, ...]


def bucket(question_id: str, seed: int) -> float:
    """Map an id to a stable point in [0, 1).

    `hash()` is deliberately not used: Python randomises string hashing per
    process unless PYTHONHASHSEED is set, so a split built with it would differ
    between two runs of the same script on the same machine.
    """
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _UINT64


def assign_split(
    question_id: str,
    seed: int,
    fractions: SplitFractions = DEFAULT_FRACTIONS,
) -> Split:
    """Which split this id belongs to. Pure, and stable under dataset growth."""
    point = bucket(question_id, seed)
    if point < fractions.train:
        return Split.TRAIN
    if point < fractions.train + fractions.dev:
        return Split.DEV
    return Split.TEST


def make_splits(
    examples: Sequence[QAExample],
    seed: int,
    fractions: SplitFractions = DEFAULT_FRACTIONS,
) -> tuple[QAExample, ...]:
    """Assign a split to every example.

    Out-of-distribution examples are returned as `Split.OOD` untouched - Musique
    and Bamboogle are held-out test sets in their entirety (PROJECT.md section
    7), and splitting them would be a category error.

    Input order is preserved, so the caller decides how the data is ordered and
    this function does not quietly become a shuffle.
    """
    check_unique_ids(examples)
    return tuple(
        example.with_split(Split.OOD)
        if example.distribution is Distribution.OUT_OF_DISTRIBUTION
        else example.with_split(assign_split(example.question_id, seed, fractions))
        for example in examples
    )


def summarize(examples: Sequence[QAExample], seed: int) -> SplitReport:
    """Counts per split. Log this with the run - it is cheap provenance."""
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        if example.split is None:
            raise DataError(f"example {example.question_id!r} has no split; call make_splits first")
        counts[example.split.value] += 1
    return SplitReport(
        seed=seed,
        scheme_version=SCHEME_VERSION,
        counts=dict(counts),
        total=len(examples),
    )


def find_leakage(examples: Sequence[QAExample]) -> tuple[LeakageGroup, ...]:
    """Questions that are the same after normalisation but land in different splits.

    Multi-hop QA datasets contain near-duplicates, and combining two of them
    (section 7 uses HotpotQA *and* 2WikiMultihopQA) adds cross-dataset overlap
    on top. Hash-based assignment keys on `question_id`, so two rows asking the
    same question with different ids will happily land on opposite sides.

    This is a reporting function, not a fixer. What to do about a group - drop
    it, or force it into one split - is a judgement call that belongs in a
    reviewed script, not buried in a library. Run it before every training run
    and put the result in the writeup.
    """
    by_question: dict[str, list[QAExample]] = defaultdict(list)
    for example in examples:
        by_question[example.normalized_question].append(example)

    groups: list[LeakageGroup] = []
    for normalized, group in sorted(by_question.items()):
        splits = {e.split.value for e in group if e.split is not None}
        if len(splits) > 1:
            groups.append(
                LeakageGroup(
                    normalized_question=normalized,
                    question_ids=tuple(sorted(e.question_id for e in group)),
                    splits=tuple(sorted(splits)),
                )
            )
    return tuple(groups)


def subsample(
    examples: Sequence[QAExample],
    split: Split,
    limit: int,
    seed: int,
) -> tuple[QAExample, ...]:
    """Take `limit` examples from one split, deterministically.

    PROJECT.md section 7: "Start with 2000-5000 train examples. RL needs far
    less data than SFT and each example is expensive."

    Selection is by lowest bucket under a salted hash rather than by slicing the
    input, for the same reason as the split itself: the chosen subset must not
    change when unrelated examples are added. Raising the limit later adds
    examples to the subset without removing any already in it, so a 2000-example
    run is a strict subset of a 5000-example one and the two are comparable.
    """
    if limit < 0:
        raise DataError(f"limit must be non-negative, got {limit}")
    candidates = [e for e in examples if e.split is split]
    ranked = sorted(candidates, key=lambda e: (bucket(e.question_id, seed + 1), e.question_id))
    return tuple(ranked[:limit])
