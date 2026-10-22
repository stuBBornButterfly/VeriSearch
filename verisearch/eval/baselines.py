"""The four baselines, and the thing that makes comparing them mean something.

PROJECT.md section 8 lists four baselines and one constraint: *"All on identical
eval sets with the identical retriever."* Section 2 makes that a binary success
criterion. The constraint is the hard part, not the baselines.

Here is how it fails without ever raising. Baseline 3 runs on Monday. The index
gets rebuilt on Wednesday with a different embedder, or a split file gets
regenerated and the examples come out in a different order, or someone evaluates
the trained model at k=5 against a baseline measured at k=3. Every number still
computes, the table still prints, and the headline claim is now comparing two
different experiments. Nothing in the output looks wrong.

So a run carries an `EvalContext` - the dataset, the split, a fingerprint over
the ordered question ids, the retriever's embedder fingerprint, and k - and
`check_comparable` refuses two contexts that differ. That turns section 2's
sixth criterion from a claim into something the eval script checks.

`joint_improvement_claim` is section 1's thesis as a function: higher accuracy
**and** lower unsupported-claim rate **at fewer retrievals** than the prompted
RAG baseline, each with an interval. All three or the claim does not hold.

No model code here. A baseline is a configuration plus a protocol, and running
one needs weights this project does not yet have hardware for.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

# Runtime import, not a TYPE_CHECKING one. `ClaimPart.delta` is a Pydantic field
# of this type, and Pydantic resolves field annotations at class-construction
# time - behind `TYPE_CHECKING` the model builds fine, type-checks fine, and
# raises `PydanticUserError` the first time anyone instantiates it. ruff's
# flake8-type-checking rules would move it; do not let them.
from verisearch.eval.metrics import ConfidenceInterval

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class EvalError(Exception):
    """Raised when an evaluation or a comparison cannot be trusted."""


class BaselineKind(StrEnum):
    """The four baselines of section 8, plus the system under test.

    `TRAINED` is in the same enum deliberately. It is evaluated by the same
    harness, under the same context, and the comparison is only valid if it was
    - keeping it in a separate type would make "run the trained model slightly
    differently" an easy mistake rather than an impossible one.
    """

    NO_RETRIEVAL = "no_retrieval"
    VANILLA_RAG = "vanilla_rag"
    PROMPTED_ITERATIVE = "prompted_iterative"
    SFT_ONLY = "sft_only"
    TRAINED = "trained"


class BaselineSpec(BaseModel):
    """One baseline as data: what it is allowed to do, and what it isolates.

    `isolates` is not documentation for its own sake. Section 8 calls baseline 3
    "the baseline that matters" and baseline 4 the one that "separates RL helped
    from training helped". If a baseline gets dropped for time, this field is
    what says which claim died with it.
    """

    model_config = _FROZEN

    kind: BaselineKind
    label: str = Field(min_length=1)
    retrieval_enabled: bool
    # None means "whatever the config's search budget is". 0 and 1 are structural
    # properties of the baseline and are not config's to change.
    max_searches: int | None = Field(ge=0)
    requires_checkpoint: bool
    isolates: str = Field(min_length=1)


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        kind=BaselineKind.NO_RETRIEVAL,
        label="Base model, no retrieval",
        retrieval_enabled=False,
        max_searches=0,
        requires_checkpoint=False,
        isolates="the floor - how much of this benchmark is answerable from parametric memory",
    ),
    BaselineSpec(
        kind=BaselineKind.VANILLA_RAG,
        label="Vanilla RAG, single retrieval",
        retrieval_enabled=True,
        max_searches=1,
        requires_checkpoint=False,
        isolates="what one retrieval on the raw question buys over no retrieval",
    ),
    BaselineSpec(
        kind=BaselineKind.PROMPTED_ITERATIVE,
        label="Prompted iterative RAG, no training",
        retrieval_enabled=True,
        max_searches=None,
        requires_checkpoint=False,
        isolates="whether RL beat a careful prompt - the comparison the whole claim rests on",
    ),
    BaselineSpec(
        kind=BaselineKind.SFT_ONLY,
        label="SFT on successful prompted trajectories",
        retrieval_enabled=True,
        max_searches=None,
        requires_checkpoint=True,
        isolates="RL helped, as distinct from training on anything at all helped",
    ),
    BaselineSpec(
        kind=BaselineKind.TRAINED,
        label="GRPO-trained policy",
        retrieval_enabled=True,
        max_searches=None,
        requires_checkpoint=True,
        isolates="the system under test",
    ),
)

_BY_KIND = {spec.kind: spec for spec in BASELINES}

# The baseline every headline number is measured against. Named rather than
# indexed so that reordering BASELINES cannot silently change what "the
# baseline" means in a report.
REFERENCE_BASELINE = BaselineKind.PROMPTED_ITERATIVE


def baseline(kind: BaselineKind) -> BaselineSpec:
    """Look up one baseline. Raises rather than returning None."""
    spec = _BY_KIND.get(kind)
    if spec is None:  # pragma: no cover - unreachable while the enum and BASELINES agree
        raise EvalError(f"no baseline spec for {kind!r}")
    return spec


def resolved_search_budget(spec: BaselineSpec, config_budget: int) -> int:
    """The search budget this baseline actually runs with.

    Baselines 1 and 2 pin their own budget - 0 and 1 are what *makes* them those
    baselines, and letting a config override them would turn "vanilla RAG" into
    something else while still labelling it vanilla RAG in the table.
    """
    if config_budget < 0:
        raise EvalError(f"search budget must be >= 0, got {config_budget}")
    if spec.max_searches is None:
        return config_budget
    return spec.max_searches


def examples_fingerprint(question_ids: Iterable[str]) -> str:
    """Stable hash over the ordered question ids.

    `hashlib`, not `hash()`: Python randomises string hashing per process unless
    `PYTHONHASHSEED` is set, and this pipeline deliberately leaves it unset, so
    `hash()` here would make two runs of the same eval set look like different
    eval sets.

    Order is included. Two runs over the same questions in different orders are
    comparable in aggregate, but a *paired* bootstrap pairs positionally, so a
    reordered set would pair each question against a different one and still
    produce a tight interval.
    """
    digest = hashlib.sha256()
    count = 0
    for question_id in question_ids:
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\x00")
        count += 1
    if count == 0:
        raise EvalError("cannot fingerprint an empty eval set")
    return digest.hexdigest()[:16]


class EvalContext(BaseModel):
    """Everything that must match for two runs to be comparable.

    Add a field here when you find something that can differ between two runs
    and change the numbers. Every field is one way the section 2 claim could be
    quietly false.
    """

    model_config = _FROZEN

    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    n_examples: int = Field(ge=1)
    examples_fingerprint: str = Field(min_length=1)
    # The index's embedder fingerprint, from `IndexMeta`. "identical retriever"
    # in section 8 means this, not "we used the same class".
    retriever_fingerprint: str = Field(min_length=1)
    k: int = Field(ge=1)


def context_for(
    question_ids: Sequence[str],
    *,
    dataset: str,
    split: str,
    retriever_fingerprint: str,
    k: int,
) -> EvalContext:
    """Build a context from the examples a run is about to evaluate."""
    if len(set(question_ids)) != len(question_ids):
        # A duplicate makes a paired comparison ambiguous and inflates whichever
        # question is doubled.
        raise EvalError("eval set contains duplicate question ids")
    return EvalContext(
        dataset=dataset,
        split=split,
        n_examples=len(question_ids),
        examples_fingerprint=examples_fingerprint(question_ids),
        retriever_fingerprint=retriever_fingerprint,
        k=k,
    )


def check_comparable(reference: EvalContext, candidate: EvalContext) -> None:
    """Refuse two runs that are not measuring the same thing.

    Section 2: "4 baselines evaluated on identical eval sets with the identical
    retriever." This is where that is enforced instead of remembered.
    """
    differences = [
        f"{field}: {getattr(reference, field)!r} vs {getattr(candidate, field)!r}"
        for field in EvalContext.model_fields
        if getattr(reference, field) != getattr(candidate, field)
    ]
    if differences:
        raise EvalError(
            "these two runs are not comparable - "
            + "; ".join(differences)
            + ". Section 8 requires identical eval sets and an identical retriever; "
            "comparing across a difference produces a number that looks fine and "
            "answers no question."
        )


class ClaimPart(BaseModel):
    """One of the three conditions in the headline claim."""

    model_config = _FROZEN

    name: str = Field(min_length=1)
    delta: ConfidenceInterval
    # What the claim needs: "increase", "decrease", or "no_increase".
    required: str = Field(min_length=1)
    holds: bool
    note: str = Field(min_length=1)


class JointClaim(BaseModel):
    """Section 1's thesis, evaluated.

    "Higher multi-hop accuracy **and** lower unsupported-claim rate **at fewer
    retrievals** than a prompted RAG baseline." Three conditions. The `and` is
    the whole point - two out of three is a different, weaker paper, and
    reporting it as the headline would be the most expensive kind of mistake
    this project can make.
    """

    model_config = _FROZEN

    parts: tuple[ClaimPart, ...]

    @property
    def holds(self) -> bool:
        return all(part.holds for part in self.parts)

    def summary(self) -> str:
        verdict = "HOLDS" if self.holds else "DOES NOT HOLD"
        lines = [f"Joint improvement claim: {verdict}"]
        lines.extend(
            f"  {'ok ' if part.holds else 'no '} {part.name}: {part.delta} ({part.note})"
            for part in self.parts
        )
        return "\n".join(lines)


def _part(
    name: str,
    delta: ConfidenceInterval,
    *,
    required: str,
) -> ClaimPart:
    """Decide one condition from a paired delta of candidate minus baseline.

    An interval that straddles zero means "no detected difference", and that is
    a failure for `increase` and `decrease` but a pass for `no_increase` - which
    is exactly what section 2 asks of the retrieval count ("lower than or equal
    to"). Treating "no difference" as a win for accuracy is the error this
    function exists to prevent.
    """
    if required == "increase":
        holds = delta.low > 0.0
        note = "needs the whole interval above zero"
    elif required == "decrease":
        holds = delta.high < 0.0
        note = "needs the whole interval below zero"
    elif required == "no_increase":
        holds = delta.high <= 0.0 or not delta.excludes_zero
        note = "needs no detected increase"
    else:  # pragma: no cover - guarded by the call sites below
        raise EvalError(f"unknown requirement {required!r}")
    return ClaimPart(name=name, delta=delta, required=required, holds=holds, note=note)


def joint_improvement_claim(
    *,
    f1_delta: ConfidenceInterval,
    unsupported_rate_delta: ConfidenceInterval,
    searches_delta: ConfidenceInterval,
) -> JointClaim:
    """Compose the three conditions. Every delta is candidate minus baseline.

    Deltas, not two point estimates: section 9 ends with "A 2-point gain on 500
    examples is noise", and the only way to act on that is a paired interval on
    the difference.
    """
    return JointClaim(
        parts=(
            _part("multi-hop F1", f1_delta, required="increase"),
            _part("unsupported-claim rate", unsupported_rate_delta, required="decrease"),
            _part("searches per question", searches_delta, required="no_increase"),
        )
    )
