"""The nightly regression gate. PROJECT.md section 10, `eval-regression`.

Section 10: "Fixed 100-example eval, asserts F1 has not dropped >2 pts." This
module decides what "dropped" means, on a fixed set, against a stored reference
run.

**Dropped means the interval says so.** `metrics.paired_bootstrap_delta` says
it exists for this stage - "F1 has not dropped more than 2 points" is only
answerable with an interval on the difference. So a nightly run fails only when
both are true:

* the point estimate fell by more than the threshold, and
* the paired interval lies entirely below zero.

A drop past the threshold whose interval still includes zero is reported as
`NOISE` and passes, loudly. Failing on the point estimate alone would make the
gate flake on exactly the noise section 9 warns about.

**What this gate can and cannot see.** At n=100 a paired F1 interval is
typically several points wide on each side, so a real 2-point drift is usually
inside it and passes as `NOISE` - by the project's own rule, "a 2-point gain on
500 examples is noise", and this is a fifth of that. The gate catches breakage:
a masking bug, a tokenizer or prompt change, a retriever that went dead - the
failures that move F1 by tens of points. Every verdict prints the interval's
half-width, so the size of drop it could have resolved is on the page rather
than assumed. To see 2 points, raise n.

**The reference is a normal eval run,** promoted on purpose: the report and
rows of a known-good checkpoint on the regression set, from `03_eval.sh`. The
candidate's context is rebuilt every night from the index and the set
themselves, never copied from the reference, so `check_comparable` compares two
independent descriptions rather than one description with itself.

Owner: Ayesha.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from verisearch.data.datasets import Split
from verisearch.data.splits import subsample
from verisearch.eval.baselines import EvalError, examples_fingerprint
from verisearch.eval.harness import compare_runs
from verisearch.eval.metrics import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, ConfidenceInterval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from verisearch.data.datasets import QAExample
    from verisearch.eval.harness import EvalRun

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class RegressionStatus(StrEnum):
    """The three things a nightly comparison can say."""

    # F1 did not fall by more than the threshold.
    HELD = "held"
    # It fell by more than the threshold, but the interval includes zero.
    NOISE = "noise"
    # It fell by more than the threshold and the interval excludes zero.
    REGRESSED = "regressed"


class RegressionVerdict(BaseModel):
    """One nightly comparison, and why it came out the way it did."""

    model_config = _FROZEN

    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    n: int = Field(ge=1)
    # A fraction of F1, 0.02 for "2 points". The CLI takes points.
    max_drop: float = Field(ge=0.0, lt=1.0)
    f1_delta: ConfidenceInterval
    status: RegressionStatus

    @property
    def resolvable_drop(self) -> float:
        """Half the interval's width: roughly the smallest drop this run could
        have told apart from noise."""
        return self.f1_delta.width / 2.0

    def summary(self) -> str:
        delta = self.f1_delta
        lines = [
            f"{self.candidate} vs reference {self.reference}, paired on n={self.n}",
            f"  F1 delta            {delta}",
            f"  threshold           -{self.max_drop * 100:.1f} pts",
            f"  resolvable          about +/-{self.resolvable_drop * 100:.1f} pts at this n",
            f"  verdict             {self.status.value}",
        ]
        if self.status is RegressionStatus.NOISE:
            lines.append(
                f"  F1 fell {-delta.point * 100:.1f} pts, past the threshold, but the interval "
                f"includes zero. Not failing on it - check the rows before the next merge."
            )
        if self.max_drop < self.resolvable_drop:
            lines.append(
                f"  note: a {self.max_drop * 100:.1f}-pt drop is inside this run's noise; "
                f"only a larger one can fail the gate"
            )
        return "\n".join(lines)


def regression_set(
    examples: Sequence[QAExample],
    n: int,
    *,
    seed: int,
    split: Split = Split.DEV,
) -> tuple[QAExample, ...]:
    """The fixed n-example set, chosen deterministically from one split.

    `splits.subsample`, so the choice is by salted hash rather than file order,
    and a reordered dev file gives the same set. A dev file that *gains*
    examples can change it - a newcomer with a lower hash displaces the last
    one in - which is why the set is written once by `regression-set` and kept,
    not rebuilt each night. Refuses rather than returning fewer than n, because
    a smaller set would quietly measure a different gate.

    Returned sorted by question id, which is the order `write_jsonl` writes and
    therefore the order every run over the file sees. The fingerprint is
    order-sensitive, so a set returned in hash order would print a fingerprint
    no context built from its own file could match.
    """
    if n < 1:
        raise EvalError(f"a regression set needs at least one example, got n={n}")
    chosen = subsample(examples, split, limit=n, seed=seed)
    if len(chosen) < n:
        raise EvalError(
            f"asked for {n} {split.value} examples, found {len(chosen)}. Examples need "
            f"their split set - is this the processed {split.value} file?"
        )
    return tuple(sorted(chosen, key=lambda example: example.question_id))


def _check_rows_are_its_own(run: EvalRun, role: str) -> None:
    """The rows must be the questions the run's context describes, in order.

    `harness.load_run` checks the row count against the context. A count is not
    an identity: a results file from another set of the same size would load,
    and every delta computed against it would be about different questions.
    """
    fingerprint = examples_fingerprint(row.question_id for row in run.results)
    if fingerprint != run.report.context.examples_fingerprint:
        raise EvalError(
            f"the {role} rows are not the questions its context describes "
            f"({fingerprint} vs {run.report.context.examples_fingerprint})"
        )


def check_regression(
    reference: EvalRun,
    candidate: EvalRun,
    *,
    max_drop: float,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> RegressionVerdict:
    """Compare tonight's run with the reference. See the module docstring."""
    if not 0.0 <= max_drop < 1.0:
        raise EvalError(f"max_drop is a fraction of F1 in [0, 1), got {max_drop}")
    _check_rows_are_its_own(reference, "reference")
    _check_rows_are_its_own(candidate, "candidate")

    # Refuses incomparable contexts and misaligned rows itself.
    comparison = compare_runs(
        reference, candidate, resamples=resamples, confidence=confidence, seed=seed
    )
    delta = comparison.f1_delta
    if delta.point >= -max_drop:
        status = RegressionStatus.HELD
    elif delta.high < 0.0:
        status = RegressionStatus.REGRESSED
    else:
        status = RegressionStatus.NOISE

    return RegressionVerdict(
        reference=comparison.reference,
        candidate=comparison.candidate,
        n=comparison.n_paired,
        max_drop=max_drop,
        f1_delta=delta,
        status=status,
    )
