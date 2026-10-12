"""NLI entailment scoring.

PROJECT.md section 5: a small DeBERTa-v3 MNLI-class model, running **offline,
not in the hot loop**. Section 6: faithfulness is the fraction of claims
entailed by the passages that were actually retrieved - verifiable, not an LLM
judge.

Three things in here are easy to get wrong and silent when wrong. All three are
pure functions, so all three are tested:

1. **Label order.** An MNLI head emits three logits. Which index means
   entailment is a property of the checkpoint, not of the task, and different
   checkpoints disagree. Hardcoding `[contradiction, neutral, entailment]`
   against a model that uses the reverse order inverts the metric completely -
   hallucinated claims score as supported. `resolve_label_order` reads the
   model's own `id2label` and maps by name, failing loud on anything it does
   not recognise. Never hardcode this.

2. **Direction.** NLI is directional: premise entails hypothesis. Here the
   passage is the premise and the claim is the hypothesis - "does the evidence
   support this claim". Reversed, it asks whether the claim supports the
   evidence, which is a different question that produces plausible numbers.

3. **One premise or many.** Each claim is checked against each passage
   separately and the best result wins, rather than concatenating every passage
   into one premise. Concatenation overruns the model's max length on realistic
   top-k, and a claim supported by passage 3 gets diluted by passages 1, 2 and
   4.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import transformers

# PROJECT.md section 5 names the class of model, not the checkpoint. base.yaml
# proposes MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli. Unpinned until it has
# actually been run, per section 12.
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

# Names seen across MNLI checkpoints, normalised to the three labels. Matched
# case-insensitively after stripping. Anything else is refused rather than
# guessed at.
_LABEL_ALIASES: dict[str, str] = {
    "entailment": "entailment",
    "entail": "entailment",
    "entailed": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "not_entailment": "contradiction",
}


class NliError(Exception):
    """Raised when entailment scoring cannot proceed. Section 3: fail loud."""


class NliLabel(StrEnum):
    """The three classes an MNLI head predicts."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class NliSpec(BaseModel):
    """Settings that change the verdicts.

    Fingerprinted for the same reason as `EmbedderSpec`: a faithfulness number
    is only comparable with another produced by the same scorer at the same
    threshold, and section 15 requires every reported number to be traceable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(default=DEFAULT_NLI_MODEL, min_length=1)
    # base.yaml: 0.5. Above this entailment probability a claim counts as
    # supported. Not a tuned value - measure it against hand-labelled
    # trajectories at P2 and record the result.
    entailment_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)

    @property
    def fingerprint(self) -> str:
        """Hash over what changes the verdicts. `batch_size` does not."""
        payload = {
            "model_name": self.model_name,
            "entailment_threshold": self.entailment_threshold,
            "max_length": self.max_length,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def resolve_label_order(id2label: Mapping[int, str]) -> tuple[NliLabel, ...]:
    """Map a model's own label config onto `NliLabel`, ordered by logit index.

    This is the guard that matters most in the module. Reading the checkpoint's
    `id2label` and matching by name is the only correct way to know which logit
    is entailment. A hardcoded order that happens to be wrong does not fail - it
    reports hallucinated claims as supported, and the training signal silently
    rewards the thing the project exists to reduce.

    `LABEL_0`-style configs are refused: they carry no information about which
    class is which, and guessing would reintroduce exactly the bug this
    prevents.
    """
    if not id2label:
        raise NliError("model config has an empty id2label; cannot determine label order")

    expected_indices = set(range(len(id2label)))
    if set(id2label) != expected_indices:
        raise NliError(f"id2label keys must be 0..{len(id2label) - 1}, got {sorted(id2label)}")

    resolved: list[NliLabel] = []
    for index in sorted(id2label):
        raw = id2label[index]
        canonical = _LABEL_ALIASES.get(str(raw).strip().casefold())
        if canonical is None:
            raise NliError(
                f"unrecognised NLI label {raw!r} at index {index}. Known: "
                f"{', '.join(sorted(set(_LABEL_ALIASES)))}. A model whose labels are "
                f"LABEL_0/1/2 carries no information about which logit is entailment - "
                f"find a checkpoint with a real label map rather than guessing."
            )
        resolved.append(NliLabel(canonical))

    if len(set(resolved)) != len(resolved):
        raise NliError(f"id2label maps two indices to the same label: {resolved}")
    if set(resolved) != set(NliLabel):
        missing = sorted(set(NliLabel) - set(resolved))
        raise NliError(f"id2label is missing {[m.value for m in missing]}")
    return tuple(resolved)


def build_pairs(claim: str, passage_texts: Sequence[str]) -> list[tuple[str, str]]:
    """One (premise, hypothesis) pair per passage.

    Passage first. NLI is directional and the question is "does this evidence
    support this claim", never the reverse. The tuple order is the whole
    contract between this function and the tokenizer call, so it is built here
    once rather than at each call site.
    """
    if not claim.strip():
        raise NliError("cannot check an empty claim")
    if not passage_texts:
        raise NliError(
            "cannot check a claim against zero passages; an answer with no retrieved "
            "evidence is a distinct case the caller must handle"
        )
    return [(passage, claim) for passage in passage_texts]


class ClaimVerdict(BaseModel):
    """The outcome for one claim, and which passage produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str
    label: NliLabel
    entailment_probability: float = Field(ge=0.0, le=1.0)
    best_passage_index: int = Field(ge=0)

    @property
    def entailed(self) -> bool:
        return self.label is NliLabel.ENTAILMENT


def best_verdict(
    claim: str,
    probabilities: Sequence[Sequence[float]],
    label_order: Sequence[NliLabel],
    *,
    threshold: float,
) -> ClaimVerdict:
    """Pick the passage that best supports a claim.

    `probabilities` is one row of class probabilities per passage, in the
    model's own logit order - which is why `label_order` has to come from
    `resolve_label_order` rather than be assumed.

    Best means highest entailment probability. A claim supported by any single
    retrieved passage is supported; it does not need agreement across passages,
    and requiring that would punish a correct answer for retrieving a diverse
    top-k.

    The label is ENTAILMENT only if that probability clears the threshold.
    Below it, the label is the more probable of NEUTRAL and CONTRADICTION -
    never ENTAILMENT, even when entailment is the raw argmax. An earlier version
    fell back to the unrestricted argmax, which meant a 0.49 entailment
    probability against a 0.5 threshold still came back entailed: the threshold
    was decorative. Section 9 counts contradiction and mere absence of support
    both as unsupported, but they mean different things when reading
    trajectories by hand, so the distinction is kept.
    """
    if not probabilities:
        raise NliError("no probabilities to choose from")
    if len(label_order) != len(NliLabel):
        raise NliError(f"label_order must cover all {len(NliLabel)} labels, got {label_order}")

    entail_index = list(label_order).index(NliLabel.ENTAILMENT)

    best_row = -1
    best_probability = -1.0
    for row_index, row in enumerate(probabilities):
        if len(row) != len(label_order):
            raise NliError(
                f"row {row_index} has {len(row)} probabilities, expected {len(label_order)}"
            )
        probability = row[entail_index]
        if probability > best_probability:
            best_probability = probability
            best_row = row_index

    row = probabilities[best_row]
    if best_probability >= threshold:
        label = NliLabel.ENTAILMENT
    else:
        # Argmax over the non-entailment classes only. Ties break towards the
        # lower logit index so the verdict is reproducible.
        others = [i for i, lab in enumerate(label_order) if lab is not NliLabel.ENTAILMENT]
        argmax = max(others, key=lambda i: (row[i], -i))
        label = label_order[argmax]

    return ClaimVerdict(
        claim=claim,
        label=label,
        entailment_probability=float(best_probability),
        best_passage_index=best_row,
    )


def faithfulness_score(verdicts: Sequence[ClaimVerdict]) -> float:
    """Fraction of claims entailed. PROJECT.md section 6, the `w_a` term.

    Raises on an empty input. An answer with no checkable claims is a real
    state, and what it scores is still an open question on the contract
    proposal - baking a default in here would quietly settle a reward-design
    decision that belongs to both owners.
    """
    if not verdicts:
        raise NliError(
            "faithfulness_score is undefined with no claims; see the open DECIDE on "
            "FaithfulnessResult.score in the types proposal"
        )
    return sum(1 for verdict in verdicts if verdict.entailed) / len(verdicts)


class TransformersNliScorer:
    """`transformers` backend. Needs `[retrieval]`; never runs in the PR pipeline.

    **Unverified.** Neither CI nor this machine has torch or transformers
    installed. Deliberately a thin shell over tokenize-forward-softmax: every
    decision that could be silently wrong lives in the pure functions above,
    which are tested.
    """

    def __init__(self, spec: NliSpec | None = None) -> None:
        self.spec = spec or NliSpec()
        # Annotated through the TYPE_CHECKING import so attribute and call
        # access resolves via the ignore_missing_imports override, rather
        # than us writing an explicit Any (banned by section 3).
        self._model: transformers.PreTrainedModel | None = None
        self._tokenizer: transformers.PreTrainedTokenizerBase | None = None
        self._label_order: tuple[NliLabel, ...] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise NliError(
                "TransformersNliScorer needs the retrieval extra: uv pip install -e '.[retrieval]'"
            ) from exc

        # Everything from here needs the extra, so it is what carries the
        # pragma - not the ImportError branch above, which is the path that
        # actually runs without it and is tested.
        self._tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
            self.spec.model_name
        )
        model = AutoModelForSequenceClassification.from_pretrained(  # pragma: no cover
            self.spec.model_name
        )
        model.eval()  # pragma: no cover
        self._model = model  # pragma: no cover
        # Read the order off the checkpoint. Never assume it.
        self._label_order = resolve_label_order(  # pragma: no cover
            dict(model.config.id2label)
        )

    @property
    def label_order(self) -> tuple[NliLabel, ...]:
        self._load()
        if self._label_order is None:  # pragma: no cover - defensive
            raise NliError("label order was not resolved")
        return self._label_order  # pragma: no cover - needs a loaded model

    def _probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[list[float]]:
        try:
            import torch
        except ImportError as exc:
            raise NliError(
                "TransformersNliScorer needs the retrieval extra: uv pip install -e '.[retrieval]'"
            ) from exc

        self._load()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise NliError("model was not loaded")

        rows: list[list[float]] = []
        for start in range(0, len(pairs), self.spec.batch_size):
            batch = pairs[start : start + self.spec.batch_size]
            encoded = tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation=True,
                max_length=self.spec.max_length,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits
            rows.extend(torch.softmax(logits, dim=-1).tolist())
        return rows

    def check_claims(
        self,
        claims: Sequence[str],
        passage_texts: Sequence[str],
    ) -> tuple[ClaimVerdict, ...]:
        """Score every claim against every passage. Offline, batched."""
        if not claims:
            raise NliError("no claims to check")
        order = self.label_order
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            pairs = build_pairs(claim, passage_texts)
            verdicts.append(
                best_verdict(
                    claim,
                    self._probabilities(pairs),
                    order,
                    threshold=self.spec.entailment_threshold,
                )
            )
        return tuple(verdicts)
