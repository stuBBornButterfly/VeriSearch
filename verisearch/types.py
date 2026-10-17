"""Shared contract between the retrieval/eval half and the RL half.

**STATUS: PROPOSAL. NOT AGREED. DO NOT MERGE.**

PROJECT.md section 3 says these interfaces are agreed in P1 and then treated as
stable, and KICKOFF.md allots 20 minutes of the first session to agreeing them
together. This file exists so that meeting starts from something concrete
instead of a blank page. Every `DECIDE:` block below is a question for that
meeting, not a decision already taken.

Ownership once agreed: shared. Changes after P1 exit need a PR both owners
approve.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Closed sets. PROJECT.md section 3: "Enums for closed sets."
# ---------------------------------------------------------------------------


class TurnKind(StrEnum):
    """What produced a turn, and therefore whether it carries loss.

    THINK, SEARCH and ANSWER are emitted by the policy. INFO is injected by the
    environment after a retrieval call - the policy did not write those tokens.
    That distinction is the whole of loss masking, so it lives in the type
    system rather than in a comment inside the rollout loop.
    """

    THINK = "think"
    SEARCH = "search"
    INFO = "info"
    ANSWER = "answer"


class Termination(StrEnum):
    """Why an episode stopped. Drives the format reward and the no-answer penalty."""

    ANSWERED = "answered"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    MAX_TURNS = "max_turns"
    PARSE_ERROR = "parse_error"


class ParseFailure(StrEnum):
    """Malformed-output categories. PROJECT.md section 3 requires a parser test for each."""

    UNCLOSED_TAG = "unclosed_tag"
    NESTED_TAG = "nested_tag"
    UNKNOWN_TAG = "unknown_tag"
    EMPTY_QUERY = "empty_query"
    NO_ANSWER = "no_answer"


class Entailment(StrEnum):
    """NLI label for one atomic claim against the retrieved passages."""

    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTED = "contradicted"


# ---------------------------------------------------------------------------
# Data crossing the boundary. PROJECT.md section 3: Pydantic v2 or frozen
# dataclass, never a raw dict. Immutable by default.
# ---------------------------------------------------------------------------

_FROZEN = ConfigDict(frozen=True, extra="forbid")

_QUERY_PUNCTUATION = re.compile(r"[^\w\s]")
_QUERY_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize_query(text: str) -> str:
    """Canonical form for deciding that two searches are the same search.

    Deliberately aggressive, for the same reason `data.normalize_question` is:
    a false positive costs one unfair -0.1, while a false negative means the
    duplicate penalty can be dodged with a space and section 6's retrieval-cost
    mechanism leaks. Accents folded, case dropped, punctuation and articles
    removed, whitespace collapsed.

    How aggressive this should be is reward design and therefore Dhrubo's call -
    strip-and-casefold is the floor he asked for and this is the ceiling. It is
    one function to change.

    Duplicated from `data.datasets.normalize_question` on purpose: `types.py` is
    the base of the dependency graph and imports from no module in the package.
    Both should move to a shared `verisearch/text.py` once one exists, along
    with `eval.metrics.normalize_title`.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    depunctuated = _QUERY_PUNCTUATION.sub(" ", folded.lower())
    return " ".join(_QUERY_ARTICLES.sub(" ", depunctuated).split())


class Passage(BaseModel):
    """One retrieved chunk. Produced by Ayesha's retriever, consumed by Dhrubo's env."""

    model_config = _FROZEN

    doc_id: str
    title: str
    text: str
    score: float | None = None
    # DECIDE: keep `score`? The reward never reads it; eval needs it for
    # recall@k, and a fake retriever has nothing meaningful to put there - hence
    # `| None`. The alternative is dropping it here and returning scores
    # alongside, which keeps Passage honest and complicates the signature.


class Turn(BaseModel):
    """One segment of a trajectory.

    DECIDE - the most important question in this file. Does a Turn carry its
    token ids?

    If yes, loss masking becomes a pure function of the trajectory, and
    `tests/test_masking.py` - mandatory and blocking per PROJECT.md section 3 -
    can assert on a plain data structure with no model and no GPU. That is the
    highest-value test in the project and it should be cheap to write.

    If no, masking lives inside the rollout loop and testing it needs a
    tokenizer and a forward pass. The bug it guards against is invisible in
    training curves, so an expensive test is a test that rots.

    Cost of yes: memory and JSONL size - eight rollouts x four turns x a few
    hundred tokens, held for a whole group. Mitigation is the empty-tuple
    default below: populate during training, leave empty when trajectories are
    built for reading or eval.
    """

    model_config = _FROZEN

    kind: TurnKind
    text: str
    token_ids: tuple[int, ...] = ()
    # Passages attach to the INFO turn that injected them, not to the SEARCH
    # turn that asked for them. Faithfulness scores an answer against the
    # passages that were actually in context when it was written.
    passages: tuple[Passage, ...] = ()

    @property
    def generated(self) -> bool:
        """True when the policy wrote these tokens, i.e. they carry loss."""
        return self.kind is not TurnKind.INFO


class Claim(BaseModel):
    """One atomic claim extracted from an answer, with its NLI verdict."""

    model_config = _FROZEN

    text: str
    label: Entailment
    # Which retrieved passages support it. Empty for NEUTRAL and CONTRADICTED.
    supporting_doc_ids: tuple[str, ...] = ()


class FaithfulnessResult(BaseModel):
    """Output of the offline NLI scorer. Ayesha's, consumed by Dhrubo's reward."""

    model_config = _FROZEN

    claims: tuple[Claim, ...]
    score: float = Field(ge=0.0, le=1.0)
    # DECIDE: what is `score` when `claims` is empty?
    #   0.0  - punishes an answer with no checkable claims. Also punishes an
    #          honest "the passages do not say", which may be behaviour we want.
    #   1.0  - vacuously faithful. Invites a degenerate "I don't know" policy.
    #   None - forces every caller to branch. Honest, and annoying.
    # A reward-design question, not a typing one, so it belongs to the session.
    # Whatever we pick, `tests/test_rewards.py` gets a case for it.

    @property
    def unsupported_rate(self) -> float:
        """Headline metric from PROJECT.md section 9. 0.0 when there are no claims."""
        if not self.claims:
            return 0.0
        unsupported = sum(1 for c in self.claims if c.label is not Entailment.ENTAILED)
        return unsupported / len(self.claims)


class Trajectory(BaseModel):
    """One complete episode. The unit everything downstream reads.

    DECIDE: flat `turns` tuple, as proposed here, or a nested structure of
    (think, search, info) steps? Flat serializes cleanly to JSONL and is easy to
    read by eye - and PROJECT.md section 15 requires reading raw trajectories
    after every significant change. Nested makes "how many searches" structural
    rather than a count. Proposing flat, with counts as computed properties.
    """

    model_config = _FROZEN

    question_id: str
    question: str
    # Plural: HotpotQA has one gold answer, but EM and F1 need alias tolerance
    # and the OOD sets (Musique, Bamboogle) ship several acceptable forms.
    gold_answers: tuple[str, ...]
    turns: tuple[Turn, ...]
    answer: str | None
    termination: Termination
    # Which parse failure ended the episode, when one did. `Termination` records
    # *that* parsing failed and `ParseFailure` had no field until now, so the
    # category existed in the type system and reached no reader: section 9 wants
    # a failure breakdown and `eval.metrics.classify_failure` cannot recover the
    # reason from anything else stored here. None for every episode that parsed.
    parse_failure: ParseFailure | None = None

    @model_validator(mode="after")
    def _a_parse_error_says_which(self) -> Self:
        """PARSE_ERROR without a reason is a row nobody can act on.

        Beyond what the review asked for - the finding was the field alone. The
        validator is what stops the field being optional in practice, which is
        how `ParseFailure` became unreachable in the first place. Drop it if the
        episode loop has a case where the category genuinely is not known.
        """
        if self.termination is Termination.PARSE_ERROR and self.parse_failure is None:
            raise ValueError(
                "termination=parse_error requires parse_failure to name the category; "
                "section 3 requires a parser test per category and section 9 a failure "
                "breakdown, and neither can be built from 'it failed'"
            )
        return self

    @property
    def search_queries(self) -> tuple[str, ...]:
        return tuple(t.text for t in self.turns if t.kind is TurnKind.SEARCH)

    @property
    def n_searches(self) -> int:
        """Computed, not stored, so it cannot drift from `turns`."""
        return len(self.search_queries)

    @property
    def n_duplicate_queries(self) -> int:
        """Feeds the -0.1 duplicate-query penalty in PROJECT.md section 6.

        Counted on a normalised form. Comparing raw strings makes the penalty
        avoidable with a trailing space, which is a reward hack the policy can
        find by accident and which would show up as a duplicate rate that falls
        during training while the behaviour does not change.
        """
        queries = self.search_queries
        return len(queries) - len({_normalize_query(q) for q in queries})

    @property
    def retrieved_passages(self) -> tuple[Passage, ...]:
        """Every passage the policy actually saw. Faithfulness scores against this."""
        return tuple(p for t in self.turns for p in t.passages)


class RewardBreakdown(BaseModel):
    """Per-component reward values plus the composed total.

    DECIDE: should `RewardComponent.__call__` return a bare float, as PROJECT.md
    section 3 writes it, or should composition produce this?

    For: PROJECT.md section 6 names four reward hacks and makes documenting them
    a deliverable. You cannot see "faithfulness went flat while correctness
    rose" in a scalar. Per-component values have to reach W&B and the trajectory
    dump, which means surviving composition.

    Against: it couples every component to a container type and dilutes the
    pure-function rule in section 3.

    Proposing: components stay `-> float` per the existing contract, and
    `rewards/compose.py` builds this. Nothing in the protocol changes.
    """

    model_config = _FROZEN

    # `Mapping`, not `dict`. `frozen=True` blocks attribute assignment and not
    # `breakdown.components["correctness"] = 9.0`, so a dict here is a mutable
    # hole in a model section 3 calls immutable - and this is the object whose
    # per-component values reach W&B and the trajectory dump.
    components: Mapping[str, float]
    penalties: Mapping[str, float] = Field(default_factory=dict)
    total: float


class ScoredTrajectory(BaseModel):
    """A trajectory plus its reward. What eval reads and what gets dumped to JSONL."""

    model_config = _FROZEN

    trajectory: Trajectory
    reward: RewardBreakdown
    faithfulness: FaithfulnessResult | None = None


# ---------------------------------------------------------------------------
# Protocols. Signatures from PROJECT.md section 3, verbatim.
# Dependency inversion: training depends on these, never on FAISS.
# ---------------------------------------------------------------------------


class Retriever(Protocol):
    """Ayesha implements (FAISS + HTTP client). Dhrubo codes against this from day one."""

    def search(self, query: str, k: int) -> list[Passage]: ...

    # DECIDE - the throughput question. GRPO generates a group of 8 rollouts at
    # once, so 8 searches arrive together, and the real retriever is an HTTP
    # service. A sync single-query protocol serializes all 8 behind one socket.
    #
    #   a) leave it sync and single. Simplest, and rollout throughput may not be
    #      retrieval-bound at group 8. Measure before deciding.
    #   b) add `search_batch(queries: list[str], k: int) -> list[list[Passage]]`
    #      as a second method. Batches the hot path, keeps fakes trivial.
    #   c) go async. Correct, and it infects the whole training loop including
    #      the vLLM calls.
    #
    # Proposing (a) now, (b) at P3 if profiling says retrieval is the
    # bottleneck, (c) only with evidence. Dhrubo owns the measurement.


class FaithfulnessScorer(Protocol):
    """Ayesha implements (NLI, offline). Never called inside the training hot loop."""

    # `Sequence`, not `list`: `Trajectory.retrieved_passages` is a tuple, so a
    # `list` parameter means every caller writes `list(...)` to get past mypy -
    # a cast that exists only to satisfy a signature. Returns stay `list`, which
    # is the concrete type an implementation actually builds.
    def score(self, answer: str, passages: Sequence[Passage]) -> FaithfulnessResult: ...


class RewardComponent(Protocol):
    """Dhrubo implements, one per term in PROJECT.md section 6. Pure: no I/O, no globals."""

    name: str

    def __call__(self, traj: Trajectory) -> float: ...


# DECIDE: add `@runtime_checkable` to these three? It lets `tests/conftest.py`
# assert the fakes satisfy the protocol via `isinstance`, which catches a fake
# drifting out of sync with the real implementation - exactly the failure mode
# that makes fakes dangerous. mypy already checks this statically at every call
# site, so it only buys a clearer test failure message.

# DECIDE: where does the GRPO group type live? Advantage is
# `(r_i - mean(r)) / std(r)` over a group of 8, so something must hold the
# group. If only `train/grpo.py` ever sees it, it belongs in `train/`. If eval
# reads groups out of a dump to inspect within-group variance, it crosses the
# boundary and belongs here. Leaning: Dhrubo's, in `train/`, until eval needs it.

# DECIDE: embeddings. PROJECT.md section 3 mandates
# `numpy.typing.NDArray[np.float32]` over bare `np.ndarray`. Nothing here needs
# an array yet - the embedder is internal to `retrieval/`. Add an `Embedding`
# alias only once a second module starts passing vectors around.
