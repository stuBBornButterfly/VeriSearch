"""Fakes for the shared contract in `verisearch.types`.

PROJECT.md section 0, risk table: *"One person blocks the other -> Protocols +
fakes from P1. Neither side ever waits on the other's implementation."* These
are those fakes. The RL half codes its episode loop and its reward components
against them from day one, with no FAISS, no HTTP service, no model, no GPU.

**Why these live in the package.** PROJECT.md section 4 puts the fakes in
`tests/conftest.py`. They are exposed there, as fixtures, but they are defined
here because `tests/` is not in mypy's `files` list - so a fake defined only in
`conftest.py` is type-checked by nothing. A fake that has silently drifted out
of sync with the protocol is worse than no fake at all: it makes the other
half's tests pass green against an interface that does not exist, and the
mismatch surfaces at integration, which in this project means P3. Defining them
inside `verisearch/` puts them under `mypy --strict` and lets them be imported
from a script or a notebook without pytest.

**Design rule for everything in this module: a fake must be able to fail.** A
retriever that always returns the gold passage makes a broken episode loop look
correct and makes a faithfulness reward look saturated. So `DEFAULT_CORPUS`
carries distractors, retrieval returns `k` results whether or not they are any
good - which is what a dense index actually does - and every failure mode the
RL half needs to exercise is reachable: no hits, fewer than `k` hits, repeated
queries, an empty query, an unanswered episode.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from verisearch.types import (
    Claim,
    Entailment,
    FaithfulnessResult,
    FaithfulnessScorer,
    Passage,
    Retriever,
    Termination,
    Trajectory,
    Turn,
    TurnKind,
)

__all__ = [
    "DEFAULT_CORPUS",
    "FakeFaithfulnessScorer",
    "FakeRetriever",
    "conformance_witnesses",
    "make_trajectory",
]

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A claim that negates something its passages assert is contradicted, not merely
# unsupported. Lexical overlap alone cannot tell those apart, so the fake uses a
# marker list. Crude on purpose - see the class docstring.
_NEGATIONS = frozenset({"not", "never", "no", "nor", "cannot", "didnt", "wasnt", "isnt"})


def _tokens(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens. Order and duplicates do not matter here."""
    return frozenset(_WORD.findall(text.lower()))


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split.

    Deliberately not `verisearch.faithfulness.claims.extract_claims`: a fake
    that shares code with the implementation it stands in for stops being an
    independent check and becomes a mirror. If the real splitter has a bug, a
    fake that imports it has the same bug and no test can see it.
    """
    return [chunk.strip() for chunk in _SENTENCE_SPLIT.split(text.strip()) if chunk.strip()]


# Three passages that genuinely answer a multi-hop question about Ada Lovelace,
# and two distractors that share vocabulary with the question while supporting
# nothing. The distractors are the point: with a corpus of only gold passages,
# `unsupported_rate` is structurally incapable of being non-zero and the
# faithfulness reward cannot be shown to work at all.
DEFAULT_CORPUS: tuple[Passage, ...] = (
    Passage(
        doc_id="d1",
        title="Ada Lovelace",
        text=(
            "Ada Lovelace was an English mathematician who wrote the first published "
            "notes on the Analytical Engine."
        ),
    ),
    Passage(
        doc_id="d2",
        title="Analytical Engine",
        text=(
            "The Analytical Engine was a mechanical general-purpose computer designed "
            "by Charles Babbage."
        ),
    ),
    Passage(
        doc_id="d3",
        title="Charles Babbage",
        text=(
            "Charles Babbage was an English polymath who originated the concept of a "
            "programmable computer."
        ),
    ),
    Passage(
        doc_id="d4",
        title="Lovelace, Texas",
        text="Lovelace is an unincorporated community in Hunt County, Texas, United States.",
    ),
    Passage(
        doc_id="d5",
        title="Engine",
        text="An engine is a machine designed to convert one form of energy into motion.",
    ),
)


class FakeRetriever:
    """Deterministic lexical retriever over an in-memory corpus.

    Satisfies the `Retriever` protocol. Ranking is the fraction of query tokens
    found in `title + text`, ties broken by `doc_id` so the order is stable
    across runs, processes and machines. A fake whose ordering shifts with the
    hash seed turns a real regression into a coin flip, which is why the tie
    break is explicit rather than left to sort stability over a set.

    Two behaviours are copied from the real thing on purpose:

    - **It returns `k` results regardless of quality.** A dense index always
      hands back its nearest neighbours, however far away they are. An off-topic
      query here returns the least-bad distractors, not an empty list, so the
      episode loop and the faithfulness scorer are both exercised on irrelevant
      context by default rather than only when a test remembers to arrange it.
    - **An empty query returns nothing** rather than raising, because the parser
      is what classifies `ParseFailure.EMPTY_QUERY`. The retriever is not the
      right place to enforce that, and a fake that raised would mask a missing
      check in the parser.

    `blind_to` forces the genuinely-zero-results path without emptying the
    corpus. Every call is recorded on `calls`, which is how the search budget
    and the duplicate-query penalty get tested with no service running.
    """

    def __init__(
        self,
        corpus: Sequence[Passage] = DEFAULT_CORPUS,
        *,
        blind_to: Iterable[str] = (),
    ) -> None:
        self._corpus = tuple(corpus)
        self._blind_to = frozenset(query.strip().lower() for query in blind_to)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[Passage]:
        """Top-`k` passages by token overlap. Records the call."""
        if k <= 0:
            # Not a realistic retriever response, a caller bug. Fail loudly in
            # the fake so it is found in a unit test rather than against a live
            # service at group 8.
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)
        self.calls.append((query, k))
        normalised = query.strip().lower()
        if not normalised or normalised in self._blind_to:
            return []
        query_tokens = _tokens(query)
        scored = [(self._overlap(query_tokens, p), p.doc_id, p) for p in self._corpus]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [passage.model_copy(update={"score": score}) for score, _, passage in scored[:k]]

    @staticmethod
    def _overlap(query_tokens: frozenset[str], passage: Passage) -> float:
        if not query_tokens:
            return 0.0
        haystack = _tokens(f"{passage.title} {passage.text}")
        return len(query_tokens & haystack) / len(query_tokens)

    @property
    def queries(self) -> tuple[str, ...]:
        """Every query seen, in order. Duplicates preserved."""
        return tuple(query for query, _ in self.calls)

    @property
    def n_searches(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        """Forget recorded calls. The corpus is unchanged."""
        self.calls.clear()


class FakeFaithfulnessScorer:
    """Deterministic lexical stand-in for the NLI scorer.

    Satisfies the `FaithfulnessScorer` protocol. A claim is ENTAILED when at
    least `support_threshold` of its tokens appear in one passage, CONTRADICTED
    when that holds but the claim also carries a negation marker, and NEUTRAL
    otherwise. This is not an NLI model and is not trying to be - it is a
    deterministic oracle with a known decision rule, which is what a reward test
    needs and what a real model cannot provide.

    `empty_claims_score` exists because `FaithfulnessResult.score` with zero
    claims is still an open question in `types.py` - 0.0 punishes an honest "the
    passages do not say", 1.0 invites a degenerate "I don't know" policy. The
    fake takes it as a constructor argument instead of picking, so the contract
    session can run all three candidate answers through the same reward code
    rather than arguing about them in the abstract.
    """

    def __init__(self, *, empty_claims_score: float = 0.0, support_threshold: float = 0.6) -> None:
        if not 0.0 <= empty_claims_score <= 1.0:
            msg = f"empty_claims_score must be in [0, 1], got {empty_claims_score}"
            raise ValueError(msg)
        if not 0.0 < support_threshold <= 1.0:
            msg = f"support_threshold must be in (0, 1], got {support_threshold}"
            raise ValueError(msg)
        self._empty_claims_score = empty_claims_score
        self._threshold = support_threshold

    def score(self, answer: str, passages: list[Passage]) -> FaithfulnessResult:
        """Label every sentence of `answer` against `passages`."""
        per_passage = {p.doc_id: _tokens(f"{p.title} {p.text}") for p in passages}
        claims: list[Claim] = []
        for sentence in _split_sentences(answer):
            claim_tokens = _tokens(sentence)
            if not claim_tokens:
                continue
            coverage = {
                doc_id: len(claim_tokens & tokens) / len(claim_tokens)
                for doc_id, tokens in per_passage.items()
            }
            best = max(coverage.values(), default=0.0)
            supporting: tuple[str, ...] = ()
            if best < self._threshold:
                label = Entailment.NEUTRAL
            elif claim_tokens & _NEGATIONS:
                label = Entailment.CONTRADICTED
            else:
                label = Entailment.ENTAILED
                # Sorted so the field is reproducible, not dict-ordered.
                supporting = tuple(
                    sorted(doc for doc, cov in coverage.items() if cov >= self._threshold)
                )
            claims.append(Claim(text=sentence, label=label, supporting_doc_ids=supporting))

        if not claims:
            return FaithfulnessResult(claims=(), score=self._empty_claims_score)
        entailed = sum(1 for claim in claims if claim.label is Entailment.ENTAILED)
        return FaithfulnessResult(claims=tuple(claims), score=entailed / len(claims))


def make_trajectory(
    *,
    question: str = "Who wrote the first notes on the Analytical Engine?",
    question_id: str = "q1",
    gold_answers: Sequence[str] = ("Ada Lovelace",),
    searches: Sequence[str] = (),
    passages_per_search: Sequence[Sequence[Passage]] | None = None,
    answer: str | None = "Ada Lovelace",
    termination: Termination | None = None,
    think: str = "I should look this up rather than guess.",
    tokens_per_turn: int = 0,
) -> Trajectory:
    """Build a well-formed `Trajectory` from a compact spec.

    Hand-constructing turns is twenty lines of boilerplate per reward test, and
    boilerplate is where a test that does not mean what it says comes from. This
    emits the canonical shape from PROJECT.md section 3 - `think`, then a
    `search`/`info` pair per query, then `answer` when there is one.

    `passages_per_search` defaults to no passages for any search, which is the
    realistic no-hits case rather than an error. Pass explicit passages - from
    `FakeRetriever.search` - when the test is about what the policy saw.

    `tokens_per_turn > 0` fills `Turn.token_ids` with a distinct deterministic
    range per turn. That is what makes `tests/test_masking.py` writable against
    a plain data structure today: the INFO turns get token ids that must carry
    zero loss weight, and asserting it needs no tokenizer and no GPU. It is the
    concrete argument for answering the `Turn.token_ids` question in `types.py`
    with yes.
    """
    if passages_per_search is not None and len(passages_per_search) != len(searches):
        msg = (
            f"passages_per_search has {len(passages_per_search)} entries "
            f"for {len(searches)} searches"
        )
        raise ValueError(msg)
    if tokens_per_turn < 0:
        msg = f"tokens_per_turn must be non-negative, got {tokens_per_turn}"
        raise ValueError(msg)

    turns: list[Turn] = []

    def add(kind: TurnKind, text: str, passages: Sequence[Passage] = ()) -> None:
        # Ids are offset by turn index so no two turns share one. That is what
        # lets a masking test tell "the mask covers the INFO span" apart from
        # "the mask covers something the right size".
        base = len(turns) * 1000
        ids = tuple(range(base, base + tokens_per_turn))
        turns.append(Turn(kind=kind, text=text, token_ids=ids, passages=tuple(passages)))

    add(TurnKind.THINK, think)
    for i, query in enumerate(searches):
        add(TurnKind.SEARCH, query)
        retrieved = passages_per_search[i] if passages_per_search is not None else ()
        add(TurnKind.INFO, "\n".join(f"{p.title}: {p.text}" for p in retrieved), retrieved)
    if answer is not None:
        add(TurnKind.ANSWER, answer)

    if termination is None:
        termination = Termination.ANSWERED if answer is not None else Termination.MAX_TURNS

    return Trajectory(
        question_id=question_id,
        question=question,
        gold_answers=tuple(gold_answers),
        turns=tuple(turns),
        answer=answer,
        termination=termination,
    )


def conformance_witnesses() -> tuple[Retriever, FaithfulnessScorer]:
    """Static proof that the fakes still satisfy the protocols.

    The annotated return type is the whole mechanism: if `FakeRetriever.search`
    or `FakeFaithfulnessScorer.score` drifts from the signature in `types.py`,
    `mypy --strict` rejects this function. That also settles the
    `@runtime_checkable` question in `types.py` - static checking already
    catches the drift that makes fakes dangerous, so the decorator would only
    improve a failure message, and only for code mypy has already cleared.
    """
    return FakeRetriever(), FakeFaithfulnessScorer()
