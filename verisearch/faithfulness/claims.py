"""Answer to checkable claims.

PROJECT.md section 6 defines faithfulness as "fraction of atomic claims entailed
by retrieved passages", and section 9 makes unsupported-claim rate the headline
hallucination metric. This module produces the denominator of both.

**What this does and does not do.** It splits an answer into sentences. True
atomic decomposition - turning "Peter Weir, born in Sydney, directed Gallipoli"
into three separately checkable facts - needs a language model. Section 6 bars
an LLM judge from the training loop, but faithfulness runs offline, so an LLM
decomposer is available later if it earns its place.

The consequence has to be stated wherever the number is reported: sentence-level
claims are **more lenient** than atomic ones. A sentence asserting two facts, one
supported and one not, can come back entailed. So the unsupported-claim rate this
produces is a **lower bound** on the true rate. A reported improvement is real; a
reported absolute is optimistic.

No model, no network, no dependencies beyond the standard library.

Owner: Ayesha.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# A pathological answer must not cost unbounded NLI calls. This runs over
# thousands of trajectories per evaluation, and an answer with 200 sentences is
# a degenerate-repetition failure, not an answer worth scoring exhaustively.
DEFAULT_MAX_CLAIMS = 32

# Shortest string treated as a claim. Below this it is a fragment - "Yes.",
# "No." - which NLI cannot evaluate in isolation and which would add noise to
# the denominator.
MIN_CLAIM_CHARS = 3

# Periods that do not end a sentence. Protected before splitting and restored
# after. Not exhaustive and not meant to be - the uppercase lookahead in
# _SENTENCE_BOUNDARY already handles the common cases, and this list only has to
# cover abbreviations that are genuinely followed by a capitalised word
# ("Dr. Weir", "St. Louis"). A missed one over-splits, costing an extra NLI call
# and a slightly stricter score; a false protection hides an unsupported claim
# inside a longer sentence. Over-splitting is the safer error, so the list stays
# short.
ABBREVIATIONS = (
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Sr",
    "Jr",
    "St",
    "Mt",
    "Inc",
    "Ltd",
    "Co",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "approx",
)

_SENTINEL = "\u0000"
# A period ends a sentence only when something that looks like a sentence
# follows it. Without the lookahead, "He worked in the U.S. and in Australia."
# splits after "U.S." and yields the fragment "and in Australia.", which NLI
# cannot evaluate and which is pure noise in the denominator of the headline
# metric. The cost is that a sentence starting with a lowercase word does not
# split; model answers do not usually do that.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s*[A-Z])")
_ALPHANUMERIC = re.compile(r"\w")
_WHITESPACE = re.compile(r"\s+")


class ClaimError(Exception):
    """Raised when claims cannot be extracted. PROJECT.md section 3: fail loud."""


def _protect(text: str) -> str:
    """Replace non-terminal periods with a sentinel so splitting ignores them."""
    protected = _DECIMAL.sub(_SENTINEL, text)
    protected = _INITIAL.sub(rf"\1{_SENTINEL}", protected)
    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(f"{abbreviation}.", f"{abbreviation}{_SENTINEL}")
    return protected


def _restore(text: str) -> str:
    return text.replace(_SENTINEL, ".")


def normalize_claim(text: str) -> str:
    """Collapse whitespace. Used for comparison, not for what gets returned."""
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, leaving abbreviations and decimals intact."""
    protected = _protect(text)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [restored for part in parts if (restored := normalize_claim(_restore(part)))]


def extract_claims(
    answer: str | None,
    *,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> tuple[str, ...]:
    """Split an answer into claims to check against retrieved passages.

    Returns an empty tuple for an answer with nothing checkable in it - no
    answer at all, whitespace, or pure punctuation. An empty tuple is a real
    and distinct state, and what faithfulness *means* in that case is still an
    open question on the contract proposal rather than something to decide here.

    Duplicates are dropped, case-insensitively. This is not tidiness: section 6
    lists "copy passages verbatim to farm faithfulness" as an expected reward
    hack, and repeating one supported sentence twenty times would otherwise
    drive the entailed fraction towards 1.0 for free. Deduplication removes that
    particular move. The length penalty in section 6 covers the rest.
    """
    if max_claims < 1:
        raise ClaimError(f"max_claims must be >= 1, got {max_claims}")
    if answer is None:
        return ()

    claims: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(answer):
        if len(sentence) < MIN_CLAIM_CHARS:
            continue
        if not _ALPHANUMERIC.search(sentence):
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(sentence)
        if len(claims) == max_claims:
            break
    return tuple(claims)


def unsupported_rate(entailed_flags: Sequence[bool]) -> float:
    """Fraction of claims not entailed. The headline metric in section 9.

    Raises on an empty input rather than returning 0.0. "No claims" and "no
    unsupported claims" are different states, and conflating them would report a
    model that answers nothing as perfectly faithful - which is the degenerate
    policy the reward design is trying to avoid rewarding.
    """
    if not entailed_flags:
        raise ClaimError(
            "unsupported_rate is undefined with no claims; the caller has to decide what "
            "an unanswerable or empty answer scores"
        )
    return sum(1 for flag in entailed_flags if not flag) / len(entailed_flags)
