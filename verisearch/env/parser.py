"""Parse one turn of policy output into typed turns and an action.

Owner: Dhrubo. PROJECT.md section 3 requires a parser test for every malformed
case, which is why every failure here is *data* (`ParseFailure`) and not an
exception: model output is untrusted input, and an episode that ends in a parse
failure is a row in the section 9 failure breakdown, not a crash.

Two rules the rest of the RL half depends on:

1. **The action ends the turn.** Text after the first `</search>` or `</answer>`
   is dropped into `ParsedTurn.trailing`. vLLM truncates returned *text* at a
   stop string, but the returned token ids include the whole final token, which
   can carry trailing characters - so a turn that stopped on `</search>` often
   does not *end* with it. Detect by position, never with `str.endswith`.
   Measured in the P1 spike: 4 turns stopped on the stop string and only 2
   decoded to a string ending in it (`spikes/NOTES.md`).
2. **An empty query is a penalty, not a crash.** `EMPTY_QUERY` comes back with
   `action = SEARCH` so the episode still spends budget on it and the reward
   still charges the -0.05 from section 6. Every other failure is fatal.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from verisearch.env.protocol import POLICY_TAGS, Tag
from verisearch.types import ParseFailure, Turn, TurnKind

__all__ = ["Action", "ParsedTurn", "parse_turn"]

#: Any tag-like token, well-formed or not. Deliberately permissive about
#: whitespace and the tag name so that `<tool>`, `< think >` and `</>` are all
#: *seen* by the scanner and classified, rather than being invisible text.
_TAG = re.compile(r"<\s*(/?)\s*([^<>\s/]*)\s*>")

_KIND_FOR_TAG = {
    Tag.THINK: TurnKind.THINK,
    Tag.SEARCH: TurnKind.SEARCH,
    Tag.ANSWER: TurnKind.ANSWER,
}


class Action(StrEnum):
    """What the policy asked the environment to do this turn."""

    SEARCH = "search"
    ANSWER = "answer"


class ParsedTurn(BaseModel):
    """The result of parsing one generation.

    `turns` holds only what the policy wrote, in order, with text stripped - so
    it can be appended to a `Trajectory` as is. The environment's `INFO` turn is
    added by `env/episode.py`, never here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turns: tuple[Turn, ...] = ()
    action: Action | None = None
    query: str | None = None
    answer: str | None = None
    failure: ParseFailure | None = None
    #: Text after the action's closing tag. Dropped from the trajectory, kept
    #: here because it is how a tokenizer boundary problem gets noticed.
    trailing: str = ""

    @property
    def fatal(self) -> bool:
        """True when the failure should end the episode.

        `EMPTY_QUERY` is the one failure that is not fatal: section 6 prices it
        at -0.05 and the policy gets to try again with its remaining budget.
        """
        return self.failure is not None and self.failure is not ParseFailure.EMPTY_QUERY


def _failed(failure: ParseFailure, turns: tuple[Turn, ...] = ()) -> ParsedTurn:
    return ParsedTurn(turns=turns, failure=failure)


def _spans(text: str) -> Iterator[tuple[re.Match[str], str, bool]]:
    """Yield `(match, name, is_closing)` for every tag-like token."""
    for match in _TAG.finditer(text):
        yield match, match.group(2).strip().lower(), bool(match.group(1))


def parse_turn(text: str) -> ParsedTurn:
    """Parse one generation. Never raises on malformed input."""
    policy_tag_values = {tag.value for tag in POLICY_TAGS}
    turns: list[Turn] = []
    open_tag: Tag | None = None
    body_start = 0

    for match, name, is_closing in _spans(text):
        if name not in policy_tag_values:
            # Unknown tag, or `<info>`: the environment owns that tag, so a
            # policy that writes one is fabricating its own evidence. There is
            # no `ParseFailure` category for forging evidence specifically -
            # raised on #2 - so it lands here, where at least it is fatal.
            return _failed(ParseFailure.UNKNOWN_TAG, tuple(turns))

        tag = Tag(name)
        if not is_closing:
            if open_tag is not None:
                return _failed(ParseFailure.NESTED_TAG, tuple(turns))
            open_tag = tag
            body_start = match.end()
            continue

        if open_tag is None:
            # A closing tag with nothing open: `foo</think>`. The opening tag is
            # missing rather than the closing one, but the shape of the bug is
            # the same - unbalanced tags - and section 3 names this category.
            return _failed(ParseFailure.UNCLOSED_TAG, tuple(turns))
        if tag is not open_tag:
            return _failed(ParseFailure.NESTED_TAG, tuple(turns))

        body = text[body_start : match.start()].strip()
        open_tag = None

        if tag is Tag.THINK:
            turns.append(Turn(kind=TurnKind.THINK, text=body))
            continue

        turns.append(Turn(kind=_KIND_FOR_TAG[tag], text=body))
        trailing = text[match.end() :]
        if tag is Tag.SEARCH:
            return ParsedTurn(
                turns=tuple(turns),
                action=Action.SEARCH,
                query=body,
                failure=ParseFailure.EMPTY_QUERY if not body else None,
                trailing=trailing,
            )
        if not body:
            # `<answer></answer>`: the tag is there, the answer is not. Section 6
            # prices "no answer by max turns" at -0.5 and this is that, declared
            # early rather than after burning the turn budget.
            return _failed(ParseFailure.NO_ANSWER, tuple(turns))
        return ParsedTurn(
            turns=tuple(turns),
            action=Action.ANSWER,
            answer=body,
            trailing=trailing,
        )

    if open_tag is not None:
        return _failed(ParseFailure.UNCLOSED_TAG, tuple(turns))
    # Thought, or prose, but asked nothing and answered nothing. Not an error in
    # the parser's sense - the text is well-formed - but the episode has no move
    # to make, so it is the same dead end as a missing answer.
    return _failed(ParseFailure.NO_ANSWER, tuple(turns))
