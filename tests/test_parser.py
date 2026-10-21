"""Parser tests. PROJECT.md section 3: every malformed case gets a case here.

Table-driven, and the tables are organised by what the episode loop does with
the result - a well-formed action, a non-fatal penalty, or a fatal failure -
because that is the distinction the loop branches on.
"""

from __future__ import annotations

import pytest

from verisearch.env.parser import Action, parse_turn
from verisearch.env.protocol import Tag, render_info, render_prompt
from verisearch.types import ParseFailure, Passage, TurnKind

# ---------------------------------------------------------------------------
# Well-formed
# ---------------------------------------------------------------------------


def test_think_then_search() -> None:
    parsed = parse_turn("<think>Need the film first.</think>\n<search>1999 Best Picture</search>")

    assert parsed.failure is None
    assert parsed.fatal is False
    assert parsed.action is Action.SEARCH
    assert parsed.query == "1999 Best Picture"
    assert [(t.kind, t.text) for t in parsed.turns] == [
        (TurnKind.THINK, "Need the film first."),
        (TurnKind.SEARCH, "1999 Best Picture"),
    ]


def test_think_then_answer() -> None:
    parsed = parse_turn("<think>The passages say Reading.</think><answer>Reading</answer>")

    assert parsed.action is Action.ANSWER
    assert parsed.answer == "Reading"
    assert parsed.turns[-1].kind is TurnKind.ANSWER


def test_action_without_thinking_is_allowed() -> None:
    """Nothing in section 1 makes <think> mandatory. The format reward can want
    it; the parser is not the place to enforce style."""
    parsed = parse_turn("<search>Sam Mendes birthplace</search>")

    assert parsed.action is Action.SEARCH
    assert [t.kind for t in parsed.turns] == [TurnKind.SEARCH]


def test_bodies_are_stripped() -> None:
    parsed = parse_turn("<search>\n  Sam Mendes birthplace \n</search>")

    assert parsed.query == "Sam Mendes birthplace"


def test_whitespace_inside_the_tag_itself_is_tolerated() -> None:
    parsed = parse_turn("< search >Ada Lovelace</ search >")

    assert parsed.action is Action.SEARCH
    assert parsed.query == "Ada Lovelace"


def test_multiple_think_turns_are_kept_in_order() -> None:
    parsed = parse_turn("<think>one</think><think>two</think><answer>x</answer>")

    assert [t.text for t in parsed.turns] == ["one", "two", "x"]


# ---------------------------------------------------------------------------
# The action ends the turn. This is the token-boundary finding from the spike.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    [
        pytest.param("", id="clean-stop"),
        pytest.param("\n\n", id="trailing-newlines"),
        pytest.param(">", id="partial-token-overshoot"),
        pytest.param("\n<info>\nfabricated\n</info>", id="model-kept-going"),
        pytest.param("<search>second query</search>", id="second-search-ignored"),
    ],
)
def test_text_after_the_closing_tag_never_changes_the_action(suffix: str) -> None:
    """vLLM truncates the returned text at the stop string, but the returned
    token ids include the whole final token, so a turn that stopped on
    `</search>` often does not end with it. Position, not `endswith`."""
    parsed = parse_turn(f"<search>first query</search>{suffix}")

    assert parsed.action is Action.SEARCH
    assert parsed.query == "first query"
    assert parsed.failure is None
    assert parsed.trailing == suffix
    assert [t.text for t in parsed.turns] == ["first query"]


def test_trailing_text_after_an_answer_is_dropped_too() -> None:
    parsed = parse_turn("<answer>Reading</answer> I hope that helps!")

    assert parsed.answer == "Reading"
    assert parsed.trailing == " I hope that helps!"


# ---------------------------------------------------------------------------
# Non-fatal: the empty query. Section 6 prices it at -0.05.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", " ", "\n\t "])
def test_empty_query_is_a_penalty_not_a_crash(query: str) -> None:
    parsed = parse_turn(f"<search>{query}</search>")

    assert parsed.failure is ParseFailure.EMPTY_QUERY
    assert parsed.fatal is False
    assert parsed.action is Action.SEARCH
    assert parsed.query == ""


# ---------------------------------------------------------------------------
# Fatal failures, one case per ParseFailure category
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("<think>unterminated", ParseFailure.UNCLOSED_TAG, id="unclosed-think"),
        pytest.param("<search>no closer", ParseFailure.UNCLOSED_TAG, id="unclosed-search"),
        pytest.param("<answer>no closer", ParseFailure.UNCLOSED_TAG, id="unclosed-answer"),
        pytest.param("stray</think>", ParseFailure.UNCLOSED_TAG, id="closing-with-nothing-open"),
        pytest.param(
            "<think><search>q</search></think>", ParseFailure.NESTED_TAG, id="search-in-think"
        ),
        pytest.param(
            "<search>a<search>b</search></search>", ParseFailure.NESTED_TAG, id="search-in-search"
        ),
        pytest.param("<think>a</search>", ParseFailure.NESTED_TAG, id="mismatched-pair"),
        pytest.param("<tool>ls</tool>", ParseFailure.UNKNOWN_TAG, id="unknown-tag"),
        pytest.param("<info>made up</info>", ParseFailure.UNKNOWN_TAG, id="policy-wrote-info"),
        pytest.param(
            "<think>ok</think><info>made up</info>",
            ParseFailure.UNKNOWN_TAG,
            id="info-after-a-real-think",
        ),
        pytest.param("<>empty name</>", ParseFailure.UNKNOWN_TAG, id="empty-tag-name"),
        pytest.param("<answer></answer>", ParseFailure.NO_ANSWER, id="empty-answer"),
        pytest.param("<think>I give up.</think>", ParseFailure.NO_ANSWER, id="thought-only"),
        pytest.param("Reading, Berkshire.", ParseFailure.NO_ANSWER, id="bare-prose"),
        pytest.param("", ParseFailure.NO_ANSWER, id="empty-generation"),
    ],
)
def test_malformed_output_is_classified(text: str, expected: ParseFailure) -> None:
    parsed = parse_turn(text)

    assert parsed.failure is expected
    assert parsed.fatal is True
    assert parsed.action is None


def test_every_parse_failure_category_is_reachable() -> None:
    """A new category must come with a parser path and a test, or this fails."""
    produced = {
        parse_turn(text).failure
        for text in (
            "<think>unterminated",
            "<think><search>q</search></think>",
            "<tool>ls</tool>",
            "<search> </search>",
            "<answer></answer>",
        )
    }

    assert produced == set(ParseFailure)


def test_turns_before_a_fatal_failure_are_kept() -> None:
    """The think that preceded the break is still something the policy wrote,
    and reading it is how a malformed turn gets diagnosed (section 15)."""
    parsed = parse_turn("<think>Let me search.</think><search>q<search>")

    assert [t.text for t in parsed.turns] == ["Let me search."]
    assert parsed.failure is ParseFailure.NESTED_TAG


# ---------------------------------------------------------------------------
# The parser against what the protocol renders - the two must agree
# ---------------------------------------------------------------------------


def test_injected_info_does_not_parse_as_a_policy_turn() -> None:
    """Round-trip guard. If `render_info` ever emitted something the parser read
    as a policy turn, injected text would start carrying loss."""
    info = render_info((Passage(doc_id="d1", title="T", text="x"),))

    assert parse_turn(info).failure is ParseFailure.UNKNOWN_TAG


def test_the_prompt_shows_the_format_it_asks_for() -> None:
    """The instructions contain `</search>`, which is also the stop sequence.

    That is safe and worth pinning: stop strings are matched against generated
    continuations, never against the prompt, so showing the format costs
    nothing. If this ever stops being true the loop breaks on turn one.
    """
    prompt = render_prompt("Where was Sam Mendes born?", search_budget=4, top_k=3)

    assert Tag.SEARCH.close in prompt
    assert "at most 4 times" in prompt
    assert "returns 3 passages" in prompt
    assert prompt.endswith("Question: Where was Sam Mendes born?")
