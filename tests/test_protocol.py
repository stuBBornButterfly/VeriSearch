"""Protocol tests: the prompt, the injected block, and the stop-sequence guard."""

from __future__ import annotations

import pytest

from verisearch.env.protocol import (
    NO_RESULTS,
    POLICY_TAGS,
    STOP_SEQUENCES,
    ProtocolError,
    Tag,
    check_stop_sequences,
    render_info,
    render_injection,
    render_prompt,
)
from verisearch.types import Passage


def _passages(n: int) -> tuple[Passage, ...]:
    return tuple(
        Passage(doc_id=f"d{i}", title=f"Title {i}", text=f"Body {i}.") for i in range(1, n + 1)
    )


def test_tags_render_both_halves() -> None:
    assert (Tag.SEARCH.open, Tag.SEARCH.close) == ("<search>", "</search>")


def test_info_is_not_a_policy_tag() -> None:
    """The environment owns `<info>`. A policy that writes one is forging
    evidence, which `parse_turn` treats as a fatal failure."""
    assert Tag.INFO not in POLICY_TAGS
    assert set(POLICY_TAGS) == {Tag.THINK, Tag.SEARCH, Tag.ANSWER}


def test_default_stop_sequences_are_exactly_the_search_closer() -> None:
    assert STOP_SEQUENCES == ("</search>",)
    check_stop_sequences(STOP_SEQUENCES)


@pytest.mark.parametrize(
    ("stops", "reason"),
    [
        pytest.param(("</search>", "</answer>"), "truncated", id="answer-closer-present"),
        pytest.param(("</answer>",), "must include", id="search-closer-missing"),
        pytest.param((), "must include", id="empty"),
    ],
)
def test_bad_stop_sequences_are_refused_with_the_reason(
    stops: tuple[str, ...], reason: str
) -> None:
    """`configs/base.yaml` shipped with `</answer>` in this list (fixed in #20).
    The guard is here so a config change cannot quietly reintroduce it."""
    with pytest.raises(ProtocolError, match=reason):
        check_stop_sequences(stops)


def test_rendered_info_is_numbered_and_in_retrieval_order() -> None:
    block = render_info(_passages(2))

    assert block == ("<info>\n[1] Title 1: Body 1.\n[2] Title 2: Body 2.\n</info>")


def test_no_hits_says_so_in_words() -> None:
    """An empty `<info></info>` block reads like no search happened. The policy
    has to be able to tell 'I searched and found nothing' apart from that."""
    assert render_info(()) == f"<info>\n{NO_RESULTS}\n</info>"


def test_the_injection_carries_its_own_separators() -> None:
    """Finding A on #31: the newlines around the block are part of what is
    injected, so an `INFO` turn's text and its ids describe the same characters."""
    passages = _passages(1)

    assert render_injection(passages) == f"\n{render_info(passages)}\n"


def test_render_info_is_deterministic() -> None:
    passages = _passages(3)

    assert render_info(passages) == render_info(passages)


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_an_empty_question_is_refused(question: str) -> None:
    with pytest.raises(ProtocolError, match="question is empty"):
        render_prompt(question, search_budget=4, top_k=3)


def test_the_prompt_tells_the_policy_not_to_write_info() -> None:
    prompt = render_prompt("Who directed it?", search_budget=2, top_k=1)

    assert "Never write <info> yourself." in prompt
