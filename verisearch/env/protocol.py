"""Prompt protocol: the tag vocabulary, the prompt, and what the environment injects.

Owner: Dhrubo. PROJECT.md section 1 fixes the trajectory shape:

    <think> reasoning </think>
    <search> query </search>
    -> environment injects <info> passages </info>
    <think> reasoning </think>
    <answer> final answer </answer>

Formatting only. Parsing lives in `env/parser.py` and control flow in
`env/episode.py`, so that this module can be imported by a test that only wants
to know what a well-formed turn looks like.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from verisearch.types import Passage

__all__ = [
    "INSTRUCTIONS",
    "NO_RESULTS",
    "POLICY_TAGS",
    "STOP_SEQUENCES",
    "ProtocolError",
    "Tag",
    "check_stop_sequences",
    "render_info",
    "render_injection",
    "render_prompt",
]


class Tag(StrEnum):
    """The four tags. `INFO` is the only one the environment writes."""

    THINK = "think"
    SEARCH = "search"
    INFO = "info"
    ANSWER = "answer"

    @property
    def open(self) -> str:
        return f"<{self.value}>"

    @property
    def close(self) -> str:
        return f"</{self.value}>"


#: Tags the policy is allowed to emit. A policy-written `<info>` is a parse
#: failure, not a turn: it would be the model fabricating its own evidence.
POLICY_TAGS: tuple[Tag, ...] = (Tag.THINK, Tag.SEARCH, Tag.ANSWER)

#: Generation stops here, and only here.
#:
#: `</answer>` is deliberately absent. TRL marks a completion truncated when its
#: last token is not EOS or pad (`trl/trainer/grpo_trainer.py:2325`), so stopping
#: on `</answer>` reports every *successfully finished* episode as clipped and
#: `mask_truncated_completions=True` would zero the loss on exactly the rollouts
#: that worked. Let the model's EOS end the answer instead; then
#: `completions/clipped_ratio` reads as "fraction that never answered", which is
#: a metric worth having. Measured in `spikes/NOTES.md`, fixed in `base.yaml` by
#: Ayesha's #20; `check_stop_sequences` keeps it fixed.
STOP_SEQUENCES: tuple[str, ...] = (Tag.SEARCH.close,)

#: What the environment injects when a search returned nothing. Said plainly on
#: purpose: the policy has to learn that evidence can be absent, and an empty
#: `<info></info>` block is easy to misread as "no search happened".
NO_RESULTS = "No passages matched that query."

INSTRUCTIONS = """\
Answer the question using evidence you retrieve.

Work in turns. In each turn you may reason inside <think></think>, then do exactly one of:

  <search>your query</search>   to retrieve passages, then stop and wait
  <answer>your answer</answer>  to finish

Retrieved passages come back to you inside <info></info>. Never write <info> yourself.
You may search at most {search_budget} times, and each search returns {top_k} passages.
Search only while the passages you have do not answer the question; answer as soon as they do.
Keep the answer short - just the fact asked for.

Question: {question}"""


class ProtocolError(Exception):
    """Raised when a configured protocol cannot work. Fail loud, PROJECT.md section 3."""


def render_prompt(question: str, *, search_budget: int, top_k: int) -> str:
    """The opening prompt for one episode.

    Takes the budget and `top_k` rather than reading a config, so the prompt the
    policy sees cannot disagree with the budget the episode loop enforces - they
    come from the same `EpisodeConfig`.
    """
    if not question.strip():
        msg = "question is empty; an episode needs something to answer"
        raise ProtocolError(msg)
    return INSTRUCTIONS.format(question=question.strip(), search_budget=search_budget, top_k=top_k)


def render_info(passages: Sequence[Passage]) -> str:
    """The `<info>` block the environment appends after a search.

    Deterministic, and in retrieval order: the policy should be able to learn
    that earlier passages are the better-scoring ones.
    """
    if not passages:
        body = NO_RESULTS
    else:
        body = "\n".join(f"[{i}] {p.title}: {p.text}" for i, p in enumerate(passages, start=1))
    return f"{Tag.INFO.open}\n{body}\n{Tag.INFO.close}"


def render_injection(passages: Sequence[Passage]) -> str:
    """Exactly what the environment appends to the context after a search.

    The `<info>` block plus the newline on each side of it. The separators are
    part of the injected text, not added around it later, so the `INFO` turn's
    text and its token ids describe the same characters - and the context the
    policy samples from is the context the trainer replays. Ayesha's finding A
    on #31: with the newlines added outside, turn 2 was sampled after
    `</search>\\n<info>` and replayed after `</search><info>`.
    """
    return f"\n{render_info(passages)}\n"


def check_stop_sequences(stop_sequences: Sequence[str]) -> None:
    """Reject a stop-sequence set that would corrupt the training signal.

    Two ways it can, both silent:

    - `</answer>` present: every finished episode is reported truncated (see
      `STOP_SEQUENCES`).
    - `</search>` missing: generation runs past the search tag, so the
      environment never gets to inject passages and the episode degenerates into
      the model writing its own `<info>`.
    """
    stops = tuple(stop_sequences)
    if Tag.SEARCH.close not in stops:
        msg = (
            f"stop sequences {stops} must include {Tag.SEARCH.close!r}, or generation runs "
            "past the search tag and the environment never injects passages"
        )
        raise ProtocolError(msg)
    if Tag.ANSWER.close in stops:
        msg = (
            f"stop sequences {stops} must not include {Tag.ANSWER.close!r}: a completion that "
            "ends on a stop string has a non-EOS final token, which TRL counts as truncated, "
            "so every finished episode would be reported clipped. Let EOS end the answer."
        )
        raise ProtocolError(msg)
