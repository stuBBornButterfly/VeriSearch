"""Five questions end to end. PROJECT.md section 3: fake retriever, CPU only,
under 60 seconds; section 10 runs it as the blocking `integration` stage.

Everything real except the two things that need a GPU or a network: the prompt,
the parser and the episode loop are Dhrubo's (`verisearch/env/`), the scoring is
`eval/metrics.py`, and the trajectories go through JSONL the way P1's exit run
writes them. The retriever is `FakeRetriever`; the policy is `ReadingPolicy`
below.

The policy is scripted, but it reads. Each step is a function of the latest
`<info>` block in the prompt it is handed, so the two-hop question can only be
answered if the environment really injected the first search's passages and
fed them back. A loop that dropped or garbled the injection would fail here,
not just in a training curve.

The five cover each way an episode ends that the loop can reach:

1. one search, answered right
2. two searches, the second built from the first's passages, answered right
3. answered from memory with no search, wrong
4. searched, then wrote prose with no tags: a parse error with no answer
5. never stopped searching: the search budget ran out

Owner: Ayesha.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from verisearch.config import load_config
from verisearch.env.episode import EpisodeConfig, Generation, Question, run_episode
from verisearch.env.protocol import STOP_SEQUENCES
from verisearch.eval.metrics import FailureReason, classify_failure, score_answer
from verisearch.testing.fakes import FakeRetriever
from verisearch.types import ParseFailure, Termination, Trajectory

pytestmark = pytest.mark.integration

# A real injected block, as `render_info` writes it. The instructions also
# mention `<info></info>` in passing; the newlines keep that from matching.
_INFO_BLOCK = re.compile(r"<info>\n(.*?)\n</info>", re.DOTALL)
_FIRST_TITLE = re.compile(r"^\[1\] ([^:]+):", re.MULTILINE)

Step = Callable[[str | None], str]


class ReadingPolicy:
    """Plays one scripted episode, one step per turn, reading what it was given.

    Each step receives the text of the latest `<info>` block in the prompt, or
    None before the first search, and returns the turn the policy writes.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        assert "</search>" in stop, "the loop must stop generation at </search>"
        assert max_new_tokens > 0
        self.prompts.append(prompt)
        blocks = _INFO_BLOCK.findall(prompt)
        latest = blocks[-1] if blocks else None
        return Generation(text=self._steps.pop(0)(latest))


def _first_title(info: str | None) -> str:
    assert info is not None, "answered from passages that were never injected"
    match = _FIRST_TITLE.search(info)
    assert match is not None, f"no passage [1] in {info!r}"
    return match.group(1)


def _extract(pattern: str) -> Callable[[str | None], str]:
    def read(info: str | None) -> str:
        assert info is not None, "read passages that were never injected"
        match = re.search(pattern, info)
        assert match is not None, f"{pattern!r} not in {info!r}"
        return match.group(1)

    return read


_machine = _extract(r"notes on the ([A-Z]\w+ [A-Z]\w+)")
_designer = _extract(r"designed by ([A-Z]\w+ [A-Z]\w+)")


def _search(query: str) -> Step:
    return lambda _info: f"<think>I need to look this up.</think>\n<search>{query}</search>"


CASES: dict[str, tuple[Question, list[Step]]] = {
    "one-hop": (
        Question(
            question_id="int-1",
            question="Who wrote the first published notes on the Analytical Engine?",
            gold_answers=("Ada Lovelace",),
        ),
        [
            _search("first published notes Analytical Engine"),
            lambda info: f"<answer>{_first_title(info)}</answer>",
        ],
    ),
    "two-hop": (
        Question(
            question_id="int-2",
            question="Who designed the machine that Ada Lovelace wrote notes on?",
            gold_answers=("Charles Babbage",),
        ),
        [
            _search("Ada Lovelace notes"),
            lambda info: f"<search>{_machine(info)} designed by</search>",
            lambda info: f"<think>Got it.</think>\n<answer>{_designer(info)}</answer>",
        ],
    ),
    "from-memory": (
        Question(
            question_id="int-3",
            question="In which county is Lovelace, Texas?",
            gold_answers=("Hunt County",),
        ),
        [lambda _info: "<think>I know this one.</think>\n<answer>Dallas County</answer>"],
    ),
    "prose": (
        Question(
            question_id="int-4",
            question="Who originated the concept of a programmable computer?",
            gold_answers=("Charles Babbage",),
        ),
        [
            _search("concept of a programmable computer"),
            lambda _info: "I think it was probably Babbage, but I am not sure.",
        ],
    ),
    "never-stops": (
        Question(
            question_id="int-5",
            question="What does an engine convert energy into?",
            gold_answers=("motion",),
        ),
        # One more search than any budget in the config allows.
        [_search(f"engine energy {n}") for n in range(10)],
    ),
}


@pytest.fixture(scope="module")
def config() -> EpisodeConfig:
    """The limits from `configs/base.yaml`, as training and eval use them.

    Except the stop sequences: `base.yaml` on this branch still lists
    `</answer>`, which `check_stop_sequences` refuses, and #20 removes it. Drop
    the override once #20 is merged.
    """
    return EpisodeConfig.from_config(load_config()).model_copy(
        update={"stop_sequences": STOP_SEQUENCES}
    )


@pytest.fixture(scope="module")
def run(config: EpisodeConfig) -> dict[str, tuple[Trajectory, FakeRetriever, float]]:
    """All five episodes, each with a fresh retriever so its calls are its own."""
    results: dict[str, tuple[Trajectory, FakeRetriever, float]] = {}
    for name, (question, steps) in CASES.items():
        retriever = FakeRetriever()
        start = time.perf_counter()
        trajectory = run_episode(question, ReadingPolicy(steps), retriever, config)
        results[name] = (trajectory, retriever, time.perf_counter() - start)
    return results


def test_the_five_questions_finish_well_inside_the_budget(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]],
) -> None:
    assert len(run) == 5
    assert sum(elapsed for _, _, elapsed in run.values()) < 60.0


def test_a_one_hop_question_is_answered_from_the_injected_passage(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]],
) -> None:
    traj, retriever, _ = run["one-hop"]

    assert traj.termination is Termination.ANSWERED
    assert traj.answer == "Ada Lovelace"
    assert traj.n_searches == 1
    assert [query for query, _ in retriever.calls] == ["first published notes Analytical Engine"]
    assert score_answer(traj.answer, traj.gold_answers).exact_match == 1.0


def test_the_second_hop_is_built_from_the_first_hops_passages(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]],
) -> None:
    """The query the second search sends exists nowhere but in the first
    search's passages - so this passes only if the loop fed them back."""
    traj, retriever, _ = run["two-hop"]

    assert [query for query, _ in retriever.calls] == [
        "Ada Lovelace notes",
        "Analytical Engine designed by",
    ]
    assert traj.termination is Termination.ANSWERED
    assert traj.answer == "Charles Babbage"
    assert score_answer(traj.answer, traj.gold_answers).exact_match == 1.0
    assert {p.title for p in traj.retrieved_passages} >= {"Ada Lovelace", "Analytical Engine"}


def test_an_answer_from_memory_is_scored_wrong_and_blamed_on_retrieval(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]],
) -> None:
    traj, retriever, _ = run["from-memory"]

    assert traj.termination is Termination.ANSWERED
    assert retriever.calls == []
    # Exact match, not F1: "Dallas County" shares "county" with the gold, so it
    # earns partial F1 while being wrong.
    assert score_answer(traj.answer, traj.gold_answers).exact_match == 0.0
    assert _failure(traj) is FailureReason.EMPTY_RETRIEVAL


def test_prose_after_a_search_is_a_parse_error_with_no_answer(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]],
) -> None:
    traj, _, _ = run["prose"]

    assert traj.termination is Termination.PARSE_ERROR
    assert traj.parse_failure is ParseFailure.NO_ANSWER
    assert traj.answer is None
    assert traj.retrieved_passages, "the search before the prose was served"
    assert _failure(traj) is FailureReason.NO_ANSWER


def test_a_policy_that_never_answers_runs_out_of_searches(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]], config: EpisodeConfig
) -> None:
    traj, retriever, _ = run["never-stops"]

    assert traj.termination is Termination.SEARCH_BUDGET_EXHAUSTED
    assert traj.answer is None
    # The search that broke the budget is in the trajectory but never reached
    # the retriever.
    assert traj.n_searches == config.search_budget + 1
    assert len(retriever.calls) == config.search_budget


def test_every_trajectory_survives_the_jsonl_round_trip(
    run: dict[str, tuple[Trajectory, FakeRetriever, float]], tmp_path: Path
) -> None:
    """P1's exit criterion is typed trajectory JSONL; this is that file, read back."""
    path = tmp_path / "trajectories.jsonl"
    written = [traj for traj, _, _ in run.values()]
    path.write_text("".join(traj.model_dump_json() + "\n" for traj in written), encoding="utf-8")

    read = [Trajectory.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]

    assert read == written


def test_two_of_five_are_right(run: dict[str, tuple[Trajectory, FakeRetriever, float]]) -> None:
    scores = [score_answer(traj.answer, traj.gold_answers) for traj, _, _ in run.values()]

    assert sum(score.exact_match for score in scores) / len(scores) == pytest.approx(0.4)


def _failure(traj: Trajectory) -> FailureReason:
    """Classified the way `eval/harness.py` does on the eval branch (#15)."""
    score = score_answer(traj.answer, traj.gold_answers)
    return classify_failure(
        exact_match_score=score.exact_match,
        has_answer=traj.answer is not None,
        hit_turn_limit=traj.termination is Termination.MAX_TURNS,
        retrieved_count=len(traj.retrieved_passages),
    )
