"""Static checks on the Jenkinsfile.

The pipeline is the one part of this project no test can execute: it runs only
on the controller, and its nightly half only on a timer build of `main`. So a
mistake in the nightly path surfaces at 2 a.m., on `main`, after an unrelated
merge. These tests pin the properties that were once wrong, as text, where the
PR pipeline does see them. Each one names the bug it exists for.

Comments are stripped before anything is checked - the Jenkinsfile explains the
old `fileExists` guard in prose, and prose is not a stage condition.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

JENKINSFILE = Path(__file__).resolve().parent.parent / "Jenkinsfile"

_TIMEOUT = re.compile(r"timeout\(time:\s*(\d+),\s*unit:\s*'MINUTES'\)")


@pytest.fixture(scope="module")
def code() -> str:
    """The Jenkinsfile with `//` comments removed."""
    lines = JENKINSFILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(line.split("//", 1)[0] for line in lines)


@pytest.fixture(scope="module")
def stages(code: str) -> dict[str, str]:
    """Stage name to body. A body runs up to the next `stage(`, which is enough
    to attribute every directive to the stage it belongs to."""
    parts = re.split(r"stage\('([^']+)'\)", code)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _gpu_stages(stages: dict[str, str]) -> dict[str, str]:
    return {name: body for name, body in stages.items() if "agent { label" in body}


def test_the_nightly_stages_exist(stages: dict[str, str]) -> None:
    """Guards the rest of this file against passing vacuously."""
    assert set(_gpu_stages(stages)) == {"smoke-train", "eval-regression"}
    assert "nightly-gate" in stages


def test_no_stage_is_gated_on_a_file_existing(code: str) -> None:
    """`fileExists('scripts/03_eval.sh')` became true at P1, when the CLI PR
    added the script, and would have run a regression eval on a controller with
    no GPU, no index and no `--regression` mode. A file existing says nothing
    about whether a stage can run."""
    assert "fileExists(" not in code


def test_the_outer_timeout_covers_every_stage(code: str) -> None:
    """It was 45 minutes - exactly the five blocking stages - so a nightly
    build was killed partway through smoke-train and never reached
    eval-regression."""
    minutes = [int(value) for value in _TIMEOUT.findall(code)]
    outer, *per_stage = minutes
    assert per_stage, "no stage timeouts found - the regex no longer matches the file"
    assert outer >= sum(per_stage), (
        f"outer timeout {outer} min is less than the {sum(per_stage)} min the stages "
        f"may take between them"
    )


def test_every_stage_has_its_own_timeout(stages: dict[str, str]) -> None:
    """The outer bound only says the build ended; a stage bound says which
    step hung."""
    missing = [name for name, body in stages.items() if not _TIMEOUT.search(body)]
    assert not missing, f"stages without a timeout: {missing}"


def test_a_stage_on_another_agent_evaluates_its_conditions_first(stages: dict[str, str]) -> None:
    """Without `beforeAgent true`, Jenkins allocates the agent before checking
    `when` - so on a night the stage is off, the build still queues for a GPU
    label that may not exist, until the outer timeout kills it."""
    for name, body in _gpu_stages(stages).items():
        assert "beforeAgent true" in body, f"{name} allocates its agent before its conditions"


def test_every_gpu_stage_has_its_own_switch_and_the_gate_prints_it(
    stages: dict[str, str],
) -> None:
    """One switch per stage, because the two become runnable at different times:
    a single GPU switch would turn eval-regression red the night smoke-train
    was enabled. And the gate prints each switch, so an "off" is a line in the
    log rather than a grey box."""
    gate = stages["nightly-gate"]
    for name, body in _gpu_stages(stages).items():
        switch = "VERISEARCH_NIGHTLY_" + name.upper().replace("-", "_")
        assert f"environment name: '{switch}', value: '1'" in body, f"{name} has no switch"
        assert switch in gate, f"nightly-gate does not report {switch}"
    assert "VERISEARCH_GPU_LABEL" in gate


def test_every_gpu_stage_installs_for_itself(stages: dict[str, str]) -> None:
    """A stage on another agent gets a fresh workspace. Without its own install
    it runs against whatever that machine happens to have."""
    for name, body in _gpu_stages(stages).items():
        assert "uv sync --locked" in body, f"{name} does not install its own environment"


def test_every_nightly_stage_is_main_only(stages: dict[str, str]) -> None:
    for name, body in stages.items():
        if "isNightly()" in body:
            assert "branch 'main'" in body, f"{name} is nightly but not restricted to main"
