"""How the shell scripts find Python.

Jenkins runs `scripts/*.sh` with `bash`, not `uv run`, so a script that calls a
bare `python` gets whatever interpreter the agent has on PATH - not necessarily
the one the project was installed into. Every script takes its interpreter from
`scripts/_python.sh` instead. These tests pin that three ways: each script goes
through the helper (read from the file), each one runs the interpreter it was
given (run under bash, against a stand-in interpreter, so no data or server is
needed), and the helper picks the right interpreter.

Owner: Ayesha.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
HELPER = SCRIPTS / "_python.sh"

# Each script that calls the CLI, and the subcommand its `--plan` run reaches.
CLI_SCRIPTS = {
    "00_prepare_data.sh": "prepare-data",
    "01_build_index.sh": "build-index",
    "03_eval.sh": "eval-context",
    "04_sweep.sh": "frontier",
    "05_serve_bench.sh": "serve-bench",
}
# Dhrubo's. It picks its interpreter in the same order, in its own file.
RESOLVES_ITS_OWN = {"02_train.sh"}

_SOURCES_HELPER = 'source "$(dirname "${BASH_SOURCE[0]}")/_python.sh"'
_BARE_PYTHON = re.compile(r"^\s*(exec\s+)?python3?\s")

#: Git Bash, for when `bash` on PATH is not the shell we want.
_GIT_BASH = (
    Path("C:/Program Files/Git/bin/bash.exe"),
    Path("C:/Program Files (x86)/Git/bin/bash.exe"),
)


def _posix_bash() -> str | None:
    """A bash that can run these scripts, or None if there is none.

    On Windows, `bash` on PATH is usually `C:/Windows/System32/bash.exe`, the
    WSL launcher, which runs in a different filesystem where neither the repo
    nor the venv exists. Same resolution as Dhrubo's `tests/test_rollout.py`
    (#31); one copy should move to `conftest.py` once both are merged.
    """
    found = shutil.which("bash")
    if found is not None and "system32" not in found.replace("\\", "/").lower():
        return found
    return next((str(path) for path in _GIT_BASH if path.is_file()), None)


@pytest.fixture
def bash() -> str:
    found = _posix_bash()
    if found is None:
        # Not a skip to make CI green: the thing under test is a shell script,
        # and a machine with no POSIX shell cannot run it. Same category as the
        # `gpu` marker.
        pytest.skip("no POSIX bash on this machine; Git Bash or a Unix shell is needed")
    return found


def _environment(**overrides: str) -> dict[str, str]:
    """The test's environment, minus any PYTHON it inherited, plus overrides."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHON"}
    env.update(overrides)
    return env


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Read from the files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CLI_SCRIPTS))
def test_each_script_takes_its_interpreter_from_the_helper(name: str) -> None:
    text = (SCRIPTS / name).read_text(encoding="utf-8")

    assert _SOURCES_HELPER in text
    bare = [line for line in text.splitlines() if _BARE_PYTHON.match(line)]
    assert bare == [], f"{name} calls python directly: {bare}"


def test_every_script_is_accounted_for() -> None:
    """A new script has to go through the helper, or be listed as resolving its own."""
    found = {path.name for path in SCRIPTS.glob("*.sh")} - {HELPER.name}

    assert found - RESOLVES_ITS_OWN == set(CLI_SCRIPTS)


# ---------------------------------------------------------------------------
# Run under bash
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(("name", "subcommand"), sorted(CLI_SCRIPTS.items()))
def test_each_script_runs_the_interpreter_it_was_given(
    bash: str, tmp_path: Path, name: str, subcommand: str
) -> None:
    """With PYTHON pointing at a stand-in that echoes its arguments, every
    script calls it, with the CLI module and the subcommand after `--plan`."""
    stand_in = _executable(tmp_path / "python", '#!/usr/bin/env bash\necho "stand-in $*"\n')
    env = _environment(
        PYTHON=stand_in.as_posix(),
        # 03_eval.sh skips eval-context when the context file exists, and stops
        # at the report step without RUN_NAME. Point it at a fresh path and stop.
        CONTEXT=(tmp_path / "context.json").as_posix(),
        CONTEXT_ONLY="1",
    )

    result = subprocess.run(
        [bash, f"scripts/{name}", "--plan"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"stand-in -m verisearch.cli --plan {subcommand}" in result.stdout


@pytest.mark.integration
def test_a_script_runs_the_real_cli_without_being_told_which_python(
    bash: str, tmp_path: Path
) -> None:
    """What Jenkins does: no PYTHON, so the helper finds the project's own.

    The corpus is a one-line file here, not `data/processed/corpus.jsonl`:
    `data/` is gitignored, so a fresh CI checkout has none, and a test that
    passes only where the data happens to exist is testing the laptop.
    """
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id": "d1", "title": "T", "text": "x"}\n', encoding="utf-8")

    result = subprocess.run(
        [bash, "scripts/01_build_index.sh", "--plan"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_environment(CORPUS=corpus.as_posix(), OUT=(tmp_path / "index").as_posix()),
    )

    assert result.returncode == 0, result.stderr
    assert "plan only, nothing written" in result.stdout


def _resolve(bash: str, root: Path, **env: str) -> str:
    """What PYTHON the helper sets, when it sits at `root/scripts/_python.sh`."""
    helper = root / "scripts" / "_python.sh"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HELPER, helper)
    result = subprocess.run(
        [bash, "-c", 'source "$1"; printf %s "$PYTHON"', "_", helper.as_posix()],
        capture_output=True,
        text=True,
        check=True,
        env=_environment(**env),
    )
    return result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    "layout",
    [".venv/bin/python", ".venv/Scripts/python.exe"],
    ids=["unix-venv", "windows-venv"],
)
def test_the_helper_prefers_the_project_venv(bash: str, tmp_path: Path, layout: str) -> None:
    _executable(tmp_path / layout, "#!/bin/sh\n")

    chosen = _resolve(bash, tmp_path)

    assert chosen.endswith(f"/{layout}")
    assert tmp_path.name in chosen


@pytest.mark.integration
def test_the_callers_python_beats_the_venv(bash: str, tmp_path: Path) -> None:
    _executable(tmp_path / ".venv/bin/python", "#!/bin/sh\n")

    assert _resolve(bash, tmp_path, PYTHON="/opt/custom/python") == "/opt/custom/python"


@pytest.mark.integration
def test_without_a_venv_the_helper_falls_back_to_path(bash: str, tmp_path: Path) -> None:
    assert _resolve(bash, tmp_path) in {"python", "python3"}
