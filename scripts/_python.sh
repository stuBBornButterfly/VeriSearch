# Sourced, not run: sets PYTHON for the scripts beside it.
#
# Jenkins calls these scripts with `bash`, not `uv run`, so a bare `python` is
# whatever the agent happens to have on PATH - not necessarily the interpreter
# the project was installed into. The order:
#
#   1. PYTHON, if the caller set it
#   2. the project venv: .venv/bin/python on Linux and macOS, or
#      .venv/Scripts/python.exe on Windows, which is where `uv venv` puts it on
#      the Jenkins laptop
#   3. `python`, then `python3`, from PATH
#
# The same order as Dhrubo's scripts/02_train.sh, plus the Windows layout. The
# venv is found from this file's location, not the working directory.
#
# The one decision these scripts make in bash: it picks the interpreter that
# would otherwise run verisearch/cli.py, so it cannot live there.
#
# Owner: Ayesha.

_verisearch_venv="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${_verisearch_venv}/bin/python" ]]; then
    PYTHON="${_verisearch_venv}/bin/python"
  elif [[ -x "${_verisearch_venv}/Scripts/python.exe" ]]; then
    PYTHON="${_verisearch_venv}/Scripts/python.exe"
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    PYTHON=python3
  fi
fi
unset _verisearch_venv
