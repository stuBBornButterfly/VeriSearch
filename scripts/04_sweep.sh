#!/usr/bin/env bash
#
# The lambda sweep. PROJECT.md section 2: "Lambda sweep of at least 4 points,
# plotted as an accuracy vs retrieval-count frontier."
#
# Three phases. Only the third runs today.
#
#   1. train  - one run per lambda. scripts/02_train.sh, Dhrubo's, needs a GPU.
#   2. eval   - one run per lambda, all against the SAME context.json. Needs a
#               trained checkpoint, so it waits on phase 1.
#   3. frontier - read every run's report, check the sweep is complete and that
#               its points were measured the same way, emit the table and the
#               figure. Runs now, given reports on disk.
#
# The manifest is the source of truth for which runs must exist. A sweep where
# one point crashed leaves three reports, and three points draw a curve that
# looks exactly like a complete one - so phase 3 refuses a partial sweep and
# names the missing lambda rather than plotting what it found.
#
# Exit 1 when the sweep measured no trade-off: searches per question have to
# fall as lambda rises, or there is no frontier and the retrieval penalty is not
# reaching the policy.
#
# The figure needs the [plot] extra:  uv pip install -e '.[plot]'
#
# Usage:
#   scripts/04_sweep.sh                       # phase 3 only, the default today
#   PHASE=frontier scripts/04_sweep.sh --plan
#   MANIFEST=configs/sweep_lambda.yaml FIGURE=results/figures/frontier.png \
#       scripts/04_sweep.sh
#
# Owner: Ayesha.
set -euo pipefail

# Which interpreter runs the CLI: see _python.sh.
# shellcheck source=scripts/_python.sh
source "$(dirname "${BASH_SOURCE[0]}")/_python.sh"

MANIFEST="${MANIFEST:-configs/sweep_lambda.yaml}"
OUT="${OUT:-results/frontier.md}"
FIGURE="${FIGURE:-}"
PHASE="${PHASE:-frontier}"

PLAN=()
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  else
    ARGS+=("$arg")
  fi
done

if [[ "${PHASE}" != "frontier" ]]; then
  echo "error: phase '${PHASE}' needs a training GPU and scripts/02_train.sh." >&2
  echo "       The hardware report (#1) rules out training on this machine, so" >&2
  echo "       phases 1 and 2 are not wired up yet. Only PHASE=frontier runs." >&2
  exit 2
fi

FIGURE_ARGS=()
if [[ -n "${FIGURE}" ]]; then
  FIGURE_ARGS=(--figure "${FIGURE}")
fi

"${PYTHON}" -m verisearch.cli "${PLAN[@]}" frontier \
  --manifest "${MANIFEST}" \
  --out "${OUT}" \
  ${FIGURE_ARGS[@]+"${FIGURE_ARGS[@]}"} \
  ${ARGS[@]+"${ARGS[@]}"}
