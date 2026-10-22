#!/usr/bin/env bash
#
# Evaluate one run. PROJECT.md section 4, and the mechanism behind section 8's
# "All on identical eval sets with the identical retriever".
#
# Two steps, and the order matters:
#
#   1. eval-context  - reads the retriever fingerprint out of the index's own
#                      meta.json, pairs it with the eval set, writes context.json.
#                      Run this ONCE per experiment. Every baseline then loads
#                      the same file.
#   2. eval-report   - aggregates the scored rows a run produced into section 9's
#                      table, against that shared context.
#
# Step 2 needs results.jsonl, which the harness writes while running a policy.
# There is no policy yet - no training GPU - so today this script is run with
# CONTEXT_ONLY=1 to produce the context, and the report half waits on P3.
#
# Usage:
#   CONTEXT_ONLY=1 scripts/03_eval.sh
#   RUN_NAME=prompted KIND=prompted_iterative RESULTS=results/raw/prompted.jsonl \
#       scripts/03_eval.sh
#   scripts/03_eval.sh --plan
#   scripts/03_eval.sh --regression --n 100 --max-f1-drop 2.0
#
# --regression is section 10's nightly gate, and is a separate path: it skips
# both steps above and compares CANDIDATE - tonight's rows on the fixed set -
# with the promoted reference run. Its inputs:
#
#   REGRESSION_EXAMPLES   the fixed set, from `python -m verisearch.cli
#                         regression-set` (once, then kept)
#   REGRESSION_REFERENCE  a directory holding report.json and results.jsonl
#                         from a known-good checkpoint on that set
#   CANDIDATE             tonight's results.jsonl on the same set. Producing it
#                         means running the policy, which needs the episode loop.
#
# A nightly agent starts from a fresh workspace, so on the GPU machine point
# these, and INDEX, at paths that outlive a build.
#
# Owner: Ayesha.
set -euo pipefail

# Which interpreter runs the CLI: see _python.sh.
# shellcheck source=scripts/_python.sh
source "$(dirname "${BASH_SOURCE[0]}")/_python.sh"

INDEX="${INDEX:-data/index/corpus}"
EXAMPLES="${EXAMPLES:-data/processed/hotpotqa_dev.jsonl}"
DATASET="${DATASET:-hotpotqa}"
SPLIT="${SPLIT:-dev}"
K="${K:-3}"

RESULTS_DIR="${RESULTS_DIR:-results/raw}"
CONTEXT="${CONTEXT:-${RESULTS_DIR}/context.json}"

RUN_NAME="${RUN_NAME:-}"
KIND="${KIND:-}"
RESULTS="${RESULTS:-}"
REPORT="${REPORT:-}"

REGRESSION_EXAMPLES="${REGRESSION_EXAMPLES:-data/regression/examples.jsonl}"
REGRESSION_REFERENCE="${REGRESSION_REFERENCE:-data/regression/reference}"
CANDIDATE="${CANDIDATE:-${RESULTS_DIR}/regression/candidate.jsonl}"

PLAN=()
ARGS=()
REGRESSION=""
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  elif [[ "$arg" == "--regression" ]]; then
    REGRESSION=1
  else
    ARGS+=("$arg")
  fi
done

# The nightly gate. Everything it decides - the set size, the threshold, what
# counts as a drop - is in eval-regression, where it is tested. --n and
# --max-f1-drop pass straight through.
if [[ -n "${REGRESSION}" ]]; then
  exec "${PYTHON}" -m verisearch.cli "${PLAN[@]}" eval-regression \
    --examples "${REGRESSION_EXAMPLES}" \
    --index "${INDEX}" \
    --reference "${REGRESSION_REFERENCE}" \
    --candidate "${CANDIDATE}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --k "${K}" \
    ${ARGS[@]+"${ARGS[@]}"}
fi

# Step 1. Skipped when the context already exists, because rebuilding it is how
# two runs come to disagree about what they measured: the file is the contract,
# not the command that made it.
if [[ -f "${CONTEXT}" && -z "${FORCE_CONTEXT:-}" ]]; then
  echo "context    ${CONTEXT} already exists, reusing it"
  echo "           set FORCE_CONTEXT=1 to rebuild - but every report already"
  echo "           written against the old one becomes incomparable"
else
  "${PYTHON}" -m verisearch.cli "${PLAN[@]}" eval-context \
    --index "${INDEX}" \
    --examples "${EXAMPLES}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --k "${K}" \
    --out "${CONTEXT}"
fi

if [[ -n "${CONTEXT_ONLY:-}" ]]; then
  echo "CONTEXT_ONLY set, stopping before the report"
  exit 0
fi

# Step 2.
if [[ -z "${RUN_NAME}" || -z "${KIND}" || -z "${RESULTS}" ]]; then
  echo "error: RUN_NAME, KIND and RESULTS are required for the report step." >&2
  echo "       KIND is one of: no_retrieval vanilla_rag prompted_iterative" >&2
  echo "                       sft_only trained" >&2
  echo "       Set CONTEXT_ONLY=1 to stop after the context." >&2
  exit 2
fi

REPORT="${REPORT:-${RESULTS_DIR}/${RUN_NAME}/report.json}"

"${PYTHON}" -m verisearch.cli "${PLAN[@]}" eval-report \
  --results "${RESULTS}" \
  --context "${CONTEXT}" \
  --run-name "${RUN_NAME}" \
  --kind "${KIND}" \
  --out "${REPORT}" \
  ${ARGS[@]+"${ARGS[@]}"}
