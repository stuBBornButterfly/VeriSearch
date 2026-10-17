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
#
# Owner: Ayesha.
set -euo pipefail

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

PLAN=()
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  else
    ARGS+=("$arg")
  fi
done

# Step 1. Skipped when the context already exists, because rebuilding it is how
# two runs come to disagree about what they measured: the file is the contract,
# not the command that made it.
if [[ -f "${CONTEXT}" && -z "${FORCE_CONTEXT:-}" ]]; then
  echo "context    ${CONTEXT} already exists, reusing it"
  echo "           set FORCE_CONTEXT=1 to rebuild - but every report already"
  echo "           written against the old one becomes incomparable"
else
  python -m verisearch.cli "${PLAN[@]}" eval-context \
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

python -m verisearch.cli "${PLAN[@]}" eval-report \
  --results "${RESULTS}" \
  --context "${CONTEXT}" \
  --run-name "${RUN_NAME}" \
  --kind "${KIND}" \
  --out "${REPORT}" \
  ${ARGS[@]+"${ARGS[@]}"}
