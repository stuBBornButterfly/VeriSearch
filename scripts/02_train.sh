#!/usr/bin/env bash
#
# Train one policy with GRPO. PROJECT.md section 4, and the engine behind P3.
#
# Owner: Dhrubo. The flags are a contract with two callers, so they do not move
# without a conversation:
#
#   1. Jenkins, nightly on main (section 10, Ayesha's file):
#        bash scripts/02_train.sh --smoke --steps 5
#      That run must be CPU-tolerant to *start* and must assert the loss is
#      finite. It uses the fake retriever and five questions from the fake
#      corpus, so it needs no index and no service - it proves the loop runs,
#      not that the model learns.
#
#   2. scripts/04_sweep.sh, once per lambda point:
#        LAMBDA=0.05 bash scripts/02_train.sh
#
# Usage:
#   bash scripts/02_train.sh --smoke --steps 5
#   QUESTIONS=data/processed/hotpotqa_train.jsonl LIMIT=20 bash scripts/02_train.sh
#   LAMBDA=0.02 OUT=checkpoints/lambda-0.02 bash scripts/02_train.sh
#   bash scripts/02_train.sh --plan
#
# Needs the [train] extra and a GPU for anything but --plan:
#   uv sync --extra dev --extra train
set -euo pipefail

CONFIGS="${CONFIGS:-configs/base.yaml configs/dev_1.5b.yaml}"
QUESTIONS="${QUESTIONS:-data/processed/hotpotqa_train.jsonl}"
OUT="${OUT:-checkpoints/grpo}"
LIMIT="${LIMIT:-}"
LAMBDA="${LAMBDA:-}"

# --steps, --smoke and --plan are forwarded; anything else is rejected here
# rather than passed along, because a typo that reaches argparse as an unknown
# flag is a failed nightly build that looks like a training failure.
STEPS=()
SMOKE=()
PLAN=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=(--smoke)
      shift
      ;;
    --plan)
      PLAN=(--plan)
      shift
      ;;
    --steps)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "error: --steps needs a positive integer, got '${2:-}'" >&2
        exit 2
      fi
      STEPS=(--steps "$2")
      shift 2
      ;;
    --steps=*)
      value="${1#*=}"
      if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "error: --steps needs a positive integer, got '$value'" >&2
        exit 2
      fi
      STEPS=(--steps "$value")
      shift
      ;;
    *)
      echo "error: unknown flag: $1" >&2
      echo "       supported: --smoke, --steps N, --plan" >&2
      exit 2
      ;;
  esac
done

# Which interpreter. Jenkins calls this script with `bash`, not `uv run`, so a
# bare `python` depends on whatever the agent happens to have activated. Prefer
# the project venv when it is there, and let PYTHON override.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    PYTHON=python3
  fi
fi

CONFIG_ARGS=()
for path in ${CONFIGS}; do
  CONFIG_ARGS+=(--config "${path}")
done

EXTRA=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ -n "${LAMBDA}" ]]; then
  EXTRA+=(--lambda-cost "${LAMBDA}")
fi

echo "train      configs: ${CONFIGS}"
echo "           out:     ${OUT}"

"${PYTHON}" -m verisearch.train.grpo \
  "${CONFIG_ARGS[@]}" \
  --questions "${QUESTIONS}" \
  --out "${OUT}" \
  ${STEPS[@]+"${STEPS[@]}"} \
  ${SMOKE[@]+"${SMOKE[@]}"} \
  ${PLAN[@]+"${PLAN[@]}"} \
  ${EXTRA[@]+"${EXTRA[@]}"}
