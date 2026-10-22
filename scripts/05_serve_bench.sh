#!/usr/bin/env bash
#
# Load-benchmark the policy server. PROJECT.md section 9's serving row -
# "throughput req/s, p50 p95 p99, at concurrency 1, 8, 32" - and the P5 exit
# criterion, "benchmark table with real numbers at 3 concurrency levels".
#
# Three processes must already be up. This script starts none of them, because
# launching, health-checking and tearing down a GPU server from bash is exactly
# the kind of decision section 3's standards cannot reach:
#
#   1. retrieval    the retrieval service (retrieval/service.py), on :8000
#   2. vLLM         serving the policy, on a third port because retrieval holds
#                   vLLM's default 8000. For example (unverified flags - check
#                   them against the installed vLLM):
#                     vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 \
#                       --enable-lora --lora-modules verisearch=<checkpoint>
#   3. the policy   serve/server.py, wired to the episode loop, on :8100
#      server
#
# What is being measured - policy, adapter, hardware - is read from the policy
# server's /health, not from flags here, so the line under the table cannot
# describe a different machine from the one that answered.
#
# Levels, request counts, warmup and timeout come from the serve: section of the
# config. Section 9's numbers are in configs/base.yaml. A cheap smoke run is an
# overlay that lowers serve.requests_per_level, passed as
# CONFIG="configs/base.yaml path/to/overlay.yaml" - it will report p99 as n/a and
# exit 1, which is the correct verdict on a smoke run.
#
# Exit codes: 0 publishable, 1 measured but not section 9's table (a failure, a
# level that never reached its concurrency, or too few requests for p99 - the
# files are still written, for diagnosis), 2 could not run at all.
#
# Usage:
#   scripts/05_serve_bench.sh --plan         # reads /health, sends no load
#   scripts/05_serve_bench.sh
#   SERVE_URL=http://gpu-box:8100 scripts/05_serve_bench.sh
#
# Owner: Ayesha.
set -euo pipefail

# Which interpreter runs the CLI: see _python.sh.
# shellcheck source=scripts/_python.sh
source "$(dirname "${BASH_SOURCE[0]}")/_python.sh"

EXAMPLES="${EXAMPLES:-data/processed/hotpotqa_dev.jsonl}"
CONFIG="${CONFIG:-configs/base.yaml}"
OUT_DIR="${OUT_DIR:-results/raw/serve}"
TABLE="${TABLE:-results/tables/serving.md}"
SERVE_URL="${SERVE_URL:-}"

PLAN=()
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  else
    ARGS+=("$arg")
  fi
done

# Word-split on purpose: CONFIG is a list of files, merged left to right.
# shellcheck disable=SC2206
CONFIGS=(${CONFIG})

URL=()
if [[ -n "${SERVE_URL}" ]]; then
  URL=(--url "${SERVE_URL}")
fi

"${PYTHON}" -m verisearch.cli "${PLAN[@]}" serve-bench \
  --examples "${EXAMPLES}" \
  --config "${CONFIGS[@]}" \
  --out "${OUT_DIR}" \
  --table "${TABLE}" \
  ${URL[@]+"${URL[@]}"} \
  ${ARGS[@]+"${ARGS[@]}"}
