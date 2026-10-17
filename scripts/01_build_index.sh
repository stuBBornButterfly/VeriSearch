#!/usr/bin/env bash
#
# Build the retrieval corpus index. PROJECT.md section 4.
#
# Needs the [retrieval] extra:   uv pip install -e '.[retrieval]'
#
# Usage:
#   scripts/01_build_index.sh                       # defaults below
#   scripts/01_build_index.sh --plan                # print the plan, write nothing
#   CORPUS=data/corpus.jsonl OUT=data/index/hotpot scripts/01_build_index.sh
#
# Every decision this makes lives in verisearch/cli.py, which mypy and pytest
# both reach. This file sets defaults and forwards arguments; if you find
# yourself adding an `if` here, it belongs in the CLI instead.
#
# Owner: Ayesha.
set -euo pipefail

CORPUS="${CORPUS:-data/processed/corpus.jsonl}"
OUT="${OUT:-data/index/corpus}"
MODEL="${MODEL:-BAAI/bge-small-en-v1.5}"
DIM="${DIM:-384}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"

# --plan is a global flag and has to precede the subcommand, so it is pulled out
# of "$@" rather than appended.
PLAN=()
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  else
    ARGS+=("$arg")
  fi
done

echo "building index from ${CORPUS} into ${OUT}"

python -m verisearch.cli "${PLAN[@]}" build-index \
  --corpus "${CORPUS}" \
  --out "${OUT}" \
  --model "${MODEL}" \
  --dim "${DIM}" \
  --index-type "${INDEX_TYPE}" \
  ${ARGS[@]+"${ARGS[@]}"}

# The fingerprint printed above is what every eval run must report. If it
# changes, every report that cited the old one is measuring a different
# retriever - see eval/baselines.py check_comparable.
