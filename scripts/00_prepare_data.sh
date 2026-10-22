#!/usr/bin/env bash
#
# Download the datasets, split them, and build the retrieval corpus.
# PROJECT.md section 7. The step before 01_build_index.sh, and the only one that
# needs the network.
#
# Needs the [data] extra:   uv sync --extra data
#
# Writes into OUT (default data/processed/):
#   <dataset>_<split>.jsonl   train/dev/test for HotpotQA and 2WikiMultihopQA,
#                             ood for Musique and Bamboogle
#   corpus.jsonl              what 01_build_index.sh embeds
#   leakage.json              near-duplicate questions that landed in two splits
#   manifest.json             provenance: seed, sources, counts, coverage
#
# Exit codes: 0 usable, 1 written but a supporting title has no document in the
# corpus (recall is capped - read the manifest), 2 could not run.
#
# Usage:
#   scripts/00_prepare_data.sh --plan            # names the downloads, no network
#   LIMIT=50 OUT=data/smoke scripts/00_prepare_data.sh   # smoke build, streamed
#   scripts/00_prepare_data.sh                   # the full pool
#
# Every decision is in verisearch/cli.py and verisearch/data/prepare.py; this
# file sets defaults and forwards arguments.
#
# Owner: Ayesha.
set -euo pipefail

# Which interpreter runs the CLI: see _python.sh.
# shellcheck source=scripts/_python.sh
source "$(dirname "${BASH_SOURCE[0]}")/_python.sh"

OUT="${OUT:-data/processed}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"

PLAN=()
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--plan" ]]; then
    PLAN=(--plan)
  else
    ARGS+=("$arg")
  fi
done

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

"${PYTHON}" -m verisearch.cli "${PLAN[@]}" prepare-data \
  --out "${OUT}" \
  --seed "${SEED}" \
  ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
  ${ARGS[@]+"${ARGS[@]}"}
