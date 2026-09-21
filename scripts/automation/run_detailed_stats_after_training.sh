#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?run id required}"
TRAIN_PID="${2:?train pid required}"
REF_MODEL="${3:-models/best.keras}"
SAMPLES="${4:-0}"

mkdir -p logs/comparisons/detailed
LOG_PATH="logs/comparisons/detailed/watch_${RUN_ID}.log"

{
  echo "[$(date)] Waiting for train PID ${TRAIN_PID}..."
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 180
    echo "[$(date)] train still running..."
  done

  MODEL_B="models/${RUN_ID}-best.keras"
  if [[ ! -f "$MODEL_B" ]]; then
    echo "[$(date)] ERROR: retrained model not found at $MODEL_B"
    exit 2
  fi

  echo "[$(date)] Training finished. Running detailed paired stats..."
  python scripts/analysis/compare_models_detailed.py \
    --model-a "$REF_MODEL" \
    --model-b "$MODEL_B" \
    --label-a "best_model" \
    --label-b "$RUN_ID" \
    --max-samples "$SAMPLES" \
    --out-dir logs/comparisons/detailed

  LATEST_MD="$(ls -t logs/comparisons/detailed/report_*.md | head -1)"
  cp "$LATEST_MD" docs/MODEL_COMPARISON_DETAILED_STATS.md
  echo "[$(date)] Copied latest report to docs/MODEL_COMPARISON_DETAILED_STATS.md"
} >> "$LOG_PATH" 2>&1
