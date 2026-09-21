#!/bin/bash
# Run detailed statistics after training completes

set -e

cd /workspace/cnn_denoiser

TRAIN_LOG="/workspace/cnn_denoiser/logs/basic_unet_retrain_20260428.log"

echo "======================================"
echo "POST-TRAINING STATISTICS GENERATION"
echo "======================================"
echo "Started: $(date)"

echo ""
echo "Validating training log before statistics..."
if [ ! -f "$TRAIN_LOG" ]; then
    echo "ERROR: Training log not found at $TRAIN_LOG"
    exit 1
fi

if grep -Eq "Out of memory|ResourceExhaustedError|TRAINING ERROR" "$TRAIN_LOG"; then
    echo "ERROR: Training appears to have failed (OOM/ResourceExhausted)."
    echo "Refusing to generate statistics from a stale or failed checkpoint."
    exit 2
fi

LAST_EPOCH=$(grep -E '^Epoch [0-9]+/' "$TRAIN_LOG" | tail -1 || true)
echo "Last epoch marker: ${LAST_EPOCH:-none found}"

# Wait for best model to exist
echo ""
echo "Verifying best model checkpoint exists..."
MODEL_B="/workspace/cnn_denoiser/models/basic_unet_retrain_20260428-best.keras"
if [ ! -f "$MODEL_B" ]; then
    echo "ERROR: Best model not found at $MODEL_B"
    exit 1
fi
echo "✓ Found: $MODEL_B"

# Reference model
MODEL_A="/workspace/cnn_denoiser/models/best.keras"
if [ ! -f "$MODEL_A" ]; then
    echo "ERROR: Reference model not found at $MODEL_A"
    exit 1
fi
echo "✓ Found: $MODEL_A"

# Create output directory
OUT_DIR="logs/comparisons/detailed"
mkdir -p "$OUT_DIR"

# Run comparison
echo ""
echo "Running detailed statistical comparison..."
echo "Model A (Reference): $MODEL_A"
echo "Model B (Retrained):  $MODEL_B"
echo "Output Directory:     $OUT_DIR"
echo ""

python scripts/analysis/compare_models_detailed.py \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --max-samples 0 \
    --out-dir "$OUT_DIR"

# Find the latest report
LATEST_REPORT=$(ls -t "$OUT_DIR"/report_*.md 2>/dev/null | head -1)

if [ -z "$LATEST_REPORT" ]; then
    echo "ERROR: No report generated"
    exit 1
fi

echo ""
echo "✓ Report generated: $LATEST_REPORT"

# Copy to docs
echo ""
echo "Copying final report to docs/MODEL_COMPARISON_DETAILED_STATS.md..."
cp "$LATEST_REPORT" docs/MODEL_COMPARISON_DETAILED_STATS.md
echo "✓ Copied successfully"

echo ""
echo "======================================"
echo "COMPLETED: $(date)"
echo "======================================"
echo ""
echo "Final report available at:"
echo "  docs/MODEL_COMPARISON_DETAILED_STATS.md"
