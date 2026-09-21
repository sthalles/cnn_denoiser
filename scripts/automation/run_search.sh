#!/bin/bash
# =============================================================================
# Hyperparameter Search Wrapper - Run for days with crash recovery
# =============================================================================
#
# Usage:
#   ./run_search.sh              # Default 50 epochs per trial
#   ./run_search.sh 30           # 30 epochs per trial
#   ./run_search.sh 10 0.002 5   # Fast screen: ~1/500 data, 5 eval samples/SNR
#   ./run_search.sh 10 0.002 5 search_results_quick
#   ./run_search.sh 50 1.0 15 search_results 5 two_stage
#   ./run_search.sh 50 1.0 15 search_results 5 two_stage 0.02 6 5
#
# Stop gracefully:
#   touch search_results/STOP    # Finishes current trial then exits
#   Ctrl+C                       # Saves state immediately
#
# Resume:
#   ./run_search.sh              # Picks up where it left off
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

EPOCHS=${1:-50}
DATASET_FRACTION=${2:-1.0}
EVAL_SAMPLES=${3:-15}
SEARCH_DIR=${4:-search_results}
PROMOTE_TOP=${5:-5}
MODE=${6:-single}
SCREEN_FRACTION=${7:-0.02}
SCREEN_EPOCHS=${8:-6}
SCREEN_EVAL_SAMPLES=${9:-5}
MAX_RESTARTS=50
RESTART_DELAY=30
LOG_DIR="$SEARCH_DIR"
LOG_FILE="${LOG_DIR}/search_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  Hyperparameter Search Runner"
echo "=============================================="
echo "  Epochs per trial:  $EPOCHS"
echo "  Dataset fraction:  $DATASET_FRACTION"
echo "  Eval samples/SNR:  $EVAL_SAMPLES"
echo "  Search dir:        $SEARCH_DIR"
echo "  Mode:              $MODE"
echo "  Promote top-N:     $PROMOTE_TOP"
echo "  Screen fraction:   $SCREEN_FRACTION"
echo "  Screen epochs:     $SCREEN_EPOCHS"
echo "  Screen eval/SNR:   $SCREEN_EVAL_SAMPLES"
echo "  Max restarts:      $MAX_RESTARTS"
echo "  Log file:          $LOG_FILE"
echo "  Stop file:         $LOG_DIR/STOP"
echo "=============================================="

# Remove any leftover stop file
rm -f "$LOG_DIR/STOP"

attempt=0
while [ $attempt -lt $MAX_RESTARTS ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "[$(date)] Attempt $attempt/$MAX_RESTARTS"

    # Run the search
    if [ "$MODE" = "two_stage" ]; then
        python "${ROOT_DIR}/5-hyperparameter_search.py" \
            --two-stage \
            --epochs "$EPOCHS" \
            --dataset-fraction "$DATASET_FRACTION" \
            --eval-samples "$EVAL_SAMPLES" \
            --promote-top "$PROMOTE_TOP" \
            --screen-fraction "$SCREEN_FRACTION" \
            --screen-epochs "$SCREEN_EPOCHS" \
            --screen-eval-samples "$SCREEN_EVAL_SAMPLES" \
            --search-dir "$SEARCH_DIR" \
            --screen-dir "${SEARCH_DIR}_quick" \
            2>&1 | tee -a "$LOG_FILE"
    else
        python "${ROOT_DIR}/5-hyperparameter_search.py" \
            --epochs "$EPOCHS" \
            --dataset-fraction "$DATASET_FRACTION" \
            --eval-samples "$EVAL_SAMPLES" \
            --search-dir "$SEARCH_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi
    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Search completed successfully!"
        break
    fi

    # Exit code 3: no GPU detected (guard in python script)
    if [ $exit_code -eq 3 ]; then
        echo "[$(date)] No GPU available. Stopping without retries."
        break
    fi

    # Check for stop file
    if [ -f "$LOG_DIR/STOP" ]; then
        echo "[$(date)] Stop file detected. Exiting."
        break
    fi

    echo "[$(date)] Search exited with code $exit_code. Restarting in ${RESTART_DELAY}s..."
    sleep $RESTART_DELAY
done

echo ""
echo "=============================================="
echo "  Search finished. Showing results:"
echo "=============================================="
python "${ROOT_DIR}/5-hyperparameter_search.py" --results --search-dir "$SEARCH_DIR"

echo ""
echo "Full log: $LOG_FILE"
