#!/usr/bin/env bash
# =============================================================================
#  run_followup.sh  —  Wait for the current search to finish, then run
#                      refinement quick-screen (winner-family permutations),
#                      then full-train the best variant of each top-3 family.
#
#  Usage:
#    ./run_followup.sh [WAIT_PID]
#
#  If WAIT_PID is given, waits for that process to finish first.
#  Then runs:
#    Stage 1: quick-screen the dynamically generated refinement families
#    Stage 2: pick the best quick-screened variant in each family
#    Stage 3: full-train those 3 finalists and compare them
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

WAIT_PID="${1:-}"
REFINE_DIR="search_results_refine"
SCREEN_DIR="search_results_refine_quick"
REFINE_LOG="search_results_refine/run_$(date +%Y%m%d_%H%M%S).log"
SOURCE_STATE="search_results_auto/search_state.json"

# ── Full-train settings ──
FULL_EPOCHS=50
FULL_FRACTION=1.0
FULL_EVAL=15
PROMOTE_TOP=3

# ── Quick-screen settings (slightly more data than initial screen,
#    since we only have 8 configs so can afford a bit more per trial) ──
SCREEN_FRACTION=0.05
SCREEN_EPOCHS=10
SCREEN_EVAL=10

echo "========================================================"
echo "  REFINEMENT FOLLOW-UP SCRIPT"
echo "  Finalists to full-train: ${PROMOTE_TOP}"
echo "  Full epochs: ${FULL_EPOCHS}"
echo "  Screen: fraction=${SCREEN_FRACTION}, epochs=${SCREEN_EPOCHS}, eval=${SCREEN_EVAL}"
echo "  Output: ${REFINE_DIR}"
echo "========================================================"

# ── Wait for the initial search to finish ──
if [[ -n "${WAIT_PID}" ]]; then
    echo ""
    echo "  Waiting for PID ${WAIT_PID} to finish..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        sleep 60
        echo "  [$(date '+%H:%M:%S')] Still waiting for PID ${WAIT_PID}..."
    done
    echo "  [$(date '+%H:%M:%S')] PID ${WAIT_PID} finished."
fi

# ── Show final results of the initial search ──
echo ""
echo "========================================================"
echo "  INITIAL SEARCH FINAL RESULTS"
echo "========================================================"
python "${ROOT_DIR}/5-hyperparameter_search.py" --results --search-dir search_results_auto || true

# ── Verify GPU is available before starting refinement ──
GPU_COUNT=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null || echo 0)
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo ""
    echo "ERROR: No GPU detected. Aborting refinement. Re-run once GPU is available:"
    echo "  ./run_followup.sh"
    exit 3
fi
echo "  GPU confirmed (${GPU_COUNT} device(s)). Starting refinement..."

mkdir -p "${REFINE_DIR}"
mkdir -p "${SCREEN_DIR}"

# ── Run quick-screen refinement search ──
echo ""
echo "========================================================"
echo "  STARTING REFINEMENT QUICK SCREEN"
echo "  Logs: ${REFINE_LOG}"
echo "========================================================"

python "${ROOT_DIR}/5-hyperparameter_search.py" \
    --refinement \
    --refine-from-state "${SOURCE_STATE}" \
    --epochs "${SCREEN_EPOCHS}" \
    --dataset-fraction "${SCREEN_FRACTION}" \
    --eval-samples "${SCREEN_EVAL}" \
    --search-dir "${SCREEN_DIR}" \
    2>&1 | tee -a "${REFINE_LOG}"

RC1=${PIPESTATUS[0]}
if [[ ${RC1} -ne 0 ]]; then
    echo ""
    echo "Quick-screen refinement failed with exit code ${RC1}."
    exit ${RC1}
fi

# ── Pick one best refinement variant per top-3 family ──
echo ""
echo "========================================================"
echo "  SELECTING FAMILY FINALISTS"
echo "========================================================"

python - <<'PY' | tee -a "${REFINE_LOG}"
import json
import math
import os

state_path = 'search_results_refine_quick/search_state.json'
out_path = 'search_results_refine/selected_families.json'

with open(state_path, 'r') as f:
    state = json.load(f)

groups = {}
for name, trial in state.get('trials', {}).items():
    if not name.startswith('ref_') or '__' not in name:
        continue
    ev = trial.get('eval_results') or {}
    score = ev.get('overall_mean_improvement')
    if score is None:
        continue
    score = float(score)
    if not math.isfinite(score):
        continue
    family = name.split('__', 1)[0]
    groups.setdefault(family, []).append((score, name))

selected = []
summary = {}
for family in sorted(groups):
    best_score, best_name = max(groups[family], key=lambda item: item[0])
    selected.append(best_name)
    summary[family] = {
        'selected_name': best_name,
        'score': best_score,
    }
    print(f"  {family:<24} -> {best_name:<40} {best_score:+.2f} dB")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({'selected': selected, 'summary': summary}, f, indent=2)
PY

SELECTED_CSV=$(python - <<'PY'
import json
with open('search_results_refine/selected_families.json', 'r') as f:
    data = json.load(f)
print(','.join(data['selected']))
PY
)

echo ""
echo "Selected finalists: ${SELECTED_CSV}" | tee -a "${REFINE_LOG}"

# ── Full-train one finalist per family ──
echo ""
echo "========================================================"
echo "  STARTING FULL TRAINING OF FAMILY FINALISTS"
echo "========================================================"

python "${ROOT_DIR}/5-hyperparameter_search.py" \
    --refinement \
    --refine-from-state "${SOURCE_STATE}" \
    --epochs "${FULL_EPOCHS}" \
    --dataset-fraction "${FULL_FRACTION}" \
    --eval-samples "${FULL_EVAL}" \
    --search-dir "${REFINE_DIR}" \
    --only-configs "${SELECTED_CSV}" \
    --screen-fraction "${SCREEN_FRACTION}" \
    --screen-epochs "${SCREEN_EPOCHS}" \
    --screen-eval-samples "${SCREEN_EVAL}" \
    2>&1 | tee -a "${REFINE_LOG}"

RC=${PIPESTATUS[0]}

# ── Show refinement results ──
echo ""
echo "========================================================"
echo "  REFINEMENT SEARCH COMPLETE  (exit code: ${RC})"
echo "========================================================"
python "${ROOT_DIR}/5-hyperparameter_search.py" --results --search-dir "${REFINE_DIR}" || true

echo ""
echo "========================================================"
echo "  FINAL COMPARISON"
echo "========================================================"
echo ""
echo "── Initial search top 4 ──"
python "${ROOT_DIR}/5-hyperparameter_search.py" --results --search-dir search_results_auto 2>/dev/null | grep -E '^\s+[0-9]' || true
echo ""
echo "── Refinement top 3 ──"
python "${ROOT_DIR}/5-hyperparameter_search.py" --results --search-dir "${REFINE_DIR}" 2>/dev/null | grep -E '^\s+[0-9]' || true

exit ${RC}
