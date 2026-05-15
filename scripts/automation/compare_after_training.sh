#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?run id required}"
TRAIN_PID="${2:?train pid required}"
SAMPLES="${3:-20}"

mkdir -p logs/comparisons

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 120
done

TS="$(date +%Y%m%d_%H%M%S)"
NEW_MODEL="models/${RUN_ID}-best.keras"
BASE_MODEL="models/best.keras"
NEW_EVAL="logs/comparisons/${RUN_ID}_eval_${TS}.txt"
BASE_EVAL="logs/comparisons/best_model_eval_${TS}.txt"
OUT_JSON="logs/comparisons/comparison_latest.json"

if [[ -f "$NEW_MODEL" ]]; then
  python scripts/legacy/3-usar_modelo_v2.py eval --model "$NEW_MODEL" --samples "$SAMPLES" > "$NEW_EVAL" 2>&1
fi

python scripts/legacy/3-usar_modelo_v2.py eval --model "$BASE_MODEL" --samples "$SAMPLES" > "$BASE_EVAL" 2>&1

python - <<'PY'
import glob
import json
import os
import re

root = '/workspace/cnn_denoiser/logs/comparisons'
new_files = sorted(glob.glob(os.path.join(root, 'basic_unet_retrain_20260428_eval_*.txt')))
base_files = sorted(glob.glob(os.path.join(root, 'best_model_eval_*.txt')))
out_path = os.path.join(root, 'comparison_latest.json')


def parse_score(path):
    if not path or not os.path.exists(path):
        return None
    text = open(path, 'r').read()
    m = re.search(r'Average SDR improvement:\s*([+-]?[0-9]*\.?[0-9]+)\s*dB', text)
    return float(m.group(1)) if m else None

new_file = new_files[-1] if new_files else None
base_file = base_files[-1] if base_files else None
new_score = parse_score(new_file)
base_score = parse_score(base_file)

payload = {
    'new_eval_file': new_file,
    'base_eval_file': base_file,
    'new_avg_sdr_improvement_db': new_score,
    'base_avg_sdr_improvement_db': base_score,
    'delta_new_minus_base_db': (new_score - base_score) if (new_score is not None and base_score is not None) else None,
}

with open(out_path, 'w') as f:
    json.dump(payload, f, indent=2)

print(json.dumps(payload, indent=2))
PY
