#!/usr/bin/env bash
#
# End-to-end orchestrator for the Park-style rebuild. Assumes:
#   - tensorflow-2026 docker container is running and bind-mounts /workspace/cnn_denoiser
#   - dataset/CleanSpeech_training and dataset/NoisySpeech_training already generated
#   - tfrecords_park/ already built
#
# Trains the U-Net cRM model and the Park R-CED model on the new dataset,
# then evaluates each on the validation split, then writes a comparison report.
#
# Usage from repo root:
#   scripts/automation/run_park_rebuild.sh
#
# Override knobs via env vars:
#   UNET_EPOCHS=40 PARK_EPOCHS=60 UNET_BATCH=32 PARK_BATCH=128 ./run_park_rebuild.sh

set -euo pipefail

UNET_EPOCHS=${UNET_EPOCHS:-40}
UNET_BATCH=${UNET_BATCH:-32}
UNET_LR=${UNET_LR:-5e-4}

PARK_EPOCHS=${PARK_EPOCHS:-60}
PARK_BATCH=${PARK_BATCH:-128}
PARK_LR=${PARK_LR:-1e-3}

EVAL_MAX_FILES=${EVAL_MAX_FILES:-300}

CONTAINER=${CONTAINER:-tensorflow-2026}
REPO=/workspace/cnn_denoiser

UNET_DIR=models/runs/unet_crm_park_data
PARK_DIR=models/runs/park_rced_park_data
REPORT=docs/MODEL_COMPARISON_PARK_VS_UNET.md

dexec() { docker exec "$CONTAINER" bash -c "cd $REPO && $1"; }

echo "== Training cRM U-Net =="
dexec "python scripts/training/train.py \
  --model unet \
  --tfrecord-root tfrecords_park \
  --out-dir $UNET_DIR \
  --epochs $UNET_EPOCHS \
  --batch-size $UNET_BATCH \
  --learning-rate $UNET_LR \
  2>&1 | tee $UNET_DIR/train.log"

echo "== Training Park R-CED FCN =="
dexec "python scripts/training/train.py \
  --model park \
  --tfrecord-root tfrecords_park \
  --out-dir $PARK_DIR \
  --epochs $PARK_EPOCHS \
  --batch-size $PARK_BATCH \
  --learning-rate $PARK_LR \
  2>&1 | tee $PARK_DIR/train.log"

echo "== Evaluating cRM U-Net =="
dexec "python scripts/evaluation/evaluate.py \
  --model unet \
  --weights $UNET_DIR/best.keras \
  --tfrecord-root tfrecords_park \
  --clean-dir dataset/CleanSpeech_training \
  --noisy-dir dataset/NoisySpeech_training \
  --manifest dataset/dataset_manifest.json \
  --out-dir $UNET_DIR/eval \
  --max-files $EVAL_MAX_FILES \
  --n-audio-samples 6"

echo "== Evaluating Park R-CED FCN =="
dexec "python scripts/evaluation/evaluate.py \
  --model park \
  --weights $PARK_DIR/best.keras \
  --tfrecord-root tfrecords_park \
  --clean-dir dataset/CleanSpeech_training \
  --noisy-dir dataset/NoisySpeech_training \
  --manifest dataset/dataset_manifest.json \
  --out-dir $PARK_DIR/eval \
  --max-files $EVAL_MAX_FILES \
  --n-audio-samples 6"

echo "== Writing comparison report =="
dexec "python scripts/evaluation/compare.py \
  --unet-eval-dir $UNET_DIR/eval \
  --park-eval-dir $PARK_DIR/eval \
  --report-out $REPORT"

echo "== DONE. Report at $REPORT =="
