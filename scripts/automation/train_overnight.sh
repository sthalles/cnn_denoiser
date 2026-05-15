#!/bin/bash
#
# Audio Denoiser - Overnight Training Script
# 
# This script runs training with automatic restart on crash.
# Useful for overnight/unattended training sessions.
#
# Usage: ./train_overnight.sh [epochs] [batch_size]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
EPOCHS=${1:-100}
BATCH_SIZE=${2:-16}
MODEL_NAME="denoiser_v2"
MAX_RESTARTS=10
RESTART_DELAY=30  # seconds to wait before restart

# Log file
LOG_DIR="${ROOT_DIR}/logs_v2"
TRAINING_LOG="${LOG_DIR}/training_session_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$TRAINING_LOG"
}

log_header() {
    echo "" | tee -a "$TRAINING_LOG"
    echo "============================================================" | tee -a "$TRAINING_LOG"
    log "$1"
    echo "============================================================" | tee -a "$TRAINING_LOG"
}

# Unset legacy keras env var
unset TF_USE_LEGACY_KERAS

# Change to project directory
cd "${ROOT_DIR}"

log_header "OVERNIGHT TRAINING SESSION STARTED"
log "Target epochs:  $EPOCHS"
log "Batch size:     $BATCH_SIZE"
log "Model name:     $MODEL_NAME"
log "Max restarts:   $MAX_RESTARTS"
log "Log file:       $TRAINING_LOG"

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    log_header "Training Attempt $((restart_count + 1)) of $MAX_RESTARTS"
    
    # Run training
    python "${ROOT_DIR}/scripts/legacy/2-criar_modelo_v2.py" \
        --name "$MODEL_NAME" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee -a "$TRAINING_LOG"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_header "${GREEN}TRAINING COMPLETED SUCCESSFULLY${NC}"
        log "All $EPOCHS epochs completed!"
        
        # Run a quick evaluation
        log "Running evaluation on test file..."
        python "${ROOT_DIR}/scripts/legacy/3-usar_modelo_v2.py" file \
            "dataset/NoisySpeech_training/noisy100_SNRdb_0.0_clnsp100.wav" \
            "artifacts/audio/enhanced_final_test.wav" \
            --model "models/${MODEL_NAME}-best.keras" 2>&1 | tee -a "$TRAINING_LOG"
        
        exit 0
    fi
    
    # Check if it was a keyboard interrupt
    if [ $exit_code -eq 130 ]; then
        log_header "${YELLOW}TRAINING STOPPED BY USER${NC}"
        exit 130
    fi
    
    # Crash detected - prepare for restart
    restart_count=$((restart_count + 1))
    log_header "${RED}TRAINING CRASHED (exit code: $exit_code)${NC}"
    log "Restart attempt $restart_count of $MAX_RESTARTS"
    
    if [ $restart_count -lt $MAX_RESTARTS ]; then
        log "Waiting $RESTART_DELAY seconds before restart..."
        log "Checkpoints will be auto-loaded on restart."
        sleep $RESTART_DELAY
        
        # Check GPU status
        log "GPU Status:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv 2>&1 | tee -a "$TRAINING_LOG"
        
        # Clear any zombie processes
        pkill -9 -f "scripts/legacy/2-criar_modelo_v2.py" 2>/dev/null || true
        sleep 2
    fi
done

log_header "${RED}MAX RESTARTS EXCEEDED${NC}"
log "Training failed after $MAX_RESTARTS attempts."
log "Check logs at: $TRAINING_LOG"
exit 1
