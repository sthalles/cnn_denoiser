# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python virtual environment: `.venv/` (Python 3.12). Activate with `source .venv/bin/activate`.
- GPU: NVIDIA RTX 4070 (12GB VRAM). All training scripts enforce GPU-only execution and will exit with code 3 if no GPU is found.
- Mixed precision (FP16) is enabled globally via `setup_gpu()` in `config.py`.

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Verify config and STFT shapes
python config.py

# Generate TFRecords from dataset
python scripts/legacy/1-gerar_tfrecord_v2.py

# Train model (from repo root)
python scripts/legacy/2-criar_modelo_v2.py

# Run inference on a file
python scripts/legacy/3-usar_modelo_v2.py

# Hyperparameter search (resumable)
python 5-hyperparameter_search.py
python 5-hyperparameter_search.py --dry-run       # Preview configs
python 5-hyperparameter_search.py --results        # Show results so far
python 5-hyperparameter_search.py --epochs 30      # Limit epochs per trial

# Automated search with crash recovery (runs from repo root)
scripts/automation/run_search.sh                   # Default 50 epochs/trial
scripts/automation/run_search.sh 10 0.002 5        # Quick screen mode
scripts/automation/run_followup.sh                 # Refinement after main search
scripts/automation/train_overnight.sh [epochs] [batch_size]

# Stop a running search gracefully
touch search_results/STOP
```

## Architecture

### Core modules (repo root)

- **`config.py`** — Single source of truth for all parameters (`Config` class) plus shared signal processing utilities: `compute_stft`, `compute_istft`, `normalize_spectrogram`, `compute_complex_mask`, `apply_complex_mask`, `segment_audio`, `overlap_add_segments`, `compute_sdr`, `setup_gpu`. All other scripts import from here.
- **`dataset_v2.py`** — TFRecord parsing and `tf.data` pipeline construction. Also installs a `StderrFilter` to suppress CUDA/XLA noise.
- **`5-hyperparameter_search.py`** — Self-contained search loop with persistent state (JSON), two-stage quick-screen + full-train promotion, NaN/Inf guards, and process-level restart on retriable failures.
- **`super_utils.py`** — Legacy utility functions from the original v1 pipeline (8kHz, magnitude-only). Do not use in new code.

### Signal processing approach (v2)

Audio is processed as complex spectrograms. The model predicts a **Complex Ratio Mask (cRM)** applied as `Clean = Noisy * Mask`. Key parameters (all in `Config`):
- 16kHz sample rate, 512-point FFT → 257 frequency bins
- 128-sample hop (75% overlap), Hann window
- 2-second segments with 50% overlap between segments
- Global normalization via `max_mag` stored in `tfrecords_v2/norm_stats.npz`

### Model (v2)

U-Net encoder-decoder with:
- Encoder filters: `[64, 128, 256, 512]`
- Residual blocks + optional attention in bottleneck
- Dilated convolutions in bottleneck
- Two output channels: real and imaginary parts of the cRM
- Loss: weighted combination of MSE on mask, SDR loss, and spectral loss

### Winning configuration

The best-found configuration is saved at `models/denoiser_final_winner.keras` (also at `models/final_training/final_winner/best.keras`). It achieved +3.13 dB average SDR improvement. Key parameters:
- `encoder_filters: [64, 128, 256, 512]`, `res_blocks: 3`, `use_attention: true`
- `bottleneck: dilated`, `dropout: 0.2`, `first_kernel: 7`
- `mask_activation: linear`, `mask_scale: 1.0`, `learning_rate: 5e-4`
- `loss_weights: [0.3, 0.4, 0.3]` (MSE, SDR, spectral)

## Dataset

- Clean speech: `dataset/CleanSpeech_training/` (~383 WAV files, 16kHz)
- Noisy speech: `dataset/NoisySpeech_training/` (4 SNR levels per file: 0, 10, 20, 30 dB)
- Noisy filename format: `noisy{N}_SNRdb_{snr}_clnsp{N}.wav`
- Preprocessed data: `tfrecords_v2/` (norm stats at `tfrecords_v2/norm_stats.npz`)

## Key Invariants

- **Always use global normalization** (`norm_stats.npz`) — per-sample normalization breaks train/inference consistency.
- **STFT parameters must match exactly** between TFRecord generation, training, and inference.
- Scripts in `scripts/legacy/` compute `ROOT_DIR` relative to their location and add it to `sys.path` — run them from the repo root or they will resolve paths correctly regardless.
- XLA/JIT compilation is intentionally disabled in the search script: mixed precision + dropout + XLA causes dtype graph failures on this workload.
- The search is stopped gracefully by creating `<search_dir>/STOP`; Ctrl+C also saves state.

## Directories

| Path | Purpose |
|---|---|
| `scripts/legacy/` | Numbered pipeline scripts (v1 and v2) |
| `scripts/automation/` | Unattended training and search orchestration |
| `notebooks/` | Jupyter exploration notebooks |
| `docs/` | Model comparison stats and run history |
| `artifacts/audio/` | Generated audio outputs (e.g., denoised samples) |
| `logs_v2/` | Training session logs |
| `models/` | Saved Keras models |
| `tfrecords_v2/` | Preprocessed dataset shards |
