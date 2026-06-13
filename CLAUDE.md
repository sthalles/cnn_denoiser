# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **GPU**: NVIDIA RTX 4070 (12 GB VRAM, compute capability 8.9, Ada Lovelace). All training enforces GPU-only execution and exits with code 3 if no GPU is found.
- **Mixed precision** (FP16) is enabled globally via `setup_gpu()` in `config.py`. Tensor cores accelerate FP16 matmuls/convs when channel dims are multiples of 8 (preferably 16).
- **Container**: `tensorflow-2026` Docker container (`nvcr.io/nvidia/tensorflow:25.02-tf2-py3`, TF 2.20 + cuDNN 9). Bind-mounts `/home/ayanami/cnn_denoiser` → `/workspace/cnn_denoiser`. Inside the container, the user maps to `ubuntu` (host `ayanami`); when writing files via the container, `chown -R ubuntu:ubuntu <path>` keeps host-side permissions sane.
- Host venv: `.venv/` (Python 3.12) exists for editor tooling but **training and evaluation should run inside the container** for consistent CUDA/cuDNN versions.
- The container has `TF_USE_LEGACY_KERAS=1` set in its image env. Strip it for current scripts: `env -u TF_USE_LEGACY_KERAS python ...` (training inherits cleanly because `setup_gpu()` calls `tf.keras.mixed_precision.set_global_policy` before any Keras object is built; evaluation must explicitly unset it).

## Common Commands

```bash
# Regenerate dataset (uses dataset/clean_train and dataset/noise_train as sources)
docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  python scripts/data/generate_dataset.py \
    --clean-source dataset/clean_train \
    --noise-source dataset/noise_train \
    --out-dir dataset \
    --noise-types AirConditioner CopyMachine Typing VacuumCleaner WasherDryer Washing \
    --snrs 0 5 10 15 \
    --total-hours 10 --duration 10 --sample-rate 16000 \
    --val-fraction 0.1 --seed 42 --clean-first"

# Build TFRecords (parallel per-worker shards; ~5 min for 10 h dataset)
docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  python scripts/data/build_tfrecords.py \
    --clean-dir dataset/CleanSpeech_training \
    --noisy-dir dataset/NoisySpeech_training \
    --manifest dataset/dataset_manifest.json \
    --out-dir tfrecords_park \
    --samples-per-shard 1024 --workers 16 --segment-overlap 0.0"

# Train any of the three models (--jit enables XLA fusion, recommended for conformer)
docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && \
  python scripts/training/train.py \
    --model {unet|park|conformer} \
    --tfrecord-root tfrecords_park \
    --out-dir models/runs/<run-name> \
    --epochs 30 --batch-size 16 --learning-rate 5e-4 --jit"

# Evaluate a trained model on the val split (SDR per input SNR; writes audio samples)
docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && env -u TF_USE_LEGACY_KERAS \
  python scripts/evaluation/evaluate.py \
    --model {unet|park|conformer} \
    --weights models/runs/<run-name>/best.keras \
    --tfrecord-root tfrecords_park \
    --clean-dir dataset/CleanSpeech_training \
    --noisy-dir dataset/NoisySpeech_training \
    --manifest dataset/dataset_manifest.json \
    --out-dir models/runs/<run-name>/eval \
    --max-files 200 --n-audio-samples 8"

# Cross-model comparison report (any number of runs)
python scripts/evaluation/compare.py \
  --eval-dir unet=models/runs/unet_crm_park_data/eval \
  --eval-dir park=models/runs/park_rced_park_data/eval \
  --eval-dir conformer=models/runs/conformer_crn_park_data/eval \
  --report-out docs/MODEL_COMPARISON_PARK_REBUILD.md

# Full end-to-end orchestrator (trains both U-Net and Park, then evaluates + compares)
scripts/automation/run_park_rebuild.sh
```

## Architecture

### Core modules (repo root)

- **`config.py`** — Single source of truth for STFT/segment/training constants (`Config` class) plus signal processing utilities: `compute_stft`, `compute_istft`, `normalize_spectrogram`, `compute_complex_mask`, `apply_complex_mask`, `segment_audio`, `overlap_add_segments`, `compute_sdr`, `setup_gpu`.
- **`dataset_v2.py`** — Legacy v2 TFRecord parsing (kept for backward compatibility; **prefer `scripts/data/tfrecord_io.py` for new work**).
- **`5-hyperparameter_search.py`** — Legacy search loop for the original U-Net winner discovery. Still functional but superseded by the current per-model training scripts.

### Pipeline modules (scripts/)

- **`scripts/data/generate_dataset.py`** — Replaces the original `dataset/noisyspeech_synthesizer.py`. Guarantees exact clip duration, resamples to target SR, splits speakers between train/val before generation, mixes at exact SNR targets, and saves a `dataset_manifest.json` so downstream stages preserve the speaker split.
- **`scripts/data/build_tfrecords.py`** — Each worker writes its **own** shards directly (no large objects cross the process boundary). Defaults to no segment overlap. Stores noisy and clean real/imag STFTs so both cRM and magnitude-domain targets can be derived at training time.
- **`scripts/data/tfrecord_io.py`** — Fast pipeline (FixedLenFeature, parallel interleave; benchmarked at ~80 batches/sec on a 4070 / Ryzen 5900x). Two factories: `make_crm_dataset` (for U-Net and Conformer) and `make_mag_dataset` (for Park R-CED).
- **`scripts/models/unet_crm.py`** — Re-implementation of the v2 winner U-Net (encoder filters `[64, 128, 256, 512]`, dilated bottleneck at 512 ch, res blocks, dropout 0.2, linear 2-channel output for cRM). ~36 M params.
- **`scripts/models/park_rced.py`** — Park & Lee (2017) R-CED: 16 conv layers, redundant filter pyramid (12-32-12), mirror skip connections, magnitude-domain target, single output channel. ~350 K params.
- **`scripts/models/conformer_crn.py`** — Tensor-core-aligned Conformer-CRN: strided conv encoder (32-64-128-256, freq downsampled), 4 Conformer blocks in the bottleneck (d_model=256, 4 heads, head_dim=64), transpose conv decoder with skips, cRM output. Input freq dim padded 257→264 so every tensor-core matmul has dims that are multiples of 8. ~12 M params.
- **`scripts/training/train.py`** — Unified entry for all three models. `--jit` enables XLA. Saves `best.keras`, `last.keras`, `training.csv`, `history.json`, `training_config.json`.
- **`scripts/evaluation/evaluate.py`** — Per-SNR SDR evaluation. Saves a few denoised WAV triplets to `<eval-dir>/audio/`. Conformer is rebuilt from code and weights are loaded separately (Lambda closures don't round-trip through `.keras` serialization).
- **`scripts/evaluation/compare.py`** — Accepts any number of `--eval-dir name=path` entries and writes a Markdown report with overall, per-SNR, and pairwise paired comparisons.

### Signal processing approach

Audio is processed as complex spectrograms.
- cRM target (U-Net, Conformer): model predicts a 2-channel **Complex Ratio Mask** `M` such that `Clean = Noisy * M`.
- Magnitude target (Park): model predicts `log1p(|clean|)`; reconstruction uses noisy phase.

Key parameters (all in `Config`):
- 16 kHz sample rate, 512-point FFT → 257 frequency bins
- 128-sample hop (75% overlap inside STFT), Hann window
- 2-second segments (251 frames). **Training pipeline uses no overlap between segments** (the legacy v2 pipeline used 50% overlap; the new builder lets you choose via `--segment-overlap`).
- Global normalization via `max_mag` stored in `<tfrecord-root>/dataset_info.json` and `norm_stats.npz`.

## Three current models

After regenerating the dataset on the `park-style-rebuild` branch and training the three models below in the `tensorflow-2026` container, the results on a random 200-file sample of the val split are:

| Model | Params | Train time | Mean ΔSDR | 0 dB | 5 dB | 10 dB | 15 dB | Weights |
|---|---|---|---|---|---|---|---|---|
| U-Net cRM | 36 M | ~22 h (32 ep) | **+11.22 dB** | +14.35 | +12.76 | +10.44 | +7.62 | `models/runs/unet_crm_park_data/best.keras` |
| Park R-CED FCN | 350 K | ~14 h (40 ep) | +8.55 dB | +11.70 | +9.91 | +7.84 | +5.03 | `models/runs/park_rced_park_data/best.keras` |
| Conformer-CRN | 12 M | **~5.3 h (30 ep)** | **+11.26 dB** | +14.43 | +12.81 | +10.44 | +7.63 | `models/runs/conformer_crn_park_data/best.keras` |

Conformer-CRN ties U-Net at roughly 1/3 the params and 1/4 the training wall-clock — the dimension choices make it the most tensor-core-efficient of the three. All three exceed the pre-rebuild +3.13 dB baseline produced by `models/best.keras`. Full report: [docs/MODEL_COMPARISON_PARK_REBUILD.md](docs/MODEL_COMPARISON_PARK_REBUILD.md).

## Dataset (regenerated on `park-style-rebuild`)

Generated by `scripts/data/generate_dataset.py`; do **not** use the legacy `dataset/noisyspeech_synthesizer.py` (variable output length, no resampling, no speaker split).

- Source: `dataset/clean_train/` (VCTK, 59 speakers, ~23 k 16 kHz mono WAVs) and `dataset/noise_train/` (selected subset, see below)
- Curated stationary noise set: AirConditioner, CopyMachine, Typing, VacuumCleaner, WasherDryer, Washing
- SNR levels: {0, 5, 10, 15} dB; ~10 h total (3 600 unique 10 s clean clips × 4 SNRs = 14 400 noisy files)
- Train/val split is **by speaker** before generation (53 train / 6 val). The split is preserved into `dataset/dataset_manifest.json` and consumed by `build_tfrecords.py` so the same speakers stay on the same side end-to-end.
- Generated files live in `dataset/CleanSpeech_training/` and `dataset/NoisySpeech_training/`; filename convention is unchanged: `clnsp{N}.wav` / `noisy{N}_SNRdb_{snr}_clnsp{N}.wav`.
- TFRecord output: `tfrecords_park/` (~70 GB, 64 train shards + 16 val shards, each ~1 GB, all gitignored).

## Key Invariants

- **Always run training/evaluation inside the `tensorflow-2026` container.** The host venv lacks current CUDA/cuDNN.
- **Use global normalization** (`max_mag` from `dataset_info.json`) — per-sample normalization breaks train/inference consistency.
- **STFT parameters must match exactly** between TFRecord generation, training, and inference.
- **Speaker split is set when generating the dataset** — neither `build_tfrecords.py` nor the trainers re-split. Regenerate the dataset if you need a different split.
- **Channel dims should be multiples of 8 (preferably 16)** so tensor-core paths fire under FP16. The Conformer model is designed around this; if you swap in a new architecture, preserve this property.
- **Conformer's `.keras` file does not round-trip Lambda closures** — load weights into a freshly built model (see `evaluate.py` for the pattern). The U-Net loads fine with `safe_mode=False`.
- **XLA (`--jit`) is safe for the Conformer** with FP16; for the U-Net the legacy advice (no XLA with mixed-precision + dropout) still applies. Park uses no JIT.

## Directories

| Path | Purpose |
|---|---|
| `scripts/data/` | Dataset generation, TFRecord build, data IO |
| `scripts/models/` | Model definitions (`unet_crm.py`, `park_rced.py`, `conformer_crn.py`) |
| `scripts/training/` | `train.py` — unified training entry point |
| `scripts/evaluation/` | `evaluate.py`, `compare.py` |
| `scripts/automation/` | Orchestration shells (`run_park_rebuild.sh`, legacy search runners) |
| `scripts/legacy/` | Pre-rebuild pipeline scripts (do not extend; kept for reference) |
| `docs/` | Project history, agent instructions, model comparison reports, `park_paper.pdf` |
| `notebooks/` | Jupyter exploration notebooks |
| `artifacts/audio/` | Generated audio outputs from older runs |
| `models/runs/<name>/` | Output of `train.py`: `best.keras`, `last.keras`, `training.csv`, `history.json`, plus `eval/` |
| `models/` | Older saved models including `models/best.keras` (pre-rebuild +3.13 dB baseline) |
| `tfrecords_park/` | Current preprocessed dataset shards (gitignored) |
| `archive/` | Stale artifacts moved during the rebuild (gitignored) |

## Known traps

- Forgetting `env -u TF_USE_LEGACY_KERAS` on evaluate.py → `ImportError: Keras cannot be imported`.
- Loading the Conformer with `tf.keras.models.load_model(..., safe_mode=False)` → `TypeError: arg 4 (defaults) must be None or tuple` because of a Lambda closure on a shape int. Rebuild the architecture and `load_weights` instead.
- Running `build_tfrecords.py` with `ProcessPoolExecutor.map` that returns results to the main process → unbounded result buffering, OOM-killed at ~18 GB RSS. The current implementation has each worker write its own shards.
- Per-process tqdm output buffers heavily when wrapped in `| tee | tail`; monitor progress via the CSV file or by counting files in `<out-dir>/train/`.
