# CNN Denoiser Project History and Workspace Reorganization

Date: 2026-04-28

## 1) What Was Built and Why

This repository evolved from single-run training scripts into a resilient, unattended model search pipeline designed to run for days, recover from crashes, resume interrupted trials, and find the best denoiser configuration by objective evaluation.

Primary objective achieved:
- Automatically train/evaluate many model configurations
- Resume after interruption/crash
- Keep GPU-only execution (no silent CPU fallback)
- Add fast screening to reduce total search time
- Narrow to best candidates and run full training

## 2) Search/Training Evolution (Chronological)

### Initial phase
- Added broad hyperparameter search space (28 named configurations)
- Added persistent search state and resumability
- Added result ranking by SDR improvement across SNR levels

### Stability hardening
- Disabled XLA/JIT in search training where mixed precision caused dtype graph failures
- Added FP16-safe behavior (dropout handling and dtype-safe layers)
- Added conservative batch sizing to avoid GPU event/file-descriptor exhaustion
- Added retriable memory-pressure error handling
- Added process-level restart flow for retriable failures

### Speed optimization
- Added dataset-fraction controls for quick viability testing
- Added quick-screen mode for short, cheap trial checks
- Added two-stage flow:
  1. Stage 1 quick-screen all candidates
  2. Stage 2 full-train only top-N promoted candidates

### Correctness and recovery improvements
- Fixed quick-screen monitor behavior to avoid bad epoch-1 lock-in during unstable validation
- Added NaN/Inf filtering in SDR evaluation paths
- Added checkpoint/backup restore callback for interrupted long runs
- Added hard GPU guard and wrapper stop behavior when GPU is unavailable

### Refinement phase
- Added refinement search mode
- Evolved refinement from static guessed combos to dynamic family-based permutations from actual top completed configs
- Added follow-up automation to:
  1. Wait for main run completion
  2. Quick-screen refinement variants
  3. Select best variant per winner family
  4. Full-train those finalists and compare

## 3) Final Search Winner

Refinement winner:
- `ref_combo_aggressive__plus_mask_linear`
- Search metric: +3.35 dB average SDR improvement (refinement comparison run)

Winning configuration parameters:
- encoder_filters: [64, 128, 256, 512]
- res_blocks: 3
- use_attention: true
- bottleneck: dilated
- dropout: 0.2
- first_kernel: 7
- mask_scale: 1.0
- mask_activation: linear
- learning_rate: 5e-4
- loss_weights: [0.3, 0.4, 0.3]

This configuration was promoted into the main search space as:
- `final_winner`

## 4) Final Full-Dataset Training Outcome

A fresh full-dataset run was executed for `final_winner` in:
- `models/final_training/`

Run summary from final log:
- Early stopped at epoch 48 (restored best epoch 38)
- Final recorded result: +3.13 dB average SDR improvement
- Per-SNR mean SDR improvements:
  - 0 dB: +7.78 dB
  - 5 dB: +6.07 dB
  - 10 dB: +3.43 dB
  - 15 dB: +0.30 dB
  - 20 dB: -1.94 dB

Saved best model:
- `models/final_training/final_winner/best.keras`

Convenience copy:
- `models/denoiser_final_winner.keras`

## 5) Cleanup Performed

Removed no-longer-needed search artifact folders after preserving the winner:
- `search_results/`
- `search_results_quick/`
- `search_results_auto/`
- `search_results_auto_quick/`
- `search_results_refine/`
- `search_results_refine_quick/`

Approximate space recovered during cleanup:
- ~35 GB

## 6) Workspace Reorganization (Current Structure)

To reduce root clutter and make intent clearer, files were reorganized into dedicated folders.

### New/organized folders
- `docs/`
  - Project documentation and run history
- `scripts/automation/`
  - Unattended and orchestration scripts
- `scripts/legacy/`
  - Older/numbered pipeline scripts and ad-hoc test utilities
- `notebooks/`
  - Jupyter notebooks
- `artifacts/audio/`
  - Generated audio outputs

### Moved files
- To `scripts/automation/`:
  - `run_search.sh`
  - `run_followup.sh`
  - `train_overnight.sh`
- To `scripts/legacy/`:
  - `1-gerar_tfrecord.py`
  - `1-gerar_tfrecord_v2.py`
  - `2-criar_modelo.py`
  - `2-criar_modelo_v2.py`
  - `3-usar_modelo.py`
  - `3-usar_modelo_v2.py`
  - `4-test_model.py`
  - `analyze_dataset.py`
  - `test_audio.py`
  - `test_audio-2.py`
- To `notebooks/`:
  - `NoteDenoiser_GuimsEdition.ipynb`
- To `artifacts/audio/`:
  - `enhanced_final_test.wav`

### Script compatibility updates after move
Automation scripts were updated to compute repository root from script location so they can run reliably from their new subfolder:
- `scripts/automation/run_search.sh`
- `scripts/automation/run_followup.sh`
- `scripts/automation/train_overnight.sh`

`train_overnight.sh` was also updated to call moved legacy scripts from `scripts/legacy/` and write evaluation audio to `artifacts/audio/`.

## 7) Recommended Operational Entry Points

Primary training/search entry points now:
- `python 5-hyperparameter_search.py ...`
- `scripts/automation/run_search.sh ...`
- `scripts/automation/run_followup.sh ...`
- `scripts/automation/train_overnight.sh ...`

For production-quality model usage, prefer:
- `models/final_training/final_winner/best.keras`
- or `models/denoiser_final_winner.keras`
