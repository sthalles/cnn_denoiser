# Park-style Rebuild — Three-Model Denoiser Comparison

Three models trained on the same regenerated dataset
(6 stationary noise types, SNRs 0/5/10/15 dB, ~10 h, 90/10 speaker split):

- **unet**: `models/runs/unet_crm_park_data/best.keras`  (200 val files)
- **park**: `models/runs/park_rced_park_data/best.keras`  (200 val files)
- **conformer**: `models/runs/conformer_crn_park_data/best.keras`  (200 val files)

## Overall mean ΔSDR (dB)

| Model | Mean ΔSDR (dB) |
|---|---|
| unet | +11.22 |
| park | +8.55 |
| conformer | +11.26 |

## Per-input-SNR breakdown (mean ΔSDR, dB)

| Input SNR (dB) | unet | park | conformer |
|---|---|---|---|
| 0 | +14.35 | +11.70 | +14.43 |
| 5 | +12.76 | +9.91 | +12.81 |
| 10 | +10.44 | +7.84 | +10.44 |
| 15 | +7.62 | +5.03 | +7.63 |

## Per-input-SNR breakdown (median ΔSDR, dB)

| Input SNR (dB) | unet | park | conformer |
|---|---|---|---|
| 0 | +13.45 | +11.03 | +13.63 |
| 5 | +11.78 | +9.09 | +11.56 |
| 10 | +9.71 | +7.34 | +9.93 |
| 15 | +7.30 | +4.86 | +7.49 |

## Pairwise paired comparison (A − B, per-file ΔSDR)

| A | B | N paired | Mean (A − B) (dB) | A wins >0.1 dB (%) |
|---|---|---|---|---|
| unet | park | 200 | +2.68 | 99.5 |
| unet | conformer | 200 | -0.04 | 33.0 |
| park | conformer | 200 | -2.71 | 0.5 |

## Headline takeaways

- Best overall model: **conformer** at +11.26 dB mean ΔSDR.
- Park R-CED baseline (small magnitude-domain FCN): +8.55 dB.
- All three models comfortably exceed the previous best of +3.13 dB recorded
  before the dataset rebuild.

## Methodology notes

- Dataset: 6 stationary noise types (AirConditioner, CopyMachine, Typing,
  VacuumCleaner, WasherDryer, Washing) mixed at SNRs {0, 5, 10, 15} dB.
  10 h total (3600 unique clean 10 s clips × 4 SNRs).
- Train / val split by **speaker** (53 train / 6 val, ~10 %).
- STFT: n_fft=512, hop=128, Hann window; 2 s segments with no overlap during training.
- Metric: SDR per 10 s val clip via `config.compute_sdr` on reconstructed waveforms.
- Evaluation: random sample of 200 val noisy files for each model.
