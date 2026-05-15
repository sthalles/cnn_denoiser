# Detailed Model Performance Statistics

Generated: 2026-04-28 06:49:06

## Models Compared

- Model A (best_model): `/workspace/cnn_denoiser/models/best.keras`
- Model B (retrained_basic_unet): `/workspace/cnn_denoiser/models/basic_unet_retrain_20260428-best.keras`
- Evaluated samples: 1532

## Overall Summary

| Metric | Mean (dB) | Median (dB) | Std (dB) | 95% CI Low | 95% CI High |
| --- | --- | --- | --- | --- | --- |
| Baseline SDR (input noisy) | 7.509 | 7.710 | 5.582 | 7.232 | 7.778 |
| SDR improvement (best_model) | 9.340 | 9.710 | 4.403 | 9.103 | 9.554 |
| SDR improvement (retrained_basic_unet) | 0.418 | 1.193 | 4.179 | 0.206 | 0.615 |
| Delta (retrained_basic_unet - best_model) | -8.923 | -8.335 | 3.687 | -9.111 | -8.739 |

## Pairwise Competition

| Statistic | Value |
| --- | --- |
| Wins (Model B better) | 0 |
| Losses (Model B worse) | 1532 |
| Ties | 0 |
| Win rate excl ties | 0.000 |
| Win rate incl ties | 0.000 |
| Delta > +0.5 dB rate | 0.000 |
| Delta > +1.0 dB rate | 0.000 |
| Delta > +2.0 dB rate | 0.000 |
| Delta < -0.5 dB rate | 1.000 |
| Delta < -1.0 dB rate | 1.000 |
| Sign-test p-value (2-sided) | 0.000 |
| Cohen's d (paired deltas) | -2.420 |
| Corr(input SDR, best_model improvement) | -0.738 |
| Corr(input SDR, retrained_basic_unet improvement) | -0.729 |

## Per-SNR Detailed Statistics

| SNR (dB) | N | Mean best_model imp | Mean retrained_basic_unet imp | Mean delta (B-A) | Win rate B | Sign-test p |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 383 | 13.527 | 4.059 | -9.468 | 0.000 | 0.000 |
| 10.0 | 383 | 10.982 | 2.213 | -8.770 | 0.000 | 0.000 |
| 20.0 | 383 | 8.035 | -0.475 | -8.510 | 0.000 | 0.000 |
| 30.0 | 383 | 4.817 | -4.126 | -8.943 | 0.000 | 0.000 |

## Distribution Statistics (Overall)

| Series | Min | P05 | P25 | Median | P75 | P95 | Max | IQR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_model | -5.851 | 1.027 | 6.875 | 9.710 | 12.228 | 15.982 | 22.070 | 5.353 |
| retrained_basic_unet | -15.060 | -8.276 | -1.822 | 1.193 | 3.500 | 5.771 | 9.618 | 5.321 |
| retrained_basic_unet - best_model | -24.127 | -15.959 | -10.913 | -8.335 | -6.559 | -3.461 | -1.040 | 4.354 |

## Subset Analysis

### low_snr_le_5db

| Metric | Value |
| --- | --- |
| N | 383 |
| Mean improvement (best_model) | 13.527 |
| Mean improvement (retrained_basic_unet) | 4.059 |
| Mean delta (B-A) | -9.468 |
| Median delta (B-A) | -9.150 |
| 95% CI delta low | -9.754 |
| 95% CI delta high | -9.186 |

### high_snr_ge_15db

| Metric | Value |
| --- | --- |
| N | 766 |
| Mean improvement (best_model) | 6.426 |
| Mean improvement (retrained_basic_unet) | -2.301 |
| Mean delta (B-A) | -8.727 |
| Median delta (B-A) | -7.916 |
| 95% CI delta low | -9.035 |
| 95% CI delta high | -8.419 |

## Artifacts

- Raw paired rows CSV: `logs/comparisons/detailed/paired_raw_20260428_053920.csv`
- JSON stats: `logs/comparisons/detailed/stats_20260428_053920.json`
- This report: `logs/comparisons/detailed/report_20260428_053920.md`
