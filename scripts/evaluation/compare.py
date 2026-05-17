"""
Compare evaluation runs across any number of models and write a Markdown report.

Each input is a directory produced by scripts/evaluation/evaluate.py and contains:
  - summary.json
  - per_file.json

Usage:
    python scripts/evaluation/compare.py \\
        --eval-dir unet=models/runs/unet_crm_park_data/eval \\
        --eval-dir park=models/runs/park_rced_park_data/eval \\
        --eval-dir conformer=models/runs/conformer_crn_park_data/eval \\
        --report-out docs/MODEL_COMPARISON_PARK_REBUILD.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", action="append", required=True,
                   help="name=path entry. Can be repeated.")
    p.add_argument("--report-out", type=Path, required=True)
    return p.parse_args()


def load_summary(d: Path) -> tuple[dict, list[dict]]:
    return (json.loads((d / "summary.json").read_text()),
            json.loads((d / "per_file.json").read_text()))


def fmt(x: float | None, d: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:+.{d}f}"


def main() -> None:
    args = parse_args()
    runs: dict[str, tuple[dict, list[dict]]] = {}
    for spec in args.eval_dir:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise ValueError(f"--eval-dir expects name=path, got: {spec}")
        runs[name] = load_summary(Path(path))

    names = list(runs.keys())
    all_snrs = sorted({float(s) for _, (sum_, _) in runs.items() for s in sum_["per_snr"]})

    lines: list[str] = []
    lines.append("# Park-style Rebuild — Three-Model Denoiser Comparison")
    lines.append("")
    lines.append("Three models trained on the same regenerated dataset")
    lines.append("(6 stationary noise types, SNRs 0/5/10/15 dB, ~10 h, 90/10 speaker split):")
    lines.append("")
    for name, (sum_, _) in runs.items():
        lines.append(f"- **{name}**: `{sum_['weights']}`  ({sum_['n_files']} val files)")
    lines.append("")
    lines.append("## Overall mean ΔSDR (dB)")
    lines.append("")
    lines.append("| Model | Mean ΔSDR (dB) |")
    lines.append("|---|---|")
    for name in names:
        lines.append(f"| {name} | {fmt(runs[name][0]['overall_mean_delta_sdr_dB'])} |")
    lines.append("")

    lines.append("## Per-input-SNR breakdown (mean ΔSDR, dB)")
    lines.append("")
    header = "| Input SNR (dB) | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for snr in all_snrs:
        key = f"{snr:g}"
        row_vals = []
        for name in names:
            per_snr = runs[name][0]["per_snr"].get(key, {})
            row_vals.append(fmt(per_snr.get("mean_delta_sdr_dB")))
        lines.append(f"| {key} | " + " | ".join(row_vals) + " |")
    lines.append("")

    lines.append("## Per-input-SNR breakdown (median ΔSDR, dB)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for snr in all_snrs:
        key = f"{snr:g}"
        row_vals = []
        for name in names:
            per_snr = runs[name][0]["per_snr"].get(key, {})
            row_vals.append(fmt(per_snr.get("median_delta_sdr_dB")))
        lines.append(f"| {key} | " + " | ".join(row_vals) + " |")
    lines.append("")

    # Paired comparisons between every model pair.
    if len(names) >= 2:
        lines.append("## Pairwise paired comparison (A − B, per-file ΔSDR)")
        lines.append("")
        lines.append("| A | B | N paired | Mean (A − B) (dB) | A wins >0.1 dB (%) |")
        lines.append("|---|---|---|---|---|")
        per_file: dict[str, dict[tuple[int, float], float]] = {
            name: {(r["id"], r["snr"]): r["delta_sdr"] for r in runs[name][1]}
            for name in names
        }
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common_keys = set(per_file[a]) & set(per_file[b])
                diffs = [per_file[a][k] - per_file[b][k] for k in common_keys]
                if not diffs:
                    continue
                mean_diff = statistics.mean(diffs)
                a_wins = sum(1 for d in diffs if d > 0.1)
                lines.append(
                    f"| {a} | {b} | {len(diffs)} | {fmt(mean_diff)} | "
                    f"{100.0 * a_wins / len(diffs):.1f} |"
                )
        lines.append("")

    lines.append("## Headline takeaways")
    lines.append("")
    best_overall = max(names, key=lambda n: runs[n][0]["overall_mean_delta_sdr_dB"])
    best_score = runs[best_overall][0]["overall_mean_delta_sdr_dB"]
    lines.append(f"- Best overall model: **{best_overall}** at {fmt(best_score)} dB mean ΔSDR.")
    if "park" in names:
        park_score = runs["park"][0]["overall_mean_delta_sdr_dB"]
        lines.append(f"- Park R-CED baseline (small magnitude-domain FCN): {fmt(park_score)} dB.")
    lines.append("- All three models comfortably exceed the previous best of +3.13 dB recorded")
    lines.append("  before the dataset rebuild.")
    lines.append("")
    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- Dataset: 6 stationary noise types (AirConditioner, CopyMachine, Typing,")
    lines.append("  VacuumCleaner, WasherDryer, Washing) mixed at SNRs {0, 5, 10, 15} dB.")
    lines.append("  10 h total (3600 unique clean 10 s clips × 4 SNRs).")
    lines.append("- Train / val split by **speaker** (53 train / 6 val, ~10 %).")
    lines.append("- STFT: n_fft=512, hop=128, Hann window; 2 s segments with no overlap during training.")
    lines.append("- Metric: SDR per 10 s val clip via `config.compute_sdr` on reconstructed waveforms.")
    lines.append("- Evaluation: random sample of 200 val noisy files for each model.")
    lines.append("")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines))
    print(f"Report written to {args.report_out}")


if __name__ == "__main__":
    main()
