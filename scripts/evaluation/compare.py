"""
Compare two evaluation runs (e.g., U-Net cRM vs Park R-CED) and write a Markdown report.

Inputs are the directories produced by scripts/evaluation/evaluate.py:
  - <eval-dir>/summary.json
  - <eval-dir>/per_file.json

Usage:
    python scripts/evaluation/compare.py \\
        --unet-eval-dir models/runs/unet_crm_park_data/eval \\
        --park-eval-dir models/runs/park_rced_park_data/eval \\
        --report-out    docs/MODEL_COMPARISON_PARK_VS_UNET.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unet-eval-dir", type=Path, required=True)
    p.add_argument("--park-eval-dir", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    return p.parse_args()


def load_summary(d: Path) -> tuple[dict, list[dict]]:
    return (json.loads((d / "summary.json").read_text()),
            json.loads((d / "per_file.json").read_text()))


def fmt(x: float | None, d=2) -> str:
    if x is None:
        return "-"
    return f"{x:+.{d}f}"


def main() -> None:
    args = parse_args()
    u_sum, u_files = load_summary(args.unet_eval_dir)
    p_sum, p_files = load_summary(args.park_eval_dir)

    snrs = sorted(set(u_sum["per_snr"].keys()) | set(p_sum["per_snr"].keys()),
                  key=lambda x: float(x))

    lines: list[str] = []
    lines.append("# U-Net cRM vs Park R-CED FCN: Evaluation Comparison")
    lines.append("")
    lines.append(f"- U-Net weights : `{u_sum['weights']}`  ({u_sum['n_files']} val files)")
    lines.append(f"- Park weights  : `{p_sum['weights']}`  ({p_sum['n_files']} val files)")
    lines.append("")
    lines.append("## Overall SDR improvement (mean over val files, dB)")
    lines.append("")
    lines.append(f"| Model | Mean ΔSDR (dB) |")
    lines.append(f"|---|---|")
    lines.append(f"| U-Net cRM | {fmt(u_sum['overall_mean_delta_sdr_dB'])} |")
    lines.append(f"| Park R-CED FCN | {fmt(p_sum['overall_mean_delta_sdr_dB'])} |")
    lines.append("")
    lines.append("## Per-SNR breakdown (mean ΔSDR, dB)")
    lines.append("")
    lines.append("| Input SNR (dB) | U-Net cRM | Park R-CED |")
    lines.append("|---|---|---|")
    for snr in snrs:
        u = u_sum["per_snr"].get(snr, {}).get("mean_delta_sdr_dB")
        p = p_sum["per_snr"].get(snr, {}).get("mean_delta_sdr_dB")
        lines.append(f"| {snr} | {fmt(u)} | {fmt(p)} |")
    lines.append("")
    # Per-file deltas (paired diff)
    by_key = {}
    for r in u_files:
        by_key[(r["id"], r["snr"])] = {"unet": r["delta_sdr"], "noisy_sdr_in": r["sdr_in"]}
    for r in p_files:
        key = (r["id"], r["snr"])
        if key in by_key:
            by_key[key]["park"] = r["delta_sdr"]
    paired = [(v["unet"] - v["park"]) for v in by_key.values() if "park" in v]
    if paired:
        mean_pair = statistics.mean(paired)
        lines.append(f"## Paired comparison (U-Net − Park, per file)")
        lines.append("")
        lines.append(f"- N paired files: {len(paired)}")
        lines.append(f"- Mean (U-Net − Park) ΔSDR: {fmt(mean_pair)} dB")
        lines.append(f"- Stdev: {statistics.pstdev(paired):.2f} dB")
        winning_unet = sum(1 for d in paired if d > 0.1)
        lines.append(f"- U-Net wins (>0.1 dB): {winning_unet} / {len(paired)} "
                     f"({100*winning_unet/max(1,len(paired)):.1f}%)")
        lines.append("")

    lines.append("## Notes on methodology")
    lines.append("")
    lines.append("- Dataset: 6 stationary noise types (AirConditioner, CopyMachine, Typing, "
                 "VacuumCleaner, WasherDryer, Washing) at SNRs {0, 5, 10, 15} dB, ~10 hours.")
    lines.append("- Train / val split: by speaker (no leakage). 16 kHz, 10 s clips.")
    lines.append("- STFT: n_fft=512, hop=128, Hann window. Segments: 2 s, no overlap during training.")
    lines.append("- Metrics: SDR (signal-to-distortion ratio in dB) using config.compute_sdr.")
    lines.append("")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines))
    print(f"Report written to {args.report_out}")


if __name__ == "__main__":
    main()
