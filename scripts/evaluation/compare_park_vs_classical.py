"""
Compare the Park R-CED model against the classical baselines (spectral
subtraction, Wiener filter) on the held-out validation split.

For each sampled (clean, noisy, snr) val triple it enhances the noisy signal
with all three methods and scores every output against the clean reference with
three metrics:
  - Delta-SDR  (improvement in Signal-to-Distortion Ratio, dB)
  - PESQ       (wideband, ITU-T P.862.2 -> MOS-LQO)
  - STOI       (Short-Time Objective Intelligibility)
The unprocessed noisy signal is scored too, as the "no processing" reference.

This is CPU-only and runs on the host (no GPU needed). The Park weights are
loaded into a freshly built float32 model (robust across TF versions / on CPU).

Usage:
    python scripts/evaluation/compare_park_vs_classical.py \\
        --weights  models/runs/park_rced_park_data/best.keras \\
        --tfrecord-root tfrecords_park \\
        --clean-dir dataset/CleanSpeech_training \\
        --noisy-dir dataset/NoisySpeech_training \\
        --manifest  dataset/dataset_manifest.json \\
        --out-dir   models/runs/park_rced_park_data/eval_compare \\
        --max-files 200 --n-audio-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config, compute_sdr  # noqa: E402
from scripts.models.park_rced import build_park_rced  # noqa: E402
from scripts.evaluation.evaluate import (  # noqa: E402
    load_audio, collect_val_files, denoise_park,
)
from scripts.evaluation.classical_baselines import (  # noqa: E402
    spectral_subtraction, wiener_filter,
)

from pesq import pesq as _pesq          # noqa: E402
from pystoi import stoi as _stoi        # noqa: E402

METHODS = ["noisy", "specsub", "wiener", "park"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tfrecord-root", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--noisy-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--n-audio-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def safe_pesq(clean: np.ndarray, est: np.ndarray) -> float:
    try:
        return float(_pesq(Config.SAMPLE_RATE, clean, est, "wb"))
    except Exception:
        return float("nan")


def safe_stoi(clean: np.ndarray, est: np.ndarray) -> float:
    try:
        return float(_stoi(clean, est, Config.SAMPLE_RATE, extended=False))
    except Exception:
        return float("nan")


def score(clean: np.ndarray, est: np.ndarray, sdr_in: float) -> dict:
    L = min(len(clean), len(est))
    clean, est = clean[:L], est[:L]
    sdr_out = compute_sdr(clean, est)
    return {
        "sdr": float(sdr_out),
        "delta_sdr": float(sdr_out - sdr_in),
        "pesq": safe_pesq(clean, est),
        "stoi": safe_stoi(clean, est),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Force float32 so CPU inference is correct/fast regardless of the policy the
    # weights were trained under (mixed_float16 on the 4070).
    tf.keras.mixed_precision.set_global_policy("float32")

    info = json.loads((args.tfrecord_root / "dataset_info.json").read_text())
    max_mag = float(info["max_mag"])
    manifest = json.loads(args.manifest.read_text())

    model = build_park_rced(input_shape=(Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 1))
    model.load_weights(str(args.weights))
    print(f"Park weights loaded from {args.weights} "
          f"({model.count_params():,} params, float32, CPU).")

    val_files = collect_val_files(args.noisy_dir, manifest)
    if args.max_files and len(val_files) > args.max_files:
        sel = rng.choice(len(val_files), size=args.max_files, replace=False)
        val_files = [val_files[int(i)] for i in sel]
    print(f"Scoring {len(val_files)} val files x {len(METHODS)} methods "
          f"x 3 metrics (SDR/PESQ/STOI).")

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    saved_examples: list[dict] = []

    per_file_rows: list[dict] = []
    # per_snr[snr][method][metric] -> list of values
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, noisy_path, snr, clnsp_name in tqdm(val_files, desc="park-vs-classical"):
        clean_path = args.clean_dir / f"{clnsp_name}.wav"
        if not clean_path.exists():
            continue
        try:
            clean = load_audio(clean_path)
            noisy = load_audio(noisy_path)
        except Exception as e:
            print(f"skip {noisy_path.name}: {e}", file=sys.stderr)
            continue
        L = min(len(clean), len(noisy))
        clean, noisy = clean[:L], noisy[:L]

        sdr_in = compute_sdr(clean, noisy)
        outputs = {
            "park": denoise_park(model, noisy, max_mag)[:L],
            "specsub": spectral_subtraction(noisy)[:L],
            "wiener": wiener_filter(noisy)[:L],
        }

        row = {"id": idx, "snr": snr}
        # "noisy" reference (no processing): delta_sdr is 0 by definition.
        row["noisy"] = {
            "sdr": float(sdr_in), "delta_sdr": 0.0,
            "pesq": safe_pesq(clean, noisy), "stoi": safe_stoi(clean, noisy),
        }
        for name, est in outputs.items():
            row[name] = score(clean, est, sdr_in)

        for method in METHODS:
            for metric, val in row[method].items():
                agg[snr][method][metric].append(val)
        per_file_rows.append(row)

        if len(saved_examples) < args.n_audio_samples:
            base = f"cmp_id{idx}_snr{snr:.0f}"
            sf.write(str(audio_dir / f"{base}_clean.wav"), clean.astype(np.float32), Config.SAMPLE_RATE)
            sf.write(str(audio_dir / f"{base}_noisy.wav"), noisy.astype(np.float32), Config.SAMPLE_RATE)
            for name, est in outputs.items():
                sf.write(str(audio_dir / f"{base}_{name}.wav"), est.astype(np.float32), Config.SAMPLE_RATE)
            saved_examples.append({"id": idx, "snr": snr, "base": base})

    def summarize(values_by_method: dict) -> dict:
        out = {}
        for method in METHODS:
            m = values_by_method[method]
            out[method] = {
                "n": len(m["sdr"]),
                "sdr": float(np.nanmean(m["sdr"])) if m["sdr"] else None,
                "delta_sdr": float(np.nanmean(m["delta_sdr"])) if m["delta_sdr"] else None,
                "pesq": float(np.nanmean(m["pesq"])) if m["pesq"] else None,
                "stoi": float(np.nanmean(m["stoi"])) if m["stoi"] else None,
            }
        return out

    per_snr_summary = {f"{snr:g}": summarize(agg[snr]) for snr in sorted(agg)}

    # overall = pool all snrs
    overall_acc: dict = defaultdict(lambda: defaultdict(list))
    for snr in agg:
        for method in METHODS:
            for metric, vals in agg[snr][method].items():
                overall_acc[method][metric].extend(vals)
    overall_summary = summarize(overall_acc)

    summary = {
        "weights": str(args.weights),
        "n_files": len(per_file_rows),
        "methods": METHODS,
        "metrics": ["sdr", "delta_sdr", "pesq", "stoi"],
        "snrs": sorted({r["snr"] for r in per_file_rows}),
        "per_snr": per_snr_summary,
        "overall": overall_summary,
        "audio_examples": saved_examples,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "per_file.json").write_text(json.dumps(per_file_rows, indent=2))

    # compact console table
    print("\n=== overall (mean over all SNRs) ===")
    print(f"{'method':10s} {'dSDR':>7s} {'PESQ':>6s} {'STOI':>6s}")
    for method in METHODS:
        s = overall_summary[method]
        print(f"{method:10s} {s['delta_sdr']:+7.2f} {s['pesq']:6.3f} {s['stoi']:6.3f}")
    print(f"\nWrote {args.out_dir/'metrics.json'} and per_file.json")


if __name__ == "__main__":
    main()
