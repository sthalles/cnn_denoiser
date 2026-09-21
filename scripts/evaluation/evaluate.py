"""
Evaluate a denoiser model on the held-out validation split.

For each (clean, noisy, snr) triple from the manifest's `val` split:
  - run the model in segments matching the training spectrogram size
  - reconstruct the denoised waveform via ISTFT + overlap-add
  - record SDR(noisy) and SDR(denoised) vs the clean reference

Reports per-SNR mean SDR improvement and an overall mean. Also writes a few
denoised WAVs to <out-dir>/audio/ for listening tests.

Usage:
    python scripts/evaluation/evaluate.py \\
        --model unet --weights models/runs/unet_crm_park_data/best.keras \\
        --tfrecord-root tfrecords_park \\
        --clean-dir dataset/CleanSpeech_training \\
        --noisy-dir dataset/NoisySpeech_training \\
        --manifest  dataset/dataset_manifest.json \\
        --out-dir   models/runs/unet_crm_park_data/eval \\
        --n-audio-samples 6 \\
        --max-files 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    Config, compute_stft, compute_istft, segment_audio,
    overlap_add_segments, compute_sdr, setup_gpu,
)  # noqa: E402

NOISY_RE = re.compile(r"noisy(\d+)_SNRdb_([\d.]+)_(clnsp\d+)\.wav$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["unet", "park", "conformer"], required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--tfrecord-root", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--noisy-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-audio-samples", type=int, default=6)
    p.add_argument("--max-files", type=int, default=200,
                   help="Limit how many val noisy files to evaluate (random sample).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_audio(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != Config.SAMPLE_RATE:
        raise RuntimeError(f"unexpected sr={sr} for {path}")
    return a


def collect_val_files(noisy_dir: Path, manifest: dict) -> list[tuple[int, Path, float, str]]:
    val_ids = {entry["id"] for entry in manifest["splits"]["val"]}
    out = []
    for p in sorted(noisy_dir.iterdir()):
        m = NOISY_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in val_ids:
            continue
        snr = float(m.group(2))
        out.append((idx, p, snr, m.group(3)))
    return out


def denoise_unet(model, noisy: np.ndarray, max_mag: float) -> np.ndarray:
    """Run the cRM U-Net on a single waveform and return the denoised waveform."""
    segments = segment_audio(noisy)
    seg_out: list[tuple[int, np.ndarray]] = []
    stfts_in = []
    seg_starts = []
    for start, seg in segments:
        stft = compute_stft(seg)
        n_real = (np.real(stft) / max_mag).astype(np.float32)
        n_imag = (np.imag(stft) / max_mag).astype(np.float32)
        stfts_in.append((stft, np.stack([n_real, n_imag], axis=-1)))
        seg_starts.append(start)

    inputs = np.stack([si[1] for si in stfts_in], axis=0)
    preds = model.predict(inputs, verbose=0)
    for (raw_stft, _), pred, start in zip(stfts_in, preds, seg_starts):
        mask_r = pred[..., 0].astype(np.float64)
        mask_i = pred[..., 1].astype(np.float64)
        est_stft = raw_stft * (mask_r + 1j * mask_i)
        est_audio = compute_istft(est_stft)
        # Trim/pad to original segment length
        if len(est_audio) >= Config.SEGMENT_SAMPLES:
            est_audio = est_audio[:Config.SEGMENT_SAMPLES]
        else:
            est_audio = np.pad(est_audio, (0, Config.SEGMENT_SAMPLES - len(est_audio)))
        seg_out.append((start, est_audio))

    return overlap_add_segments(seg_out, Config.SEGMENT_SAMPLES, Config.SEGMENT_HOP, len(noisy))


def denoise_park(model, noisy: np.ndarray, max_mag: float) -> np.ndarray:
    """Run the Park magnitude-domain FCN. Phase is reused from the noisy STFT."""
    segments = segment_audio(noisy)
    seg_out: list[tuple[int, np.ndarray]] = []
    stfts_in = []
    seg_starts = []
    for start, seg in segments:
        stft = compute_stft(seg)
        mag = np.abs(stft)
        normalized = mag / max_mag
        in_log = np.log1p(normalized).astype(np.float32)[..., None]
        stfts_in.append((stft, in_log))
        seg_starts.append(start)

    inputs = np.stack([si[1] for si in stfts_in], axis=0)
    preds = model.predict(inputs, verbose=0)

    for (raw_stft, _), pred, start in zip(stfts_in, preds, seg_starts):
        pred_logmag = pred[..., 0].astype(np.float64)
        pred_mag_norm = np.maximum(np.expm1(pred_logmag), 0.0)
        pred_mag = pred_mag_norm * max_mag
        # combine predicted magnitude with noisy phase
        phase = np.angle(raw_stft)
        est_stft = pred_mag * np.exp(1j * phase)
        est_audio = compute_istft(est_stft)
        if len(est_audio) >= Config.SEGMENT_SAMPLES:
            est_audio = est_audio[:Config.SEGMENT_SAMPLES]
        else:
            est_audio = np.pad(est_audio, (0, Config.SEGMENT_SAMPLES - len(est_audio)))
        seg_out.append((start, est_audio))

    return overlap_add_segments(seg_out, Config.SEGMENT_SAMPLES, Config.SEGMENT_HOP, len(noisy))


def main() -> None:
    args = parse_args()
    setup_gpu()
    rng = np.random.default_rng(args.seed)

    info = json.loads((args.tfrecord_root / "dataset_info.json").read_text())
    max_mag = float(info["max_mag"])
    manifest = json.loads(args.manifest.read_text())

    # Conformer's Lambda layers carry closures that don't round-trip through
    # keras saving; rebuild the architecture from code and load weights instead.
    if args.model == "conformer":
        from scripts.models.conformer_crn import build_conformer_crn  # noqa: E402
        model = build_conformer_crn(input_shape=(Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 2))
        model.load_weights(str(args.weights))
    else:
        # safe_mode=False so Lambda layers (used in U-Net) can be deserialized.
        model = tf.keras.models.load_model(str(args.weights), compile=False, safe_mode=False)
    print(f"Loaded {args.model} from {args.weights}")

    val_files = collect_val_files(args.noisy_dir, manifest)
    if args.max_files and len(val_files) > args.max_files:
        sel = rng.choice(len(val_files), size=args.max_files, replace=False)
        val_files = [val_files[int(i)] for i in sel]

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    save_audio_count = args.n_audio_samples

    per_snr: dict[float, list[float]] = {}
    per_file_rows: list[dict] = []

    for idx, noisy_path, snr, clnsp_name in tqdm(val_files, desc=f"eval {args.model}"):
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
        if args.model in ("unet", "conformer"):
            # Both produce a cRM; same downstream reconstruction.
            est = denoise_unet(model, noisy, max_mag)
        else:
            est = denoise_park(model, noisy, max_mag)
        est = est[:L]

        sdr_in = compute_sdr(clean, noisy)
        sdr_out = compute_sdr(clean, est)
        delta = sdr_out - sdr_in
        per_snr.setdefault(snr, []).append(delta)
        per_file_rows.append({
            "id": idx, "noisy": noisy_path.name, "snr": snr,
            "sdr_in": float(sdr_in), "sdr_out": float(sdr_out), "delta_sdr": float(delta),
        })

        if save_audio_count > 0:
            base = f"{args.model}_id{idx}_snr{snr:.0f}"
            sf.write(str(audio_dir / f"{base}_clean.wav"), clean.astype(np.float32), Config.SAMPLE_RATE)
            sf.write(str(audio_dir / f"{base}_noisy.wav"), noisy.astype(np.float32), Config.SAMPLE_RATE)
            sf.write(str(audio_dir / f"{base}_denoised.wav"), est.astype(np.float32), Config.SAMPLE_RATE)
            save_audio_count -= 1

    summary = {
        "model": args.model,
        "weights": str(args.weights),
        "n_files": len(per_file_rows),
        "overall_mean_delta_sdr_dB": float(np.mean([r["delta_sdr"] for r in per_file_rows]))
        if per_file_rows else None,
        "per_snr": {
            f"{snr:g}": {
                "n": len(vals),
                "mean_delta_sdr_dB": float(np.mean(vals)),
                "median_delta_sdr_dB": float(np.median(vals)),
            } for snr, vals in sorted(per_snr.items())
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "per_file.json").write_text(json.dumps(per_file_rows, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
