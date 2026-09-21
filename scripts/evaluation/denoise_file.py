"""
Denoise an arbitrary WAV with a trained model (default: Park R-CED).

Two modes:

1. Denoise a file that is *already* noisy:
       python scripts/evaluation/denoise_file.py \
           --input my_noisy_clip.wav \
           --out-dir tcc/apresentacao/audio_demo

   Writes <out-dir>/<stem>_denoised.wav.

2. Demo mode -- take a CLEAN recording, mix in one of the trained noise
   types at a target SNR, then denoise. Produces the three tracks needed
   for the video (normal / noisy / denoised), all at 16 kHz so an A/B
   swap only differs by the noise and the network, not by bandwidth:

       python scripts/evaluation/denoise_file.py \
           --input narracao.wav \
           --mix-noise AirConditioner --snr 0 \
           --out-dir tcc/apresentacao/audio_demo

   Writes <stem>_normal.wav, <stem>_noisy.wav, <stem>_denoised.wav.

The mixing (RMS-normalize to -25 dBFS, scale noise to the target SNR, then
one peak-safe scaling shared by all tracks) mirrors scripts/data/generate_dataset.py
so the demo is faithful to how the training data was built.

Run inside the container, unsetting legacy keras:
    docker exec tensorflow-2026 bash -c "cd /workspace/cnn_denoiser && env -u TF_USE_LEGACY_KERAS python scripts/evaluation/denoise_file.py ..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config, setup_gpu  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Input WAV (clean recording in demo mode, else any noisy clip).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", choices=["unet", "park", "conformer"], default="park")
    p.add_argument("--weights", type=Path, default=Path("models/runs/park_rced_park_data/best.keras"))
    p.add_argument("--tfrecord-root", type=Path, default=Path("tfrecords_park"),
                   help="Used only to read max_mag from dataset_info.json.")
    # demo mode
    p.add_argument("--mix-noise", type=str, default=None,
                   help="Noise type to mix in (e.g. AirConditioner). Enables 3-track demo mode.")
    p.add_argument("--noise-source", type=Path, default=Path("dataset/noise_train"))
    p.add_argument("--snr", type=float, default=0.0, help="Target SNR in dB for demo mode.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_mono_16k(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != Config.SAMPLE_RATE:
        a = resample_poly(a, Config.SAMPLE_RATE, sr).astype(np.float32)
    return a


def rms_normalize(x: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(x ** 2) + eps))
    return x * (10 ** (target_dbfs / 20) / max(rms, eps))


def take_noise(noise_source: Path, noise_type: str, n: int, rng: np.random.Generator) -> np.ndarray:
    files = sorted(noise_source.glob(f"{noise_type}_*.wav"))
    if not files:
        raise SystemExit(f"No noise files for type '{noise_type}' in {noise_source}")
    f = files[int(rng.integers(len(files)))]
    noise = load_mono_16k(f)
    if len(noise) < n:  # loop to cover the recording
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    off = int(rng.integers(0, len(noise) - n + 1))
    return noise[off:off + n]


def build_model(model_name: str, weights: Path):
    import tensorflow as tf
    if model_name == "conformer":
        from scripts.models.conformer_crn import build_conformer_crn
        m = build_conformer_crn(input_shape=(Config.N_FREQ_BINS, Config.N_TIME_FRAMES, 2))
        m.load_weights(str(weights))
        return m
    return tf.keras.models.load_model(str(weights), compile=False, safe_mode=False)


def main() -> None:
    args = parse_args()
    setup_gpu()
    rng = np.random.default_rng(args.seed)

    info = json.loads((args.tfrecord_root / "dataset_info.json").read_text())
    max_mag = float(info["max_mag"])

    from scripts.evaluation.evaluate import denoise_park, denoise_unet  # reuse exact reconstruction

    model = build_model(args.model, args.weights)
    print(f"Loaded {args.model} from {args.weights}")

    signal = load_mono_16k(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    if args.mix_noise:
        clean_norm = rms_normalize(signal)
        noise_norm = rms_normalize(take_noise(args.noise_source, args.mix_noise, len(signal), rng))
        eps = 1e-12
        noise_rms = float(np.sqrt(np.mean(noise_norm ** 2) + eps))
        clean_rms = float(np.sqrt(np.mean(clean_norm ** 2) + eps))
        noise_scale = np.sqrt(clean_rms ** 2 / (10 ** (args.snr / 10)) / max(noise_rms ** 2, eps))
        mix = clean_norm + noise_norm * noise_scale
        peak = max(float(np.max(np.abs(clean_norm))), float(np.max(np.abs(mix))))
        scale = 0.99 / peak if peak > 0.99 else 1.0
        normal = (clean_norm * scale).astype(np.float32)
        noisy = (mix * scale).astype(np.float32)
    else:
        normal = signal
        noisy = signal

    denoise = denoise_park if args.model == "park" else denoise_unet
    denoised = denoise(model, noisy.astype(np.float64), max_mag)[:len(noisy)].astype(np.float32)

    sr = Config.SAMPLE_RATE
    if args.mix_noise:
        sf.write(str(args.out_dir / f"{stem}_normal.wav"), normal, sr, subtype="PCM_16")
        sf.write(str(args.out_dir / f"{stem}_noisy.wav"), noisy, sr, subtype="PCM_16")
        sf.write(str(args.out_dir / f"{stem}_denoised.wav"), denoised, sr, subtype="PCM_16")
        print(f"Wrote 3 tracks (normal/noisy/denoised) to {args.out_dir} at SNR {args.snr:g} dB, noise={args.mix_noise}")
    else:
        sf.write(str(args.out_dir / f"{stem}_denoised.wav"), denoised, sr, subtype="PCM_16")
        print(f"Wrote {stem}_denoised.wav to {args.out_dir}")


if __name__ == "__main__":
    main()
