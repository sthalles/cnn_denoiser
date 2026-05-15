"""
Generate clean/noisy WAV pairs for training the denoiser.

Replaces dataset/noisyspeech_synthesizer.py with stricter guarantees:
- Every output is exactly TARGET_DURATION seconds at TARGET_SR Hz, mono.
- Source files are resampled if needed.
- For each unique clean+noise pairing, mixes at every SNR level (same content,
  different noise levels) -- so models see identical speech at multiple SNRs.
- Speakers are split into train/val before generation (no speaker leakage).
- Output filenames match the legacy v2 convention so existing parsers work:
    clean:  CleanSpeech_training/clnsp{N}.wav
    noisy:  NoisySpeech_training/noisy{N}_SNRdb_{snr}_clnsp{N}.wav

Usage:
    python scripts/data/generate_dataset.py \\
        --clean-source dataset/clean_train \\
        --noise-source dataset/noise_train \\
        --out-dir       dataset \\
        --noise-types AirConditioner CopyMachine Typing VacuumCleaner WasherDryer Washing \\
        --snrs 0 5 10 15 \\
        --total-hours 10 \\
        --duration 10 \\
        --sample-rate 16000 \\
        --val-fraction 0.1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


SPEAKER_RE = re.compile(r"^(p\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-source", required=True, type=Path,
                   help="Directory containing source clean speech WAVs (e.g., VCTK).")
    p.add_argument("--noise-source", required=True, type=Path,
                   help="Directory containing source noise WAVs (one type per filename prefix).")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output root. Will write CleanSpeech_training/, NoisySpeech_training/ inside.")
    p.add_argument("--noise-types", nargs="+", required=True,
                   help="Noise filename prefixes to include (e.g., AirConditioner CopyMachine).")
    p.add_argument("--snrs", nargs="+", type=float, required=True,
                   help="SNR levels in dB to generate per clip (e.g., 0 5 10 15).")
    p.add_argument("--total-hours", type=float, required=True,
                   help="Approximate total hours of noisy audio to generate (counted once per SNR).")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Exact duration per output clip (seconds). Default: 10.")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Target sample rate. Default: 16000.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of speakers to hold out for validation. Default: 0.1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean-first", action="store_true",
                   help="Wipe out-dir/CleanSpeech_training and NoisySpeech_training before generating.")
    return p.parse_args()


def load_resample(path: Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(f"librosa needed to resample {path} ({sr} -> {target_sr})")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def speaker_of(filename: str) -> str | None:
    m = SPEAKER_RE.match(filename)
    return m.group(1) if m else None


def split_speakers(clean_files: list[Path], val_fraction: float, rng: random.Random) -> tuple[set[str], set[str]]:
    speakers = sorted({speaker_of(p.name) for p in clean_files if speaker_of(p.name)})
    rng.shuffle(speakers)
    n_val = max(1, int(round(len(speakers) * val_fraction)))
    val = set(speakers[:n_val])
    train = set(speakers[n_val:])
    return train, val


def gather_clean(clean_dir: Path, speakers: set[str]) -> list[Path]:
    out = []
    for p in sorted(clean_dir.glob("*.wav")):
        spk = speaker_of(p.name)
        if spk and spk in speakers:
            out.append(p)
    return out


def gather_noise(noise_dir: Path, types: Iterable[str]) -> dict[str, list[Path]]:
    by_type: dict[str, list[Path]] = {}
    for t in types:
        files = sorted(noise_dir.glob(f"{t}_*.wav"))
        if not files:
            print(f"WARNING: no files found for noise type '{t}'", file=sys.stderr)
            continue
        by_type[t] = files
    if not by_type:
        raise RuntimeError("No noise files matched any of the requested types.")
    return by_type


def take_clean_segment(rng: random.Random,
                       clean_files: list[Path],
                       sample_rate: int,
                       target_samples: int) -> tuple[np.ndarray, list[str]]:
    """Return exactly `target_samples` of speech. Concatenates short clips of the same
    speaker (with 100ms silence) when one file is not long enough."""
    while True:
        idx = rng.randrange(len(clean_files))
        seed_file = clean_files[idx]
        seed_spk = speaker_of(seed_file.name)
        if not seed_spk:
            continue
        same_spk_files = [p for p in clean_files if speaker_of(p.name) == seed_spk]
        rng.shuffle(same_spk_files)

        audio = np.zeros(0, dtype=np.float32)
        used = []
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
        for f in same_spk_files:
            try:
                a = load_resample(f, sample_rate)
            except Exception as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            if len(a) == 0:
                continue
            audio = np.concatenate([audio, a, silence]) if len(audio) else a
            used.append(f.name)
            if len(audio) >= target_samples:
                break

        if len(audio) < target_samples:
            continue  # not enough speech for this speaker -- try another

        # Random crop inside the concatenated audio so we don't always start at 0.
        max_start = len(audio) - target_samples
        start = rng.randint(0, max_start) if max_start > 0 else 0
        seg = audio[start:start + target_samples].copy()
        return seg, used


def take_noise_segment(rng: random.Random,
                       noise_by_type: dict[str, list[Path]],
                       sample_rate: int,
                       target_samples: int) -> tuple[np.ndarray, str]:
    """Return exactly `target_samples` of noise, by random offset (looping if needed)."""
    types = list(noise_by_type.keys())
    chosen_type = rng.choice(types)
    files = noise_by_type[chosen_type]
    f = rng.choice(files)
    a = load_resample(f, sample_rate)
    if len(a) == 0:
        raise RuntimeError(f"empty noise file {f}")
    if len(a) < target_samples:
        # tile
        reps = (target_samples // len(a)) + 1
        a = np.tile(a, reps)
    start = rng.randint(0, max(0, len(a) - target_samples))
    seg = a[start:start + target_samples].copy()
    return seg, f"{chosen_type}/{f.name}"


def rms_normalize(x: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(x ** 2) + eps))
    return x * (10 ** (target_dbfs / 20) / max(rms, eps))


def mix_all_snrs(clean_norm: np.ndarray, noise_norm: np.ndarray, snrs: list[float]
                 ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Both inputs are already normalized to the same RMS (e.g., -25 dBFS).
    Returns (clean_out, [noisy_at_each_snr]). All outputs share a single peak-safe
    scaling so the saved clean matches the clean inside every noisy mix."""
    eps = 1e-12
    noise_rms = float(np.sqrt(np.mean(noise_norm ** 2) + eps))
    clean_rms = float(np.sqrt(np.mean(clean_norm ** 2) + eps))
    mixes = []
    for snr in snrs:
        noise_scale = np.sqrt(clean_rms ** 2 / (10 ** (snr / 10)) / max(noise_rms ** 2, eps))
        mix = clean_norm + noise_norm * noise_scale
        mixes.append(mix)
    # One peak-safe scaling across clean + all mixes
    peak = max(float(np.max(np.abs(clean_norm))),
               max(float(np.max(np.abs(m))) for m in mixes))
    scale = 0.99 / peak if peak > 0.99 else 1.0
    clean_out = (clean_norm * scale).astype(np.float32)
    mixes = [(m * scale).astype(np.float32) for m in mixes]
    return clean_out, mixes


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    target_samples = int(args.duration * args.sample_rate)

    clean_root = args.out_dir / "CleanSpeech_training"
    noisy_root = args.out_dir / "NoisySpeech_training"
    if args.clean_first:
        for d in (clean_root, noisy_root):
            if d.exists():
                shutil.rmtree(d)
    clean_root.mkdir(parents=True, exist_ok=True)
    noisy_root.mkdir(parents=True, exist_ok=True)

    all_clean = sorted(args.clean_source.glob("*.wav"))
    if not all_clean:
        raise RuntimeError(f"No clean WAVs in {args.clean_source}")
    train_spk, val_spk = split_speakers(all_clean, args.val_fraction, rng)
    train_clean = gather_clean(args.clean_source, train_spk)
    val_clean = gather_clean(args.clean_source, val_spk)
    print(f"Speakers: train={len(train_spk)} val={len(val_spk)}")
    print(f"Clean files: train={len(train_clean)} val={len(val_clean)}")

    noise_by_type = gather_noise(args.noise_source, args.noise_types)
    for t, fs in noise_by_type.items():
        print(f"  noise[{t}]: {len(fs)} files")

    # We count duration once per SNR (each clip is duplicated to every SNR level).
    # So number of UNIQUE 10s clips = total_seconds / duration.
    total_unique = int(round(args.total_hours * 3600 / args.duration))
    # Split unique clips proportional to val fraction
    n_val = max(1, int(round(total_unique * args.val_fraction)))
    n_train = max(1, total_unique - n_val)
    print(f"Generating {n_train} train + {n_val} val unique clips "
          f"x {len(args.snrs)} SNRs = {(n_train + n_val) * len(args.snrs)} noisy outputs.")

    manifest = {
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "snrs": list(args.snrs),
        "total_hours": args.total_hours,
        "train_speakers": sorted(train_spk),
        "val_speakers": sorted(val_spk),
        "noise_types": list(noise_by_type.keys()),
        "splits": {"train": [], "val": []},
    }

    counter = 0
    for split_name, clean_pool, n_items in (("train", train_clean, n_train),
                                            ("val", val_clean, n_val)):
        for i in range(n_items):
            counter += 1
            clean_seg, clean_used = take_clean_segment(rng, clean_pool, args.sample_rate, target_samples)
            noise_seg, noise_used = take_noise_segment(rng, noise_by_type, args.sample_rate, target_samples)

            clean_norm = rms_normalize(clean_seg)
            noise_norm = rms_normalize(noise_seg)
            clean_out, mixes = mix_all_snrs(clean_norm, noise_norm, list(args.snrs))

            clean_name = f"clnsp{counter}.wav"
            write_wav(clean_root / clean_name, clean_out, args.sample_rate)

            per_clip_snrs = []
            for snr, noisy in zip(args.snrs, mixes):
                # filename matches legacy v2 expectation: noisy{N}_SNRdb_{snr}_clnsp{N}.wav
                noisy_name = f"noisy{counter}_SNRdb_{float(snr)}_clnsp{counter}.wav"
                write_wav(noisy_root / noisy_name, noisy, args.sample_rate)
                per_clip_snrs.append(float(snr))

            manifest["splits"][split_name].append({
                "id": counter,
                "clean": clean_name,
                "clean_source": clean_used,
                "noise_source": noise_used,
                "snrs": per_clip_snrs,
            })
            if counter % 100 == 0:
                print(f"  [{split_name}] {i + 1}/{n_items} clips (total written: {counter})", flush=True)

    manifest_path = args.out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")
    print(f"Done. {counter} unique clips, {counter * len(args.snrs)} noisy files.")


if __name__ == "__main__":
    generate(parse_args())
