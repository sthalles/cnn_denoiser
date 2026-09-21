"""
Audio Denoiser - Model Testing & Evaluation (v2)

Easy-to-use script to test a trained denoiser model.

Usage:
    # Quick test (10 samples, default SNRs):
    python 4-test_model.py

    # Full test (50 samples):
    python 4-test_model.py --samples 50

    # Custom SNR levels:
    python 4-test_model.py --snr 0 5 10 15 20

    # Specific model:
    python 4-test_model.py --model models/denoiser_v2-best.keras

    # Save denoised audio samples:
    python 4-test_model.py --save-audio

    # Use training data pairs instead of synthesizing:
    python 4-test_model.py --use-training-data
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import re
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import numpy as np
from config import (
    Config, load_audio, save_audio, compute_sdr, compute_snr, setup_gpu
)

# Suppress CUDA/XLA warnings
class StderrFilter:
    def __init__(self, original):
        self.original = original
        self.patterns = [
            'gpu_timer', 'cuda_dnn', 'Unable to register', 'ptxas',
            'NUMA', 'StreamExecutor', 'TF-TRT', 'TensorRT',
            'successful NUMA', 'XLA service'
        ]
    def write(self, msg):
        if not any(p in msg for p in self.patterns):
            self.original.write(msg)
    def flush(self):
        self.original.flush()
    def fileno(self):
        return self.original.fileno()

sys.stderr = StderrFilter(sys.stderr)


def find_model():
    """Find the best available model file"""
    candidates = [
        os.path.join(Config.MODEL_DIR, 'denoiser_v2-best.keras'),
        os.path.join(Config.MODEL_DIR, 'denoiser_v2-final.keras'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def mix_audio(clean, noise, target_snr_db):
    """Mix clean speech with noise at a target SNR level"""
    # Match lengths
    if len(noise) < len(clean):
        repeats = (len(clean) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    # Compute current powers
    clean_power = np.mean(clean ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8

    # Scale noise to achieve target SNR
    target_snr_linear = 10 ** (target_snr_db / 10)
    scale = np.sqrt(clean_power / (noise_power * target_snr_linear))
    scaled_noise = noise * scale

    # Mix
    noisy = clean + scaled_noise

    # Prevent clipping
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy * (0.99 / peak)
        clean = clean * (0.99 / peak)

    return noisy, clean


def load_test_files(clean_dir, noise_dir, max_clean=None, max_noise=None):
    """Load test audio files"""
    clean_files = sorted([
        os.path.join(clean_dir, f)
        for f in os.listdir(clean_dir)
        if f.endswith('.wav')
    ])
    noise_files = sorted([
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith('.wav')
    ])

    if max_clean:
        clean_files = clean_files[:max_clean]
    if max_noise:
        noise_files = noise_files[:max_noise]

    return clean_files, noise_files


def run_test(args):
    """Main test routine"""
    # ── Find model ──
    model_path = args.model or find_model()
    if not model_path or not os.path.exists(model_path):
        print("ERROR: No model found. Specify with --model")
        sys.exit(1)

    # ── Import denoiser (triggers TF/GPU init) ──
    from importlib import import_module
    usar = import_module('3-usar_modelo_v2')
    denoiser = usar.AudioDenoiser(model_path)

    snr_levels = args.snr
    results = []

    if args.use_training_data:
        # ── Evaluate on pre-mixed training pairs ──
        from config import get_file_pairs
        pairs = get_file_pairs(
            args.clean_dir_train or Config.CLEAN_DIR,
            args.noisy_dir_train or Config.NOISY_DIR
        )
        if not pairs:
            print("ERROR: No training data pairs found.")
            sys.exit(1)

        np.random.seed(42)
        np.random.shuffle(pairs)
        pairs = pairs[:args.samples]

        print(f"\nEvaluating on {len(pairs)} training data pairs...")
        print("-" * 70)

        for i, (clean_path, noisy_path, snr) in enumerate(pairs):
            try:
                clean_audio, _ = load_audio(clean_path)
                noisy_audio, _ = load_audio(noisy_path)

                min_len = min(len(clean_audio), len(noisy_audio))
                clean_audio = clean_audio[:min_len]
                noisy_audio = noisy_audio[:min_len]

                denoised = denoiser.denoise_audio(noisy_audio, show_progress=False)
                min_len = min(len(clean_audio), len(denoised))

                sdr_before = compute_sdr(clean_audio[:min_len], noisy_audio[:min_len])
                sdr_after = compute_sdr(clean_audio[:min_len], denoised[:min_len])

                results.append({
                    'clean_file': os.path.basename(clean_path),
                    'noise_type': 'training_mix',
                    'snr': snr,
                    'sdr_before': sdr_before,
                    'sdr_after': sdr_after,
                    'improvement': sdr_after - sdr_before,
                })

                print(f"  [{i+1:3d}/{len(pairs)}] SNR={snr:5.1f}dB | "
                      f"SDR: {sdr_before:+6.2f} → {sdr_after:+6.2f} dB "
                      f"(Δ {sdr_after - sdr_before:+.2f})")

            except Exception as e:
                print(f"  [{i+1:3d}/{len(pairs)}] ERROR: {e}")

    else:
        # ── Synthesize noisy audio from clean_test + noise_test ──
        clean_dir = args.clean_dir or 'dataset/clean_test'
        noise_dir = args.noise_dir or 'dataset/noise_test'

        if not os.path.isdir(clean_dir) or not os.path.isdir(noise_dir):
            print(f"ERROR: Test directories not found: {clean_dir}, {noise_dir}")
            sys.exit(1)

        clean_files, noise_files = load_test_files(clean_dir, noise_dir)
        print(f"\nFound {len(clean_files)} clean files, {len(noise_files)} noise files")
        print(f"SNR levels: {snr_levels} dB")
        print(f"Samples per SNR: {args.samples}")

        # Randomly pair clean + noise files
        np.random.seed(42)
        total_tests = len(snr_levels) * args.samples
        test_idx = 0

        for snr_db in snr_levels:
            print(f"\n{'─' * 70}")
            print(f"  Testing at SNR = {snr_db} dB")
            print(f"{'─' * 70}")

            for i in range(args.samples):
                test_idx += 1
                clean_idx = np.random.randint(0, len(clean_files))
                noise_idx = np.random.randint(0, len(noise_files))

                try:
                    clean_audio, _ = load_audio(clean_files[clean_idx])
                    noise_audio, _ = load_audio(noise_files[noise_idx])

                    # Skip very short files
                    if len(clean_audio) < Config.SAMPLE_RATE:
                        continue

                    noisy_audio, clean_audio = mix_audio(clean_audio, noise_audio, snr_db)

                    # Denoise
                    denoised = denoiser.denoise_audio(noisy_audio, show_progress=False)

                    min_len = min(len(clean_audio), len(denoised))
                    sdr_before = compute_sdr(clean_audio[:min_len], noisy_audio[:min_len])
                    sdr_after = compute_sdr(clean_audio[:min_len], denoised[:min_len])

                    clean_name = os.path.basename(clean_files[clean_idx])
                    noise_name = os.path.basename(noise_files[noise_idx]).replace('.wav', '')

                    results.append({
                        'clean_file': clean_name,
                        'noise_type': noise_name,
                        'snr': snr_db,
                        'sdr_before': sdr_before,
                        'sdr_after': sdr_after,
                        'improvement': sdr_after - sdr_before,
                    })

                    print(f"  [{test_idx:3d}/{total_tests}] {clean_name[:20]:20s} + {noise_name[:20]:20s} | "
                          f"SDR: {sdr_before:+6.2f} → {sdr_after:+6.2f} dB "
                          f"(Δ {sdr_after - sdr_before:+.2f})")

                    # Optionally save audio
                    if args.save_audio:
                        out_dir = args.output_dir or 'test_output'
                        os.makedirs(out_dir, exist_ok=True)
                        tag = f"snr{snr_db}_{clean_name.replace('.wav', '')}_{noise_name}"
                        save_audio(noisy_audio, os.path.join(out_dir, f"noisy_{tag}.wav"))
                        save_audio(denoised[:len(noisy_audio)], os.path.join(out_dir, f"denoised_{tag}.wav"))
                        save_audio(clean_audio, os.path.join(out_dir, f"clean_{tag}.wav"))

                except Exception as e:
                    print(f"  [{test_idx:3d}/{total_tests}] ERROR: {e}")

    if not results:
        print("\nNo results to display.")
        return

    # ═══════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n")
    print("═" * 70)
    print("  MODEL TEST RESULTS")
    print("═" * 70)
    print(f"  Model: {model_path}")
    print(f"  Total samples tested: {len(results)}")
    print("═" * 70)

    # Per-SNR breakdown
    snr_groups = {}
    for r in results:
        key = r['snr']
        if key not in snr_groups:
            snr_groups[key] = []
        snr_groups[key].append(r)

    header = f"{'SNR (dB)':>10} │ {'Samples':>7} │ {'SDR Before':>10} │ {'SDR After':>10} │ {'Δ SDR':>10} │ {'Min Δ':>8} │ {'Max Δ':>8}"
    print(f"\n{header}")
    print("─" * len(header))

    for snr_key in sorted(snr_groups.keys()):
        group = snr_groups[snr_key]
        n = len(group)
        avg_before = np.mean([r['sdr_before'] for r in group])
        avg_after = np.mean([r['sdr_after'] for r in group])
        avg_imp = np.mean([r['improvement'] for r in group])
        min_imp = np.min([r['improvement'] for r in group])
        max_imp = np.max([r['improvement'] for r in group])

        print(f"{snr_key:10.1f} │ {n:7d} │ {avg_before:+10.2f} │ {avg_after:+10.2f} │ {avg_imp:+10.2f} │ {min_imp:+8.2f} │ {max_imp:+8.2f}")

    print("─" * len(header))

    # Overall summary
    all_before = np.mean([r['sdr_before'] for r in results])
    all_after = np.mean([r['sdr_after'] for r in results])
    all_imp = np.mean([r['improvement'] for r in results])
    std_imp = np.std([r['improvement'] for r in results])
    median_imp = np.median([r['improvement'] for r in results])

    print(f"{'OVERALL':>10} │ {len(results):7d} │ {all_before:+10.2f} │ {all_after:+10.2f} │ {all_imp:+10.2f} │        │        ")
    print("═" * len(header))

    print(f"\n  Average SDR Improvement:  {all_imp:+.2f} dB")
    print(f"  Median SDR Improvement:   {median_imp:+.2f} dB")
    print(f"  Std Dev:                  {std_imp:.2f} dB")
    print(f"  SDR Before (avg):         {all_before:+.2f} dB")
    print(f"  SDR After  (avg):         {all_after:+.2f} dB")

    # Per noise-type breakdown (if synthesized)
    if not args.use_training_data:
        noise_groups = {}
        for r in results:
            # Group by noise category (strip trailing number)
            noise_base = re.sub(r'_\d+$', '', r['noise_type'])
            if noise_base not in noise_groups:
                noise_groups[noise_base] = []
            noise_groups[noise_base].append(r)

        if len(noise_groups) > 1:
            print(f"\n{'─' * 50}")
            print(f"  Per Noise Type Breakdown")
            print(f"{'─' * 50}")
            print(f"  {'Noise Type':<25} │ {'Samples':>7} │ {'Avg Δ SDR':>10}")
            print(f"  {'─' * 25}─┼─{'─' * 7}─┼─{'─' * 10}")
            for noise_type in sorted(noise_groups.keys()):
                group = noise_groups[noise_type]
                avg = np.mean([r['improvement'] for r in group])
                print(f"  {noise_type:<25} │ {len(group):7d} │ {avg:+10.2f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test a trained denoiser model and display SDR results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 4-test_model.py                          # Quick test (10 samples)
  python 4-test_model.py --samples 50             # Thorough test
  python 4-test_model.py --snr 0 5 10 15 20      # Custom SNR levels
  python 4-test_model.py --save-audio             # Save denoised samples
  python 4-test_model.py --use-training-data      # Test on training pairs
        """
    )

    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (auto-detects if not specified)')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of test samples per SNR level (default: 10)')
    parser.add_argument('--snr', type=float, nargs='+', default=[0, 5, 10, 15, 20],
                        help='SNR levels in dB to test (default: 0 5 10 15 20)')
    parser.add_argument('--clean-dir', type=str, default=None,
                        help='Clean test audio directory (default: dataset/clean_test)')
    parser.add_argument('--noise-dir', type=str, default=None,
                        help='Noise test audio directory (default: dataset/noise_test)')
    parser.add_argument('--clean-dir-train', type=str, default=None,
                        help='Clean training directory (for --use-training-data)')
    parser.add_argument('--noisy-dir-train', type=str, default=None,
                        help='Noisy training directory (for --use-training-data)')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save noisy/denoised/clean audio files')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Directory for saved audio (default: test_output)')
    parser.add_argument('--use-training-data', action='store_true',
                        help='Use pre-mixed training data pairs instead of synthesizing')

    args = parser.parse_args()
    run_test(args)
