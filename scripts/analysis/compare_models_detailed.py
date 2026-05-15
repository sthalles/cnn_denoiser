#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_file_pairs, load_audio, compute_sdr


def load_legacy_inference_module():
    module_path = os.path.join(ROOT_DIR, 'scripts', 'legacy', '3-usar_modelo_v2.py')
    spec = importlib.util.spec_from_file_location('legacy_inference_v2', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(arr, p):
    if len(arr) == 0:
        return float('nan')
    return float(np.percentile(np.asarray(arr, dtype=np.float64), p))


def bootstrap_mean_ci(values, rng, n_boot=2000, alpha=0.05):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return float('nan'), float('nan')
    if vals.size == 1:
        v = float(vals[0])
        return v, v
    means = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        means[i] = np.mean(sample)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def summarize(values, rng):
    vals = np.asarray(values, dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return {
            'n': 0,
            'mean': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'p05': float('nan'),
            'p25': float('nan'),
            'p75': float('nan'),
            'p95': float('nan'),
            'iqr': float('nan'),
            'sem': float('nan'),
            'ci95_low': float('nan'),
            'ci95_high': float('nan'),
        }

    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    ci_low, ci_high = bootstrap_mean_ci(vals, rng)
    p25 = percentile(vals, 25)
    p75 = percentile(vals, 75)

    return {
        'n': n,
        'mean': float(np.mean(vals)),
        'std': std,
        'median': float(np.median(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'p05': percentile(vals, 5),
        'p25': p25,
        'p75': p75,
        'p95': percentile(vals, 95),
        'iqr': float(p75 - p25),
        'sem': sem,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
    }


def sign_test_pvalue_two_sided(wins, losses):
    n = wins + losses
    if n == 0:
        return float('nan')
    z = (wins - n / 2.0) / math.sqrt(n / 4.0)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def fmt(x, nd=3):
    if x is None:
        return 'N/A'
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 'N/A'
    return f"{x:.{nd}f}"


def write_table_md(f, rows, headers):
    f.write('| ' + ' | '.join(headers) + ' |\n')
    f.write('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')
    for row in rows:
        f.write('| ' + ' | '.join(row) + ' |\n')
    f.write('\n')


def main():
    parser = argparse.ArgumentParser(description='Detailed paired model comparison for SDR denoising performance')
    parser.add_argument('--model-a', required=True, help='Path to model A (reference)')
    parser.add_argument('--model-b', required=True, help='Path to model B (challenger)')
    parser.add_argument('--label-a', default='best_model', help='Display label for model A')
    parser.add_argument('--label-b', default='retrained_basic_unet', help='Display label for model B')
    parser.add_argument('--clean-dir', default='dataset/CleanSpeech_training')
    parser.add_argument('--noisy-dir', default='dataset/NoisySpeech_training')
    parser.add_argument('--max-samples', type=int, default=0, help='0 means all matched files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='logs/comparisons/detailed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.model_a):
        raise FileNotFoundError(f"Model A not found: {args.model_a}")
    if not os.path.exists(args.model_b):
        raise FileNotFoundError(f"Model B not found: {args.model_b}")

    legacy = load_legacy_inference_module()
    AudioDenoiser = legacy.AudioDenoiser

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    pairs = get_file_pairs(args.clean_dir, args.noisy_dir)
    if args.max_samples and args.max_samples > 0:
        random.shuffle(pairs)
        pairs = pairs[:args.max_samples]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_csv_path = os.path.join(args.out_dir, f'paired_raw_{ts}.csv')
    json_path = os.path.join(args.out_dir, f'stats_{ts}.json')
    md_path = os.path.join(args.out_dir, f'report_{ts}.md')

    print(f"Loading model A: {args.model_a}")
    denoiser_a = AudioDenoiser(args.model_a)
    print(f"Loading model B: {args.model_b}")
    denoiser_b = AudioDenoiser(args.model_b)

    rows = []

    for idx, (clean_path, noisy_path, snr) in enumerate(pairs, start=1):
        try:
            clean_audio, _ = load_audio(clean_path)
            noisy_audio, _ = load_audio(noisy_path)

            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]

            den_a = denoiser_a.denoise_audio(noisy_audio, show_progress=False)
            den_b = denoiser_b.denoise_audio(noisy_audio, show_progress=False)

            min_len_a = min(len(clean_audio), len(den_a))
            min_len_b = min(len(clean_audio), len(den_b))

            sdr_before = float(compute_sdr(clean_audio[:min_len], noisy_audio[:min_len]))
            sdr_a = float(compute_sdr(clean_audio[:min_len_a], den_a[:min_len_a]))
            sdr_b = float(compute_sdr(clean_audio[:min_len_b], den_b[:min_len_b]))

            imp_a = sdr_a - sdr_before
            imp_b = sdr_b - sdr_before
            delta = imp_b - imp_a

            rows.append({
                'file': os.path.basename(noisy_path),
                'clean_path': clean_path,
                'noisy_path': noisy_path,
                'snr': float(snr),
                'sdr_before': sdr_before,
                'sdr_after_a': sdr_a,
                'sdr_after_b': sdr_b,
                'improvement_a': imp_a,
                'improvement_b': imp_b,
                'delta_b_minus_a': delta,
            })

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(pairs)} files")

        except Exception as e:
            print(f"Skipping {noisy_path}: {e}")

    if not rows:
        raise RuntimeError('No rows were successfully evaluated.')

    with open(raw_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    imp_a_all = [r['improvement_a'] for r in rows]
    imp_b_all = [r['improvement_b'] for r in rows]
    delta_all = [r['delta_b_minus_a'] for r in rows]
    sdr_before_all = [r['sdr_before'] for r in rows]

    stats = {
        'meta': {
            'timestamp': ts,
            'model_a': args.model_a,
            'model_b': args.model_b,
            'label_a': args.label_a,
            'label_b': args.label_b,
            'clean_dir': args.clean_dir,
            'noisy_dir': args.noisy_dir,
            'requested_max_samples': args.max_samples,
            'evaluated_samples': len(rows),
            'raw_csv': raw_csv_path,
            'report_md': md_path,
        },
        'overall': {
            'baseline_sdr': summarize(sdr_before_all, rng),
            'improvement_a': summarize(imp_a_all, rng),
            'improvement_b': summarize(imp_b_all, rng),
            'delta_b_minus_a': summarize(delta_all, rng),
        },
        'per_snr': {},
        'pairwise_competition': {},
    }

    by_snr = defaultdict(list)
    for r in rows:
        by_snr[r['snr']].append(r)

    for snr in sorted(by_snr.keys()):
        grp = by_snr[snr]
        ia = [x['improvement_a'] for x in grp]
        ib = [x['improvement_b'] for x in grp]
        d = [x['delta_b_minus_a'] for x in grp]
        sb = [x['sdr_before'] for x in grp]

        wins = sum(1 for x in d if x > 0)
        losses = sum(1 for x in d if x < 0)
        ties = sum(1 for x in d if x == 0)
        win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else float('nan')

        stats['per_snr'][str(snr)] = {
            'baseline_sdr': summarize(sb, rng),
            'improvement_a': summarize(ia, rng),
            'improvement_b': summarize(ib, rng),
            'delta_b_minus_a': summarize(d, rng),
            'wins_b': wins,
            'losses_b': losses,
            'ties': ties,
            'win_rate_b_excl_ties': win_rate,
            'sign_test_pvalue_two_sided': sign_test_pvalue_two_sided(wins, losses),
        }

    wins = sum(1 for x in delta_all if x > 0)
    losses = sum(1 for x in delta_all if x < 0)
    ties = sum(1 for x in delta_all if x == 0)
    n_non_tie = wins + losses

    stats['pairwise_competition'] = {
        'wins_b': wins,
        'losses_b': losses,
        'ties': ties,
        'win_rate_b_excl_ties': float(wins / n_non_tie) if n_non_tie else float('nan'),
        'win_rate_b_incl_ties': float(wins / len(delta_all)),
        'loss_rate_b_incl_ties': float(losses / len(delta_all)),
        'delta_gt_0_5db_rate': float(sum(1 for x in delta_all if x > 0.5) / len(delta_all)),
        'delta_gt_1_0db_rate': float(sum(1 for x in delta_all if x > 1.0) / len(delta_all)),
        'delta_gt_2_0db_rate': float(sum(1 for x in delta_all if x > 2.0) / len(delta_all)),
        'delta_lt_minus_0_5db_rate': float(sum(1 for x in delta_all if x < -0.5) / len(delta_all)),
        'delta_lt_minus_1_0db_rate': float(sum(1 for x in delta_all if x < -1.0) / len(delta_all)),
        'sign_test_pvalue_two_sided': sign_test_pvalue_two_sided(wins, losses),
        'cohens_d_paired': float(np.mean(delta_all) / np.std(delta_all, ddof=1)) if len(delta_all) > 1 and np.std(delta_all, ddof=1) > 0 else float('nan'),
        'corr_input_sdr_vs_imp_a': float(np.corrcoef(np.asarray(sdr_before_all), np.asarray(imp_a_all))[0, 1]) if len(rows) > 1 else float('nan'),
        'corr_input_sdr_vs_imp_b': float(np.corrcoef(np.asarray(sdr_before_all), np.asarray(imp_b_all))[0, 1]) if len(rows) > 1 else float('nan'),
    }

    low_snr_rows = [r for r in rows if r['snr'] <= 5.0]
    high_snr_rows = [r for r in rows if r['snr'] >= 15.0]

    def subset_stats(subrows):
        if not subrows:
            return None
        ia = [x['improvement_a'] for x in subrows]
        ib = [x['improvement_b'] for x in subrows]
        d = [x['delta_b_minus_a'] for x in subrows]
        return {
            'n': len(subrows),
            'improvement_a': summarize(ia, rng),
            'improvement_b': summarize(ib, rng),
            'delta_b_minus_a': summarize(d, rng),
        }

    stats['subsets'] = {
        'low_snr_le_5db': subset_stats(low_snr_rows),
        'high_snr_ge_15db': subset_stats(high_snr_rows),
    }

    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    with open(md_path, 'w') as f:
        f.write('# Detailed Model Performance Statistics\n\n')
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write('## Models Compared\n\n')
        f.write(f"- Model A ({args.label_a}): `{args.model_a}`\n")
        f.write(f"- Model B ({args.label_b}): `{args.model_b}`\n")
        f.write(f"- Evaluated samples: {len(rows)}\n\n")

        f.write('## Overall Summary\n\n')
        overall_rows = [
            ['Baseline SDR (input noisy)', fmt(stats['overall']['baseline_sdr']['mean']), fmt(stats['overall']['baseline_sdr']['median']), fmt(stats['overall']['baseline_sdr']['std']), fmt(stats['overall']['baseline_sdr']['ci95_low']), fmt(stats['overall']['baseline_sdr']['ci95_high'])],
            [f"SDR improvement ({args.label_a})", fmt(stats['overall']['improvement_a']['mean']), fmt(stats['overall']['improvement_a']['median']), fmt(stats['overall']['improvement_a']['std']), fmt(stats['overall']['improvement_a']['ci95_low']), fmt(stats['overall']['improvement_a']['ci95_high'])],
            [f"SDR improvement ({args.label_b})", fmt(stats['overall']['improvement_b']['mean']), fmt(stats['overall']['improvement_b']['median']), fmt(stats['overall']['improvement_b']['std']), fmt(stats['overall']['improvement_b']['ci95_low']), fmt(stats['overall']['improvement_b']['ci95_high'])],
            [f"Delta ({args.label_b} - {args.label_a})", fmt(stats['overall']['delta_b_minus_a']['mean']), fmt(stats['overall']['delta_b_minus_a']['median']), fmt(stats['overall']['delta_b_minus_a']['std']), fmt(stats['overall']['delta_b_minus_a']['ci95_low']), fmt(stats['overall']['delta_b_minus_a']['ci95_high'])],
        ]
        write_table_md(f, overall_rows, ['Metric', 'Mean (dB)', 'Median (dB)', 'Std (dB)', '95% CI Low', '95% CI High'])

        f.write('## Pairwise Competition\n\n')
        comp = stats['pairwise_competition']
        comp_rows = [
            ['Wins (Model B better)', str(comp['wins_b'])],
            ['Losses (Model B worse)', str(comp['losses_b'])],
            ['Ties', str(comp['ties'])],
            ['Win rate excl ties', fmt(comp['win_rate_b_excl_ties'])],
            ['Win rate incl ties', fmt(comp['win_rate_b_incl_ties'])],
            ['Delta > +0.5 dB rate', fmt(comp['delta_gt_0_5db_rate'])],
            ['Delta > +1.0 dB rate', fmt(comp['delta_gt_1_0db_rate'])],
            ['Delta > +2.0 dB rate', fmt(comp['delta_gt_2_0db_rate'])],
            ['Delta < -0.5 dB rate', fmt(comp['delta_lt_minus_0_5db_rate'])],
            ['Delta < -1.0 dB rate', fmt(comp['delta_lt_minus_1_0db_rate'])],
            ['Sign-test p-value (2-sided)', fmt(comp['sign_test_pvalue_two_sided'])],
            ["Cohen's d (paired deltas)", fmt(comp['cohens_d_paired'])],
            [f"Corr(input SDR, {args.label_a} improvement)", fmt(comp['corr_input_sdr_vs_imp_a'])],
            [f"Corr(input SDR, {args.label_b} improvement)", fmt(comp['corr_input_sdr_vs_imp_b'])],
        ]
        write_table_md(f, comp_rows, ['Statistic', 'Value'])

        f.write('## Per-SNR Detailed Statistics\n\n')
        snr_rows = []
        for snr in sorted(stats['per_snr'].keys(), key=lambda x: float(x)):
            rec = stats['per_snr'][snr]
            snr_rows.append([
                snr,
                str(rec['improvement_a']['n']),
                fmt(rec['improvement_a']['mean']),
                fmt(rec['improvement_b']['mean']),
                fmt(rec['delta_b_minus_a']['mean']),
                fmt(rec['win_rate_b_excl_ties']),
                fmt(rec['sign_test_pvalue_two_sided'])
            ])
        write_table_md(f, snr_rows, ['SNR (dB)', 'N', f'Mean {args.label_a} imp', f'Mean {args.label_b} imp', 'Mean delta (B-A)', 'Win rate B', 'Sign-test p'])

        f.write('## Distribution Statistics (Overall)\n\n')
        dist_rows = []
        for label, key in [(args.label_a, 'improvement_a'), (args.label_b, 'improvement_b'), (f'{args.label_b} - {args.label_a}', 'delta_b_minus_a')]:
            s = stats['overall'][key]
            dist_rows.append([
                label,
                fmt(s['min']), fmt(s['p05']), fmt(s['p25']), fmt(s['median']), fmt(s['p75']), fmt(s['p95']), fmt(s['max']), fmt(s['iqr'])
            ])
        write_table_md(f, dist_rows, ['Series', 'Min', 'P05', 'P25', 'Median', 'P75', 'P95', 'Max', 'IQR'])

        f.write('## Subset Analysis\n\n')
        for subset_name, subset in stats['subsets'].items():
            f.write(f"### {subset_name}\n\n")
            if not subset:
                f.write('No samples in this subset.\n\n')
                continue
            subset_rows = [
                ['N', str(subset['n'])],
                [f"Mean improvement ({args.label_a})", fmt(subset['improvement_a']['mean'])],
                [f"Mean improvement ({args.label_b})", fmt(subset['improvement_b']['mean'])],
                ['Mean delta (B-A)', fmt(subset['delta_b_minus_a']['mean'])],
                ['Median delta (B-A)', fmt(subset['delta_b_minus_a']['median'])],
                ['95% CI delta low', fmt(subset['delta_b_minus_a']['ci95_low'])],
                ['95% CI delta high', fmt(subset['delta_b_minus_a']['ci95_high'])],
            ]
            write_table_md(f, subset_rows, ['Metric', 'Value'])

        f.write('## Artifacts\n\n')
        f.write(f"- Raw paired rows CSV: `{raw_csv_path}`\n")
        f.write(f"- JSON stats: `{json_path}`\n")
        f.write(f"- This report: `{md_path}`\n")

    print(f"Wrote raw CSV: {raw_csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown report: {md_path}")


if __name__ == '__main__':
    main()
