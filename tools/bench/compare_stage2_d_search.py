"""Audit and compare two complete default Stage2 matrices, retaining both sample counts.

This is a screening comparison, not a paired causal experiment. Original logs,
samples and outliers are never rewritten or discarded. Retests remain separate.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/test'))
from check_stage2_version import audit
from bench_stage2_version import load


def matrix(study, repeats):
    checked = audit(study, repeats)
    plan = load(study / 'plan.json')
    cases = {c['id']: c for c in plan['cases']}
    groups = defaultdict(list)
    for r in plan['runs']:
        groups[r['case'], r['B2'], r['engine']].append(r)
    rows = {}
    for key, runs in groups.items():
        case = cases[key[0]]; first = runs[0]
        rows[key] = dict(case=key[0], bits=case['bits'], exponent=case['exponent'],
            B1=case['b1'], sigma=case['sigma'], B2=key[1], nominal_B2=first['nominal_B2'],
            engine=key[2], N_hex=case['N_hex'], save_bytes=Path(case['save']).read_bytes(),
            seconds=statistics.median(r['engine_seconds'] for r in runs),
            min_seconds=min(r['engine_seconds'] for r in runs),
            max_seconds=max(r['engine_seconds'] for r in runs),
            process_seconds=statistics.median(r['process_seconds'] for r in runs),
            D=first['D'], P=first['P'], carrier=first['carrier_exponent'],
            run_ids=[r['id'] for r in runs])
    return plan, rows, checked


def compare(baseline, candidate, relative=.05, absolute=.25):
    if set(baseline) != set(candidate) or len(baseline) != 52:
        raise ValueError('Need the same 52 exact conditions')
    result = []
    for key, before in baseline.items():
        after = candidate[key]
        for field in ('N_hex', 'save_bytes', 'bits', 'B1', 'sigma', 'nominal_B2', 'carrier'):
            if before[field] != after[field]:
                raise ValueError('Comparison identity differs: ' + field + ': ' + str(key))
        delta = after['seconds'] - before['seconds']
        ratio = after['seconds'] / before['seconds']
        result.append({k: after[k] for k in ('case', 'bits', 'exponent', 'B1', 'sigma', 'B2', 'nominal_B2', 'engine')} |
            dict(baseline_seconds=before['seconds'], candidate_seconds=after['seconds'],
                 baseline_min=before['min_seconds'], baseline_max=before['max_seconds'],
                 candidate_min=after['min_seconds'], candidate_max=after['max_seconds'],
                 ratio=ratio, change_percent=100*(ratio-1), delta_seconds=delta,
                 baseline_process_seconds=before['process_seconds'], candidate_process_seconds=after['process_seconds'],
                 baseline_D=before['D'], baseline_P=before['P'], D=after['D'], P=after['P'],
                 same_plan=(before['D'], before['P']) == (after['D'], after['P']),
                 retest_required=ratio > 1+relative and delta > absolute))
    return sorted(result, key=lambda r: (r['nominal_B2'], r['bits'], r['engine']))


def plot(rows, versions, counts, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    bounds = sorted({r['nominal_B2'] for r in rows})
    fig, axes = plt.subplots(1, len(bounds), figsize=(16, 6.5), sharey=True)
    for ax, bound in zip(np.atleast_1d(axes), bounds):
        selected = [r for r in rows if r['nominal_B2'] == bound]
        for mode, label, color in (('gpu', 'Ordinary N', '#24699c'), ('gpu-carrier', 'Mersenne carrier', '#d16a32')):
            cells = sorted((r for r in selected if r['engine'] == mode), key=lambda r: r['bits'])
            ax.plot(np.arange(13), [r['change_percent'] for r in cells], '-o', color=color, label=label, ms=4)
            low = [100*(r['candidate_min']/r['baseline_max']-1) for r in cells]
            high = [100*(r['candidate_max']/r['baseline_min']-1) for r in cells]
            ax.fill_between(np.arange(13), low, high, color=color, alpha=.13)
        ax.axhline(0, color='#555', lw=1)
        ax.set_xticks(np.arange(13), [str(r['bits']) for r in cells], rotation=55)
        ax.set_title(f'Request B2 {bound:.1e}; exact adjusted bounds per input')
        ax.set_xlabel('Actual cofactor N (bits)'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Full engine time change (%); positive = slower')
    fig.suptitle(f'ECM-ELF {versions[1]} (n={counts[1]}) vs {versions[0]} (n={counts[0]}) | default D, no tune')
    fig.text(.5, .025, 'RTX 4060 Laptop, GPU1, SM 1500 MHz | 6300 / 640 / 256 MiB | Stage2 init + main incl. checks and GCD\n'
        'Separate sessions; no causal claim. Shading combines observed min/max, not a confidence interval. All samples retained.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(target.with_suffix('.' + ext), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline-repeats', type=int, default=3)
    p.add_argument('--candidate-repeats', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path)
    p.add_argument('--retest-relative', type=float, default=.05)
    p.add_argument('--retest-seconds', type=float, default=.25)
    args = p.parse_args()
    if min(args.retest_relative, args.retest_seconds) < 0:
        p.error('Retest thresholds cannot be negative')
    first, before, first_audit = matrix(args.baseline, args.baseline_repeats)
    last, after, last_audit = matrix(args.candidate, args.candidate_repeats)
    for field in ('device', 'arena_mb', 'fold_mb', 'batch_mb'):
        if first[field] != last[field]:
            raise ValueError('Benchmark protocol differs: ' + field)
    rows = compare(before, after, args.retest_relative, args.retest_seconds)
    totals = {}
    for mode in ('gpu', 'gpu-carrier', 'all'):
        selected = [r for r in rows if mode == 'all' or r['engine'] == mode]
        a = sum(r['baseline_seconds'] for r in selected); b = sum(r['candidate_seconds'] for r in selected)
        totals[mode] = dict(baseline_seconds=a, candidate_seconds=b, change_percent=100*(b/a-1),
                            matrix_nonregression=b <= a)
    def identity(path, plan, repeats, checked):
        return dict(study=str(path.resolve()), version=plan['program_version'], repeats=repeats, audit=checked,
            exe_sha256=hashlib.sha256(Path(plan['exe']).read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256((path/'plan.json').read_bytes()).hexdigest(), gpu=plan['gpu'])
    summary = dict(complete=True, screening_only=True,
        baseline=identity(args.baseline, first, args.baseline_repeats, first_audit),
        candidate=identity(args.candidate, last, args.candidate_repeats, last_audit),
        primary='stage2_full_wall.total=init+main; mandatory checks and GCD included; Stage1 excluded',
        retest_rule=dict(relative=args.retest_relative, absolute_seconds=args.retest_seconds, both_required=True),
        all_default_plans_unchanged=all(r['same_plan'] for r in rows), totals=totals,
        retests=[r for r in rows if r['retest_required']], rows=rows,
        limitations='Independent historical sessions; n=1 is screening, not variance estimation. Retests must retain original samples.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    with (args.output/'cells.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    shutil.copy2(__file__, args.output/'analysis_snapshot.py')
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=False)
        plot(rows, (first['program_version'], last['program_version']),
             (args.baseline_repeats, args.candidate_repeats), args.figures/'default-matrix-change')
    print(json.dumps(dict(totals=totals, unchanged_plans=summary['all_default_plans_unchanged'],
                          retests=len(summary['retests'])), indent=2))


if __name__ == '__main__':
    main()
