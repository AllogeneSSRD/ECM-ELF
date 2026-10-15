"""Summarize exact outer-plan samples and fixed-D full Stage2 A/B."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    operator = json.loads(a.operator.read_text())
    stage2 = json.loads(a.stage2.read_text())
    if not operator['complete'] or not stage2['complete'] or stage2['identity']['variant'] != 'mid-outer':
        raise ValueError('complete verified outer studies required')
    a.output.mkdir(parents=True, exist_ok=True)
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in stage2['summaries']:
        rows.append(dict(case=s['case'], off_seconds=s['off']['engine_seconds'],
                         on_seconds=s['on']['engine_seconds'], gain_percent=s['gain_percent'],
                         off_mad=s['off']['engine_seconds_mad'], on_mad=s['on']['engine_seconds_mad'],
                         off_peak_mib=s['off']['full_peak_bytes']/2**20,
                         on_peak_mib=s['on']['full_peak_bytes']/2**20,
                         samples_per_side=s['off']['samples']))
    for row in rows:
        # Propagated MAD is a dispersion scale, not a confidence interval.
        row['gain_mad_scale'] = 100*row['on_seconds']/row['off_seconds']*math.hypot(
            row['on_mad']/row['on_seconds'], row['off_mad']/row['off_seconds'])
    baseline = sum(r['off_seconds'] for r in rows)
    candidate = sum(r['on_seconds'] for r in rows)
    summary = dict(operator_binary_sha256=operator['binary_sha256'],
                   stage2_binary_sha256=stage2['identity']['identity']['binary_sha256'],
                   operator_samples=sum(len(s['samples']) for s in operator['statistics']),
                   operator_checked_words=sum(int(r['checked_words']) for s in operator['statistics'] for r in s['samples']),
                   stage2_samples=sum(2*r['samples_per_side'] for r in rows),
                   sum_case_medians_off_seconds=baseline, sum_case_medians_on_seconds=candidate,
                   matrix_gain_percent=100*(1-candidate/baseline), rows=rows,
                   timing='Stage2 engine init+main; sum of case medians; not a single production workload')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (a.output/'stage2.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 2]})
    for batch in (1, 4):
        selected = [s for s in operator['statistics'] if s['batch'] == batch and s['operation'] == 2 and 21 <= s['k'] <= 23]
        axes[0].plot([s['k'] for s in selected], [s['modes']['2']['gain_percent'] for s in selected], 'o-', label=f'batch={batch}')
    axes[0].axhline(0, color='#555555', linewidth=.7)
    axes[0].set(xticks=[21, 22, 23], xlabel='NTT log2(length)', ylabel='Reduction in elapsed time (%)', title='Full convolution: forced M6 vs M4')
    axes[0].legend()
    labels = [r['case'].replace('_', '\n', 1) for r in rows]
    bars = axes[1].bar(range(len(rows)), [r['gain_percent'] for r in rows],
                       yerr=[r['gain_mad_scale'] for r in rows], capsize=3,
                       error_kw={'linewidth': .8, 'alpha': .7},
                       color=['#27836e' if r['gain_percent'] >= 0 else '#b95756' for r in rows])
    axes[1].axhline(0, color='#555555', linewidth=.7)
    for bar, row in zip(bars, rows):
        value = row['gain_percent']
        offset=row['gain_mad_scale']+.2
        axes[1].text(bar.get_x()+bar.get_width()/2, value+(offset if value >= 0 else -offset), f'{value:.1f}',
                     ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
    axes[1].set(xticks=range(len(rows)), xticklabels=labels, ylabel='Reduction in engine init+main (%)',
                title=f"Fixed D Stage2: k21–23 M6 (matrix {summary['matrix_gain_percent']:+.2f}%)\nError bars: propagated MAD scale, not CI")
    axes[1].set_ylim(min(r['gain_percent']-r['gain_mad_scale'] for r in rows)-1,
                     max(r['gain_percent']+r['gain_mad_scale'] for r in rows)+1)
    axes[1].tick_params(axis='x', labelsize=7, rotation=42)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('RTX 4060 Laptop | default 55 W / 1800 MHz cap | no clock/power changes')
    fig.tight_layout()
    fig.savefig(a.figure, dpi=180, bbox_inches='tight')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
