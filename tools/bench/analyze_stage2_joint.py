"""Visualize joint-plan scans, full Stage2 A/B and GPU-gap attribution."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator', type=Path, nargs='+', required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--gaps', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    a = p.parse_args()
    operators = [json.loads(path.read_text()) for path in a.operator]
    full = json.loads(a.stage2.read_text())
    if any(not r['complete'] or not r.get('joint') for r in operators) or not full['complete'] or full['identity']['variant'] != 'joint-outer':
        raise ValueError('complete joint studies required')
    shapes = [s for report in operators for s in report['statistics']]
    if len({(s['k'], s['batch'], s['operation']) for s in shapes}) != len(shapes):
        raise ValueError('duplicate formal operator shape')
    a.output.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    operator_rows = []
    for s in shapes:
        reference = '10' if 21 <= s['k'] <= 23 else '0'
        best = min(s['modes'], key=lambda m: s['modes'][m]['median_seconds'])
        operator_rows.append(dict(k=s['k'], batch=s['batch'], reference_mode=int(reference), best_mode=int(best),
            best_gain_percent=100*(1-s['modes'][best]['median_seconds']/s['modes'][reference]['median_seconds']),
            reference_seconds=s['modes'][reference]['median_seconds'], best_seconds=s['modes'][best]['median_seconds']))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    picture = None
    for k, ax in zip(range(20, 28), axes.flat):
        selected = next((s for s in shapes if s['k'] == k and s['batch'] == 1 and s['operation'] == 2), None)
        if selected is None:
            ax.set_title(f'k={k}; not measured')
            ax.axis('off')
            continue
        reference = '10' if 21 <= k <= 23 else '0'
        values = np.array([100*(1-selected['modes'][str(m)]['median_seconds']/selected['modes'][reference]['median_seconds']) for m in range(1, 17)]).reshape(4, 4)
        picture = ax.imshow(values, cmap='RdYlGn', vmin=-40, vmax=8, aspect='auto')
        for row in range(4):
            for col in range(4):
                ax.text(col, row, f'{values[row,col]:+.1f}', ha='center', va='center', fontsize=9)
        ax.set(xticks=range(4), xticklabels=['M5', 'M6', 'M7', 'M8'], yticks=range(4),
               yticklabels=['tile10', 'tile11', 'tile12', 'tile13'], title=f'k={k}; ref='+('tile12/M6' if reference == '10' else 'production'))
    if picture is not None:
        fig.colorbar(picture, ax=axes, label='Elapsed-time reduction (%) | colors clipped below -40 / above +8', shrink=.8)
    fig.suptitle('Full convolution, batch1, CTA512 | RTX 4060 Laptop default55W / 1800MHz cap')
    fig.savefig(a.figures/'stage2_joint_heatmap_20261015.png', dpi=170)
    plt.close(fig)
    rows = []
    for s in full['summaries']:
        x, y = s['off'], s['on']
        rows.append(dict(case=s['case'], off_seconds=x['engine_seconds'], on_seconds=y['engine_seconds'],
            gain_percent=s['gain_percent'], off_mad=x['engine_seconds_mad'], on_mad=y['engine_seconds_mad'],
            off_peak_mib=x['full_peak_bytes']/2**20, on_peak_mib=y['full_peak_bytes']/2**20,
            samples_per_side=x['samples']))
    for name, selected in [('operator', operator_rows), ('stage2', rows)]:
        with (a.output/(name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    baseline, candidate = sum(r['off_seconds'] for r in rows), sum(r['on_seconds'] for r in rows)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')
    errors = [100*r['on_seconds']/r['off_seconds']*math.hypot(r['on_mad']/r['on_seconds'], r['off_mad']/r['off_seconds']) for r in rows]
    ax.bar(range(len(rows)), [r['gain_percent'] for r in rows], yerr=errors, capsize=3,
           color=['#27836e' if r['gain_percent'] >= 0 else '#b95756' for r in rows])
    ax.axhline(0, color='#555555', linewidth=.7)
    ax.set(xticks=range(len(rows)), xticklabels=[r['case'] for r in rows], ylabel='Reduction in engine init+main (%)',
           title=f'Joint tile/outer vs middle-M6 | fixed D matrix {100*(1-candidate/baseline):+.3f}%\n4 samples/side; propagated MAD scale, not CI')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(a.figures/'stage2_joint_full_20261015.png', dpi=170)
    plt.close(fig)
    summary = dict(operator_samples=sum(len(s['samples']) for s in shapes),
        operator_checked_words=sum(int(r['checked_words']) for s in shapes for r in s['samples']),
        operator_binaries=[r['binary_sha256'] for r in operators], stage2_binary=full['identity']['identity']['binary_sha256'],
        stage2_samples=sum(2*r['samples_per_side'] for r in rows), matrix_off_seconds=baseline,
        matrix_on_seconds=candidate, matrix_gain_percent=100*(1-candidate/baseline), rows=rows)
    if a.gaps:
        gaps = json.loads(a.gaps.read_text())
        summary['gaps'] = gaps
        parts = list(gaps['partition_seconds'].items())[:9]
        ranges = list(gaps['outside_runtime_by_range_seconds'].items())[:9]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for ax, data, title in zip(axes, [parts, ranges], ['GPU-gap intersection with CUDA APIs', 'Outside runtime API: innermost CPU range']):
            ax.barh([k.replace('_v3020', '').replace('_v9000', '') for k, v in data][::-1], [v*1000 for k, v in data][::-1], color='#428a9b')
            ax.set(xlabel='Milliseconds (profiled; excluded from A/B)', title=title)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle(f"Own-process GPU event gaps: {gaps['gap_seconds']:.3f}s / {gaps['window_seconds']:.3f}s; NVTX ranges={gaps['nvtx_ranges']}")
        fig.savefig(a.figures/'stage2_prep_gaps_20261015.png', dpi=170)
        plt.close(fig)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'gaps')}, indent=2))


if __name__ == '__main__':
    main()
