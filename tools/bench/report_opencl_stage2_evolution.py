"""Compare a frozen OpenCL baseline, current OpenCL and current CUDA A/B."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics as st


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder):
    manifest = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise RuntimeError('Incomplete A/B matrix')
    groups = defaultdict(list)
    for line in (folder / 'measurements.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        log = Path(r['log'])
        if sha(log) != r['log_sha256'] or sha(log.parent / 'results.jsonl') != r['result_sha256']:
            raise RuntimeError('Modified raw evidence')
        if r['source_build'].lower() != manifest['binaries'][r['backend']]['sha256'].lower():
            raise RuntimeError('Binary identity mismatch')
        groups[(r['bits'], r['B2'], r['backend'])].append(r)
    for samples in groups.values():
        if len(samples) != manifest['repeats'] or len({(r['save_sha256'], r['D'], r['P']) for r in samples}) != 1:
            raise RuntimeError('Changing geometry/input or wrong repeat count')
    return manifest, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.figure.exists():
        p.error('Choose new derived output paths')
    old_manifest, old = load(a.baseline)
    current_manifest, current = load(a.current)
    versions = [old_manifest['versions']['opencl'], current_manifest['versions']['opencl'], current_manifest['versions']['cuda']]
    versions = [v.removeprefix('ECM-ELF ') for v in versions]
    if set(old) != set(current):
        raise RuntimeError('Matrices do not cover the same cases')
    cells = []
    for bits, b2, backend in sorted(current):
        if backend != 'opencl':
            continue
        groups = [old[(bits, b2, 'opencl')], current[(bits, b2, 'opencl')], current[(bits, b2, 'cuda')]]
        if len({(r['save_sha256'], r['D'], r['P']) for g in groups for r in g}) != 1:
            raise RuntimeError('Versions/backends used different inputs or D/P')
        r = {'bits': bits, 'B2': b2, 'D': groups[0][0]['D'], 'P': groups[0][0]['P']}
        for name, samples in zip(('old_opencl', 'opencl', 'cuda'), groups):
            values = [s['engine_seconds'] for s in samples]
            r.update({name: st.median(values), name + '_min': min(values), name + '_max': max(values),
                      name + '_n': len(values), name + '_process': st.median(s['process_seconds'] for s in samples)})
        r['speedup'] = r['old_opencl'] / r['opencl']
        r['reduction_percent'] = 100 * (1 - r['opencl'] / r['old_opencl'])
        r['opencl_over_cuda'] = r['opencl'] / r['cuda']
        r['process_over_cuda'] = r['opencl_process'] / r['cuda_process']
        cells.append(r)
    a.out.mkdir(parents=True)
    with (a.out / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    totals = {name: sum(c[name] for c in cells) for name in ('old_opencl', 'opencl', 'cuda')}
    summary = {'baseline_manifest': old_manifest, 'current_manifest': current_manifest,
               'cells': cells, 'sum_engine_medians': totals,
               'sum_process_medians': {name: sum(c[name + '_process'] for c in cells) for name in totals},
               'baseline_measurements_sha256': sha(a.baseline / 'measurements.jsonl'),
               'current_measurements_sha256': sha(a.current / 'measurements.jsonl')}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR', str(a.out / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounds = sorted({c['B2'] for c in cells})
    fig, axes = plt.subplots(2, len(bounds), figsize=(13, 8), layout='constrained', squeeze=False)
    for col, b2 in enumerate(bounds):
        group = sorted((c for c in cells if c['B2'] == b2), key=lambda c: c['bits'])
        x = [c['bits'] for c in group]
        for name, label, color in [('old_opencl', f'OpenCL {versions[0]} (frozen)', '#94a3b8'),
                                   ('opencl', f'OpenCL {versions[1]}', '#ea580c'), ('cuda', f'CUDA {versions[2]} (replayed)', '#2563eb')]:
            y = [c[name] for c in group]
            axes[0, col].errorbar(x, y, yerr=[[c[name] - c[name + '_min'] for c in group],
                                                   [c[name + '_max'] - c[name] for c in group]],
                                  marker='o', capsize=3, label=label, color=color)
        axes[0, col].set(title=f'B2 = {b2:,}; D = {group[0]["D"]:,}', xlabel='Actual cofactor bits',
                         ylabel='Full engine seconds (log scale)', yscale='log')
        axes[0, col].legend()
        axes[1, col].plot(x, [c['speedup'] for c in group], marker='o', label='Old OpenCL / new OpenCL')
        axes[1, col].plot(x, [c['opencl_over_cuda'] for c in group], marker='s', label='New OpenCL / CUDA')
        axes[1, col].axhline(1, color='#888', linestyle='--')
        axes[1, col].set(xlabel='Actual cofactor bits', ylabel='Time ratio (definitions in legend)')
        axes[1, col].legend()
        for ax in axes[:, col]:
            ax.set_xticks(range(0, 9000, 2000)); ax.grid(alpha=.2)
    sample = next(samples[0] for key, samples in current.items() if key[2] == 'opencl')
    device = re.search(r'stage2_opencl:.*? name=(.*?) target_bits=', Path(sample['log']).read_text(encoding='utf-8-sig'))
    device = device[1] if device else 'Selected OpenCL device'
    with (a.current / 'gpu.csv').open(encoding='utf-8') as file:
        clocks = {r['clocks.current.graphics [MHz]'].strip() for r in csv.DictReader(file, skipinitialspace=True)}
    clock = next(iter(clocks)) + ' observed' if len(clocks) == 1 else 'mixed sampled clocks'
    fig.suptitle('Stage2 point scheduling + giant-base reuse + context program cache\n'
                 f'{device} | {clock} | n={current_manifest["repeats"]} | no discarded warmup | full curves/checks')
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=170); plt.close(fig)
    print(json.dumps({'sum_engine_medians': totals, 'cells': cells}))


if __name__ == '__main__':
    main()
