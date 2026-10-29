"""Verify frozen multi-curve evidence, summarize throughput and plot costs.

Each sample is a cohort of fixed total work, not an individual curve. CPU is
job CPU seconds / cohort wall; memory is simultaneous Windows job commit.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(folder):
    manifest = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise RuntimeError('Incomplete experiment')
    for binary in manifest['binaries'].values():
        root = Path(binary['exe']).parent
        for file, expected in binary['files'].items():
            if sha(root / file) != expected:
                raise RuntimeError('Modified frozen build: ' + str(root / file))
    for inp in manifest['inputs']:
        for save in inp['saves']:
            if sha(Path(save['path'])) != save['sha256']:
                raise RuntimeError('Modified input save')
    rows = [json.loads(s) for s in (folder / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    telemetry = defaultdict(list)
    with (folder / 'gpu.csv').open(encoding='utf-8') as file:
        for sample in csv.DictReader(file):
            telemetry[sample['cohort']].append(sample)
    if len(rows) != manifest['cohorts']:
        raise RuntimeError('Cohort count mismatch')
    for row in rows:
        samples = telemetry[row['name']]
        row['sampled_device_used_MiB'] = max(int(re.search(r'\d+', s['memory_used'])[0]) for s in samples)
        if row['curves'] != manifest['curves'] or row['accounting']['active_processes']:
            raise RuntimeError('Incomplete process tree')
        backend = 'cuda' if row['mode'] == 'cuda' else 'opencl'
        for task in row['tasks']:
            log = Path(task['log'])
            if sha(log) != task['log_sha256'] or sha(log.parent / 'results.jsonl') != task['result_sha256']:
                raise RuntimeError('Modified raw log/result')
            result = json.loads((log.parent / 'results.jsonl').read_text(encoding='utf-8-sig'))
            if (result['source-build'].lower() != manifest['binaries'][backend]['sha256'] or
                    result['status'] != 'NF' or result['d'] != manifest['D'] or result['b2'] != manifest['B2']):
                raise RuntimeError('Binary, verdict or fixed-shape mismatch')
    return manifest, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.figure.exists():
        p.error('Choose new derived paths')
    groups, identities = defaultdict(list), {}
    sources = []
    for folder in a.run:
        manifest, rows = read(folder)
        sources.append({'folder': str(folder), 'manifest_sha256': sha(folder / 'completed.json')})
        for row in rows:
            key = (manifest['D'], manifest['B2'], row['bits'], row['mode'], row['parallel'])
            inp = next(i for i in manifest['inputs'] if i['bits'] == row['bits'])
            identity = (tuple(s['sha256'] for s in inp['saves']), manifest['curves'], manifest['B1'],
                        tuple(sorted((k, v['sha256']) for k, v in manifest['binaries'].items())),
                        tuple(sorted(manifest['inherited_environment'].items())))
            if key in identities and identities[key] != identity:
                raise RuntimeError('Cannot pool different inputs/builds/environments')
            identities[key] = identity
            groups[key].append(row)
    cells = []
    for (d, b2, bits, mode, count), rows in sorted(groups.items()):
        rates = [r['curves_per_second'] for r in rows]
        cells.append({'D': d, 'B2': b2, 'bits': bits, 'mode': mode, 'workers': count, 'n': len(rows),
                      'curves_per_s': st.median(rates), 'rate_min': min(rates), 'rate_max': max(rates),
                      'wall_s': st.median(r['wall_seconds'] for r in rows),
                      'cpu_cores': st.median(r['cpu_core_equivalents'] for r in rows),
                      'cpu_s_per_curve': st.median(r['accounting']['cpu_seconds'] / r['curves'] for r in rows),
                      'peak_job_commit_MiB': st.median(r['accounting']['peak_job_commit_bytes'] / 2**20 for r in rows),
                      'sampled_device_used_MiB': st.median(r['sampled_device_used_MiB'] for r in rows),
                      'curve_process_s': st.median(st.median(t['process_seconds'] for t in r['tasks']) for r in rows),
                      'curve_engine_s': st.median(st.median(t['full']['total'] for t in r['tasks']) for r in rows)})
    for cell in cells:
        baseline = next((r for r in cells if all(r[k] == cell[k] for k in ('D', 'B2', 'bits', 'mode')) and r['workers'] == 1), None)
        cell['speedup'] = cell['curves_per_s'] / baseline['curves_per_s'] if baseline else None
    a.out.mkdir(parents=True)
    (a.out / 'summary.json').write_text(json.dumps({'sources': sources, 'cells': cells}, indent=2), encoding='utf-8')
    with (a.out / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shapes = sorted({(r['D'], r['B2'], r['bits']) for r in cells})
    fig, axes = plt.subplots(4, len(shapes), figsize=(6 * len(shapes), 13), squeeze=False)
    colors = {'opencl_device': '#cc6b2f', 'opencl_cpu': '#238873', 'cuda': '#326eae'}
    labels = {'opencl_device': 'OpenCL: GPU points', 'opencl_cpu': 'OpenCL: CPU points', 'cuda': 'CUDA'}
    for column, (d, b2, bits) in enumerate(shapes):
        for mode, color in colors.items():
            selected = [r for r in cells if (r['D'], r['B2'], r['bits'], r['mode']) == (d, b2, bits, mode)]
            if not selected:
                continue
            selected.sort(key=lambda r: r['workers'])
            x = [r['workers'] for r in selected]
            median = [r['curves_per_s'] for r in selected]
            axes[0, column].errorbar(x, median, yerr=[[r['curves_per_s']-r['rate_min'] for r in selected],
                [r['rate_max']-r['curves_per_s'] for r in selected]], label=labels[mode], color=color, marker='o', capsize=4)
            for row, metric in [(1, 'cpu_cores'), (2, 'peak_job_commit_MiB'), (3, 'sampled_device_used_MiB')]:
                axes[row, column].plot(x, [r[metric] for r in selected], color=color, marker='o')
        axes[0, column].set_title(f'N={bits} bits; D={d}; B2={b2:g}')
        for row in range(4):
            axes[row, column].set_xlabel('Concurrent worker limit (fixed total curves)')
            axes[row, column].grid(alpha=.25)
            axes[row, column].set_xticks(sorted({r['workers'] for r in cells}))
        axes[0, column].legend(fontsize=8)
        for row, label in enumerate(('Complete curves/s', 'Job CPU core equivalents', 'Peak simultaneous job commit (MiB)',
                                     'Sampled NVML device-used peak (MiB)')):
            axes[row, column].set_ylabel(label)
    fig.suptitle('Independent curve processes: median; throughput bars = min/max cohorts\n'
                 'Each n is a fixed-work cohort; job commit is not VRAM; NVML is a sampled device total', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figure, dpi=170)
    plt.close(fig)
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
