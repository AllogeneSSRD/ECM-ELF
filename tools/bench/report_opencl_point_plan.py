"""Verify same-work point-plan A/B evidence and plot full/kernel times.

Each sample is a complete curve. Component times are nested in the full engine,
never added to it. Raw files are read only; derived outputs require new paths.
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
from bench_edwards_gpu_stage1 import integer_expression

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    p.add_argument('--plot-d', type=int, default=2310)
    p.add_argument('--log-time', action='store_true', help='Logarithmic time axes for widely separated widths')
    a = p.parse_args()
    if a.out.exists() or a.figure.exists():
        p.error('Use new derived paths')
    if len({folder.resolve() for folder in a.run}) != len(a.run):
        p.error('Each evidence directory may be supplied only once')
    groups, geometry, sources = defaultdict(list), defaultdict(set), []
    for folder in a.run:
        manifest = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        binaries = {v['name']: v for v in manifest['variants']}
        for spec in binaries.values():
            root = Path(spec['exe']).parent
            for file, digest in spec['files'].items():
                if sha(root / file) != digest:
                    raise RuntimeError('Frozen binary/resource changed')
        rows = [json.loads(s) for s in (folder / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
        if 'matrix_plan' in manifest:
            plan = manifest['matrix_plan']
            expected = {(e, b, repeat, variant) for e in plan['exponents'] for b in plan['B2']
                        for repeat in range(manifest['repeats']) for variant in binaries}
            actual = {(r['exponent'], r['B2'], r['repeat'], r['variant']) for r in rows}
            if actual != expected or len(rows) != len(expected):
                raise RuntimeError('Complete variant matrix differs from its plan')
        run_groups = defaultdict(int)
        for r in rows:
            run_groups[r['bits'], r['B2'], r['D'], r['P'], r['variant']] += 1
        shapes = {key[:4] for key in run_groups}
        if any(run_groups[(*shape, variant)] != manifest['repeats'] for shape in shapes for variant in binaries):
            raise RuntimeError('Missing/repeated cells in a completed matrix')
        sources.append({'folder': str(folder), 'manifest_sha256': sha(folder / 'completed.json')})
        for r in rows:
            log = Path(r['log'])
            if sha(log) != r['log_sha256'] or sha(log.parent / 'results.jsonl') != r['result_sha256']:
                raise RuntimeError('Raw evidence changed')
            runtime = re.search(r'^stage2_opencl:.*?target_bits=(\d+)', log.read_text(encoding='utf-8-sig'), re.M)
            if not runtime or int(runtime[1]) != r['bits']:
                raise RuntimeError('Reported width differs from the actual saved N')
            result = json.loads((log.parent / 'results.jsonl').read_text(encoding='utf-8-sig'))
            if (result['source-build'].lower() != binaries[r['variant']]['sha256'] or result['status'] != 'NF'
                    or result['d'] != r['D'] or result['b2'] != r['B2'] or r['checks']['bad'] != 0
                    or result.get('factors', []) or result['errors']['invariant'] or result['stage2-engine']['bad_factors']
                    or result['stage2-engine']['carrier_exponent'] != 0 or r['checks']['startup_cases'] < 1
                    or r['checks']['checked_words'] < 1):
                raise RuntimeError('Binary, geometry or verdict mismatch')
            saved = folder / 'inputs' / Path(result['stage2-engine']['save']).name
            if sha(saved) != r['save_sha256']:
                raise RuntimeError('Frozen save changed')
            line = next(s for s in saved.read_text(encoding='utf-8-sig').splitlines() if 'SIGMA=' in s)
            point = dict(part.strip().split('=', 1) for part in line.split(';') if part.strip())
            if int(result['N']) != integer_expression(point['N']) or result['b1'] != int(float(point['B1'])):
                raise RuntimeError('Result differs from the exact saved N/B1')
            if point['METHOD'] == 'ECM-EDWARDS-AM' and result.get('Edwards', {}).get('sigma') != int(point['SIGMA']):
                raise RuntimeError('Saved Edwards curve identity differs')
            key = r['bits'], r['B2'], r['D'], r['P']
            geometry[key].add((r['save_sha256'], r['ntt_lengths'], r['checks']['checked_words'],
                               r['points']['points'], r['points']['d2h_bytes']))
            groups[(*key, r['variant'])].append(r)
    if any(len(values) != 1 for values in geometry.values()):
        raise RuntimeError('Variants changed the input, NTT work, points or original GMP coverage')
    cells = []
    for (bits, b2, d, degree, variant), rows in sorted(groups.items()):
        if len({r['binary_sha256'] for r in rows}) != 1:
            raise RuntimeError('Do not pool changing binaries')
        cell = {'bits': bits, 'B2': b2, 'D': d, 'P': degree, 'variant': variant, 'n': len(rows)}
        for name, extract in [('engine_s', lambda r: r['full']['total']), ('process_s', lambda r: r['process_seconds']),
                ('point_kernel_s', lambda r: r['points']['kernel']), ('point_launches', lambda r: r['points']['launches']),
                ('giant_s', lambda r: r['split']['giant']),
                ('chain_points', lambda r: r.get('giant_chain',{}).get('points',0)),
                ('chain_seed_pairs', lambda r: r.get('giant_chain',{}).get('seed_pairs',0)),
                ('chain_recoveries', lambda r: r.get('giant_chain',{}).get('recovery_commands',0)),
                ('chain_guard_s', lambda r: r.get('giant_chain',{}).get('guard_and_recovery_seconds',0)),
                ('giant_kernel_s', lambda r: r.get('giant_chain',{}).get('giant_kernel',0)),
                ('chain_max_command_s', lambda r: r.get('giant_chain',{}).get('max_kernel_seconds',0)),
                ('max_command_s', lambda r: r['point_command']['max_kernel_seconds']),
                ('owned_peak_bytes', lambda r: r['payload']['peak_bytes'])]:
            values = [extract(r) for r in rows]
            cell.update({name: st.median(values), name + '_min': min(values), name + '_max': max(values)})
        cells.append(cell)
    a.out.mkdir(parents=True)
    (a.out / 'summary.json').write_text(json.dumps({'sources': sources, 'cells': cells}, indent=2), encoding='utf-8')
    with (a.out / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    selected = [r for r in cells if r['D'] == a.plot_d]
    bounds = sorted({r['B2'] for r in selected})
    if not bounds:
        raise RuntimeError('No cells match plot D')
    variants = sorted({r['variant'] for r in selected})
    width = .8 / len(variants)
    colors = {'legacy': '#8b8b8b', 'default': '#258674', 'manual768': '#c97b38'}
    fig, axes = plt.subplots(2, len(bounds), figsize=(6 * len(bounds), 7), squeeze=False)
    for column, b2 in enumerate(bounds):
        bits = sorted({r['bits'] for r in selected if r['B2'] == b2})
        x = np.arange(len(bits))
        for j, variant in enumerate(variants):
            rows = [next((r for r in selected if (r['bits'], r['B2'], r['variant']) == (n, b2, variant)), None) for n in bits]
            for row, key in [(0, 'engine_s'), (1, 'point_kernel_s')]:
                values = [r[key] if r else float('nan') for r in rows]
                error = [[r[key]-r[key+'_min'] if r else 0 for r in rows], [r[key+'_max']-r[key] if r else 0 for r in rows]]
                axes[row, column].bar(x+(j-(len(variants)-1)/2)*width, values, width,
                    color=colors.get(variant), label=variant, yerr=error, capsize=2)
        axes[0, column].set_title(f'D={a.plot_d}; B2={b2:g}')
        for row in range(2):
            axes[row, column].set_xticks(x, bits); axes[row, column].set_xlabel('Actual target N bits')
            if a.log_time:
                axes[row, column].set_yscale('log')
            axes[row, column].grid(axis='y', alpha=.2)
    axes[0, 0].set_ylabel('Complete engine (s)'); axes[1, 0].set_ylabel('Nested point kernels (s)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Point execution plan: median, min/max; no discarded warmup'+(' | log time' if a.log_time else '')+'\nSame saves, NTT work, points and independent GMP coverage')
    fig.tight_layout(rect=(0, 0, 1, .92))
    a.figure.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.figure, dpi=170); plt.close(fig)
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
