"""Summarise frozen square-layout experiments without pooling unlike inputs.

Operator results are medians of three process medians (five event samples).
Each real Stage1 layout is normalised to its own interleaved CGBN control.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def load(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    assert data['complete'] and not data.get('failures'), path
    return data


def operators(folder):
    data = load(folder / 'summary.json')
    rows = []
    measured = [r for r in data['runs'] if 'round' in r]
    for count, mode in sorted({(r['count'], r['mode']) for r in measured}):
        group = [r for r in measured if (r['count'], r['mode']) == (count, mode)]
        assert len(group) == 3 and {r['round'] for r in group} == {0, 1, 2}
        assert all(len(r['samples_ms']) == 5 for r in group)
        assert len({r['recurrence'] for r in group}) == 1
        ms = statistics.median(r['median_ms'] for r in group)
        rows.append(dict(study=folder.name, bits=data['bits'], tpi=data['tpi'],
            count=count, mode=mode, native=count == data['native_count'],
            recurrence=group[0]['recurrence'], median_ms=ms,
            process_medians_ms=[r['median_ms'] for r in group],
            squares_per_second=count * group[0]['recurrence'] / (ms * .001),
            registers=int(group[0]['fields']['regs']), local_bytes=int(group[0]['fields']['local']),
            exe_sha256=data['exe_sha256']))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path('data/experiments/cgbn_square_layout_20261102'))
    p.add_argument('--output', type=Path, default=Path('data/benchmarks/cgbn_square_layout_20261102'))
    p.add_argument('--figures', type=Path, default=Path('data/figures/cgbn_square_layout_20261102'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    folders = [a.evidence / f'{prefix}{bits}' for prefix in ('perf', 'raw', 'joinfirst_perf')
               for bits in (8192, 16384)]
    report = dict(operators=[], stage1=[], qualification=[], sanitizers=[], ncu=[],
                  timing='event / complete Stage1 GPU interval; no warmup; profiler timings excluded')
    for folder in folders:
        report['operators'].extend(operators(folder))
    for name in ('stage1_layout0', 'stage1_layout3'):
        folder = a.evidence / name
        data = load(folder / 'summary.json')
        assert data['b1'] == 1000 and data['rounds'] == 3 and len(data['runs']) == 12
        assert all(r['matched_full_batch'] for r in data['runs'])
        assert all(r['container'] == 16384 and r['tpi'] == 32 and r['curves'] == 192 for r in data['runs'])
        assert sum(len(v) for v in data['oracle'].values()) == 4
        report['stage1'].append(dict(study=name, **data))
        folders.append(folder)
    first, second = report['stage1']
    assert first['sigma'] == second['sigma']
    assert first['identity']['baseline']['exe_sha256'] == second['identity']['baseline']['exe_sha256']
    inputs = [json.loads((a.evidence / s['study'] / 'inputs.json').read_text(encoding='utf-8'))
              for s in report['stage1']]
    assert inputs[0] == inputs[1], 'Stage1 comparison must use the same exact modulus'
    for name in ('check512', 'check8064', 'check8192', 'check16384',
                 'joinfirst_check8192', 'joinfirst_check16384', 'final_check512'):
        data = load(a.evidence / name / 'summary.json')
        report['qualification'].append(dict(study=name, calls=len(data['runs']),
            passed=sum(not r.get('fault_rejected', False) for r in data['runs']),
            rejected_faults=sum(bool(r.get('fault_rejected')) for r in data['runs'])))
    for name in ('sanitizer_mem', 'sanitizer_sync'):
        data = load(a.evidence / name / 'summary.json')
        assert data['suite'] == 'sanitizer' and data['sanitizer']['local_connection'] == 'named-pipes'
        assert {r['mode'] for r in data['runs']} == {18, 19, 23}
        report['sanitizers'].append(dict(study=name, calls=len(data['runs']), sanitizer=data['sanitizer']))
    profile = load(a.evidence / 'ncu8192/summary.json')
    for item in profile['calls']:
        with (Path(item['folder']) / 'metrics.csv').open(encoding='utf-8-sig') as file:
            rows = [r for r in csv.DictReader(file) if r.get('ID', '').isdigit()]
        assert len(rows) == 1
        keys = ('smsp__inst_executed.sum', 'launch__registers_per_thread')
        report['ncu'].append(dict(mode=item['mode'], metrics={k: float(rows[0][k].replace(',', '')) for k in keys}))
    all_clock, busy_clock = collections.Counter(), collections.Counter()
    for folder in folders:
        for path in folder.rglob('telemetry.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                clock = row.get('sm_clock_mhz')
                if clock is not None:
                    all_clock[clock] += 1
                    if row.get('gpu_percent', 0) >= 20:
                        busy_clock[clock] += 1
    report['clocks_mhz'] = dict(sorted(all_clock.items()))
    report['busy_clocks_mhz'] = dict(sorted(busy_clock.items()))
    report['stage1_resources'] = json.loads((a.evidence / 'stage1_resources.json').read_text(encoding='utf-8'))
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    cols = ['study', 'bits', 'tpi', 'count', 'mode', 'native', 'recurrence', 'median_ms',
            'squares_per_second', 'registers', 'local_bytes', 'exe_sha256']
    with (a.output / 'operators.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['operators'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    fig.subplots_adjust(top=.82, bottom=.25, wspace=.33)
    modes = (16, 18, 19, 23)
    labels = ('Original split', 'Both swaps', 'Cross first', 'Join first')
    colours = ('#536980', '#178578', '#c06a32', '#9a6499')
    for index, mode in enumerate(modes):
        times = []
        for bits in (8192, 16384):
            rows = [r for r in report['operators'] if r['study'] == f'joinfirst_perf{bits}' and r['native']]
            times.append(next(r['median_ms'] for r in rows if r['mode'] == mode) /
                         next(r['median_ms'] for r in rows if r['mode'] == 0))
        axes[0].bar([x + (index-1.5)*.19 for x in (0, 1)], times, .19,
                    label=labels[index], color=colours[index])
    axes[0].set_xticks((0, 1), ('8192 / TPI16', '16384 / TPI32'))
    axes[0].set_title('Montgomery square: native batch')
    axes[0].set_xlabel('Container bits / threads per integer')
    axes[0].legend(fontsize=8)
    for index, stage in enumerate(report['stage1']):
        values = [next(r['gpu_ratio'] for r in stage['aggregates'] if r['model'] == model)
                  for model in ('mont0', 'ed')]
        axes[1].bar([x + (index-.5)*.3 for x in (0, 1)], values, .3,
                    label='Original split' if index == 0 else 'Cross first',
                    color=colours[0 if index == 0 else 2])
    axes[1].set_xticks((0, 1), ('Montgomery PARAM0', 'Edwards NAF w6'))
    axes[1].set_title('Complete Stage1: 16384-bit container')
    axes[1].set_xlabel('B1=1000; 192 curves per complete call')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(1, color='#444444', linestyle=':', linewidth=1)
        ax.set_ylim(0, 1.5)
        ax.set_ylabel('Time / original CGBN (lower is better)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('RTX 4060 Laptop: split-square layouts and live ranges', fontsize=13)
    fig.text(.5, .03, '3 process rounds, no warmup; operator: 5 event samples per round\n'
             'Stage1 uses its own interleaved control; Mont includes preparation, Edwards is NAF loop only\n'
             'GPU timing excludes independent CPU verification; clocks recorded, no settings changed',
             ha='center', fontsize=8)
    for ext in ('png', 'svg'):
        fig.savefig(a.figures / f'cgbn-square-layout-4060lp.{ext}', dpi=170)
    for stage in report['stage1']:
        for row in stage['aggregates']:
            print(stage['study'], row['model'], row['gpu_ratio'], row['groups'])
    print('qualification', report['qualification'])
    print('clocks', report['clocks_mhz'], 'busy', report['busy_clocks_mhz'])


if __name__ == '__main__':
    main()
