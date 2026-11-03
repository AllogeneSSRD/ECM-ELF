"""Audit independent crossover timings and plot ratios by width and batch."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics


def read(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    assert data['complete'] and not data.get('failures'), path
    return data


def ncu_totals(folder):
    data = read(folder/'summary.json')
    rows = []
    for call in data['calls']:
        with (Path(call['folder'])/'metrics.csv').open(encoding='utf-8-sig') as file:
            values = list(csv.DictReader(file))
        units = next(r for r in values if r.get('ID') == '')
        kernels = [r for r in values if r.get('ID', '').isdigit()]
        names = ('smsp__inst_executed.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
                 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum', 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')
        totals = {}
        for name in names:
            # Nsight CSV scales each column independently; different captures
            # may use byte/Kbyte/Mbyte. Normalise SI prefixes before adding.
            factor = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9,
                      'inst': 1, 'sector': 1}[units[name]]
            totals[name] = sum(float(r[name].replace(',', '')) for r in kernels)*factor
        rows.append(dict(backend=call['backend'], kernel_count=len(kernels), totals=totals,
                         original_units={name: units[name] for name in names}))
    return dict(bits=data['bits'], count=data['count'], scope=data['scope'], calls=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path('data/experiments/cgbn_ntt_crossover_20261103'))
    p.add_argument('--output', type=Path, default=Path('data/benchmarks/cgbn_ntt_crossover_20261103'))
    p.add_argument('--figures', type=Path, default=Path('data/figures/cgbn_ntt_crossover_20261103'))
    p.add_argument('--extra', type=Path, nargs='*', default=[], help='Additional independent width studies with final/check_matrix')
    a = p.parse_args()
    final = read(a.evidence/'final/summary.json')
    check = read(a.evidence/'check_matrix/summary.json')
    studies = [a.evidence, *a.extra]
    for folder in a.extra:
        extra = read(folder/'final/summary.json')
        assert extra['suite'] == 'final' and extra['warmups'] == 0
        final['runs'].extend(extra['runs'])
        final['builds'].extend(extra['builds'])
        check['runs'].extend(read(folder/'check_matrix/summary.json')['runs'])
    assert final['suite'] == 'final' and final['warmups'] == 0
    report = dict(cells=[], qualification=dict(calls=len(check['runs']),
        rejected_faults=sum(bool(r.get('fault_rejected')) for r in check['runs'])),
        builds=final['builds'], timing=final['timing'], warmups=0)
    cells = sorted({(r['bits'], r['count'], r['op']) for r in final['runs']})
    rounds = {r['round'] for r in final['runs']}
    assert rounds == set(range(len(rounds))) and len(rounds) >= 3
    report['process_rounds'] = len(rounds)
    for bits, count, op in cells:
        rows = [r for r in final['runs'] if (r['bits'], r['count'], r['op']) == (bits, count, op)]
        assert len(rows) == 3*len(rounds) and len({r['input_hash'] for r in rows}) == 1
        assert all(r['kind'] == 0 and len(r['samples_ms']) == 5 for r in rows)
        groups = {}
        for backend, name in ((0, 'cgbn'), (1, 'ntt_direct'), (2, 'ntt_graph')):
            group = [r for r in rows if r['backend'] == backend]
            assert len(group) == len(rounds) and {r['round'] for r in group} == rounds
            assert len({(r['tpi'], r['tile'], r['iterations'], r['exe_sha256']) for r in group}) == 1
            times = [r['batch_ms'] for r in group]
            fields = group[0]['fields']
            groups[name] = dict(batch_ms=statistics.median(times), min_ms=min(times), max_ms=max(times),
                process_medians_ms=times, tpi=group[0]['tpi'], tile=int(fields['actual_tile']),
                length=int(fields['length']), bpw=int(fields['bpw']), iterations=group[0]['iterations'],
                regs=int(fields['cgbn_regs']), local_bytes=int(fields['cgbn_local_bytes']),
                products_per_second=count/(statistics.median(times)*.001),
                setup_seconds=statistics.median(float(r['fields']['setup_seconds']) for r in group),
                graph_seconds=statistics.median(float(r['fields']['graph_seconds']) for r in group),
                exe_sha256=group[0]['exe_sha256'])
        base = groups['cgbn']
        for name in ('ntt_graph', 'ntt_direct'):
            item = groups[name]
            item['ratio'] = item['batch_ms']/base['batch_ms']
            item['ratio_low'] = item['min_ms']/base['max_ms']
            item['ratio_high'] = item['max_ms']/base['min_ms']
            item['clear_win'] = item['ratio_high'] < 1
        report['cells'].append(dict(bits=bits, count=count, op=op, groups=groups))
    report['triangular'] = []
    for bits in (512, 8192):
        path = a.evidence/f'triangular_perf{bits}/summary.json'
        if not path.is_file():
            continue
        data = read(path)
        for count in sorted({r['count'] for r in data['runs'] if 'round' in r}):
            groups = {}
            for mode in (0, 24):
                rows = [r for r in data['runs'] if r['mode'] == mode and r['count'] == count and 'round' in r]
                assert len(rows) == 3 and {r['round'] for r in rows} == {0, 1, 2}
                groups[mode] = dict(median_ms=statistics.median(r['median_ms'] for r in rows),
                    registers=int(rows[0]['fields']['regs']), local_bytes=int(rows[0]['fields']['local']))
            report['triangular'].append(dict(bits=bits, count=count, native=count == data['native_count'],
                groups=groups, ratio=groups[24]['median_ms']/groups[0]['median_ms']))
    clocks, busy = collections.Counter(), collections.Counter()
    timed_folders = [s/'final' for s in studies] + [a.evidence/f'triangular_perf{b}' for b in (512,8192)]
    for folder in timed_folders:
        for path in folder.rglob('telemetry.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row.get('sm_clock_mhz') is not None:
                    clocks[row['sm_clock_mhz']] += 1
                    if row.get('gpu_percent', 0) >= 20:
                        busy[row['sm_clock_mhz']] += 1
    report['clocks_mhz'], report['busy_clocks_mhz'] = dict(sorted(clocks.items())), dict(sorted(busy.items()))
    report['ncu'] = [ncu_totals(a.evidence/f'ncu_integer{b}_v2') for b in (8192, 131072)
                     if (a.evidence/f'ncu_integer{b}_v2/summary.json').is_file()]
    a.output.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    flat = []
    for cell in report['cells']:
        for name, row in cell['groups'].items():
            flat.append(dict(bits=cell['bits'], count=cell['count'], op=cell['op'], backend=name, **row))
    cols = ['bits', 'count', 'op', 'backend', 'batch_ms', 'min_ms', 'max_ms', 'tpi', 'tile',
            'length', 'bpw', 'iterations', 'regs', 'local_bytes', 'ratio', 'ratio_low', 'ratio_high', 'clear_win']
    with (a.output/'timings.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, cols, extrasaction='ignore');writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    counts = sorted({r['count'] for r in report['cells']})
    for plot in ('ratio', 'time'):
        fig, axes = plt.subplots(2, len(counts), figsize=(4.1*len(counts), 7.2), squeeze=False)
        fig.subplots_adjust(top=.88, bottom=.17, wspace=.36, hspace=.52)
        for op in (0, 1):
            for column, count in enumerate(counts):
                ax = axes[op][column]
                rows = [r for r in report['cells'] if r['count'] == count and r['op'] == op]
                for name, label, colour in [('cgbn', 'CGBN', '#586979'), ('ntt_direct', 'NTT direct', '#c16c3b'),
                                             ('ntt_graph', 'NTT graph', '#16857c')]:
                    if plot == 'ratio' and name == 'cgbn':
                        continue
                    values = [r['groups'][name]['ratio'] if plot == 'ratio' else r['groups'][name]['batch_ms']*1000/count for r in rows]
                    low = [r['groups'][name]['ratio_low'] if plot == 'ratio' else r['groups'][name]['min_ms']*1000/count for r in rows]
                    high = [r['groups'][name]['ratio_high'] if plot == 'ratio' else r['groups'][name]['max_ms']*1000/count for r in rows]
                    x = [r['bits']/1024 for r in rows]
                    ax.plot(x, values, marker='o', label=label, color=colour)
                    ax.fill_between(x, low, high, color=colour, alpha=.13)
                if plot == 'ratio':
                    ax.axhline(1, color='#555555', linestyle=':', linewidth=1)
                ax.set_xscale('log', base=2);ax.set_yscale('log')
                ax.set_xticks([r['bits']/1024 for r in rows], [f"{r['bits']/1024:g}" for r in rows])
                ax.set_xlabel('Operand width (Ki bits)');ax.grid(alpha=.18)
                ax.set_title(('Full product' if op == 0 else 'Montgomery product')+f' / batch {count}')
                if column == 0:
                    ax.set_ylabel('NTT / CGBN time' if plot == 'ratio' else 'Microseconds per product')
                if plot == 'ratio':
                    ax.yaxis.set_major_locator(FixedLocator((.02,.05,.1,.2,.5,1,2,5,10)))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.legend(fontsize=8)
        fig.suptitle('RTX 4060 Laptop: exact integer NTT vs cooperative CGBN', fontsize=13)
        fig.text(.5, .025, f'Frozen screening winners, {len(rounds)} independent process rounds, 5 event samples, no discarded warmup\n'
                 'Resident dependent chain; pack/carry/unpack/REDC included; setup and verification excluded\n'
                 'Shading is min/max process medians, not a confidence interval; direct reuses graph-selected layout',
                 ha='center', fontsize=8)
        for ext in ('png', 'svg'):
            fig.savefig(a.figures/f'cgbn-ntt-{plot}-4060lp.{ext}', dpi=170)
    for cell in report['cells']:
        g = cell['groups']
        print(cell['bits'], cell['count'], cell['op'], 'CGBN', g['cgbn']['batch_ms'],
              'NTT', g['ntt_graph']['batch_ms'], 'ratio', g['ntt_graph']['ratio'],
              'range', (g['ntt_graph']['ratio_low'], g['ntt_graph']['ratio_high']))
    print('triangular', report['triangular'])
    print('clocks', report['clocks_mhz'], 'busy', report['busy_clocks_mhz'])


if __name__ == '__main__':
    main()
