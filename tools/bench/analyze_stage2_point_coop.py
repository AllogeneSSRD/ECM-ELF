"""Audit completed cooperative field experiments; no GPU activity."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ncu_summary(study):
    meta = json.loads((study/'summary.json').read_text())
    if not meta['complete'] or not meta['diagnostic']:
        raise ValueError('Need completed diagnostic captures')
    for name, key in [('metrics.csv','metrics_sha256'),('trace.ncu-rep','report_sha256')]:
        if hashlib.sha256((study/name).read_bytes()).hexdigest() != meta[key]:
            raise ValueError('NCU evidence changed')
    lines = (study/'metrics.csv').read_text(encoding='utf-8-sig').splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"'))
    rows = list(csv.DictReader(lines[header:]))
    units = rows[0]
    values = [r for r in rows if r['ID'].isdigit()]
    if len(values) != 1:
        raise ValueError('Expected exactly one selected kernel')
    row = values[0]
    number = lambda name: float(row[name].replace(',',''))
    metrics = dict(active_warps_percent=number('sm__warps_active.avg.pct_of_peak_sustained_active'),
        scheduler_issue_percent=number('smsp__issue_active.avg.pct_of_peak_sustained_active'),
        eligible_warps=number('smsp__warps_eligible.avg.per_cycle_active'),
        local_load_sectors=number('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),
        local_store_sectors=number('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'),
        long_scoreboard_ratio=number('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),
        registers_per_thread=number('launch__registers_per_thread'),
        blocks=number('launch__grid_size'),threads_per_block=number('launch__block_size'))
    unit = units['gpu__time_duration.sum']
    metrics['diagnostic_seconds'] = number('gpu__time_duration.sum')*{'s':1,'ms':.001,'us':.000001}[unit]
    fields = meta['reference']['fields']
    k,c,tail = (int(fields[name]) for name in ('chains','length','tail'))
    metrics['xadds'] = (k-1)*max(c-2,0)+max(tail-2,0)
    return dict(study=str(study),fields=fields,metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--extra', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--ncu-scalar', type=Path)
    p.add_argument('--ncu-coop', type=Path)
    a = p.parse_args()
    groups = defaultdict(list)
    identities = []
    calls = 0
    for study in [a.study, *a.extra]:
        report = json.loads((study / 'results.json').read_text(encoding='utf-8-sig'))
        if not report['complete'] or report['failures'] or report.get('exploratory') or report['conditions'].get('probe','field') != 'field' or report['conditions']['suite'] != 'performance':
            raise ValueError('Need a clean completed performance experiment')
        binary = study / 'bin' / Path(report['conditions']['exe']).name
        if hashlib.sha256(binary.read_bytes()).hexdigest() != report['exe_sha256'] or \
           hashlib.sha256((binary.parent / 'gmp-10.dll').read_bytes()).hexdigest() != report['dll_sha256']:
            raise ValueError('Frozen binary/DLL changed')
        identities.append(dict(study=str(study), sha256=report['exe_sha256'], conditions=report['conditions']))
        calls += len(report['rows'])
        for row in report['rows']:
            f = row['fields']
            if row['returncode'] or row['fault'] or int(f['bad']):
                raise ValueError('Invalid arithmetic sample')
            if len(f['samples_ms'].split(',')) != 5 or float(f['median_ms']) <= 0:
                raise ValueError('Missing timing samples')
            key = (report['exe_sha256'], *[int(f[k]) for k in ('bits', 'count', 'repeats', 'kind', 'operation', 'tpi')])
            groups[key].append(row)
    cells = []
    for key, rows in sorted(groups.items()):
        identity, bits, count, repeats, kind, operation, tpi = key
        reference = groups[(*key[:-1], 0)]
        ms = statistics.median(float(r['fields']['median_ms']) for r in rows)
        baseline = statistics.median(float(r['fields']['median_ms']) for r in reference)
        cells.append(dict(exe_sha256=identity, bits=bits, count=count, repeats=repeats, kind=kind, operation=operation,
                          tpi=tpi, n=len(rows), median_ms=ms, scalar_ms=baseline,
                          input_pattern='zero boundary' if count==1 else 'boundary and seeded random canonical inputs',
                          on_vs_scalar=ms/baseline, operations_per_second=1000*count*repeats/ms,
                          registers=sorted({int(r['fields']['registers']) for r in rows}),
                          local_bytes=sorted({int(r['fields']['local_bytes']) for r in rows}),
                          evidence=[r['directory'] for r in rows]))
    a.output.mkdir(parents=True, exist_ok=False)
    a.figures.mkdir(parents=True, exist_ok=False)
    result = dict(complete=True, identities=identities, cells=cells,
                  timing='Median of five retained events per process, then median across processes')
    if bool(a.ncu_scalar) != bool(a.ncu_coop):
        raise ValueError('Both NCU captures required')
    if a.ncu_scalar:
        captures = [ncu_summary(a.ncu_scalar),ncu_summary(a.ncu_coop)]
        for key in ('sha256','bits','nw','chains','length','tail','seed_stride'):
            if captures[0]['fields'][key] != captures[1]['fields'][key]:
                raise ValueError('NCU work/output differs: '+key)
        result['ncu'] = captures
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(__file__, a.output / 'analyzer_snapshot.py')
    for count in sorted({r['count'] for r in cells}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for kind, family in ((0, 'ordinary'), (1, 'Mersenne fold')):
            for operation, name in ((0, 'multiply'), (1, 'square')):
                ax = axes[kind, operation]
                for tpi in sorted({r['tpi'] for r in cells if r['tpi']}):
                    rows = [r for r in cells if (r['count'], r['kind'], r['operation'], r['tpi']) ==
                            (count, kind, operation, tpi)]
                    rows.sort(key=lambda r: r['bits'])
                    if len({r['bits'] for r in rows}) != len(rows):
                        raise ValueError('Overlapping candidate binaries; compare in separate figures')
                    ax.plot([r['bits'] for r in rows], [r['on_vs_scalar'] for r in rows],
                            '.-', label=f'CGBN TPI {tpi}')
                ax.axhline(1, color='gray', linestyle=':', label='own scalar control')
                ax.set_title(f'{family}: {name}')
                ax.set_ylabel('Time / own scalar control (lower is better)')
                ax.set_xlabel('Arithmetic modulus bits')
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
        pattern='zero boundary input' if count==1 else 'boundary and random canonical inputs'
        fig.suptitle(f'RTX 4060 Laptop | {count} independent instances | field probe only\n'
                     f'{pattern}; all event samples retained; n in summary')
        fig.tight_layout()
        fig.savefig(a.figures / f'point-coop-count{count}.png', dpi=150)
        fig.savefig(a.figures / f'point-coop-count{count}.svg')
        plt.close(fig)
    if a.ncu_scalar:
        labels=['scalar TPI 1','cooperative TPI 16']
        values=[r['metrics'] for r in captures]
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,key,title in [(axes[0,0],'active_warps_percent','Active warps (% of peak)'),
                             (axes[0,1],'scheduler_issue_percent','Scheduler issue active (%)'),
                             (axes[1,1],'diagnostic_seconds','Profiled kernel time (s; diagnostic)')]:
            bars=ax.bar(labels,[r[key] for r in values],color=['#777777','#247fae'])
            ax.bar_label(bars,fmt='%.2f');ax.set_title(title);ax.set_ylim(0,max(r[key] for r in values)*1.2)
        ax=axes[1,0]
        loads=[r['local_load_sectors']/r['xadds']/1000 for r in values]
        stores=[r['local_store_sectors']/r['xadds']/1000 for r in values]
        ax.bar(labels,loads,label='local loads');ax.bar(labels,stores,bottom=loads,label='local stores')
        ax.set_title('Local sectors per xADD (thousands)');ax.legend(fontsize=8)
        fig.suptitle('NCU: same 65536-point chain, 2048 chains x 32, 16319-bit Mersenne\n'
                     'Kernel replay and profiler times excluded from performance qualification')
        fig.tight_layout();fig.savefig(a.figures/'point-coop-ncu.png',dpi=150)
        fig.savefig(a.figures/'point-coop-ncu.svg');plt.close(fig)
    print(f'Audited {calls} calls, {len(cells)} cells')


if __name__ == '__main__':
    main()
