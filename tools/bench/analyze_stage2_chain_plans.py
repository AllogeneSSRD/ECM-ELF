"""Audit frozen full-chain probes and compare TPI/container plans; no GPU work."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_stage2_point_coop import ncu_summary


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--ncu-study', type=Path, action='append', default=[])
    a = p.parse_args()
    groups, digests = defaultdict(list), defaultdict(set)
    identities, calls = [], set()
    for study in a.study:
        report = load(study/'results.json')
        cond = report['conditions']
        if not report['complete'] or report['failures'] or report.get('exploratory') or \
                cond['probe'] != 'chain' or cond['suite'] != 'performance':
            raise ValueError('Need clean completed chain performance studies')
        binary = study/'bin'/Path(cond['exe']).name
        for path, field in ((binary, 'exe_sha256'), (binary.parent/'gmp-10.dll', 'dll_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != report[field]:
                raise ValueError('Frozen binary/DLL changed')
        identities.append(dict(study=str(study), exe_sha256=report['exe_sha256'],
                               device=report['device'], conditions=cond))
        for row in report['rows']:
            f = row['fields']
            path = Path(row['directory']).resolve()
            if str(path) in calls:
                raise ValueError('Duplicate invocation')
            calls.add(str(path))
            if row['returncode'] or row['fault'] or int(f['bad']):
                raise ValueError('Invalid chain sample')
            samples = list(map(float, f['samples_ms'].split(',')))
            if len(samples) != 5 or min(samples) <= 0 or \
                    abs(statistics.median(samples)-float(f['median_ms'])) > .000002:
                raise ValueError('Missing or inconsistent event samples')
            shape = tuple(int(f[k]) for k in ('bits', 'kind', 'chains', 'length', 'tail', 'seed_stride'))
            digests[shape].add(f['sha256'])
            capacity = int(f.get('capacity_bits', 16384))
            groups[(*shape, int(f['tpi']), capacity)].append(row)
    if any(len(v) != 1 for v in digests.values()):
        raise ValueError('Complete chain outputs differ between plans')
    cells = []
    for key, rows in sorted(groups.items()):
        bits, kind, chains, length, tail, stride, tpi, capacity = key
        reference = groups.get((*key[:6], 16, 16384))
        if not reference:
            raise ValueError('Missing fixed-16k TPI16 control')
        ms = statistics.median(float(r['fields']['median_ms']) for r in rows)
        base = statistics.median(float(r['fields']['median_ms']) for r in reference)
        cells.append(dict(bits=bits, kind=kind, chains=chains, length=length, tail=tail,
            seed_stride=stride, tpi=tpi, capacity_bits=capacity, n=len(rows), median_ms=ms,
            min_ms=min(float(r['fields']['median_ms']) for r in rows),
            max_ms=max(float(r['fields']['median_ms']) for r in rows),
            time_over_tpi16=ms/base, speedup=base/ms,
            registers=sorted({int(r['fields']['registers']) for r in rows}),
            local_bytes=sorted({int(r['fields']['local_bytes']) for r in rows}),
            evidence=[r['directory'] for r in rows]))
    a.output.mkdir(parents=True, exist_ok=False)
    a.figures.mkdir(parents=True, exist_ok=False)
    result = dict(complete=True, identities=identities, calls=len(calls), cells=cells,
        timing='Median of all five event samples per process, then median across processes')
    captures=[ncu_summary(study) for study in a.ncu_study]
    if len({tuple(r['fields'][k] for k in ('kind','chains','length','tail','seed_stride')) for r in captures})>1:
        raise ValueError('NCU captures use different chain workloads')
    for capture in captures:
        f=capture['fields']
        shape=tuple(int(f[k]) for k in ('bits','kind','chains','length','tail','seed_stride'))
        if f['sha256'] not in digests.get(shape,set()):
            raise ValueError('NCU output/work differs from retained unprofiled chains')
    result['ncu']=captures
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(__file__, a.output/'analyzer_snapshot.py')
    for shape in sorted({(c['chains'], c['length'], c['tail'], c['seed_stride']) for c in cells}):
        chains, length, tail, stride = shape
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, kind, label in zip(axes, (0, 1), ('ordinary modulus', 'Mersenne fold')):
            selected = [c for c in cells if (c['chains'], c['length'], c['tail'], c['seed_stride'], c['kind']) == (*shape, kind)]
            for tpi, capacity in sorted({(c['tpi'], c['capacity_bits']) for c in selected}):
                rows = sorted([c for c in selected if (c['tpi'], c['capacity_bits']) == (tpi, capacity)], key=lambda c:c['bits'])
                ax.plot([c['bits'] for c in rows], [c['time_over_tpi16'] for c in rows], '.-',
                        label=f'TPI{tpi}, {capacity}-bit capacity')
            ax.axhline(1, color='gray', linestyle=':')
            ax.set_title(label); ax.set_xlabel('Arithmetic modulus bits')
            ax.set_ylabel('Time / fixed-16k TPI16 (lower is better)')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle(f'RTX 4060 Laptop | {chains} chains x {length} points | tail {tail}, seed stride {stride}\n'
                     'Full xADD chain probe; no seed/normalization/tree costs; all samples retained')
        fig.tight_layout()
        stem=f'chain-plans-k{chains}-c{length}'
        fig.savefig(a.figures/(stem+'.png'), dpi=150)
        fig.savefig(a.figures/(stem+'.svg')); plt.close(fig)
    if captures:
        labels=[f"p{r['fields']['bits']}\nTPI{r['fields']['tpi']} / {r['fields'].get('capacity_bits','16384')}b" for r in captures]
        metrics=[r['metrics'] for r in captures]
        if any(r['xadds']<=0 for r in metrics):raise ValueError('Cannot normalize a chain without xADDs')
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for ax,key,title in [(axes[0,0],'active_warps_percent','Active warps (% of peak)'),
                            (axes[0,1],'scheduler_issue_percent','Scheduler issue active (%)'),
                            (axes[1,1],'diagnostic_seconds','Profiled kernel (s; diagnostic)')]:
            palette=['#777777','#247fae','#247fae','#729845']
            bars=ax.bar(labels,[r[key] for r in metrics],color=[palette[i%len(palette)] for i in range(len(metrics))])
            ax.bar_label(bars,fmt='%.2f');ax.set_title(title);ax.set_ylim(0,max(r[key] for r in metrics)*1.2)
        ax=axes[1,0]
        loads=[r['local_load_sectors']/r['xadds']/1000 for r in metrics]
        stores=[r['local_store_sectors']/r['xadds']/1000 for r in metrics]
        ax.bar(labels,loads,label='local loads');ax.bar(labels,stores,bottom=loads,label='local stores')
        ax.set_title('Local sectors per xADD (thousands)');ax.legend(fontsize=8)
        fig.suptitle('NCU chain probes | arithmetic width, TPI and capacity in labels\n'
                     'Same chain counts and lengths; profiling excluded from performance samples')
        fig.tight_layout();fig.savefig(a.figures/'chain-plans-ncu.png',dpi=150)
        fig.savefig(a.figures/'chain-plans-ncu.svg');plt.close(fig)
    print('Audited',len(calls),'calls;',len(cells),'cells')


if __name__ == '__main__':
    main()
