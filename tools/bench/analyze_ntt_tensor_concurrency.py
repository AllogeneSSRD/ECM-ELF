"""Scientific figures and summaries for same-card CUDA/Tensor NTT concurrency."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, action='append', required=True)
    p.add_argument('--profile', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figure-prefix', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    groups, rows, telemetry, checked, samples = [], [], [], 0, 0
    for path in a.experiment:
        data = json.loads((path / 'measurements.json').read_text())
        if not data['complete']:
            raise ValueError('Incomplete experiment')
        with (path / 'summary.csv').open() as f:
            summary = list(csv.DictReader(f))
        for run in data['runs']:
            selected = {r['name']: {k: float(v) if k in ('median_seconds','mad_seconds','convolutions_per_s') else v for k, v in r.items()}
                        for r in summary if r['case'] == run['case']['name']}
            row = dict(experiment=path.name, execution=data['identity']['execution'], **run['case'], **{k: v for k, v in run['comparison'].items() if k != 'case'})
            for name, r in selected.items():
                row[name+'_seconds'] = r['median_seconds']
                row[name+'_mad'] = r['mad_seconds']
            rows.append(row)
            checked += sum(int(r['checked_words']) for r in run['raw'])
            samples += len(run['raw'])
        groups.append(data)
        with (path / 'telemetry.csv').open() as f:
            for r in csv.DictReader(f):
                r = {k.strip(): v.strip() for k, v in r.items()}
                if not r.get('timestamp'):
                    continue
                try:
                    if float(r['utilization.gpu [%]'].split()[0]) < 50:
                        continue
                    telemetry.append(dict(mhz=float(r['clocks.current.sm [MHz]'].split()[0]),
                        watts=float(r['power.draw [W]'].split()[0]), celsius=float(r['temperature.gpu'].split()[0])))
                except (ValueError, KeyError):
                    continue
    profiles = {path.name: json.loads((path / 'summary.json').read_text()) for path in a.profile}
    def spread(key):
        values = [r[key] for r in telemetry if key != 'watts' or r[key] <= 150]
        return dict(min=min(values), median=st.median(values), max=max(values), samples=len(values)) if values else None
    meta = dict(samples=samples, checked_words=checked, mhz=spread('mhz'), watts=spread('watts'),
                celsius=spread('celsius'), outlier_power_samples=sum(r['watts'] > 150 for r in telemetry))
    (a.output / 'summary.json').write_text(json.dumps(dict(rows=rows, telemetry=meta, profiles=profiles), indent=2))
    with (a.output / 'summary.csv').open('w', newline='') as f:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    graph = [r for r in rows if r['execution']=='graph'] or rows
    labels = [f"2^{r['k']} x {r['batch']} / CTA{r.get('threads',256)}" for r in graph]
    y = np.arange(len(graph))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, len(y)*.42)))
    for name, delta, color in [('ct40_gain_vs_best_cuda', -.18, '#888888'), ('cn40_gain_vs_best_cuda', .18, '#287aaf')]:
        axes[0].barh(y+delta, [r[name] for r in graph], height=.34, label='CUDA+T40' if name.startswith('ct') else 'CUDA+N96r64', color=color)
    axes[0].set_yticks(y, labels, fontsize=8);axes[0].invert_yaxis();axes[0].axvline(0,color='#555555',lw=.7)
    axes[0].set_xlabel('Time saved vs best pure-CUDA pair (%); positive is faster');axes[0].legend()
    axes[0].set_title('Same total work: two independent convolutions')
    gains = [r['narrow_gain_vs_t40'] for r in graph]
    axes[1].barh(y, gains, color=['#287aaf' if v>0 else '#ba6658' for v in gains])
    axes[1].set_yticks(y, labels, fontsize=8);axes[1].invert_yaxis();axes[1].axvline(0,color='#555555',lw=.7)
    for pos, gain in zip(y, gains):
        axes[1].text(gain+(.03 if gain>=0 else -.03), pos, f'{gain:+.2f}%', va='center', ha='left' if gain>=0 else 'right', fontsize=8)
    axes[1].set_xlim(min(gains)-.5,max(gains)+.8)
    axes[1].set_xlabel('Single convolution: N96r64 time saved vs Tensor40 (%)')
    axes[1].set_title('Local candidate improvement is not a CUDA win')
    fig.suptitle('CUDA/Tensor same-card concurrency | RTX 4060 Laptop | default 55 W / 1800 MHz cap\nFork/join; interleaved samples per scene; all A/B output words checked')
    fig.tight_layout(rect=[0,0,1,.94]);a.figure_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(a.figure_prefix)+'.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1, len(a.experiment), figsize=(7*len(a.experiment),5),squeeze=False)
    axes=axes[0]
    for index, path in enumerate(a.experiment):
        points = sorted((r for r in rows if r['experiment']==path.name and r['batch']==1 and r.get('threads',256)==256),key=lambda r:r['k'])
        for name,label,style in [('CC','CUDA/CUDA','-o'),('CT40','CUDA/T40','-s'),('CN40','CUDA/N96r64','-^')]:
            axes[index].plot([r['k'] for r in points], [r[name+'_serial_seconds']/r[name+'_parallel_seconds'] for r in points], style, label=label)
        axes[index].axhline(1,color='#555555',lw=.7)
        axes[index].set_xlabel('log2 NTT length, batch=1');axes[index].set_ylabel('Serial / concurrent makespan (larger is better)')
        axes[index].set_title(path.name+' | two equal-size jobs');axes[index].legend()
    fig.tight_layout();fig.savefig(str(a.figure_prefix)+'_efficiency.png',dpi=160);plt.close(fig)
    systems = [(path, profiles[path.name]) for path in a.profile if 'concurrency' in profiles[path.name]]
    if systems:
        fig, axes = plt.subplots(len(systems),1,figsize=(14,2.25*len(systems)),squeeze=False)
        for (path, data), ax in zip(systems, axes[:,0]):
            c=sqlite3.connect(path/'trace.sqlite')
            raw=c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start').fetchall();c.close()
            first=min(r[0] for r in raw)
            for group,color,label in [(False,'#287aaf','CUDA tile/outer'),(True,'#d58b40','Tensor tile')]:
                intervals=[((start-first)/1e6,(end-start)/1e6) for start,end,name in raw if ('tc_goldilocks_' in name)==group and (start-first)<1500000]
                ax.broken_barh(intervals, (int(group)*.6,.4),facecolors=color)
            ax.set_xlim(0,1.5);ax.set_yticks([.2,.8],['CUDA','Tensor']);ax.set_xlabel('ms from first captured kernel')
            d=data['concurrency'];ax.set_title(path.name+f" | overlap / Tensor union = {100*d['overlap_fraction_of_tensor']:.1f}% (whole captured range)",fontsize=10)
        fig.suptitle('Systems node timeline: overlap does not establish same-SM co-issue')
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(str(a.figure_prefix)+'_timeline.png',dpi=160);plt.close(fig)
    print(json.dumps(meta,indent=2))


if __name__ == '__main__':
    main()
