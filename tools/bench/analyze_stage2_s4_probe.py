"""Summarize verified S4 event samples without discarding first calls or events."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','output','figures'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();d=json.loads((a.study/'summary.json').read_text())
    if not d['complete'] or d['failures']:raise ValueError('Incomplete S4 study')
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    groups={}
    for r in d['rows']:
        if int(r['bad']) or int(r['gpu_bad']):raise ValueError('Fault injection is not a performance sample')
        key=tuple(int(r[k]) for k in ('bits','coeffs','kind','bpw','batches','backend','threads'))
        groups.setdefault(key,[]).append(r)
    rows=[]
    for key,values in sorted(groups.items()):
        times=[float(v) for r in values for v in r['samples_ms'].split(',') if v]
        medians=[float(r['median_ms']) for r in values]
        row=dict(zip(('bits','coeffs','kind','bpw','batches','backend','threads'),key))
        row.update(processes=len(values),events=len(times),median_ms=statistics.median(medians),
                   min_event_ms=min(times),max_event_ms=max(times),
                   checked_words=sum(int(r['checked_words']) for r in values),
                   shared_bytes=int(values[0]['shared']),registers=int(values[0]['regs']),
                   local_bytes=int(values[0]['local']))
        row['coefficients_per_s']=(row['coeffs']-2)*row['batches']*1000/row['median_ms']
        rows.append(row)
    result=dict(complete=True,source=str(a.study.resolve()),exe_sha256=d['exe_sha256'],
                modulus_sha256=d['modulus_sha256'],device=d['device'],rows=rows,
                scope='Medians of per-process event medians; all events retained. Not complete Stage2 time.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'events.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={0:'Scalar',8192:'Scalar, 8192-coefficient windows',32:'Warp / two-pass',
            33:'Warp / merged',34:'Warp / merged + divisor cache'}
    for bits in sorted({r['bits'] for r in rows}):
        selected=[r for r in rows if r['bits']==bits]
        if len({(r['kind'],r['bpw'],r['batches']) for r in selected})!=1:
            raise ValueError('Separate modulus/digit/batch configurations before plotting throughput')
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        mixed_threads=len({r['threads'] for r in selected})>1
        for backend,threads in sorted({(r['backend'],r['threads']) for r in selected}):
            group=sorted((r for r in selected if (r['backend'],r['threads'])==(backend,threads)),key=lambda r:r['coeffs'])
            x=[r['coeffs']-2 for r in group];y=[r['median_ms'] for r in group]
            label=labels.get(backend,str(backend))+(f' / T{threads}' if mixed_threads else '')
            axes[0].plot(x,y,'o-',label=label)
            axes[1].plot(x,[r['coefficients_per_s']/1000 for r in group],'o-',label=label)
        for ax in axes:
            ax.set_xlabel('Active coefficients per batch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('CUDA event milliseconds');axes[1].set_ylabel('Thousand coefficients / second')
        fig.suptitle(f'{bits}-bit S4; mixed digit patterns, no warmup; all outputs checked with GMP')
        for ext in ('png','svg'):fig.savefig(a.figures/f's4_probe_{bits}.{ext}',dpi=180)
        plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','device')}))


if __name__=='__main__':main()
