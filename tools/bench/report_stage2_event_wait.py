"""Summarize complete event-wait studies; verify frozen logs before plotting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('measurements',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    p.add_argument('--compare',choices=['readback','hybrid','blocking'],default='readback')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh/empty output directory')
    rows=[];sources={}
    for path in a.measurements:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not data['complete']:raise ValueError('Incomplete study: '+str(path))
        sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        for run in data['runs']:
            for key in ('log','debug','gpu_log'):
                if key not in run:continue
                hash_key='gpu_sha256' if key=='gpu_log' else key+'_sha256'
                if hashlib.sha256(Path(run[key]).read_bytes()).hexdigest()!=run[hash_key]:
                    raise ValueError('Evidence changed: '+run[key])
        for case in sorted({r['case']['id'] for r in data['runs']}):
            case_runs=[r for r in data['runs'] if r['case']['id']==case]
            def workload(run):
                m=run['multiply']
                log=Path(run['log']).read_text(encoding='utf-8',errors='replace')
                peak=re.findall(r'^ntt_workspace_stats: .*?full_peak_bytes=(\d+)',log,re.M)
                if len(peak)!=1:raise ValueError('Missing module peak')
                return (run['signature'],[m[k] for k in ('launches','poly_muls','coeffs_reduced',
                    'gmp_selftest_cases','gmp_checked','full_checks')],peak[0])
            expected=workload(case_runs[0])
            if any(workload(r)!=expected for r in case_runs):
                raise ValueError('Workload/check/module-peak mismatch: '+case)
            for variant in data['controls']['variants']:
                runs=[r for r in data['runs'] if r['case']['id']==case and r['variant']==variant]
                if len(runs)!=data['controls']['repeats']:raise ValueError('Missing repeats')
                full=[float(r['full']['total']) for r in runs]
                cpu=[r['cpu']['logical_cores'] for r in runs]
                rows.append(dict(study=path.parent.name,case=case,bits=runs[0]['case']['bits'],
                    B2=data['controls']['b2'],D=data['controls']['d'],carrier=data['controls']['carrier'],
                    fold_mb=data['controls']['fold_mb'],variant=variant,samples=len(runs),
                    full_seconds=statistics.median(full),full_min=min(full),full_max=max(full),
                    logical_cores=statistics.median(cpu),
                    lifecycle_seconds=statistics.median(r['cpu']['wall_seconds'] for r in runs)))
    rows.sort(key=lambda r:(r['bits'],r['B2'],r['D'],r['variant']))
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(sources=sources,rows=rows),indent=2),encoding='utf-8')
    for row in rows:print(row)
    if a.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        baseline=[r for r in rows if r['variant']=='fifo']
        low=[next((r for r in rows if (r['study'],r['case'],r['variant'])==(b['study'],b['case'],a.compare)),None) for b in baseline]
        if any(r is None for r in low):raise ValueError('Need fifo/comparison pairs for chart')
        x=list(range(len(baseline)));fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axes[0].bar([i-.18 for i in x],[r['logical_cores'] for r in baseline],width=.36,label='Default FIFO')
        label={'readback':'Readback sleep (experimental)','hybrid':'Low CPU option','blocking':'Native blocking'}[a.compare]
        axes[0].bar([i+.18 for i in x],[r['logical_cores'] for r in low],width=.36,label=label)
        axes[0].set_ylabel('Process CPU / wall (logical cores)');axes[0].legend();axes[0].grid(axis='y',alpha=.25)
        overhead=[100*(r['full_seconds']/b['full_seconds']-1) for b,r in zip(baseline,low)]
        axes[1].bar(x,overhead,color='#c27939');axes[1].axhline(0,color='black',linewidth=.8)
        axes[1].set_ylabel('Engine elapsed change (%)');axes[1].grid(axis='y',alpha=.25)
        axes[1].set_xticks(x,[f"{b['bits']} bits\nB2={b['B2']:.2g}" for b in baseline])
        fig.suptitle('RTX 4060 Laptop @ 1500 MHz | fixed D | n=2 per policy\nCPU reduction and elapsed-time tradeoff; no warmup discarded')
        fig.tight_layout();a.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.figure,dpi=180);plt.close(fig)


if __name__=='__main__':main()
