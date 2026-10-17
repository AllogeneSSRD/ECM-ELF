"""Plot retained full Stage2 samples; never normalize threads, clock or bounds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    a=p.parse_args();source=a.study/'summary.json';data=json.loads(source.read_bytes())
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not data['complete'] or data['failures']:raise ValueError('Study incomplete or failed')
    if len(data['runs'])!=data['expected_invocations']:raise ValueError('Invocation count mismatch')
    if sha(a.study/'bin/ecm_cpu_stage2.exe')!=data['exe_sha256'] or sha(a.study/'bin/gmp-10.dll')!=data['dll_sha256']:
        raise ValueError('Frozen binary changed')
    if sha(a.study/'source_manifest.json')!=data['source_manifest_sha256']:raise ValueError('Source manifest changed')
    manifest=json.loads((a.study/'source_manifest.json').read_bytes())
    for name,digest in manifest.items():
        path=a.study/'bin'/Path(name).name if name.startswith('linked/') else a.study/'source'/name
        if sha(path)!=digest:raise ValueError('Frozen source/library changed: '+name)
    for case in data['cases']:
        if sha(case['save'])!=case['save_sha256']:raise ValueError('Frozen save changed')
    groups={};versions=set()
    for row in data['runs']:
        if not row['full_sample']:raise ValueError('Early factors cannot enter full NF timing plot')
        result_path=Path(row['prefix']).with_suffix('.jsonl')
        if sha(result_path)!=row['result_sha256']:raise ValueError('Raw result changed')
        result=json.loads(result_path.read_bytes());engine=result['stage2-engine']
        versions.add(result['program']['version'])
        if result['status']!='NF' or engine['backend']!='gwpoly' or engine['seconds']!=row['native_seconds']:
            raise ValueError('Timing/result mismatch')
        case=next(c for c in data['cases'] if c['exponent']==row['exponent'])
        if (result['exponent'],result['b1'],result['b2'],result['Edwards']['sigma'])!=(case['exponent'],case['b1'],row['B2'],case['sigma']):
            raise ValueError('Mathematical input mismatch')
        key=(row['exponent'],row['nominal_B2'],row['threads'])
        groups.setdefault(key,[]).append(row)
    exponents=sorted({k[0] for k in groups});bounds=sorted({k[1] for k in groups})
    if len(versions)!=1:raise ValueError('Mixed program versions')
    program_version=versions.pop()
    if data['threads']!=[1,4] or len(bounds)!=2:raise ValueError('Plot requires two bounds and threads 1/4')
    rows=[]
    for (exponent,bound,threads),samples in sorted(groups.items()):
        if len(samples)!=data['repeats']:raise ValueError('Unequal cell count')
        def median(field):return statistics.median(r[field] for r in samples)
        values=[r['native_seconds'] for r in samples]
        metrics=samples[0]['metrics']
        if any(r['metrics']['D']!=metrics['D'] or r['metrics']['fft_length']!=metrics['fft_length'] for r in samples):
            raise ValueError('Geometry changed within cell')
        rows.append(dict(exponent=exponent,bits=samples[0]['bits'],B1=samples[0]['B1'],B2=samples[0]['B2'],
            nominal_B2=bound,threads=threads,n=len(values),seconds=statistics.median(values),
            min_seconds=min(values),max_seconds=max(values),cpu_wall_ratio=median('cpu_wall_ratio'),
            peak_working_set_mib=median('peak_working_set_bytes')/1048576,
            process_seconds=median('process_seconds'),prime95_4t_seconds=samples[0]['prime95_4t_seconds'],
            D=int(metrics['D']),P=int(metrics['P']),fft_length=int(metrics['fft_length']),
            max_roundoff=max(float(r['metrics']['maxerr']) for r in samples),
            gw4_over_prime95=statistics.median(values)/samples[0]['prime95_4t_seconds'] if threads==4 else None))
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source,a.output/'study.snapshot.json');shutil.copy2(__file__,a.output/'plot.snapshot.py')
    with (a.output/'cells.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    lookup={(r['exponent'],r['nominal_B2'],r['threads']):r for r in rows}
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'figure.dpi':140})
    colors=['#727c8e','#65a9d8','#d58b39'];x=np.arange(len(exponents));width=.24
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,bound in zip(axes,bounds):
        for j,threads in enumerate([0,1,4]):
            cells=[lookup[e,bound,threads or 4] for e in exponents]
            values=[r['prime95_4t_seconds'] if not threads else r['seconds'] for r in cells]
            err=None if not threads else [[r['seconds']-r['min_seconds'] for r in cells],[r['max_seconds']-r['seconds'] for r in cells]]
            bars=ax.bar(x+(j-1)*width,values,width,color=colors[j],label=['Prime95 4T (n=1)',f'GW poly 1T (n={data["repeats"]})',f'GW poly 4T (n={data["repeats"]})'][j],yerr=err,capsize=3)
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=8)
        ax.set_yscale('log');ax.set_ylim(bottom=min(r['seconds'] for r in rows)/2,top=max(r['seconds'] for r in rows)*2)
        ax.set_xticks(x,[f'M{e}\nN: {lookup[e,bound,4]["bits"]} bits' for e in exponents])
        ax.set_title(f'Request B2 = {bound:,}\nExact adjusted B2 retained per input');ax.set_ylabel('Full Stage2 seconds (log scale)')
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle(f'ECM-ELF {program_version} GW polynomial Stage2 vs Prime95 31.4b05')
    condition=data['conditions']
    caption=f'{condition["cpu"]} | declared fixed {condition["clock_ghz"]:g} GHz | affinity 0x{data["affinity_mask"]:X} | GW budget {data["memory_mib"]} MiB'
    fig.text(.5,.02,caption+'\nSame N / Edwards sigma / B1 / actual B2. First sample retained. Error bars: min-max. No normalization.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    for ext in ('png','svg'):fig.savefig(a.figures/f'times.{ext}')
    plt.close(fig)
    selected=[lookup[e,b,4] for b in bounds for e in exponents]
    labels=[f'M{r["exponent"]}\nB2 {r["nominal_B2"]:.1e}' for r in selected]
    x=np.arange(len(selected));fig,axes=plt.subplots(1,2,figsize=(13,5))
    speedups=[lookup[r['exponent'],r['nominal_B2'],1]['seconds']/r['seconds'] for r in selected]
    bars=axes[0].bar(x-.18,speedups,.36,color=colors[2],label='Measured 1T / 4T speedup')
    axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
    axes[0].bar(x+.18,[r['cpu_wall_ratio'] for r in selected],.36,color=colors[1],label='4T CPU time / process wall')
    axes[0].axhline(1,color='#777',lw=.8);axes[0].set_ylabel('Ratio');axes[0].legend(fontsize=8)
    for threads,offset,color in [(1,-.18,colors[1]),(4,.18,colors[2])]:
        axes[1].bar(x+offset,[lookup[r['exponent'],r['nominal_B2'],threads]['peak_working_set_mib'] for r in selected],.36,color=color,label=f'{threads}T peak working set')
    axes[1].axhline(data['memory_mib'],color='#777',ls='--',lw=.8,label='Planning budget (not hard cap)')
    axes[1].set_ylabel('Single-process peak working set / MiB');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels,fontsize=8)
    fig.suptitle('Measured parallel scaling and process memory')
    fig.text(.5,.02,'Median of retained samples. CPU occupancy is not speedup. Memory peaks are never summed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ('png','svg'):fig.savefig(a.figures/f'scaling_memory.{ext}')
    plt.close(fig)
    summary=dict(complete=True,cells=rows,conditions=data['conditions'],exe_sha256=data['exe_sha256'],
        source_manifest_sha256=data['source_manifest_sha256'],study_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.figures.iterdir())})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(data["runs"])} calls, {len(rows)} cells, four plot artifacts')


if __name__=='__main__':main()
