"""Validate and plot the complete fixed-D OpenCL/CUDA comparison."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics as st

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists() or a.figure.exists():
        raise SystemExit('Choose new derived output paths')
    manifest=json.loads((a.run/'completed.json').read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise RuntimeError('Incomplete matrix')
    rows=[json.loads(s) for s in (a.run/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=defaultdict(list)
    for row in rows:
        path=Path(row['log'])
        if sha(path)!=row['log_sha256'] or sha(path.parent/'results.jsonl')!=row['result_sha256']:
            raise RuntimeError('Modified raw evidence: '+str(path))
        if row['source_build'].lower()!=manifest['binaries'][row['backend']]['sha256'].lower():
            raise RuntimeError('Executed binary identity mismatch')
        groups[(row['bits'],row['exponent'],row['B2'],row['backend'])].append(row)
    cells=[]
    for (bits,exponent,b2,backend),samples in sorted(groups.items()):
        if len(samples)!=manifest['repeats'] or len({r['save_sha256'] for r in samples})!=1:
            raise RuntimeError('Wrong sample count or changing input')
        cells.append(dict(bits=bits,exponent=exponent,B2=b2,backend=backend,n=len(samples),D=samples[0]['D'],P=samples[0]['P'],
            engine_seconds=st.median(r['engine_seconds'] for r in samples),
            min_engine_seconds=min(r['engine_seconds'] for r in samples),max_engine_seconds=max(r['engine_seconds'] for r in samples),
            process_seconds=st.median(r['process_seconds'] for r in samples),
            init_seconds=st.median(r['init_seconds'] for r in samples),main_seconds=st.median(r['main_seconds'] for r in samples)))
    lookup={(r['bits'],r['B2'],r['backend']):r for r in cells}
    paired=[]
    for r in cells:
        if r['backend']!='opencl':
            continue
        cuda=lookup[(r['bits'],r['B2'],'cuda')]
        assert (r['D'],r['P'])==(cuda['D'],cuda['P'])
        paired.append(dict(bits=r['bits'],exponent=r['exponent'],B2=r['B2'],D=r['D'],P=r['P'],
            cuda_seconds=cuda['engine_seconds'],opencl_seconds=r['engine_seconds'],
            opencl_over_cuda=r['engine_seconds']/cuda['engine_seconds'],
            cuda_process_seconds=cuda['process_seconds'],opencl_process_seconds=r['process_seconds'],
            process_ratio=r['process_seconds']/cuda['process_seconds']))
    a.out.mkdir(parents=True)
    for name,data in [('cells',cells),('pairs',paired)]:
        with (a.out/(name+'.csv')).open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={'manifest':manifest,'samples':len(rows),'cells':cells,'pairs':paired,
        'sum_medians':{b:sum(r['engine_seconds'] for r in cells if r['backend']==b) for b in ['cuda','opencl']}}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(a.out/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounds=sorted({r['B2'] for r in cells})
    fig,axes=plt.subplots(2,len(bounds),figsize=(6.5*len(bounds),8),squeeze=False,layout='constrained')
    for col,b2 in enumerate(bounds):
        ax=axes[0,col]
        for backend,color in [('cuda','#2563eb'),('opencl','#ea580c')]:
            line=sorted((r for r in cells if r['B2']==b2 and r['backend']==backend),key=lambda x:x['bits'])
            x=[r['bits'] for r in line];y=[r['engine_seconds'] for r in line]
            ax.errorbar(x,y,yerr=[[r['engine_seconds']-r['min_engine_seconds'] for r in line],[r['max_engine_seconds']-r['engine_seconds'] for r in line]],marker='o',label=backend.upper(),color=color,capsize=4)
        ax.set(title=f'B2 = {b2:,}; D = {manifest["D"]:,}',ylabel='Full engine seconds (log scale)',yscale='log',xlabel='Actual cofactor bits')
        ax.set_xticks(range(0,9000,2000));ax.grid(alpha=.2);ax.legend()
        ax=axes[1,col]
        line=sorted((r for r in paired if r['B2']==b2),key=lambda x:x['bits'])
        x=[r['bits'] for r in line]
        ax.plot(x,[r['opencl_over_cuda'] for r in line],marker='o',label='Engine total')
        ax.plot(x,[r['process_ratio'] for r in line],marker='s',label='Full process')
        ax.axhline(1,color='#777',ls='--');ax.set(xlabel='Actual cofactor bits',ylabel='OpenCL / CUDA (higher = slower OpenCL)')
        ax.set_xticks(range(0,9000,2000));ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'Native OpenCL vs CUDA Stage2 | RTX 4060 Laptop | n={manifest["repeats"]}\nNo discarded warmup; identical save/D/B2; includes engine initialization and checks')
    a.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.figure,dpi=170);plt.close(fig)
    print(json.dumps({'samples':len(rows),'sum_medians':result['sum_medians'],'pairs':paired},ensure_ascii=False))

if __name__=='__main__':
    main()
