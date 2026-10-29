"""Summarise CUDA/GMP component timings and existing Stage2 phase evidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics as st

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_samples(folder):
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    if not manifest['passed']:
        raise ValueError('Probe did not pass')
    for name,digest in manifest['artifacts'].items():
        if sha(folder/name)!=digest:
            raise ValueError('Modified evidence: '+name)
    text=(folder/'samples.csv').read_text(encoding='utf-8')
    rows=list(csv.DictReader(line for line in text.splitlines() if not line.startswith('#')))
    groups={}
    for row in rows:
        key=(row['operation'],int(row['bits']),int(row['count']),row['implementation'])
        groups.setdefault(key,[]).append(row)
    summary=[]
    for (op,bits,count,impl),samples in sorted(groups.items()):
        wall=[float(r['wall_seconds']) for r in samples]
        summary.append(dict(operation=op,bits=bits,count=count,implementation=impl,n=len(samples),
            median_ms=st.median(wall)*1000,min_ms=min(wall)*1000,max_ms=max(wall)*1000,
            kernel_median_ms=st.median(float(r['kernel_seconds']) for r in samples)*1000,
            root_inverse_median_ms=st.median(float(r['cpu_root_inverse_seconds']) for r in samples)*1000))
    return manifest,rows,summary

def num(pattern,text):
    match=re.search(pattern,text)
    if not match:
        raise ValueError('Missing field: '+pattern)
    return float(match.group(1))

def old_phases(root):
    groups={}
    evidence=[]
    for folder in ['final_wide','final_long','production_head']:
        file=root/folder/'measurements.json'
        ledger=json.loads(file.read_text(encoding='utf-8-sig'))
        if not ledger['complete']:
            raise ValueError('Incomplete historical matrix')
        evidence.append(dict(path=str(file),sha256=sha(file),binary_sha256=ledger['identity']['binary_sha256']))
        for run in ledger['runs']:
            if run['variant']!='fifo':
                continue
            texts={}
            for field in ['log','debug']:
                # Basenames permit moving an intact evidence directory.
                path=root/folder/Path(run[field]).name
                if sha(path)!=run[field+'_sha256']:
                    raise ValueError('Modified historical log: '+str(path))
                texts[field]=path.read_text(encoding='utf-8-sig')
                evidence.append(dict(path=str(path),sha256=sha(path)))
            t,d=texts['log'],texts['debug']
            bits=int(num(r'N_bits=(\d+)',t))
            B2=int(num(r'real_shape:.*?B2=(\d+)',t))
            D=int(num(r'real_shape: D=(\d+)',t))
            P=int(num(r'real_shape:.*?baby_points=(\d+)',t))
            total=num(r'stage2_full_wall:.*?total=([\d.]+)',t)
            values=dict(total_s=total,init_s=num(r'stage2_full_wall:.*?init=([\d.]+)',t),
                main_s=num(r'stage2_full_wall:.*?main=([\d.]+)',t),
                baby_affine_s=num(r'real_baby:.*?affine=([\d.]+)',t),
                baby_inverse_s=num(r'baby_device:.*?invert=([\d.]+)',d),
                giant_inverse_s=num(r'real_batched_gleaves:.*?invert=([\d.]+)',d))
            for key in ['giant','gtrees','fold','descent','inv','accum']:
                values[key+'_s']=num(r'real_batched_split:.*?'+key+r'=([\d.]+)',t)
            for key,value in list(values.items()):
                if key!='total_s':
                    values[key[:-2]+'_percent']=100*value/total
            groups.setdefault((bits,B2,D,P),[]).append(values)
    summary=[]
    for (bits,B2,D,P),rows in sorted(groups.items()):
        summary.append(dict(bits=bits,B2=B2,D=D,P=P,blocks=(P+63)//64,n=len(rows),
            **{k:st.median(r[k] for r in rows) for k in rows[0]}))
    return summary,evidence

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plot(summary,phases,out):
    os.environ.setdefault('MPLCONFIGDIR',str(out.parent/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    lookup={(r['operation'],r['bits'],r['count'],r['implementation']):r for r in summary}
    bits=[512,2203,4423,8191]
    fig,axes=plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
    for axis,op,cpu,gpu,title in [
        (axes[0,0],'normalize','cpu_batch256','cuda_hybrid','Normalization: resident X/Z to host coefficients'),
        (axes[1,0],'accumulate','cpu_products_gcd','cuda_products_cpu_gcd','Block products + GCD: host leaves to host GCDs')]:
        for count,marker in [(1024,'o'),(8192,'s')]:
            ratios=[lookup[(op,b,count,gpu)]['median_ms']/lookup[(op,b,count,cpu)]['median_ms'] for b in bits]
            axis.plot(bits,ratios,marker=marker,label=f'{count:,} points')
        axis.axhline(1,color='#888',ls='--',lw=1)
        axis.set(title=title,xlabel='Modulus bits',ylabel='CUDA wall / CPU wall (higher favors CPU)',yscale='log')
        axis.set_xticks(bits)
        axis.grid(alpha=.2)
        axis.legend()
    ax=axes[0,1]
    for b in bits:
        counts=[1,16,128,1024]
        ratios=[lookup[('gcd',b,c,'cuda_binary_experimental')]['median_ms']/lookup[('gcd',b,c,'cpu_gmp')]['median_ms'] for c in counts]
        ax.plot(counts,ratios,marker='o',label=f'{b} bits')
    ax.axhline(1,color='#888',ls='--',lw=1)
    ax.set(title='GCD: experimental CUDA binary vs GMP',xlabel='GCD count',ylabel='CUDA wall / CPU wall',xscale='log',yscale='log')
    ax.set_xticks([1,16,128,1024],['1','16','128','1024'])
    ax.grid(alpha=.2)
    ax.legend()
    ax=axes[1,1]
    fields=['baby_affine_percent','baby_inverse_percent','giant_inverse_percent','accum_percent']
    matrix=np.array([[p[k] for p in phases] for k in fields])
    im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=5)
    for y in range(4):
        for x in range(len(phases)):
            ax.text(x,y,f'{matrix[y,x]:.2f}%',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(len(phases)),[str(p['bits']) for p in phases],rotation=35)
    ax.set_yticks(range(4),['Baby normalization*','CPU baby inverse','CPU giant inverse','Products + GCD*'])
    ax.set(title='Existing full Stage2: diagnostic share of engine total',xlabel='Actual cofactor bits; * contains more than CPU GCD')
    fig.colorbar(im,ax=ax,label='% of full Stage2',shrink=.75)
    fig.suptitle('CUDA Stage2 CPU/GPU tail comparison | 4060 Laptop | n=3\nComponent ratios and historical phase shares have different clocks/inputs',fontsize=13)
    fig.savefig(out,dpi=170)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--history',type=Path,default=Path('data/experiments/stage2_host_wait_20261025'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists() or a.figure.exists():
        raise SystemExit('Choose new derived output paths')
    manifest,samples,summary=read_samples(a.run)
    phases,evidence=old_phases(a.history)
    a.out.mkdir(parents=True)
    a.figure.parent.mkdir(parents=True,exist_ok=True)
    write_csv(a.out/'components.csv',summary)
    write_csv(a.out/'phases.csv',phases)
    write_csv(a.out/'samples.csv',samples)
    result={'probe':manifest,'recorded_samples':len(samples),
            'recorded_verified_words':sum(int(r['verified_words']) for r in samples),
            'priming_excluded_from_samples':True,'components':summary,'phases':phases,'historical_evidence':evidence,
            'notes':['Phase percentages are per-run ratios followed by median, not ratios of medians.',
                     'Diagnostic phase counters overlap; never sum them as a 100% partition.',
                     'accum is an upper bound for GCD, including products/transfers/other work.',
                     'Experimental binary GCD is not an optimized CUDA GCD library or a production kernel.']}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    plot(summary,phases,a.figure)
    print(json.dumps({'out':str(a.out),'figure':str(a.figure),'samples':len(samples)},ensure_ascii=False))

if __name__=='__main__':
    main()
