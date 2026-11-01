"""Audit production-header ladder probes; compare retained GPU event samples."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    a=p.parse_args();groups=defaultdict(lambda:defaultdict(list));identities=[];devices=set();busy_clocks=set();busy_samples=0
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for study in a.study:
        data=json.loads((study/'results.json').read_text(encoding='utf-8'))
        if not data['complete'] or data['failures'] or data['exploratory']:
            raise ValueError('Need clean, completed performance evidence')
        if data['conditions']['probe']!='ladder' or data['conditions']['suite']!='performance':
            raise ValueError('Correctness calls are not performance samples')
        exe=study/'bin'/Path(data['conditions']['exe']).name
        if sha(exe)!=data['exe_sha256'] or sha(exe.parent/'gmp-10.dll')!=data['dll_sha256']:
            raise ValueError('Frozen binary changed')
        identities.append(dict(study=str(study),exe_sha256=data['exe_sha256'],dll_sha256=data['dll_sha256'],device=data['device']))
        devices.add((data['device']['uuid'],data['device']['name']))
        for line in (study/'telemetry.jsonl').read_text(encoding='utf-8').splitlines():
            sample=json.loads(line)
            if sample.get('gpu_percent',0)>=50:
                busy_clocks.add(sample['sm_clock_mhz']);busy_samples+=1
        for row in data['rows']:
            f=row['fields'];samples=list(map(float,f['samples_ms'].split(',')))
            if row['fault'] or row['returncode'] or int(f['bad']) or len(samples)!=5 or min(samples)<=0:
                raise ValueError('Invalid sample/checks')
            checked_points=min(17,int(f['points'])) if int(f['points'])>37 else int(f['points'])
            if int(f['checked_words'])!=checked_points*2*int(f['nw']):raise ValueError('Incomplete distributed GMP coverage')
            median=statistics.median(samples)
            if abs(median-float(f['median_ms']))>2e-6:raise ValueError('Median differs')
            key=tuple(int(f[k]) for k in ('bits','points','scalar_bits','kind','normal'))
            groups[key][int(f['tpi'])].append(dict(ms=median,sha256=f['sha256'],directory=row['directory'],
                capacity_bits=int(f['capacity_bits']),registers=int(f['registers']),local_bytes=int(f['local_bytes'])))
    cells=[]
    for key,variants in sorted(groups.items()):
        if set(variants)!={0,32}:raise ValueError('Missing scalar/cooperative treatment')
        if len({r['sha256'] for rows in variants.values() for r in rows})!=1:raise ValueError('Point outputs differ')
        if min(map(len,variants.values()))<2:raise ValueError('Need two opposite-order processes')
        values={str(tpi):dict(n=len(rows),median_ms=statistics.median(r['ms'] for r in rows),
                            capacity_bits=rows[0]['capacity_bits'],registers=rows[0]['registers'],
                            local_bytes=rows[0]['local_bytes'],evidence=[r['directory'] for r in rows]) for tpi,rows in variants.items()}
        cells.append(dict(zip(('bits','points','scalar_bits','kind','normal'),key),variants=values,
                          ratio=values['32']['median_ms']/values['0']['median_ms']))
    if not cells or len(devices)!=1:raise ValueError('Need nonempty single-device study')
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    result=dict(complete=True,identities=identities,cells=cells,busy_sm_clocks_mhz=sorted(busy_clocks),busy_samples=busy_samples,
                timing='All five CUDA events retained per process; medians across opposite-order processes; allocation/GMP/readback excluded',
                verification='17 distributed points per throughput call (all if <=37); complete output SHA compared across treatments; full GMP correctness separately')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4));labels=[]
    for i,cell in enumerate(cells):
        labels.append(f"{cell['bits']}b / {'Mersenne' if cell['kind']==1 else 'ordinary'}")
        ax.bar(i,cell['ratio'],color='#247fae')
        ax.text(i,cell['ratio']+.015,f"{cell['ratio']:.3f}",ha='center',fontsize=9)
    ax.set_xticks(range(len(cells)),labels,rotation=25,ha='right');ax.axhline(1,color='gray',ls=':')
    ax.set_ylabel('TPI32 / scalar CUDA event time');ax.set_ylim(0,max(1.15,max(c['ratio'] for c in cells)*1.15))
    ax.set_title('Production-header binary ladder | '+next(iter(devices))[1]+'\n'
                 'Same input and complete output; batch size, scalar bits and n in summary')
    fig.tight_layout();fig.savefig(a.figures/'stage2-baby-ladder-ratios.png',dpi=160);plt.close(fig)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
