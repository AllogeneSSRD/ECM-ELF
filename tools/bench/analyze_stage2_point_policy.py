"""Audit bounded point-policy experiments and plot complete-engine phase costs."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import matplotlib.pyplot as plt
import numpy as np
from analyze_stage2_point_routes import audited
from bench_stage2_layout_matrix import phases
from bench_stage2_version import write


def point_work(row,d,words):
    first=1;seed=chain=ladder=0;chunks=[]
    for route in row['giant_policies']:
        n=int(route['npts']);chunks.append(n)
        if route['route']=='chain':
            length=int(route['block'])
            for offset in range(0,n,length):
                seed+=5+11*((first+offset).bit_length()-1)
                chain+=6*max(0,min(length,n-offset)-2)
        else:
            ladder+=sum(5+11*((i*d).bit_length()-1) for i in range(first,first+n))
        first+=n
    return dict(seed_modmuls=seed,chain_modmuls=chain,ladder_modmuls=ladder,total_modmuls=seed+chain+ladder,
                point_chunks=chunks,points=first-1,coordinate_payload_bytes=max(chunks)*16*words)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True);p.add_argument('--publish',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=('tail_abba02','wide_abba01','tail_leafcheck01','wide_leafcheck01','final_tail_holdout01','final_tail_holdout02')
    studies={name:audited(a.root/name) for name in names}
    hashes={}
    leaf_words=0
    for name in ('tail_leafcheck01','wide_leafcheck01'):
        for row in studies[name]['runs']:
            digest=row['stats']['target_descent_values'];key=studies[name]['case']['id']+'/'+str(row['actual_B2'])
            value={k:digest[k] for k in ('words','hash')}
            if key in hashes and hashes[key]!=value:raise ValueError('Equal-interval normalized leaves differ')
            hashes[key]=value;leaf_words+=int(row['stats']['device_gleaf']['checked_leaf_words'])
    colors=['#67b8a4','#a8d6c7','#d8ba6c','#7a5a9c','#3887ba','#e18b52','#abcf6b','#d786a8','#bfc4ca']
    keys=['baby','ftree_init','inv','giant_combined','gtrees','fold','descent','accum','other']
    labels=['Baby','F tree / init other','Inverse','Giant + leaf prep / waits','G trees','Fold','Descent','Accum / GCD','Other']
    fig,axes=plt.subplots(1,2,figsize=(15,6));tables={}
    for ax,name,title in zip(axes,names[:2],('7995-bit target / 8011-bit carrier','16272-bit target / 16319-bit carrier')):
        study=studies[name]
        if study['diagnostic']:raise ValueError('Diagnostics entered timing population')
        modes=list(dict.fromkeys(study['sequence']));table=[]
        for mode in modes:
            rows=[r for r in study['runs'] if r['mode']==mode];parts=[];works=[]
            for r in rows:
                d=phases(r['engine_seconds'],float(r['clock']['init']),r['stats']['real_baby'],r['stats']['real_batched_split'])
                gleaves=float(r['stats']['real_batched_wall']['gleaves'])
                d['giant_combined']=d.pop('giant')+gleaves;d['other']-=gleaves
                if d['other']<-.012:raise ValueError('Point preparation overlaps another phase')
                parts.append(d);works.append(point_work(r,study['D'],(int(r['shape']['S_bits'])+63)//64))
            if any(w!=works[0] for w in works):raise ValueError('Repeated point work differs')
            table.append(dict(mode=mode,n=len(rows),mean=statistics.mean(r['engine_seconds'] for r in rows),
                min=min(r['engine_seconds'] for r in rows),max=max(r['engine_seconds'] for r in rows),
                requested_B2=rows[0]['requested_B2'],actual_B2=rows[0]['actual_B2'],work=works[0],
                phases={k:statistics.mean(x[k] for x in parts) for k in keys}))
        tables[name]=table;x=np.arange(len(modes));bottom=np.zeros(len(modes))
        for key,label,color in zip(keys,labels,colors):
            y=np.array([t['phases'][key] for t in table]);ax.bar(x,y,bottom=bottom,label=label,color=color);bottom+=y
        for i,t in enumerate(table):
            t['change_percent']=100*(t['mean']/table[0]['mean']-1)
            ax.vlines(i,t['min'],t['max'],color='black',lw=1.5)
            ax.text(i,t['max']+.6,f'{t["mean"]:.3f}s\n{t["change_percent"]:+.2f}%',ha='center',fontsize=10)
        ax.set_xticks(x,[dict(baseline='Legacy / 64',split6='6P / 64',balanced='Balanced / 64',rounded='Rounded B2 / 64',chain32='Legacy / 32')[m] for m in modes])
        ax.set(title=title,ylabel='Complete engine seconds (init + main)',ylim=(0,max(t['max'] for t in table)*1.2))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,fontsize=8)
    fig.subplots_adjust(top=.80,bottom=.20,wspace=.18)
    fig.text(.5,.035,'D180180 / P17280 | B1=50,000 | Requested B2=27,015,978,990 | RTX4060 Laptop, 1500MHz | n=2 each\nRounded B2=31,134,743,640 (172800 points vs 149940). Extra correctness runs excluded; component timers do not imply GPU kernel times.',ha='center',fontsize=9)
    a.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.figure,dpi=160);plt.close(fig)
    if a.publish:
        a.publish.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.figure,a.publish)
    result=dict(complete=True,tables=tables,hashes=hashes,checked_leaf_words=leaf_words,
        studies={n:dict(exe_sha256=s['exe_sha256'],audit=s['audit']) for n,s in studies.items()})
    write(a.output/'summary.json',result);shutil.copy2(__file__,a.output/'analysis_snapshot.py')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
