"""Audit frozen square matrices and plot whole-curve timings, without GPU work."""
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
from matplotlib.ticker import MultipleLocator


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--extra',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--public-figures',type=Path)
    a=p.parse_args();groups=defaultdict(lambda:defaultdict(list));identities=[];experiments=set()
    for study in [a.study,*a.extra]:
        data=load(study/'results.json')
        experiments.add(data.get('experiment','square'))
        if not data['complete'] or data['failures'] or data['diagnostic']:raise ValueError('Need clean completed curves')
        for name,binary in data['variants'].items():
            if sha(Path(binary['exe']))!=binary['sha256']:raise ValueError('Frozen binary changed')
            identities.append(dict(study=str(study),variant=name,exe_sha256=binary['sha256'],
                                   dll_sha256=sha(Path(binary['exe']).parent/'gmp-10.dll')))
        for row in data['rows']:
            if row['seconds']<=0 or row['clock']['clean']!='1':raise ValueError('Invalid timer')
            groups[row['key']][row['treatment']].append(row)
    if len(experiments)!=1:raise ValueError('Do not mix different arithmetic experiments')
    experiment=experiments.pop()
    cells=[]
    for key,variants in sorted(groups.items(),key=lambda x:(x[1]['on'][0]['family'],x[1]['on'][0]['bits'])):
        if set(variants)!=set(('stock','off','on')):raise ValueError('Missing treatment: '+key)
        signatures={tuple(r['signature']) for rows in variants.values() for r in rows}
        if len(signatures)!=1:raise ValueError('Workload/root mismatch: '+key)
        ref=variants['on'][0];values={}
        for name,rows in variants.items():
            times=[r['seconds'] for r in rows]
            points=[float(r['stats']['real_baby']['ladder'])+float(r['stats']['real_baby']['affine'])+
                    float(r['stats']['real_batched_split']['giant'])+float(r['stats']['real_batched_wall']['gleaves']) for r in rows]
            values[name]=dict(n=len(rows),median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),
                              median_point_seconds=statistics.median(points),evidence=[r['directory'] for r in rows])
        cells.append(dict(key=key,family=ref['family'],bits=ref['bits'],variants=values,
                          on_vs_stock=values['on']['median_seconds']/values['stock']['median_seconds'],
                          on_vs_off=values['on']['median_seconds']/values['off']['median_seconds']))
    totals={name:sum(c['variants'][name]['median_seconds'] for c in cells) for name in ('stock','off','on')}
    result=dict(complete=True,experiment=experiment,identities=identities,timing='engine init+main, no discarded runs',cells=cells,totals=totals,
                on_vs_stock=totals['on']/totals['stock'],on_vs_off=totals['on']/totals['off'],
                off_vs_stock=totals['off']/totals['stock'],
                default_regressions=[c['key'] for c in cells if c['variants']['off']['median_seconds']>1.05*c['variants']['stock']['median_seconds'] and
                                     c['variants']['off']['median_seconds']-c['variants']['stock']['median_seconds']>.020],
                regressions=[c['key'] for c in cells if c['on_vs_stock']>1.05 and
                             c['variants']['on']['median_seconds']-c['variants']['stock']['median_seconds']>.020])
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
    for column,family in enumerate(('ordinary','carrier')):
        rows=[c for c in cells if c['family']==family];x=[c['bits'] for c in rows]
        labels=('0.56.4 stock','0.57.1 square off','0.57.1 square on') if experiment=='square' else ('0.57.2 stock','0.57.3 cooperative off','0.57.3 cooperative TPI16')
        for name,label,color in zip(('stock','off','on'),labels,('#777777','#df8a32','#247fae')):
            axes[0,column].plot(x,[c['variants'][name]['median_seconds'] for c in rows],'.-',label=label,color=color)
        axes[0,column].set_title(family);axes[0,column].set_ylabel('Engine init + main (s)');axes[0,column].set_ylim(bottom=0)
        axes[0,column].legend(fontsize=9)
        axes[1,column].plot(x,[100*(1-c['on_vs_stock']) for c in rows],'o-',label='Time decrease vs stock',color='#247fae')
        axes[1,column].plot(x,[100*(1-c['on_vs_off']) for c in rows],'.--',label='Time decrease vs disabled path',color='#df8a32')
        axes[1,column].axhline(0,color='gray',linestyle=':');axes[1,column].set_ylabel('Whole-curve time decrease (%)')
        axes[1,column].set_xlabel('Target modulus bits');axes[1,column].legend(fontsize=9)
        for row in range(2):
            ax=axes[row,column];ax.grid(alpha=.25);ax.xaxis.set_major_locator(MultipleLocator(2000));ax.tick_params(axis='x',rotation=45)
    fig.suptitle('RTX 4060 Laptop | 1500 MHz target | fixed D/layout/chain | small B2\n'
                 'Independent processes; per-cell medians; n in summary (screening n=1, retests retained)',fontsize=12)
    stem='stage2-square-curves' if experiment=='square' else 'stage2-giant-coop-curves'
    fig.tight_layout(rect=(0,0,1,.92));png=a.figures/(stem+'.png');fig.savefig(png,dpi=160)
    fig.savefig(a.figures/(stem+'.svg'));plt.close(fig)
    if a.public_figures:
        a.public_figures.mkdir(parents=True,exist_ok=True);shutil.copy2(png,a.public_figures/png.name)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','identities')},indent=2))


if __name__=='__main__':
    main()
