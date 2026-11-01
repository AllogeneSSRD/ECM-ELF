"""Audit frozen square matrices and plot whole-curve timings, without GPU work."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
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
    p.add_argument('--stem',help='Figure filename stem; preserves a separate qualified comparison')
    a=p.parse_args();groups=defaultdict(lambda:defaultdict(list));identities=[];experiments=set();tpis=set()
    for study in [a.study,*a.extra]:
        data=load(study/'results.json')
        experiments.add(data.get('experiment','square'))
        tpis.add(data.get('coop_tpi',16))
        if not data['complete'] or data['failures'] or data['diagnostic']:raise ValueError('Need clean completed curves')
        for name,binary in data['variants'].items():
            if sha(Path(binary['exe']))!=binary['sha256']:raise ValueError('Frozen binary changed')
            identities.append(dict(study=str(study),variant=name,exe_sha256=binary['sha256'],
                                   dll_sha256=sha(Path(binary['exe']).parent/'gmp-10.dll')))
        for row in data['rows']:
            if row['seconds']<=0 or row['clock']['clean']!='1':raise ValueError('Invalid timer')
            treatment=row['treatment']
            if data.get('experiment')=='giant-coop' and data.get('coop_control')==16 and treatment=='off':
                treatment='middle'  # Independent paired retest of TPI16 vs the candidate.
            groups[row['key']][treatment].append(row)
    if len(experiments)!=1:raise ValueError('Do not mix different arithmetic experiments')
    experiment=experiments.pop()
    if len(tpis)!=1:raise ValueError('Candidate TPIs differ')
    coop_tpi=tpis.pop()
    middle=any('middle' in variants for variants in groups.values())
    cells=[]
    for key,variants in sorted(groups.items(),key=lambda x:(x[1]['on'][0]['family'],x[1]['on'][0]['bits'])):
        expected={'stock','off','on'}|({'middle'} if middle else set())
        if set(variants)!=expected:raise ValueError('Missing treatment: '+key)
        signatures={tuple(r['signature']) for rows in variants.values() for r in rows}
        if len(signatures)!=1:raise ValueError('Workload/root mismatch: '+key)
        ref=variants['on'][0];values={}
        for name,rows in variants.items():
            times=[r['seconds'] for r in rows]
            curve_walls=[]
            for r in rows:
                if 'curve_wall_seconds' in r:curve_walls.append(r['curve_wall_seconds'])
                else:
                    matches=re.findall(r'^curve_done: .* wall=([0-9.]+) s$',(Path(r['directory'])/'console.log').read_text(encoding='utf-8'),re.M)
                    if len(matches)!=1:raise ValueError('Missing complete curve wall timer')
                    curve_walls.append(float(matches[0]))
            points=[float(r['stats']['real_baby']['ladder'])+float(r['stats']['real_baby']['affine'])+
                    float(r['stats']['real_batched_split']['giant'])+float(r['stats']['real_batched_wall']['gleaves']) for r in rows]
            values[name]=dict(n=len(rows),median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),
                              median_curve_wall_seconds=statistics.median(curve_walls),
                              median_point_seconds=statistics.median(points),evidence=[r['directory'] for r in rows])
        cells.append(dict(key=key,family=ref['family'],bits=ref['bits'],variants=values,
                          on_vs_stock=values['on']['median_seconds']/values['stock']['median_seconds'],
                          on_vs_off=values['on']['median_seconds']/values['off']['median_seconds']))
    names=('stock','off','middle','on') if middle else ('stock','off','on')
    totals={name:sum(c['variants'][name]['median_seconds'] for c in cells) for name in names}
    curve_totals={name:sum(c['variants'][name]['median_curve_wall_seconds'] for c in cells) for name in names}
    result=dict(complete=True,experiment=experiment,coop_tpi=coop_tpi,identities=identities,
                timing='engine init+main excludes point startup checks; complete curve wall reported separately; no discarded runs',
                cells=cells,totals=totals,curve_wall_totals=curve_totals,
                on_vs_stock=totals['on']/totals['stock'],on_vs_off=totals['on']/totals['off'],
                off_vs_stock=totals['off']/totals['stock'],
                default_regressions=[c['key'] for c in cells if c['variants']['off']['median_seconds']>1.05*c['variants']['stock']['median_seconds'] and
                                     c['variants']['off']['median_seconds']-c['variants']['stock']['median_seconds']>.020],
                regressions=[c['key'] for c in cells if c['on_vs_stock']>1.05 and
                             c['variants']['on']['median_seconds']-c['variants']['stock']['median_seconds']>.020])
    if middle:
        result['on_vs_middle']=totals['on']/totals['middle']
        for c in cells:c['on_vs_middle']=c['variants']['on']['median_seconds']/c['variants']['middle']['median_seconds']
    a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    stem=a.stem or ('stage2-square-curves' if experiment=='square' else 'stage2-giant-coop-curves')
    labels={'stock':'stock reference','off':'candidate scalar default','middle':'candidate fixed-16k TPI16',
            'on':f'candidate cooperative TPI{coop_tpi}'} if experiment=='giant-coop' else {'stock':'0.56.4 stock','off':'0.57.1 square off','on':'0.57.1 square on'}
    colors={'stock':'#777777','off':'#df8a32','middle':'#729845','on':'#247fae'}
    for metric,suffix,title in [('median_seconds','','Engine init + main'),
                                 ('median_curve_wall_seconds','-curve-wall','Complete curve wall')]:
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
        for column,family in enumerate(('ordinary','carrier')):
            rows=[c for c in cells if c['family']==family];x=[c['bits'] for c in rows]
            for name in names:
                axes[0,column].plot(x,[c['variants'][name][metric] for c in rows],'.-',label=labels[name],color=colors[name])
            axes[0,column].set_title(family);axes[0,column].set_ylabel(title+' (s)');axes[0,column].set_ylim(bottom=0)
            axes[0,column].legend(fontsize=9)
            for ref,label,color in [('stock','stock','#247fae'),('off','disabled path','#df8a32')]+([('middle','TPI16','#729845')] if middle else []):
                gain=[100*(1-c['variants']['on'][metric]/c['variants'][ref][metric]) for c in rows]
                axes[1,column].plot(x,gain,'.--',label='Time decrease vs '+label,color=color)
            axes[1,column].axhline(0,color='gray',linestyle=':');axes[1,column].set_ylabel(title+' decrease (%)')
            axes[1,column].set_xlabel('Target modulus bits');axes[1,column].legend(fontsize=9)
            for row in range(2):
                ax=axes[row,column];ax.grid(alpha=.25);ax.xaxis.set_major_locator(MultipleLocator(2000));ax.tick_params(axis='x',rotation=45)
        fig.suptitle('RTX 4060 Laptop | 1500 MHz target | fixed D/layout/chain | small B2\n'
                     'Independent processes; per-cell medians; n in summary (screening n=1, retests retained)',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.92));png=a.figures/(stem+suffix+'.png');fig.savefig(png,dpi=160)
        fig.savefig(a.figures/(stem+suffix+'.svg'));plt.close(fig)
        if a.public_figures:
            a.public_figures.mkdir(parents=True,exist_ok=True);shutil.copy2(png,a.public_figures/png.name)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','identities')},indent=2))


if __name__=='__main__':
    main()
