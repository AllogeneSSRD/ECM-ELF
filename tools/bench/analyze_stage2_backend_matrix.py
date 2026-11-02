"""Re-audit held-out point-backend curves and publish small-B2 comparisons."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from bench_stage2_backend_matrix import capture,learned
from bench_stage2_version import load,write
from bench_stage2_point_routes import sha
from bench_stage2_runtime_matrix import completed_records
from analyze_stage2_point_matrix import partition


def audit(root):
    plan=load(root/'plan.json')
    if not plan['complete'] or plan['failures'] or len(plan['cells'])!=26 or len(plan['training'])!=26 or len(plan['decisions'])!=26:
        raise ValueError('Need complete 13-width ordinary/carrier evidence')
    if sha(root/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(root/'bin/gmp-10.dll')!=plan['dll_sha256'] or sha(root/'source_plan.json')!=plan['source_sha256']:
        raise ValueError('Frozen source/binary mismatch')
    signatures={};clocks=Counter();checked=0;selftests=0;calls=0
    for entry in plan['training']+plan['evaluations']:
        cell=next(c for c in plan['cells'] if c['id']==entry['cell']);case=next(c for c in plan['cases'] if c['id']==cell['case'])
        folder=Path(entry['folder'])
        if sha(Path(case['save']))!=case['save_sha256']:raise ValueError('Save changed')
        for name,digest in entry['file_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Evidence changed: '+str(folder/name))
        rows=capture(plan,root,folder,case,cell)
        if rows!=entry['rows']:raise ValueError('Capture differs from raw receipt/log')
        if entry['label']=='training' and not learned(completed_records(folder/'result.jsonl'),8192<(cell['carrier'] or case['bits'])<=16384):
            raise ValueError('Incomplete backend qualification')
        for row in rows:
            signature=(row['root'],row['ntt'],row['s4']['coeffs_reduced'],[(p['npts'],p['route'],p['block']) for p in row['policies']])
            if cell['id'] in signatures and signatures[cell['id']]!=signature:raise ValueError('Point backend changed mathematics/workload')
            signatures[cell['id']]=signature;checked+=int(row['s4']['gmp_checked'])
            selftests+=int(row['s4']['gmp_selftest_cases']);partition(row)
        calls+=len(rows)
        for sample in (folder/'telemetry.jsonl').read_text(encoding='utf-8').splitlines():
            sample=json.loads(sample)
            if sample.get('gpu_percent',0)>=50:clocks[sample['sm_clock_mhz']]+=1
    cells=[]
    for cell in plan['cells']:
        case=next(c for c in plan['cases'] if c['id']==cell['case']);decision=next(d for d in plan['decisions'] if d['cell']==cell['id'])
        frozen=root/'profiles'/(cell['id']+'.frozen.toml')
        if sha(frozen)!=decision['profile_sha256']:raise ValueError('Frozen tuning profile changed')
        values={}
        for mode in ('scalar','auto'):
            entries=[e for e in plan['evaluations'] if e['cell']==cell['id'] and (e['label']==mode or e['label'].startswith(mode+'_retest'))]
            if len(entries) not in (1,3):raise ValueError('Incomplete held-out treatment/retests')
            rows=[e['rows'][0] for e in entries]
            expected=(0,0) if mode=='scalar' else (decision['selection']['baby_coop'],decision['selection']['giant_coop'])
            if any((r['baby'],r['giant'])!=expected for r in rows):raise ValueError('Held-out execution differs from frozen decision')
            values[mode]=dict(n=len(rows),seconds=st.median(r['seconds'] for r in rows),
                minimum=min(r['seconds'] for r in rows),maximum=max(r['seconds'] for r in rows),
                curve_wall=st.median(r['curve_wall'] for r in rows),
                phases={key:st.mean(partition(r)[key] for r in rows) for key in partition(rows[0])})
        cells.append(dict(cell=cell['id'],case=case['id'],bits=case['bits'],exponent=case['exponent'],B1=case['b1'],B2=cell['B2'],
            family='carrier' if cell['carrier'] else 'ordinary',carrier=cell['carrier'],arithmetic_bits=cell['carrier'] or case['bits'],
            D=cell['D'],P=cell['P'],layout=cell['layout'],chain=cell['chain'],baby=decision['selection']['baby_coop'],
            giant=decision['selection']['giant_coop'],training_score=decision['selection']['comparison_seconds'],
            historical_seconds=cell['reference_seconds'],values=values,change_percent=100*(values['auto']['seconds']/values['scalar']['seconds']-1)))
    best=[];reference=load(root/'reference_layout.json') if plan['reference_layout_sha256'] else None
    if reference and sha(root/'reference_layout.json')!=plan['reference_layout_sha256']:raise ValueError('Historical reference changed')
    for case in plan['cases']:
        candidates=[r for r in cells if r['case']==case['id']]
        # The family is chosen from frozen training, never from held-out minima.
        winner=min(candidates,key=lambda r:r['training_score']);row=dict(winner)
        if reference:
            old=next(r for r in reference['rows'] if r['case']==case['id'])
            if (old['bits'],old['B1'],old['B2'])!=(row['bits'],row['B1'],row['B2']):raise ValueError('Historical input mismatch')
            row.update(layout_reference=old['tuned']['median_seconds'],prime95_seconds=old['prime95_seconds'])
        best.append(row)
    totals={mode:sum(r['values'][mode]['seconds'] for r in cells) for mode in ('scalar','auto')}
    training=[r for e in plan['training'] for r in e['rows']]
    evaluation=[r for e in plan['evaluations'] for r in e['rows']]
    return dict(complete=True,version=plan['program_version'],reference_version=plan['reference_version'],exe_sha256=plan['exe_sha256'],source=str(root),calls=calls,gmp_checked=checked,selftests=selftests,
        training_calls=len(training),training_engine_seconds=sum(r['seconds'] for r in training),evaluation_calls=len(evaluation),
        curve_wall_totals={mode:sum(r['values'][mode]['curve_wall'] for r in cells) for mode in ('scalar','auto')},
        busy_clocks=dict(clocks),protocol=plan['protocol'],cells=cells,best=best,totals=totals,
        historical_sum=sum(r['historical_seconds'] for r in cells),change_percent=100*(totals['auto']/totals['scalar']-1),
        regressions=[r['cell'] for r in cells if r['values']['auto']['seconds']>r['values']['scalar']['seconds']*1.05 and
            r['values']['auto']['seconds']-r['values']['scalar']['seconds']>.02])


def figures(result,out):
    out.mkdir(parents=True,exist_ok=False);version=result['version'];stem='stage2-backend-'+version+'-low'
    def save(fig,name,top=1):
        fig.text(.5,.008,'RTX 4060 Laptop | target 1500 MHz | B1=50000 | fixed D/layout/chain | held out n=1; flagged retests n=3 | no warmup',
                 ha='center',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.045,1,top));fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)
    for logarithmic in (False,True):
        suffix='log' if logarithmic else 'linear'
        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        for ax,family in zip(axes,('ordinary','carrier')):
            rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=[r['bits'] for r in rows]
            ax.plot(x,[r['historical_seconds'] for r in rows],'.--',color='#777777',label=result['reference_version']+' scalar reference (n=1)')
            for mode,color,label in [('scalar','#c46a19',version+' scalar/scalar'),('auto','#008879',version+' frozen automatic choice')]:
                ax.plot(x,[r['values'][mode]['seconds'] for r in rows],'o-',color=color,label=label,ms=3)
                ax.fill_between(x,[r['values'][mode]['minimum'] for r in rows],[r['values'][mode]['maximum'] for r in rows],alpha=.12,color=color)
            ax.set(title=family,xlabel='Target N bits',ylabel='Stage2 engine init + main (s)');ax.xaxis.set_major_locator(MultipleLocator(1000))
            if logarithmic:ax.set_yscale('log')
            else:ax.set_ylim(bottom=0)
            ax.grid(alpha=.2);ax.legend(fontsize=8);ax.tick_params(axis='x',rotation=45)
        save(fig,stem+'-families-'+suffix)
        rows=sorted(result['best'],key=lambda r:r['bits']);x=[r['bits'] for r in rows];fig,ax=plt.subplots(figsize=(12,5.2))
        ax.plot(x,[r['values']['auto']['seconds'] for r in rows],'o-',lw=2,color='#008879',label=version+' frozen best known parameters (held out)')
        ax.plot(x,[r['values']['scalar']['seconds'] for r in rows],'.--',color='#c46a19',label=version+' scalar at the same family / geometry')
        for mode,color in [('scalar','#c46a19'),('auto','#008879')]:
            ax.fill_between(x,[r['values'][mode]['minimum'] for r in rows],[r['values'][mode]['maximum'] for r in rows],color=color,alpha=.12)
        if 'layout_reference' in rows[0]:
            ax.plot(x,[r['layout_reference'] for r in rows],'.--',color='#5073a7',label='0.54.7 tuned reference (n=3)')
            ax.plot(x,[r['prime95_seconds'] for r in rows],'s:',color='#333333',ms=3,label='Prime95 4T exact-N historical reference (n=1)')
        ax.set(xlabel='Target N bits',ylabel='Stage2 engine init + main (s)',title='Small B2: 13 exact cofactor inputs; training and held-out samples separate')
        ax.xaxis.set_major_locator(MultipleLocator(1000));ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend(fontsize=8)
        if logarithmic:ax.set_yscale('log')
        else:ax.set_ylim(bottom=0)
        save(fig,stem+'-best-times-'+suffix)
        fig,axes=plt.subplots(1,2,figsize=(14,5.2))
        for ax,family in zip(axes,('ordinary','carrier')):
            rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=[r['bits'] for r in rows]
            for mode,color in [('scalar','#c46a19'),('auto','#008879')]:
                ax.plot(x,[r['values'][mode]['curve_wall'] for r in rows],'o-',ms=3,color=color,label=version+' '+mode)
            ax.set(title=family,xlabel='Target N bits',ylabel='curve_done.wall (s; includes startup checks / planning)')
            ax.xaxis.set_major_locator(MultipleLocator(1000));ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if logarithmic:ax.set_yscale('log')
            else:ax.set_ylim(bottom=0)
        save(fig,stem+'-curve-wall-'+suffix)
    fig,axes=plt.subplots(2,2,figsize=(15,9));keys=['baby','ftree_init','inv','giant_combined','gtrees','fold','descent','accum','other'];colors=plt.get_cmap('tab10').colors
    labels=['Baby + normalization','F tree + init','Polynomial inverse','Giant + leaf preparation','G trees','Fold','Descent','Product + GCD','Other']
    if set(keys)!=set(result['cells'][0]['values']['auto']['phases']):raise ValueError('Unmapped phase')
    for column,family in enumerate(('ordinary','carrier')):
        rows=sorted((r for r in result['cells'] if r['family']==family),key=lambda r:r['bits']);x=np.arange(len(rows))
        for line in (0,1):
            ax=axes[line,column];base=np.zeros(len(rows))
            for i,key in enumerate(keys):
                y=np.array([r['values']['auto']['phases'][key] for r in rows]);total=np.array([sum(r['values']['auto']['phases'].values()) for r in rows])
                if line:y=100*y/total
                ax.bar(x,y,bottom=base,label=labels[i],color=colors[i%len(colors)],width=.8);base+=y
            ax.set_xticks(x,[str(r['bits']) for r in rows],rotation=60);ax.set(title=family,ylabel='Time share (%)' if line else 'Stage2 phase time (s)')
            if line:ax.set_ylim(0,100)
            ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=5,fontsize=8)
    fig.suptitle('Frozen automatic backend: mutually exclusive phases; equally spaced input cases',y=.995)
    save(fig,stem+'-phase-shares',.89)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','output','figures'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--publish',type=Path);a=p.parse_args();result=audit(a.study.resolve())
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'summary.json',result);shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    with (a.output/'cells.csv').open('w',newline='',encoding='utf-8') as f:
        columns=['bits','family','B2','D','P','layout','chain','baby','giant','scalar_seconds','auto_seconds','change_percent','n']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in result['cells']:
            row={k:r[k] for k in columns if k in r};row.update(scalar_seconds=r['values']['scalar']['seconds'],auto_seconds=r['values']['auto']['seconds'],n=r['values']['auto']['n']);writer.writerow(row)
    lines=['| N bits | 实际B2 | 路径 | D / P | chain | baby / giant | 标量s | auto s | n |',
        '|---:|---:|---|---|---:|---|---:|---:|---:|']
    for r in sorted(result['best'],key=lambda r:r['bits']):
        lines.append(f'| {r["bits"]} | {r["B2"]} | {r["family"]} | {r["D"]} / {r["P"]} | {r["chain"]} | {r["baby"]} / {r["giant"]} | {r["values"]["scalar"]["seconds"]:.3f} | {r["values"]["auto"]["seconds"]:.3f} | {r["values"]["auto"]["n"]} |')
    (a.output/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');figures(result,a.figures)
    if a.publish:
        a.publish.mkdir(parents=True,exist_ok=True)
        for f in a.figures.glob('*.png'):shutil.copy2(f,a.publish/f.name)
    print({k:result[k] for k in ('complete','calls','gmp_checked','busy_clocks','totals','change_percent','regressions')})


if __name__=='__main__':main()
