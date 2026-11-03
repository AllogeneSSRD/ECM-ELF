"""Audit large-B2 frozen calls and plot complete-engine and disjoint phase costs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import statistics as st

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from bench_stage2_large_matrix import sha,workload
from analyze_stage2_point_matrix import partition
from bench_stage2_point_matrix import records
from bench_stage2_version import load,write


def audit(root):
    plan=load(root/'plan.json')
    assert plan['complete'] and not plan['failures'] and len(plan['cells'])==26
    assert sha(root/'bin/ecm_cuda_stage2.exe')==plan['exe_sha256']
    identity=plan['identity']
    assert sha(root/'bin/gmp-10.dll')==identity['dll_sha256']
    assert sha(root/'bin/sources.zip')==identity['sources_sha256']
    clocks=Counter();powers=[];checked=selftests=0;uuids=set();samples=0
    for row in plan['runs']:
        cell=next(c for c in plan['cells'] if c['id']==row['cell'])
        case=next(c for c in plan['cases'] if c['id']==cell['case']);folder=Path(row['folder'])
        assert sha(case['save'])==case['save_sha256']
        for name,digest in row['file_sha256'].items():assert sha(folder/name)==digest,(folder,name)
        before,after=(load(folder/name) for name in ('before.json','after.json'))
        assert before['uuid']==after['uuid'];uuids.add(before['uuid'])
        raw=records(plan,folder,case,cell)[0]
        for k in ('seconds','clock','stats','s4','policies'):assert raw[k]==row[k],(folder,k)
        assert abs(sum(row['phases'].values())-row['seconds'])<1e-8
        checked+=int(row['s4']['gmp_checked']);selftests+=int(row['s4']['gmp_selftest_cases'])
        for line in (folder/'telemetry.jsonl').read_text().splitlines():
            sample=json.loads(line)
            samples+=1
            if sample.get('gpu_percent',0)>=50:
                clocks[sample['sm_clock_mhz']]+=1
                if sample.get('power_mw') is not None:powers.append(sample['power_mw']/1000)
    assert len(uuids)==1,'Device changed during study'
    return plan,dict(calls=len(plan['runs']),gmp_checked=checked,selftests=selftests,busy_clocks=dict(clocks),
                     gpu_uuid=next(iter(uuids)),telemetry_samples=samples,
                     busy_power_w=[min(powers),max(powers)] if powers else None)


def selected(plan,cell,kernel):
    rows=[r for r in plan['runs'] if r['cell']==cell['id'] and r['kernel']==kernel]
    assert len(rows) in (1,3),(cell['id'],kernel,len(rows))
    assert all(workload(r)==workload(rows[0]) for r in rows)
    # Odd-sized median is one real call, so its disjoint phases still close.
    row=sorted(rows,key=lambda r:r['seconds'])[len(rows)//2]
    return dict(seconds=row['seconds'],curve_wall=row['curve_wall'],n=len(rows),
                min_seconds=min(r['seconds'] for r in rows),max_seconds=max(r['seconds'] for r in rows),
                phases=row['phases'],s4_seconds=float(row['s4']['t_reduce']),folder=row['folder'],
                workspace=row['diagnostics']['ntt_workspace_stats'],
                fold=row['diagnostics']['real_batched_folddevice'],ntt=row['ntt'])


def summarize(baseline,paired,reference):
    base,base_audit=audit(baseline);pair,pair_audit=audit(paired) if paired else (None,None)
    cpu=load(reference)['prime95_rows']
    rows=[]
    for cell in base['cells']:
        case=next(c for c in base['cases'] if c['id']==cell['case'])
        row=dict(cell=cell['id'],target_bits=case['bits'],exponent=case['exponent'],carrier=cell['carrier'],
                 B1=case['b1'],B2=cell['B2'],D=cell['D'],P=cell['P'],layout=cell['layout'],chain=cell['chain'],
                 baseline=selected(base,cell,2))
        matches=[r for r in cpu if r['N_hex'].lower()==case['N_hex'].lower() and r['b1']==case['b1'] and
                 r['b2']==cell['B2'] and r['b2_requested']==260000000000]
        assert len(matches)==1 and matches[0]['benchmark_eligible'] and matches[0]['status']=='complete'
        assert int(matches[0]['s'])==case['sigma'] and matches[0]['curve_type']=='Edwards'
        row['prime95_seconds']=matches[0]['s2_total_time']
        if pair:
            c=next(c for c in pair['cells'] if c['id']==cell['id']);assert c==cell
            previous_row=next(r for r in base['runs'] if r['cell']==cell['id'])
            assert all(workload(r)==workload(previous_row) for r in pair['runs'] if r['cell']==cell['id'])
            row['control']=selected(pair,c,2);row['cached']=selected(pair,c,3)
            row['change_percent']=100*(row['cached']['seconds']/row['control']['seconds']-1)
        rows.append(row)
    sums={}
    for family in ('ordinary','carrier','all','wide_ordinary'):
        subset=[r for r in rows if family=='all' or (family=='wide_ordinary' and not r['carrier'] and r['target_bits']>8192)
                or (family=='ordinary' and not r['carrier']) or (family=='carrier' and r['carrier'])]
        sums[family]={name:sum(r[name]['seconds'] for r in subset) for name in (('baseline','control','cached') if pair else ('baseline',))}
    return dict(complete=True,rows=rows,sums=sums,baseline_sha256=base['exe_sha256'],paired_sha256=pair['exe_sha256'] if pair else None,
                prime95_reference_sha256=sha(reference),audit=dict(baseline=base_audit,paired=pair_audit))


def geometry_study(baseline,geometry,tail):
    """Keep targeted D/tail evidence separate from the frozen 26-cell matrix."""
    base=load(baseline/'plan.json');g=load(geometry/'summary.json');t=load(tail/'summary.json')
    assert g['complete'] and t['complete'] and not g['failures'] and not t['failures']
    assert g['exe_sha256']==t['exe_sha256']
    groups={}
    for plan in (g,t):
        for row in plan['runs']:
            if row.get('diagnostic'):continue
            bits=row.get('target_bits',16272);d=row.get('D',390390);p=row.get('P',37440)
            case=next(c for c in base['cases'] if c['bits']==bits)
            cell=dict(next(c for c in base['cells'] if c['case']==case['id'] and c['carrier']),D=d,P=p)
            reread=records(plan,Path(row['folder']),case,cell)[0]
            assert reread['seconds']==row['seconds'] and reread['s4']==row['stats']['s4_multiply_stats']
            balanced=plan is t;assert reread['balance']==balanced
            groups.setdefault((bits,d,p,balanced),[]).append(reread)
    result=[]
    for (bits,d,p,balanced),rows in sorted(groups.items()):
        assert len(rows)==3
        median=sorted(rows,key=lambda r:r['seconds'])[1]
        case=next(c for c in base['cases'] if c['bits']==bits)
        original=next(c for c in base['cells'] if c['case']==case['id'] and c['carrier'])
        result.append(dict(target_bits=bits,carrier=original['carrier'],B1=case['b1'],B2=original['B2'],
                           D=d,P=p,layout=original['layout'],chain=original['chain'],tail_balance=balanced,
                           effective_balance=median['effective_balance'],n=3,seconds=median['seconds'],
                           min_seconds=min(r['seconds'] for r in rows),max_seconds=max(r['seconds'] for r in rows),
                           phases=partition(median),ladder_points=sum(int(x['npts']) for x in median['policies'] if x['route']=='ladder')))
    diag=[r for r in t['runs'] if r.get('diagnostic')]
    assert len(diag)==2 and diag[0]['leaf']==diag[1]['leaf'] and int(diag[0]['leaf']['words'])>0
    for r in diag:
        s=r['stats'];leaf=s['device_gleaf']
        assert int(leaf['checked_groups'])==int(leaf['groups']) and int(leaf['checked_leaf_words'])==int(leaf['device_leaf_words'])
        assert int(s['real_giant_seed']['checked_words'])>0
    return dict(rows=result,geometry_sha256=sha(geometry/'summary.json'),tail_sha256=sha(tail/'summary.json'),
                normalized_leaf=diag[0]['leaf'],diagnostic_B2=diag[0]['B2'])


PHASES={'baby':'Baby points + affine','ftree_init':'F-tree + mandatory selftests','inv':'Polynomial inverse',
        'giant_combined':'Giant points + leaves','gtrees':'G trees','fold':'Fold','descent':'Descent',
        'accum':'Products + GCD','other':'Other engine work'}


def figures(report,out):
    out.mkdir(parents=True,exist_ok=True);paired=report['paired_sha256'] is not None
    for scale in ('linear','log'):
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,carrier in zip(axes,(False,True)):
            rows=[r for r in report['rows'] if bool(r['carrier'])==carrier];x=[r['target_bits'] for r in rows]
            for name,label,style in [('baseline','Frozen baseline (kernel 2 request)','--o'),('control','New binary / kernel 2 request','-s'),('cached','New binary / kernel 3 request','-o')]:
                if name!='baseline' and not paired:continue
                line,=ax.plot(x,[r[name]['seconds'] for r in rows],style,label=label,ms=3,lw=1.5)
                ax.fill_between(x,[r[name]['min_seconds'] for r in rows],[r[name]['max_seconds'] for r in rows],
                                color=line.get_color(),alpha=.12)
            ax.plot(x,[r['prime95_seconds'] for r in rows],':^',color='#555555',lw=1.3,ms=3,
                    label='Prime95 historical n=1 / HX370 4 threads')
            ax.set(xlabel='Actual target N bits',ylabel='Complete engine seconds',yscale=scale,
                   title='Mersenne carrier (native reducer)' if carrier else 'Ordinary modulus')
            if scale=='linear':ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MultipleLocator(1000));ax.set_xlim(left=0);ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle('Large B2 ≈ 2.6e11 • RTX 4060 Laptop • 1500 MHz target • no warmup')
        fig.savefig(out/f'stage2_large_b2_{scale}.png',dpi=170);plt.close(fig)
    names=('control','cached') if paired else ('baseline',)
    fig,axes=plt.subplots(len(names),2,figsize=(15,4*len(names)),squeeze=False,layout='constrained')
    colors=plt.get_cmap('tab20').colors
    for ri,name in enumerate(names):
        for ax,carrier in zip(axes[ri],(False,True)):
            rows=[r for r in report['rows'] if bool(r['carrier'])==carrier];x=np.arange(len(rows));bottom=np.zeros(len(rows))
            for i,(key,label) in enumerate(PHASES.items()):
                values=np.array([100*r[name]['phases'][key]/r[name]['seconds'] for r in rows])
                ax.bar(x,values,bottom=bottom,label=label,color=colors[2*i],width=.8);bottom+=values
            ax.set(ylim=(0,100),ylabel='Engine time (%)',title=f'{name} / '+('Mersenne carrier' if carrier else 'Ordinary modulus'))
            ax.set_xticks(x,[str(r['target_bits']) for r in rows],rotation=45,ha='right');ax.set_xlabel('Actual target N bits')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
    fig.suptitle('Mutually exclusive engine phases • S4 event time is nested and excluded from stacking')
    fig.savefig(out/'stage2_large_b2_phase_percent.png',dpi=170);plt.close(fig)
    if paired:
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        rows=[r for r in report['rows'] if not r['carrier'] and r['target_bits']>8192];x=np.arange(len(rows))
        ax.bar(x-.18,[r['control']['seconds'] for r in rows],.36,label='Kernel 2')
        ax.bar(x+.18,[r['cached']['seconds'] for r in rows],.36,label='Kernel 3 / cached divisor')
        for i,r in enumerate(rows):ax.text(i,max(r[n]['seconds'] for n in ('control','cached'))+1,f"{r['change_percent']:+.2f}%",ha='center')
        ax.set_xticks(x,[str(r['target_bits']) for r in rows]);ax.set(xlabel='Actual target N bits',ylabel='Complete engine seconds',
            title='Fixed D/P/layout/point arithmetic • all original samples retained')
        ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(out/'stage2_large_b2_s4_ab.png',dpi=170);plt.close(fig)
    if report.get('geometry'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        for ax,bits in zip(axes,(12285,16272)):
            rows=[r for r in report['geometry']['rows'] if r['target_bits']==bits];ref=rows[0]['seconds']
            x=np.arange(len(rows));y=np.array([r['seconds'] for r in rows])
            errors=np.array([[r['seconds']-r['min_seconds'] for r in rows],[r['max_seconds']-r['seconds'] for r in rows]])
            ax.bar(x,y,yerr=errors,capsize=4,color=['#417ca4','#d9893c','#4b9a71'][:len(rows)],width=.6)
            for i,r in enumerate(rows):
                if i:ax.text(i,r['max_seconds']+1,f"{100*(r['seconds']/ref-1):+.2f}%",ha='center')
            ax.set_xticks(x,[f"D={r['D']}\nP={r['P']}\ntail={'on' if r['tail_balance'] else 'off'}" for r in rows])
            ax.set(ylabel='Complete engine seconds',title=f'{bits}-bit target / Mersenne carrier',ylim=(0,max(r['max_seconds'] for r in rows)*1.14))
            ax.grid(axis='y',alpha=.2)
        fig.suptitle('Targeted D/P and short-tail study • n=3 medians, min/max • same B2, chain32, layout1')
        fig.savefig(out/'stage2_large_b2_geometry.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--paired',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--prime95-reference',type=Path,default=Path('data/experiments/edwards_stage2_reference_20261017/matrix_01/plan.json'))
    p.add_argument('--geometry',type=Path);p.add_argument('--tail-study',type=Path)
    a=p.parse_args();report=summarize(a.baseline,a.paired,a.prime95_reference);a.output.mkdir(parents=True,exist_ok=True)
    if bool(a.geometry)!=bool(a.tail_study):p.error('--geometry and --tail-study are required together')
    if a.geometry:report['geometry']=geometry_study(a.baseline,a.geometry,a.tail_study)
    report['analyzer_sha256']=sha(__file__)
    report['baseline_plan_sha256']=sha(a.baseline/'plan.json')
    report['paired_plan_sha256']=sha(a.paired/'plan.json') if a.paired else None
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    write(a.output/'summary.json',report)
    with (a.output/'matrix.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['target_bits','exponent','carrier','B1','B2','D','P','layout','chain','baseline_seconds','control_seconds','cached_seconds','control_n','cached_n','change_percent','prime95_seconds'])
        writer.writeheader()
        for r in report['rows']:
            row={k:r[k] for k in writer.fieldnames if k in r}
            for name in ('baseline','control','cached'):
                if name in r:row[name+'_seconds']=r[name]['seconds']
            for name in ('control','cached'):
                if name in r:row[name+'_n']=r[name]['n']
            writer.writerow(row)
    if report.get('geometry'):
        keys=('target_bits','carrier','B1','B2','D','P','layout','chain','tail_balance','effective_balance','n','seconds','min_seconds','max_seconds','ladder_points')
        with (a.output/'geometry.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            writer.writerows({k:r[k] for k in keys} for r in report['geometry']['rows'])
    figures(report,a.figures);print(json.dumps(dict(sums=report['sums'],audit=report['audit']),indent=2))


if __name__=='__main__':main()
