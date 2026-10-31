"""Re-audit frozen chain learning and independent tail-balance matrix results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
from bench_stage2_point_matrix import records
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_point_routes import sha
from bench_stage2_version import load,write
from bench_stage2_layout_matrix import phases


def signature(row):
    return [(int(p['npts']),p['route'],int(p['block'])) for p in row['policies']]


def ntt_counts(folder):
    text=(folder/'engine.log').read_text(encoding='utf-8')
    return [dict((int(a),int(b)) for a,b in re.findall(r'(\d+):(\d+)',s)) for s in text.splitlines() if s.startswith('ntt_length_counts:')]


def partition(row):
    p=phases(row['seconds'],float(row['clock']['init']),row['stats']['real_baby'],row['stats']['real_batched_split'])
    leaves=float(row['stats']['real_batched_wall']['gleaves'])
    p['giant_combined']=p.pop('giant')+leaves;p['other']-=leaves
    if p['other']<-.012:raise ValueError('Nested phase clocks overlap')
    if p['other']<0:p['giant_combined']+=p['other'];p['other']=0
    if abs(sum(p.values())-row['seconds'])>1e-8:raise ValueError('Phase partition does not close')
    return p


def audit(root):
    plan=load(root/'plan.json')
    if not plan['complete'] or plan['failures'] or len(plan['cells'])!=26 or len(plan['evaluations'])!=52:
        raise ValueError('Need all26 cells and52 held-out calls')
    cells={c['id'] for c in plan['cells']}
    if len(cells)!=26 or len(plan['learning'])!=26 or {r['cell'] for r in plan['learning']}!=cells or len(plan['decisions'])!=26:
        raise ValueError('Incomplete chain learning/decisions')
    if len({(r['cell'],r['balance']) for r in plan['evaluations']})!=52:raise ValueError('Duplicate/missing A/B cell')
    if sha(root/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(root/'frozen_tune.toml')!=plan['frozen_sha256']:
        raise ValueError('Frozen program/profile changed')
    if sha(root/'bin/gmp-10.dll')!=plan['dll_sha256'] or sha(root/'initial_tune.toml')!=plan['seed_sha256']:
        raise ValueError('Frozen dependency/prior changed')
    checked=selftests=0;clocks=Counter()
    for section in ('learning','evaluations'):
        for entry in plan[section]:
            cell=next(c for c in plan['cells'] if c['id']==entry['cell'])
            case=next(c for c in plan['cases'] if c['id']==cell['case']);folder=Path(entry['folder'])
            if sha(case['save'])!=case['save_sha256']:raise ValueError('Frozen save changed')
            reread=records(plan,folder,case,cell);original=entry['rows'] if section=='learning' else [entry['row']]
            if reread!=original:raise ValueError('Captured records differ from raw receipts/logs')
            counts=ntt_counts(folder)
            if len(counts)!=len(reread) or any(not c for c in counts):raise ValueError('Missing NTT operation counts')
            for row in reread:
                checked+=int(row['s4']['gmp_checked']);selftests+=int(row['s4']['gmp_selftest_cases'])
            for line in (folder/'telemetry.jsonl').read_text(encoding='utf-8').splitlines():
                sample=json.loads(line)
                if sample.get('gpu_percent',0)>=50:clocks[sample['sm_clock_mhz']]+=1
    table=[]
    for cell in plan['cells']:
        case=next(c for c in plan['cases'] if c['id']==cell['case'])
        train=next(e for e in plan['learning'] if e['cell']==cell['id'])
        by_chain={c:[r['seconds'] for r in train['rows'] if r['chain']==c] for c in (32,64)}
        if not all(by_chain.values()):raise ValueError('Chain comparison incomplete')
        decision=next(e for e in plan['decisions'] if e['cell']==cell['id'])
        off=next(e for e in plan['evaluations'] if e['cell']==cell['id'] and not e['balance'])
        on=next(e for e in plan['evaluations'] if e['cell']==cell['id'] and e['balance'])
        if off['chain']!=on['chain'] or off['chain']!=decision['chain']:raise ValueError('Chain changed inside tail A/B')
        if ntt_counts(Path(off['folder']))!=ntt_counts(Path(on['folder'])):raise ValueError('Polynomial work changed')
        a,b=off['row'],on['row']
        raw_off=completed_records(Path(off['folder'])/'result.jsonl')[0]['stage2-engine']['seconds']
        raw_on=completed_records(Path(on['folder'])/'result.jsonl')[0]['stage2-engine']['seconds']
        table.append(dict(cell=cell['id'],exponent=case['exponent'],target_bits=case['bits'],carrier=cell['carrier'],
            B2=cell['B2'],D=cell['D'],P=cell['P'],chain=decision['chain'],
            training={str(c):dict(n=len(v),mean=st.mean(v)) for c,v in by_chain.items()},
            off_seconds=a['seconds'],on_seconds=b['seconds'],ratio=b['seconds']/a['seconds'],
            off_curve_seconds=raw_off,on_curve_seconds=raw_on,
            changed_schedule=signature(a)!=signature(b),off_schedule=signature(a),on_schedule=signature(b),
            off_phases=partition(a),on_phases=partition(b)))
    total_off=sum(t['off_seconds'] for t in table);total_on=sum(t['on_seconds'] for t in table)
    return plan,dict(complete=True,exe_sha256=plan['exe_sha256'],frozen_sha256=plan['frozen_sha256'],rows=table,
        sum_off_seconds=total_off,sum_on_seconds=total_on,sum_change_percent=100*(total_on/total_off-1),
        sum_off_curve_seconds=sum(r['off_curve_seconds'] for r in table),sum_on_curve_seconds=sum(r['on_curve_seconds'] for r in table),
        changed_schedules=sum(r['changed_schedule'] for r in table),
        arithmetic=dict(gmp_checked=checked,selftest_cases=selftests),busy_clocks=dict(clocks))


def figures(summary,out):
    out.mkdir(parents=True,exist_ok=True)
    for scale in ('linear','log'):
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        for ax,carrier in zip(axes,(False,True)):
            rows=[r for r in summary['rows'] if bool(r['carrier'])==carrier];x=[r['target_bits'] for r in rows]
            ax.plot(x,[r['training']['64']['mean'] for r in rows],'--',color='#a0a6ad',label='Chain64 training')
            ax.plot(x,[r['training']['32']['mean'] for r in rows],':',color='#b386c5',label='Chain32 training')
            ax.plot(x,[r['off_seconds'] for r in rows],'-o',color='#2a75ab',label='Frozen chain / balance off',ms=4)
            ax.plot(x,[r['on_seconds'] for r in rows],'-s',color='#d0873a',label='Frozen chain / balance on',ms=3)
            ax.set(yscale=scale,xlabel='Actual target N bits',ylabel='Complete engine seconds',
                title='Mersenne carrier' if carrier else 'Ordinary target modulus')
            ax.set_xticks(range(0,17000,1000));ax.tick_params(axis='x',rotation=45,labelsize=8);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('RTX4060 Laptop | fixed prior D/layout, minimum repeats=1 | small B2 full-width matrix')
        fig.tight_layout();fig.savefig(out/f'stage2-point-matrix-0.56.0-times-{scale}.png',dpi=160);plt.close(fig)
    keys=['baby','ftree_init','inv','giant_combined','gtrees','fold','descent','accum','other']
    colors=['#67b8a4','#a8d6c7','#d8ba6c','#7a5a9c','#3887ba','#e18b52','#abcf6b','#d786a8','#bfc4ca']
    labels=['Baby','F tree / init other','Inverse','Giant + leaf prep / waits','G trees','Fold','Descent','Accum / GCD','Other']
    fig,axes=plt.subplots(1,2,figsize=(17,6))
    for ax,carrier in zip(axes,(False,True)):
        rows=[r for r in summary['rows'] if bool(r['carrier'])==carrier];x=np.arange(len(rows))
        for side,offset in (('off',-.2),('on',.2)):
            bottom=np.zeros(len(rows))
            for key,color,label in zip(keys,colors,labels):
                y=np.array([100*r[side+'_phases'][key]/r[side+'_seconds'] for r in rows])
                ax.bar(x+offset,y,bottom=bottom,width=.38,color=color,label=label if side=='off' else None);bottom+=y
        ax.set_xticks(x,[str(r['target_bits']) for r in rows],rotation=45,fontsize=8)
        ax.set(ylim=(0,100),ylabel='Complete-engine time share (%)',xlabel='Actual target N bits (left off / right on)',
            title='Mersenne carrier' if carrier else 'Ordinary target modulus')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,fontsize=8)
    fig.subplots_adjust(top=.8,bottom=.18,wspace=.2)
    fig.savefig(out/'stage2-point-matrix-0.56.0-phases.png',dpi=160);plt.close(fig)


def extra_evidence(root,plan,result,path,control=False):
    report=load(path/'summary.json')
    if not report['complete']:raise ValueError('Incomplete extra evidence')
    if control and report['exe_sha256']!=plan['exe_sha256']:raise ValueError('Control binary differs')
    if not control and report['source_plan_sha256']!=sha(root/'plan.json'):raise ValueError('Recheck source differs')
    table=[]
    for name in ([report['cell']['id']] if control else report['cells']):
        cell=report['cell'] if control else next(c for c in plan['cells'] if c['id']==name)
        case=next(c for c in plan['cases'] if c['id']==cell['case'])
        entries=report['runs'] if control else [r for r in report['runs'] if r['cell']==name]
        for entry in entries:
            if records(plan,Path(entry['folder']),case,cell)!=[entry['row']]:raise ValueError('Extra raw evidence differs')
        for chain in ((64,32) if control else (entries[0]['row']['chain'],)):
            off=[e['row']['seconds'] for e in entries if e['row']['chain']==chain and not e['balance']]
            on=[e['row']['seconds'] for e in entries if e['row']['chain']==chain and e['balance']]
            if not control:
                original=next(r for r in result['rows'] if r['cell']==name)
                off=[original['off_seconds']]+off;on=[original['on_seconds']]+on
            table.append(dict(cell=name,chain=chain,n_off=len(off),n_on=len(on),off_samples=off,on_samples=on,
                off_mean=st.mean(off),on_mean=st.mean(on),change_percent=100*(st.mean(on)/st.mean(off)-1)))
    output=dict(source=str(path),rows=table)
    if not control:
        off=result['sum_off_seconds'];on=result['sum_on_seconds']
        for row in table:
            original=next(r for r in result['rows'] if r['cell']==row['cell'])
            off+=row['off_mean']-original['off_seconds'];on+=row['on_mean']-original['on_seconds']
        output.update(matrix_mean_sum_off=off,matrix_mean_sum_on=on,matrix_change_percent=100*(on/off-1))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--publish',type=Path)
    p.add_argument('--rechecks',type=Path);p.add_argument('--control',type=Path)
    a=p.parse_args();plan,result=audit(a.root);a.output.mkdir(parents=True,exist_ok=False)
    if a.rechecks:result['rechecks']=extra_evidence(a.root,plan,result,a.rechecks)
    if a.control:result['control']=extra_evidence(a.root,plan,result,a.control,True)
    write(a.output/'summary.json',result);shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    figures(result,a.figures)
    lines=['| N bits | carrier p | D / P | chain | chain64 s | chain32 s | off s | on s | on/off | schedule changed |',
           '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in result['rows']:
        lines.append(f'| {r["target_bits"]} | {r["carrier"]} | {r["D"]} / {r["P"]} | {r["chain"]} | {r["training"]["64"]["mean"]:.3f} | {r["training"]["32"]["mean"]:.3f} | {r["off_seconds"]:.3f} | {r["on_seconds"]:.3f} | {r["ratio"]:.4f} | {r["changed_schedule"]} |')
    (a.output/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if a.publish:
        a.publish.mkdir(parents=True,exist_ok=True)
        for f in a.figures.glob('stage2-point-matrix-0.56.0-*.png'):shutil.copy2(f,a.publish/f.name)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
