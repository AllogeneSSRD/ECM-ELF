"""Re-read fixed-D route evidence and separate kernel time from host clocks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import sqlite3
import statistics

import matplotlib.pyplot as plt
import numpy as np
from bench_stage2_version import load, write
from bench_stage2_n_scaling import fields
from bench_stage2_layout_matrix import phases
from bench_stage2_point_routes import sha,validate_clock
from result_format import normalize_result


def audited(path):
    report=load(path/'summary.json')
    if not report['complete'] or len(report['runs'])!=len(report['sequence']):
        raise ValueError('Need all planned calls: '+str(path))
    if sha(path/'bin/ecm_cuda_stage2.exe')!=report['exe_sha256'] or sha(path/'point.save')!=report['save_sha256']:
        raise ValueError('Frozen input/build changed')
    busy=[];checked=selftests=0
    for r in report['runs']:
        p=Path(r['prefix']);raw=load(p.with_suffix('.jsonl'))
        if raw['status']!='NF' or raw['source-build']!=report['exe_sha256']:
            raise ValueError('NF/build receipt differs')
        receipt=normalize_result(raw);case=report['case']
        actual_b2=r.get('actual_B2',report['B2'])
        for key,value in dict(N_hex=case['N_hex'],B1=case['b1'],B2=actual_b2,sigma=case['sigma'],
                              carrier_exponent=case['exponent'],curve='ed').items():
            if receipt[key]!=value:raise ValueError('Saved-point/target identity differs')
        if raw['d']!=report['D'] or raw['poly-size']!=report['P']:
            raise ValueError('Fixed D/P differs')
        text=p.with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+p.with_suffix('.debug.log').read_text(encoding='utf-8')
        clock=fields(text,'stage2_full_wall');s4=fields(text,'s4_multiply_stats')
        if clock!=r['clock'] or s4!=r['s4'] or float(clock['total'])!=r['engine_seconds']:
            raise ValueError('Captured metadata differs from raw log')
        validate_clock(clock,report.get('leaf_checks',False))
        if min(int(s4[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0 or any(int(s4[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
            raise ValueError('Arithmetic checks failed/missing')
        for name,value in r['stats'].items():
            if fields(text,name,False)!=value:raise ValueError('Phase/geometry metadata differs')
        routes=[dict(re.findall(r'(\w+)=(\S+)',line)) for line in re.findall(r'(?m)^giant_chain_policy: (.*)$',text)]
        if sum(int(x['npts']) for x in routes)!=actual_b2//report['D']+2:
            raise ValueError('Point routing does not cover the declared grid')
        if r['mode']=='split6' and (int(r['stats']['giant_chunk_plan']['points'])!=6*report['P'] or any(x['route']!='chain' for x in routes)):
            raise ValueError('Six-P hypothesis was not exercised')
        if r['mode'].startswith('chain') and any(x['route']!='chain' or x['block']!=r['mode'][5:] for x in routes):
            raise ValueError('Requested chain length was not exercised')
        checked+=int(s4['gmp_checked']);selftests+=int(s4['gmp_selftest_cases'])
        busy += [x for x in (json.loads(s) for s in (p.parent/'telemetry.jsonl').read_text().splitlines())
                 if x.get('gpu_percent',0)>=50]
    report['audit']=dict(full_NF=len(report['runs']),gmp_checked=checked,selftests=selftests,
        busy_clocks=dict(Counter(x['sm_clock_mhz'] for x in busy)))
    return report


def profile(path):
    db=sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.text_factory=lambda b:b.decode('utf-8','replace')
    kernels=[dict(r) for r in db.execute('''select s.value name,k.start,k.end,
        (k.end-k.start)/1e9 seconds,k.registersPerThread regs,k.localMemoryPerThread local_bytes,
        k.gridX grid,k.blockX threads,k.deviceId device
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start''')]
    if {r['device'] for r in kernels}!={1}:raise ValueError('Profiler captured another GPU')
    chain=[r for r in kernels if 's2g_chain_kernel<' in r['name']]
    # Actual giant ladders occur after the first chain; startup/baby ladders do not.
    tail=[r for r in kernels if 's2g_ladder_kernel<' in r['name'] and chain and r['start']>chain[0]['end']]
    intervals=[]
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        intervals += list(db.execute('select start,end from '+table+' where deviceId=1'))
    intervals=sorted((r[0],r[1]) for r in intervals)
    left,right=intervals[0];first=left;union=0
    for a,b in intervals[1:]:
        if a>right:union+=right-left;left,right=a,b
        else:right=max(right,b)
    union+=right-left;span=right-first
    grouped={}
    for k in kernels:
        key=k['name'];g=grouped.setdefault(key,dict(name=key,calls=0,seconds=0,regs=k['regs'],local_bytes=k['local_bytes'],threads=k['threads']))
        g['calls']+=1;g['seconds']+=k['seconds']
    result=dict(kernels=sorted(grouped.values(),key=lambda r:r['seconds'],reverse=True),
        chain_launches=chain,tail_ladder_launches=tail,
        tail_ladder_seconds=sum(r['seconds'] for r in tail),
        window=dict(span_seconds=span/1e9,event_union_seconds=union/1e9,no_own_event_seconds=(span-union)/1e9),
        scope='Own first-to-last GPU event window; includes startup selftests. Not full-wall, whole-card idle, or a CPU activity measurement.')
    db.close();return result


def ncu_metrics(root):
    report=load(root/'wide_ncu/summary.json')
    if not report['complete'] or report['clock_control']!='none':
        raise ValueError('Need a completed no-clock-control hardware diagnostic')
    text=(root/'wide_ncu/imported_metrics.csv').read_text(encoding='utf-8-sig').splitlines()
    start=next(i for i,line in enumerate(text) if line.startswith('"ID"'))
    rows=list(csv.DictReader(text[start:]))
    if len(rows)!=2 or rows[1]['Device']!='1' or rows[1]['Grid Size']!='(34, 1, 1)':
        raise ValueError('Unexpected NCU kernel population')
    data=rows[1];units=rows[0]
    names=('launch__block_size','launch__grid_size','launch__registers_per_thread',
        'sm__warps_active.avg.pct_of_peak_sustained_active','sm__warps_active.avg.per_cycle_active',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
        'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum')
    return {k:dict(value=float(data[k].replace(',','')),unit=units[k]) for k in names}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True);p.add_argument('--publish',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.figures.mkdir(parents=True,exist_ok=False)
    names=('tail_abba','wide_chain_probe','tail_nsys','wide_nsys','tail_leafcheck_verified','wide_leafcheck_verified')
    studies={name:audited(a.root/name) for name in names}
    hardware=ncu_metrics(a.root)
    if any(studies[k]['diagnostic'] for k in names[:2]) or any(not studies[k]['diagnostic'] for k in names[2:]):
        raise ValueError('Timing and diagnostic roles mixed')
    for name in names[-2:]:
        hashes=[r['stats']['target_descent_values'] for r in studies[name]['runs']]
        if not hashes or not all(h==hashes[0] and int(h['words'])>0 for h in hashes):
            raise ValueError('Target-ring leaf diagnostics differ')
    profiles={}
    for name in ('tail_nsys','wide_nsys'):
        profiles[name]={r['mode']:profile(Path(r['prefix']).parent/'capture.sqlite') for r in studies[name]['runs']}
    colors=['#67b8a4','#a8d6c7','#d8ba6c','#7a5a9c','#3887ba','#e18b52','#abcf6b','#d786a8','#bfc4ca']
    keys=['baby','ftree_init','inv','giant_combined','gtrees','fold','descent','accum','other']
    labels=['Baby','F tree / init other','Inverse','Giant + leaf prep / waits','G trees','Fold','Descent','Accum / GCD','Other']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    tables={}
    for ax,name,title in zip(axes,names[:2],('7995-bit target / 8011-bit carrier','16272-bit target / 16319-bit carrier')):
        study=studies[name];modes=list(dict.fromkeys(study['sequence']));table=[]
        for mode in modes:
            rows=[r for r in study['runs'] if r['mode']==mode];timing=[]
            for r in rows:
                d=phases(r['engine_seconds'],float(r['clock']['init']),r['stats']['real_baby'],r['stats']['real_batched_split'])
                gleaves=float(r['stats']['real_batched_wall']['gleaves'])
                d['giant_combined']=d.pop('giant')+gleaves;d['other']-=gleaves
                if d['other']<-.012:raise ValueError('Point preparation overlaps another phase')
                timing.append(d)
            table.append(dict(mode=mode,n=len(rows),mean=statistics.mean(r['engine_seconds'] for r in rows),
                min=min(r['engine_seconds'] for r in rows),max=max(r['engine_seconds'] for r in rows),
                phases={k:statistics.mean(x[k] for x in timing) for k in keys}))
        base=table[0]['mean']
        for t in table:t['change_percent']=100*(t['mean']/base-1)
        tables[name]=table;bottom=np.zeros(len(modes));x=np.arange(len(modes))
        for key,label,color in zip(keys,labels,colors):
            y=np.array([t['phases'][key] for t in table]);ax.bar(x,y,bottom=bottom,label=label,color=color);bottom+=y
        for i,t in enumerate(table):
            ax.vlines(i,t['min'],t['max'],color='black',lw=1.5)
            ax.text(i,t['max']+.6,f'{t["mean"]:.3f}s\n{t["change_percent"]:+.2f}%',ha='center',fontsize=10)
        ax.set_xticks(x,[m.replace('baseline','chain64\n(default chunks)').replace('split6','chain64\n(6P chunks)') for m in modes])
        ax.set(title=title,ylabel='Full engine seconds (init + main)',ylim=(0,max(t['max'] for t in table)*1.2))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=5,fontsize=8)
    fig.subplots_adjust(top=.80,bottom=.2,wspace=.18)
    fig.text(.5,.04,'Fixed D180180 / P17280 | B1=50,000 | B2=27,015,978,990 | GPU1 RTX4060 Laptop | n=2 each; all first calls retained\nSame device objects; right panel uses an isolated host adapter. Nsight and extra leaf-check runs are excluded.',ha='center',fontsize=9)
    stem='stage2-point-routes-0.54.7'
    for ext in ('png','svg'):fig.savefig(a.figures/(stem+'.'+ext),dpi=160)
    plt.close(fig)
    if a.publish:
        a.publish.mkdir(parents=True,exist_ok=True);shutil.copy2(a.figures/(stem+'.png'),a.publish/(stem+'.png'))
    result=dict(complete=True,studies=studies,profiles=profiles,tables=tables,ncu=hardware)
    write(a.output/'summary.json',result);shutil.copy2(__file__,a.output/'analysis_snapshot.py')
    print(json.dumps(dict(complete=True,tables=tables)))


if __name__=='__main__':main()
