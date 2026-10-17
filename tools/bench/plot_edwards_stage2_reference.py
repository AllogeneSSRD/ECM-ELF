"""Summarize frozen exact-N Edwards measurements and render optimization references.

Historical Prime95 4-thread measurements remain distinct from libecm thread limit.
No B2/thread/power normalization, nested timer addition, or projected samples.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import textwrap

from edwards_stage2_reference_scope import measurement_scope


def stats(values):
    median=statistics.median(values)
    return dict(n=len(values),median=median,min=min(values),max=max(values),
                mad=statistics.median(abs(x-median) for x in values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--thread-probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--scope',choices=('all','gpu'),default='all',help='Finalize GPU paths while retaining CPU samples as a separate reference')
    a=p.parse_args();source_raw=(a.study/'plan.json').read_bytes();plan=json.loads(source_raw)
    conditions_path=a.study/'cpu_conditions.json'
    cpu_conditions=json.loads(conditions_path.read_text(encoding='utf-8')) if conditions_path.exists() else {}
    scope=measurement_scope(plan,a.scope,a.allow_partial)
    if plan['failures']:raise ValueError('Recorded measurement failures require review')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    for target in (a.output,a.figures):
        if target.exists() and any(target.iterdir()):raise ValueError('Output must be empty; preserve prior evidence: '+str(target))
        target.mkdir(parents=True,exist_ok=True)
    cases={c['id']:c for c in plan['cases']};groups={}
    for row in plan['runs']:
        prefix=Path(row['prefix'])
        if hashlib.sha256(prefix.with_suffix('.jsonl').read_bytes()).hexdigest()!=row['result_sha256']:raise ValueError('Raw result changed')
        if row['full_sample']:groups.setdefault((row['case'],row['B2'],row['engine']),[]).append(row)
    rows=[]
    for key,samples in groups.items():
        case,b2,engine=key;c=cases[case]
        reference=next(r for r in plan['prime95_rows'] if r['curve_type']=='Edwards' and r.get('N_hex')==c['N_hex'] and r['b2']==b2 and int(r['s'])==c['sigma'])
        cpu_plans=[]
        if engine.startswith('cpu'):
            for sample in samples:
                log=Path(sample['prefix']).with_suffix('.driver.log').read_text(encoding='utf-8')
                match=re.search(r'dF=(\d+), k=(\d+), d=(\d+), d2=(\d+), i0=(-?\d+)',log)
                if not match:raise ValueError('Missing libecm polynomial plan')
                entry=dict(zip(('dF','k','d','d2','i0'),map(int,match.groups())))
                if entry not in cpu_plans:cpu_plans.append(entry)
        rows.append(dict(case=case,exponent=c['exponent'],bits=c['bits'],B1=c['b1'],B2=b2,nominal_B2=reference['b2_requested'],
                         sigma=c['sigma'],engine=engine,thread_limit=plan['thread_limit'] if engine.startswith('cpu') else None,
                         seconds=stats([r['engine_seconds'] for r in samples]),process_seconds=stats([r['process_seconds'] for r in samples]),
                         cpu_wall_ratio=stats([r['parent_cpu_wall_ratio'] for r in samples]) if engine.startswith('cpu') else None,
                         prime95_seconds=reference['s2_total_time'],prime95_n=1,
                         prime95_available_mb=reference['avail_mem'],prime95_reported_used_mb=reference['using_mem'],
                         historical_prime95_ratio=reference['s2_total_time']/statistics.median(r['engine_seconds'] for r in samples),
                         init_median=statistics.median(float(r['phase_seconds']['init']) for r in samples) if engine.startswith('gpu') else None,
                         main_median=statistics.median(float(r['phase_seconds']['main']) for r in samples) if engine.startswith('gpu') else None,
                         cpu_plans=cpu_plans,
                         D=sorted(set(r['D'] for r in samples)) if engine.startswith('gpu') else [],
                         P=sorted(set(r['P'] for r in samples)) if engine.startswith('gpu') else []))
    busy=[]
    for r in plan['runs']:
        if r['engine'].startswith('gpu'):
            samples=json.loads(Path(r['prefix']).with_suffix('.telemetry.json').read_text(encoding='utf-8'))
            busy.extend(s for s in samples if s.get('gpu_percent') is not None and s['gpu_percent']>=50)
    telemetry={}
    for key in ('sm_clock_mhz','temperature_c','power_mw','used_bytes'):
        values=[s[key] for s in busy if s.get(key) is not None]
        if key=='power_mw':
            valid=[x for x in values if 0<x<=250000];telemetry['power_samples_rejected']=len(values)-len(valid);values=valid
        if values:telemetry[key]=stats(values)
    summary=dict(complete=scope['complete'],reporting_scope=scope,original_matrix_complete=plan['complete'],
                 source_sha256=hashlib.sha256(source_raw).hexdigest(),
                 source_study=str(a.study.resolve()),binaries=plan['binaries'],dll_sha256=plan['dll_sha256'],
                 prime95_exe_sha256=plan.get('prime95_exe_sha256'),protocol=plan['protocol'],rows=rows,
                 measured_runs=len(plan['runs']),full_runs=sum(r['full_sample'] for r in plan['runs']),
                 factor_runs=[r['id'] for r in plan['runs'] if not r['full_sample']],failures=plan['failures'],busy_gpu_telemetry=telemetry,
                 cpu_conditions=cpu_conditions,thread_probe_sha256=hashlib.sha256(a.thread_probe.read_bytes()).hexdigest(),
                 scope_validator_sha256=hashlib.sha256(Path(__file__).with_name('edwards_stage2_reference_scope.py').read_bytes()).hexdigest(),
                 cpu_conditions_sha256=hashlib.sha256(conditions_path.read_bytes()).hexdigest() if conditions_path.exists() else None)
    (a.output/'study.snapshot.json').write_bytes(source_raw)
    if conditions_path.exists():(a.output/'cpu_conditions.snapshot.json').write_bytes(conditions_path.read_bytes())
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    columns=['exponent','bits','B1','B2','nominal_B2','sigma','engine','thread_limit','n','median_seconds','min_seconds','max_seconds','process_median_seconds','cpu_wall_ratio','prime95_4T_seconds','reference_ratio']
    with (a.output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:
            d={k:r[k] for k in columns if k in r};s=r['seconds']
            d.update(n=s['n'],median_seconds=s['median'],min_seconds=s['min'],max_seconds=s['max'],
                     process_median_seconds=r['process_seconds']['median'],cpu_wall_ratio=r['cpu_wall_ratio']['median'] if r['cpu_wall_ratio'] else '',
                     prime95_4T_seconds=r['prime95_seconds'],reference_ratio=r['historical_prime95_ratio'])
            w.writerow(d)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors={'gpu':'#217c89','gpu-carrier':'#4765a4','cpu-ntt':'#ac692f','cpu-plain':'#7b516b'}
    labels={'gpu':'GPU1 ordinary N','gpu-carrier':'GPU1 Mersenne carrier','cpu-ntt':f'libecm NTT (limit {plan["thread_limit"]})',
            'cpu-plain':f'libecm plain (limit {plan["thread_limit"]})'}
    bounds=sorted({r['b2_requested'] for r in plan['prime95_rows'] if r['curve_type']=='Edwards'})
    clock=telemetry.get('sm_clock_mhz');clock_note=(f'busy SM clock {clock["min"]:.0f}..{clock["max"]:.0f} MHz' if clock else 'clock telemetry unavailable')
    cpu_note=(f'{cpu_conditions["declared_fixed_clock_mhz"]/1000:g} GHz user-set' if cpu_conditions.get('declared_fixed_clock_mhz') else 'frequency not verified')
    caption=('Exact recovered N, Edwards sigma, B1=50000 and adjusted B2 matched; new samples include first invocation.\n'
             'Prime95 31.4b05: historical 4-thread / 4 GHz, log budget 4096 MB, n=1 per cell. libecm: thread limit '+str(plan['thread_limit'])+
             ', '+cpu_note+', 4096 MiB planner budget. GPU1 RTX4060 Laptop, '+clock_note+'. No normalization.')
    identity=('Study 2026-10-17; EXE SHA256 prefixes: CPU '+plan['binaries']['cpu']['sha256'][:12]+
              ', GPU '+plan['binaries']['gpu']['sha256'][:12]+'. Plan '+summary['source_sha256'][:12]+'.')
    artifacts=[]
    def save(fig,name):
        if not scope['complete'] and fig._suptitle is not None and 'PARTIAL' not in fig._suptitle.get_text():
            fig._suptitle.set_text(fig._suptitle.get_text()+' [PARTIAL SCOPE]')
        for ext in ('png','svg'):
            path=a.figures/(name+'.'+ext);fig.savefig(path,dpi=175,bbox_inches='tight');artifacts.append(str(path.resolve()))
        plt.close(fig)
    def footer(fig,text):
        lines=[part for line in text.splitlines() for part in textwrap.wrap(line,width=185)]
        fig.text(.02,.015,'\n'.join(lines),fontsize=8,va='bottom')
        # Reserve physical space for evidence captions instead of overlapping
        # the x labels or expanding a negative-position footer into the plot.
        bottom=(.14*len(lines)+.28)/fig.get_figheight()
        fig.tight_layout(rect=(0,bottom,1,.94))
    def reference_rows(bound,model):
        return sorted([r for r in plan['prime95_rows'] if r['curve_type']==model and r['b2_requested']==bound and r['benchmark_eligible']],key=lambda r:r['actual_input_bits'])
    fig,axes=plt.subplots(1,len(bounds),figsize=(14,5),squeeze=False)
    for ax,bound in zip(axes[0],bounds):
        ref=reference_rows(bound,'Edwards')
        ax.plot([r['actual_input_bits'] for r in ref],[r['s2_total_time'] for r in ref],color='#333333',ls='--',marker='D',markersize=4,label='Prime95 Edwards (4 threads)')
        for engine in scope['engines']:
            part=sorted([r for r in rows if r['nominal_B2']==bound and r['engine']==engine],key=lambda r:r['bits'])
            if not part:continue
            v=[r['seconds']['median'] for r in part]
            ax.errorbar([r['bits'] for r in part],v,yerr=[[x-r['seconds']['min'] for x,r in zip(v,part)],[r['seconds']['max']-x for x,r in zip(v,part)]],
                        color=colors[engine],marker='o',markersize=3,capsize=2,label=labels[engine])
        ax.set_title(f'Requested B2={bound:.1e}; per-case adjusted B2 retained')
        ax.set_xlabel('Actual modulus N (bits)');ax.set_ylabel('Complete native Stage2 time (seconds, log scale)')
        ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Edwards Stage2: '+('GPU elapsed-time comparison' if a.scope=='gpu' else 'CPU / GPU elapsed-time reference'))
    sample_counts=sorted({r['seconds']['n'] for r in rows if r['engine'] in scope['engines']})
    footer(fig,caption+'\nNew runs: n='+','.join(map(str,sample_counts))+
             ', median with min/max whiskers; thread counts differ from historical Prime95.\n'+identity)
    save(fig,'edwards_times')
    if a.scope=='gpu' and scope['cpu_reference_invocations']:
        fig,axes=plt.subplots(1,len(bounds),figsize=(14,5),squeeze=False)
        for ax,bound in zip(axes[0],bounds):
            ref=reference_rows(bound,'Edwards')
            ax.plot([r['actual_input_bits'] for r in ref],[r['s2_total_time'] for r in ref],
                    color='#333333',ls='--',marker='D',markersize=4,label='Prime95 Edwards (4 threads, n=1)')
            for engine in ('cpu-ntt','cpu-plain'):
                part=sorted([r for r in rows if r['nominal_B2']==bound and r['engine']==engine],key=lambda r:r['bits'])
                counts=','.join(map(str,sorted({r['seconds']['n'] for r in part})))
                # Individual cells have uneven coverage: show observations without
                # drawing a line through unmeasured conditions.
                for i,r in enumerate(part):
                    s=r['seconds'];v=s['median']
                    ax.errorbar(r['bits'],v,yerr=[[v-s['min']],[s['max']-v]],fmt='o',
                                color=colors[engine],capsize=2,label=labels[engine]+f', n={counts}' if i==0 else None)
            ax.set_title(f'CPU reference coverage; nominal B2={bound:.1e}')
            ax.set_xlabel('Actual modulus N (bits)');ax.set_ylabel('Complete native Stage2 (seconds, log scale)')
            ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('CPU Stage2 reference only: partial coverage; collection stopped')
        footer(fig,caption+'\nMeasured CPU cells only; no interpolation or projected timings. CPU background load overlapped development.\n'+identity)
        save(fig,'cpu_reference')
    # Historical model context includes all four user-supplied screen logs.
    fig,axes=plt.subplots(1,len(bounds),figsize=(13,4.5),squeeze=False)
    for ax,bound in zip(axes[0],bounds):
        for model,color in [('Edwards','#217c89'),('Montgomery','#ac692f')]:
            ref=reference_rows(bound,model)
            ax.plot([r['actual_input_bits'] for r in ref],[r['s2_total_time'] for r in ref],marker='o',markersize=4,color=color,label=model)
        ax.set_title(f'Prime95 4 threads, requested B2={bound:.1e}')
        ax.set_xlabel('Recovered actual N (bits)');ax.set_ylabel('Init + main + GCD (seconds)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Historical Prime95 model context: independent curve families')
    footer(fig,'Prime95 31.4b05, 4 GHz, B1=50000; four immutable 4Core@4GHz logs plus result snapshots. Each point n=1.\n'
             'Sigma, actual B2 and polynomial plans may differ between models. Plan '+summary['source_sha256'][:12]+'.')
    save(fig,'prime95_models')
    # Init/main are disjoint. Do not stack nested preparation or operator timers.
    fig,axes=plt.subplots(len(bounds),1,figsize=(14,7),squeeze=False)
    for ax,bound in zip(axes[:,0],bounds):
        ids=sorted({r['case'] for r in rows if r['nominal_B2']==bound and r['engine'].startswith('gpu')},key=lambda c:cases[c]['bits'])
        for shift,engine in [(-.19,'gpu'),(.19,'gpu-carrier')]:
            for i,c in enumerate(ids):
                ss=[r for r in rows if r['case']==c and r['nominal_B2']==bound and r['engine']==engine]
                if not ss:continue
                r=ss[0];ax.bar(i+shift,r['init_median'],.36,color=colors[engine],alpha=.45,label=labels[engine]+' init' if i==0 else None)
                ax.bar(i+shift,r['main_median'],.36,bottom=r['init_median'],color=colors[engine],label=labels[engine]+' main' if i==0 else None)
        ax.set_xticks(range(len(ids)),[str(cases[c]['bits']) for c in ids]);ax.set_xlabel('Actual N (bits)');ax.set_ylabel('Stage2 phase time (seconds)')
        ax.set_title(f'GPU phase medians; nominal B2={bound:.1e}');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('GPU1 optimization reference: initialization versus main computation')
    footer(fig,'B1=50000, GPU1 RTX4060 Laptop, '+clock_note+'. Only disjoint native phases shown; component medians need not sum to median total.\n'
             'Nested operator counters remain in raw evidence. Ordinary/carrier D and polynomial sizes may differ.\n'+identity)
    save(fig,'gpu_phases')
    probe=json.loads(a.thread_probe.read_text(encoding='utf-8'));assert probe['complete']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,label in zip(axes,['engine_seconds','cpu_wall_ratio'],['Complete CPU Stage2 (seconds)','Aggregate CPU time / process wall']):
        for idx,backend in enumerate(('ntt','plain')):
            for off,t in [(-.17,1),(.17,4)]:
                values=[r[metric] for r in probe['runs'] if r['backend']==backend and r['thread_limit']==t];v=statistics.median(values)
                ax.bar(idx+off,v,.31,label=f'thread limit {t}' if idx==0 else None,color='#217c89' if t==1 else '#ac692f',
                       yerr=[[v-min(values)],[max(values)-v]],capsize=3)
        ax.set_xticks([0,1],['NTT','plain']);ax.set_ylabel(label);ax.set_xlabel('libecm backend');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('CPU parallelism probe: configured thread limit versus measured behavior')
    footer(fig,'Ryzen AI9 HX370, '+cpu_note+', M2203, B1=1000, B2=1e9, 4096 MiB planner budget; 3 samples per cell, median and min/max.\n'
             'CPU/wall is occupancy, not useful-work efficiency. EXE '+probe['exe_sha256'][:12]+'. Probe '+summary['thread_probe_sha256'][:12]+'.')
    save(fig,'cpu_threads')
    (a.figures/'manifest.json').write_text(json.dumps(dict(artifacts=artifacts,reporting_scope=scope,study_sha256=summary['source_sha256'],scope_validator_sha256=summary['scope_validator_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Rendered',len(artifacts),'figures; accepted full samples',summary['full_runs'])


if __name__=='__main__':main()
