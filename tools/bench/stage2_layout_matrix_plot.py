"""Audit and plot the bounded layout/tune matrix via plot_stage2_version.py."""
import csv
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics as st
import matplotlib.pyplot as plt
import numpy as np
from bench_stage2_version import load, check_gpu_processes, foreign_gpu_processes
from bench_stage2_runtime_matrix import queue_rows,log_rows
from bench_stage2_layout_matrix import enrich,phases,route,sha
from stage2_d_catalog import phi
from check_stage2_version import audit as audit_defaults
from result_format import normalize_result
from bench_stage2_n_scaling import fields,number
from bench_stage2_version import shape_fields

NAMES=['baby','ftree_init','inv','giant','gtrees','fold','descent','accum','other']
LABELS=['Baby points','F tree / init other','Polynomial inverse','Giant points','G trees',
        'Fold','Descent','Accumulation / GCD','Main other']
COLORS=['#67b8a4','#a8d6c7','#d8ba6c','#7a5a9c','#3887ba','#e18b52','#abcf6b','#d786a8','#bfc4ca']
DETAIL_NAMES=NAMES[:4]+['gleaves']+NAMES[4:]
DETAIL_LABELS=LABELS[:4]+['Giant leaf prep / sync']+LABELS[4:]
DETAIL_COLORS=COLORS[:4]+['#b59bcc']+COLORS[4:]


def current_detail(row):
    wall=row['stats'].get('real_batched_wall')
    if not wall:
        wall=log_rows((Path(row['folder'])/'debug.log').read_text(encoding='utf-8'),'real_batched_wall')[row['index']-1]
    detail=dict(row['phases']);value=float(wall['gleaves'])
    if value>detail['other']+.012:raise ValueError('Giant leaf preparation overlaps an accounted component')
    detail['gleaves']=value;detail['other']-=value
    if detail['other']<0:
        detail['ftree_init']+=detail['other'];detail['other']=0
    return detail


def describe(rows,phases_key='phases'):
    times=[r['engine_seconds'] for r in rows]
    result=dict(seconds=st.mean(times),median_seconds=st.median(times),n=len(times),
        min_seconds=min(times),max_seconds=max(times),stddev_seconds=st.stdev(times) if len(times)>1 else 0,
        phases={k:st.mean(r[phases_key][k] for r in rows) for k in NAMES},
        fractions={k:st.mean(r[phases_key][k]/r['engine_seconds'] for r in rows) for k in NAMES})
    if all('parent_seconds' in r for r in rows):
        outer=[]
        for r in rows:
            parent=r['parent_seconds'];core=r['selection'].get('runtime_prepare',{}).get('total_seconds',0)
            residual=parent-r['engine_seconds']-core
            if residual<-.005:raise ValueError('Outer timing overlaps engine')
            outer.append({k:v/parent for k,v in r[phases_key].items()}|
                         dict(selection_core=core/parent,outer_other=residual/parent))
        result['parent_seconds']=st.mean(r['parent_seconds'] for r in rows)
        result['selection_core_seconds']=st.mean(r['selection'].get('runtime_prepare',{}).get('total_seconds',0) for r in rows)
        result['outer_other_seconds']=result['parent_seconds']-result['seconds']-result['selection_core_seconds']
        result['outer_fractions']={k:st.mean(r[k] for r in outer) for k in outer[0]}
    if all('detailed' in r for r in rows):
        result['detail_phases']={k:st.mean(r['detailed'][k] for r in rows) for k in DETAIL_NAMES}
        result['detail_fractions']={k:st.mean(r['detailed'][k]/r['engine_seconds'] for r in rows) for k in DETAIL_NAMES}
    return result


def defaults(plan):
    result={}
    for r in plan['runs']:
        value=dict(r)
        value['phases']=phases(r['engine_seconds'],float(r['phase_seconds']['init']),
                              r['stats']['real_baby'],r['stats']['real_batched_split'])
        value['detailed']=current_detail(value)
        result.setdefault((r['case'],r['engine']),[]).append(value)
    return result


def old_runtime(study,bound):
    plan=load(study/'plan.json');runs={r['id']:r for r in plan['runs']}
    cells={r['id']:r for r in plan['cells']};out={}
    for decision in plan['decisions']:
        cell=cells[decision['cell']]
        if cell['nominal_B2']!=bound:continue
        selected=[]
        for key in decision['run_ids']:
            row=dict(runs[key]);baby=log_rows((Path(row['folder'])/'engine.log').read_text(encoding='utf-8'),'real_baby')[row['index']-1]
            row['phases']=phases(row['engine_seconds'],float(row['phase_seconds']['init']),baby,row['stats']['real_batched_split'])
            selected.append(row)
        out[cell['case']]=describe(selected)|dict(selection=decision['selection'],version=plan['program_version'],
            seconds=decision['sample']['mean_seconds'],stddev_seconds=decision['sample']['stddev_seconds'])
    return out


def audit(plan,study):
    if not plan['complete'] or plan['failures'] or plan['schema']!='layout-matrix-1':
        raise ValueError('Need a complete failure-free bounded matrix')
    if len(plan['cases'])!=13 or len(plan['cells'])!=13 or plan['validation_repeats']!=3:
        raise ValueError('Wrong width/B2/held-out scope')
    if any(c['nominal_B2']!=plan['nominal_B2'] for c in plan['cells']) or len({c['case'] for c in plan['cells']})!=13:
        raise ValueError('Mixed B2 bounds or duplicate input in lower-bound matrix')
    if sha(study/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(study/'bin/gmp-10.dll')!=plan['dll_sha256']:
        raise ValueError('Frozen executable/dependency changed')
    if sha(study/'frozen_tune.toml')!=plan['frozen_sha256'] or sha(study/'initial_tune.toml')!=plan['seed_sha256']:
        raise ValueError('Frozen performance input changed')
    cases={c['id']:c for c in plan['cases']};cells={c['id']:c for c in plan['cells']}
    decisions={r['cell']:r for r in plan['decisions']}
    checked=selftests=calls=0
    for group,count in [('primes',2),('queues',None),('validations',3)]:
        if len(plan[group])!=13 or {q['cell'] for q in plan[group]}!=set(cells):
            raise ValueError('Incomplete/duplicate matrix wave')
        for q in plan[group]:
            cell=cells[q['cell']];case=cases[cell['case']];folder=Path(q['folder'])
            if count is not None and q['completed']!=count:raise ValueError('Wrong invocation count')
            if group=='queues' and q['completed']>plan['learning_limit']-1:raise ValueError('Learning budget exceeded')
            requested=cell['seed_route']['D'] if group=='primes' else 0
            rows=enrich(folder,queue_rows(folder,case,cell,plan['version'],q['completed'],
                       2 if group=='primes' else 3 if group=='validations' else 48,requested_d=requested))
            receipts=[json.loads(s) for s in (folder/'result.jsonl').read_text(encoding='utf-8-sig').splitlines()]
            if any(r['source-build']!=plan['exe_sha256'] for r in receipts):
                raise ValueError('Result executable identity differs')
            if rows!=q['runs']:raise ValueError('Parsed receipts differ from captured metadata')
            for r in rows:
                if r['P']!=phi(r['D'])//2:raise ValueError('Invalid independent D/P geometry')
                checked+=int(r['s4']['gmp_checked']);selftests+=int(r['s4']['gmp_selftest_cases']);calls+=1
                if group=='validations' and route(r['selection'])!=route(decisions[cell['id']]['selection']):
                    raise ValueError('Validation route differs from frozen decision')
            for name in ('before.json','after.json'):
                s=load(folder/name)
                if s['allowed_processes']!=plan['allowed_processes'] or s['shared_allowed']:
                    raise ValueError('GPU process protocol differs')
                if foreign_gpu_processes(s['processes'],s['uuid'],study,plan['allowed_processes']):
                    raise ValueError('Unexpected external GPU context')
    return dict(complete=True,full_calls=calls,independent_calls=39,gmp_checked=checked,
                selftests=selftests,arithmetic_errors=0,defaults=audit_defaults(Path(plan['defaults']),1))


def supplements(plan,baseline,study,name):
    path=study/name/'summary.json'
    if not path.exists():return []
    report=load(path)
    if not report['complete']:raise ValueError('Supplemental matrix is incomplete')
    control=name=='legacy_controls';expected=1 if control else 2
    if len(report['runs'])!=expected*len(report['candidates']):raise ValueError('Supplemental sample count differs')
    originals={(r['case'],r['engine'],r['B2']):r for r in baseline['runs']};cases={c['id']:c for c in baseline['cases']}
    if control:
        eligible={r['id'] for r in originals.values() if int(r['stats']['groot_layout']['balanced'])!=
            int((int(r['shape']['S_bits'])+63)//64>128 and (r['B2']//r['D']+2)//r['P']>=4)}
        if set(report['candidates'])!=eligible:raise ValueError('Missing layout-control cells')
    seen=set()
    for r in report['runs']:
        key=(r['case'],r['engine'],r['B2']);old=originals[key];case=cases[r['case']];prefix=Path(r['prefix'])
        identity=key+(r['repeat'],)
        if identity in seen:raise ValueError('Duplicate supplemental invocation')
        seen.add(identity)
        raw=load(prefix.with_suffix('.jsonl'));receipt=normalize_result(raw)
        if raw['status']!='NF' or receipt['bad_factors'] or receipt['factors'] or raw['source-build']!=plan['exe_sha256']:
            raise ValueError('Supplemental NF/arithmetic/identity failure')
        expected_identity=dict(N_hex=case['N_hex'],B1=case['b1'],B2=r['B2'],sigma=case['sigma'],
            curve='ed',carrier_exponent=old['carrier_exponent'])
        if any(receipt[k]!=v for k,v in expected_identity.items()):raise ValueError('Supplemental target changed')
        text=prefix.with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+prefix.with_suffix('.debug.log').read_text(encoding='utf-8')
        clock=fields(text,'stage2_full_wall');check=fields(text,'s4_multiply_stats');shape=shape_fields(text)
        if clock!=r['phase_seconds'] or check!=r['s4'] or shape!=r['shape'] or clock['clean']!='1':raise ValueError('Supplemental raw clocks differ')
        if float(clock['total'])!=r['engine_seconds'] or abs(float(clock['total'])-float(clock['init'])-float(clock['main']))>.003:
            raise ValueError('Supplemental engine clock does not close')
        if any(int(check[k]) for k in ('gmp_check_bad','gmp_selftest_bad')) or min(int(check[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0:
            raise ValueError('Supplemental arithmetic checks absent')
        for label in ('real_baby','real_batched_split','groot_layout'):
            if fields(text,label)!=r['stats'][label]:raise ValueError('Supplemental phase/layout differs')
        if (number(shape['D']),number(shape['P']))!=(old['D'],old['P']):raise ValueError('Supplemental geometry differs')
        legacy=int((int(shape['S_bits'])+63)//64>128 and (r['B2']//r['D']+2)//r['P']>=4)
        expected_layout=legacy if control else int(old['stats']['groot_layout']['balanced'])
        if int(r['stats']['groot_layout']['balanced'])!=expected_layout:raise ValueError('Supplemental layout differs')
        if control:
            original_text=Path(old['prefix']).with_suffix('.engine.log').read_text(encoding='utf-8')+'\n'+Path(old['prefix']).with_suffix('.debug.log').read_text(encoding='utf-8')
            before,after=fields(original_text,'real_batched_rootfold',False),fields(text,'real_batched_rootfold',False)
            if not before or not after or int(before.get('digest_words',0))<=0:
                raise ValueError('Supplemental G-root digest absent')
            if any(before[k]!=after[k] for k in ('digest_words','digest_sum','digest_xor','digest_kind')):
                raise ValueError('Supplemental G-root digest mismatch')
        for suffix in ('.before.json','.after.json'):
            snap=load(prefix.with_suffix(suffix))
            if snap['allowed_processes']!=plan['allowed_processes'] or snap['shared_allowed'] or foreign_gpu_processes(
                snap['processes'],snap['uuid'],Path(old['prefix']).parents[1],plan['allowed_processes']):
                raise ValueError('Supplemental GPU process protocol differs')
    return report['runs']


def plot(args):
    from plot_stage2_version import save,errorline
    study=args.layout_study.resolve();plan=load(study/'plan.json');checked=audit(plan,study)
    defaults_plan=load(Path(plan['defaults'])/'plan.json');base=defaults(defaults_plan)
    rescreens=supplements(plan,defaults_plan,study,'rescreens')
    controls=supplements(plan,defaults_plan,study,'legacy_controls')
    checked['supplemental_calls']=len(rescreens)+len(controls)
    checked['total_calls']=checked['full_calls']+checked['defaults']['invocations']+checked['supplemental_calls']
    checked['all_gmp_checked']=checked['gmp_checked']+checked['defaults']['gmp_checked']+sum(int(r['s4']['gmp_checked']) for r in rescreens+controls)
    checked['all_selftests']=checked['selftests']+checked['defaults']['gmp_selftest_cases']+sum(int(r['s4']['gmp_selftest_cases']) for r in rescreens+controls)
    if rescreens:
        for r in rescreens:
            value=dict(r);value['phases']=phases(r['engine_seconds'],float(r['phase_seconds']['init']),r['stats']['real_baby'],r['stats']['real_batched_split'])
            value['detailed']=current_detail(value)
            base[r['case'],r['engine']].append(value)
    oldstudy=Path(plan['seed_profile']).parent;prior=old_runtime(oldstudy,plan['nominal_B2'])
    cases=sorted(plan['cases'],key=lambda r:r['bits']);cells={r['case']:r for r in plan['cells']}
    decisions={r['cell']:r for r in plan['decisions']};valid={r['cell']:r for r in plan['validations']}
    rows=[]
    for case in cases:
        cell=cells[case['id']];decision=decisions[cell['id']]['selection']
        normal=describe(base[case['id'],'gpu']);carrier=describe(base[case['id'],'gpu-carrier'])
        tuned=describe([r|dict(detailed=current_detail(r)) for r in valid[cell['id']]['runs']]);before=prior[case['id']]
        fastest=min((normal,carrier),key=lambda r:r['median_seconds'])
        ref=next(r for r in defaults_plan['prime95_rows'] if r['N_hex']==case['N_hex'] and
                 r['curve_type']=='Edwards' and r['b2']==cell['B2'])
        cpu=ref['s2_total_time']
        rows.append(dict(case=case['id'],bits=case['bits'],exponent=case['exponent'],B1=case['b1'],
            B2=cell['B2'],normal=normal,carrier=carrier,default_best=fastest,tuned=tuned,prior=before,
            D=decision['D'],P=decision['P'],carrier_exponent=decision['carrier_exponent'],
            groot_balanced=decision['groot_balanced'],profile_samples=decision['runtime_samples'],
            learning_calls=2+next(q['completed'] for q in plan['queues'] if q['cell']==cell['id']),
            ordinary_reduced_calls=sum(r['selection']['runtime_allocation']['ordinary_reduced'] for group in ('primes','queues')
                for q in plan[group] if q['cell']==cell['id'] for r in q['runs']),
            change_prior_percent=100*(tuned['seconds']/before['seconds']-1),
            change_default_percent=100*(tuned['seconds']/fastest['median_seconds']-1),prime95_seconds=cpu))
    args.output.mkdir(parents=True,exist_ok=False);args.figures.mkdir(parents=True,exist_ok=False)
    x=np.array([r['bits'] for r in rows]);prefix=f'stage2-layout-{plan["version"]}-low'
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(17,6.4))
    for ax,log in zip(axes,(False,True)):
        for field,label,color,style in [('normal','Current ordinary default','#3684ad','-'),
            ('carrier','Current carrier default','#d27c3a','-'),('prior',f'{rows[0]["prior"]["version"]} selected training mean','#7b7e82','--'),
            ('tuned',f'{plan["version"]} held-out tune (n=3)','#4b9b53','-')]:
            source=[dict(r[field],seconds=r[field]['median_seconds'] if field in ('normal','carrier') else r[field]['seconds']) for r in rows]
            errorline(ax,x,source,label,color,style=style)
        ax.plot(x,[r['prime95_seconds'] for r in rows],'s:',color='#333333',label='Prime95 4T (historical n=1)',ms=3)
        ax.set(xlabel='Actual N bits',ylabel='Full Stage2 seconds',yscale='log' if log else 'linear',xlim=(0,17000))
        ax.set_xticks(range(0,17001,1000));ax.tick_params(axis='x',rotation=45)
        if not log:ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle(f'ECM-ELF {plan["version"]}: lower B2, exact Edwards matrix')
    save(fig,prefix+'-times',args,'RTX 4060 Laptop GPU1 | B1=50,000 | actual B2=27.0..27.6 billion | raw seconds, no frequency normalization\nDefaults n=1 (flagged cells rechecked); held-out tune n=3, min-max. Prior rows were selected on training observations.')
    fig,axes=plt.subplots(3,1,figsize=(18,11),sharex=True)
    xx=np.arange(len(rows))
    for ax,field,title in zip(axes,('prior','default_best','tuned'),
        (f'{rows[0]["prior"]["version"]} selected learning observations',f'{plan["version"]} faster default family (post-hoc reference)',
         f'{plan["version"]} frozen tune, independent validation (n=3)')):
        bottom=np.zeros(len(rows))
        for name,label,color in zip(NAMES,LABELS,COLORS):
            y=np.array([r[field]['fractions'][name]*100 for r in rows])
            ax.bar(xx,y,bottom=bottom,color=color,width=.78,label=label);bottom+=y
        ax.set(title=title,ylabel='Engine time (%)',ylim=(0,100));ax.grid(axis='x',visible=False)
    axes[-1].set_xticks(xx,[str(r['bits']) for r in rows]);axes[-1].set_xlabel('Actual N bits (categorical spacing)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=9,bbox_to_anchor=(.5,.974),fontsize=8)
    fig.suptitle('Lower B2: non-overlapping Stage2 phase shares',y=.995)
    save(fig,prefix+'-phase-shares-common',args,'Shared historical partition: giant leaf preparation remains in Main other because old logs lack this detail.\nMean of per-call fractions; init and main components never overlap. Larger shares need not mean more seconds.')
    fig,axes=plt.subplots(2,1,figsize=(18,8),sharex=True)
    for ax,field,title in zip(axes,('default_best','tuned'),
        (f'{plan["version"]} faster default family (post-hoc reference)',f'{plan["version"]} frozen tune, independent validation (n=3)')):
        bottom=np.zeros(len(rows))
        for name,label,color in zip(DETAIL_NAMES,DETAIL_LABELS,DETAIL_COLORS):
            y=np.array([r[field]['detail_fractions'][name]*100 for r in rows])
            ax.bar(xx,y,bottom=bottom,color=color,width=.78,label=label);bottom+=y
        ax.set(title=title,ylabel='Engine time (%)',ylim=(0,100));ax.grid(axis='x',visible=False)
    axes[-1].set_xticks(xx,[str(r['bits']) for r in rows]);axes[-1].set_xlabel('Actual N bits (categorical spacing)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=10,bbox_to_anchor=(.5,.99),fontsize=8)
    save(fig,prefix+'-phase-shares',args,'Current-version detailed partition: Giant leaf prep / sync is separate from point launch, G trees and the residual.\nHost-scope clocks: asynchronous device work can be charged to the next blocking boundary. These are not isolated GPU-kernel timings.')
    fig,axes=plt.subplots(2,1,figsize=(18,8),sharex=True)
    for ax,field,title in zip(axes,('prior','tuned'),(f'{rows[0]["prior"]["version"]} selected learning observations',f'{plan["version"]} frozen tune, independent validation')):
        bottom=np.zeros(len(rows))
        for name,label,color in zip(NAMES+['selection_core','outer_other'],
            LABELS+['Selection core','Startup / validation / finalization other'],COLORS+['#e7c7a4','#5c626a']):
            y=np.array([r[field]['outer_fractions'][name]*100 for r in rows])
            ax.bar(xx,y,bottom=bottom,color=color,width=.78,label=label);bottom+=y
        ax.set(title=title,ylabel='Curve wall time (%)',ylim=(0,100));ax.grid(axis='x',visible=False)
    axes[-1].set_xticks(xx,[str(r['bits']) for r in rows]);axes[-1].set_xlabel('Actual N bits (categorical spacing)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=6,bbox_to_anchor=(.5,.99),fontsize=8)
    save(fig,prefix+'-wall-shares',args,'Curve_done.wall boundary; no process-exit polling time. Selection core is nested in total planning, not added again.\nThe residual includes worker launch, input/profile reading, planning outside the core, and publication. Historical sampling policy is stated separately.')
    totals={key:sum(r[key]['median_seconds'] if key in ('normal','carrier','default_best') else r[key]['seconds'] for r in rows)
            for key in ('normal','carrier','default_best','prior','tuned')}
    totals['parent_prior']=sum(r['prior']['parent_seconds'] for r in rows)
    totals['parent_tuned']=sum(r['tuned']['parent_seconds'] for r in rows)
    busy=[]
    for q in plan['validations']:
        busy += [r for r in (json.loads(s) for s in (Path(q['folder'])/'telemetry.jsonl').read_text().splitlines())
                 if r.get('gpu_percent',0)>=50]
    summary=dict(complete=True,version=plan['version'],scope='lower B2 only',audit=checked,rows=rows,totals=totals,
        clocks=Counter(r['sm_clock_mhz'] for r in busy),exe_sha256=plan['exe_sha256'],
        learning_calls=sum(q['completed'] for group in ('primes','queues') for q in plan[group]),
        learning_engine_seconds=sum(r['engine_seconds'] for group in ('primes','queues') for q in plan[group] for r in q['runs']),
        reduced_widths=[r['bits'] for r in rows if r['ordinary_reduced_calls']],
        phase_totals={k:{name:sum(r[k]['phases'][name] for r in rows) for name in NAMES}
                      for k in ('default_best','prior','tuned')})
    if controls:
        table=[]
        for r in controls:
            auto=describe(base[r['case'],r['engine']])
            table.append(dict(case=r['case'],engine=r['engine'],D=r['D'],P=r['P'],
                auto_seconds=auto['median_seconds'],legacy_seconds=r['engine_seconds'],
                auto_gtrees=auto['phases']['gtrees'],legacy_gtrees=float(r['stats']['real_batched_split']['gtrees'])))
        summary['layout_controls']=dict(cells=len(table),rows=table,
            auto_seconds=sum(r['auto_seconds'] for r in table),legacy_seconds=sum(r['legacy_seconds'] for r in table),
            auto_gtrees=sum(r['auto_gtrees'] for r in table),legacy_gtrees=sum(r['legacy_gtrees'] for r in table),
            note='Only changed narrow layouts are independently controlled; unchanged defaults are not rerun.')
    write= lambda name,value:(args.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    write('summary.json',summary)
    text=[f'## {plan["version"]}：较小B2的布局与运行时tune', '',
        '范围为13份精确Edwards余因子，实际N318–16272 bits、B1=50000，名义B2=26000000000；实际B2沿用Prime95输入，不按B2或频率换算。仅本档完成当前验证，较大档仍是冻结参考。', '',
        f'GPU1 RTX4060 Laptop、CUDA13.3/SM89、驱动610.88、PTX3/addsub1；arena6300/fold640/batch256 MiB。二进制SHA256 `{plan["exe_sha256"]}`。测试未修改GPU或CPU设置，未验证当前CPU频率；旧CPU/GPU频率条件见下方冻结资料。独立复测的高利用率NVML时钟分布为{dict(summary["clocks"])}。仅放行用户批准的Codex界面PID及冻结路径，其他外部同卡任务拒绝。', '',
        f'当前tune主成绩来自冻结性能文件后的39次独立NF调用，每格n=3；完整引擎均值之和为 **{totals["tuned"]:.3f} s**。旧0.39.3选型观察均值之和为{totals["prior"]:.3f} s，变化{100*(totals["tuned"]/totals["prior"]-1):+.2f}%；相对本轮默认两类型逐格较快的事后参考{totals["default_best"]:.3f} s，变化{100*(totals["tuned"]/totals["default_best"]-1):+.2f}%。旧数据参与过选型，当前数据为独立复测，不把两者当作同口径因果A/B。', '',
        f'![较小B2当前独立tune、默认与旧参考](attachments/{prefix}-times.png)', '',
        '| N bits | 实际B2 | 默认普通 | 默认承载 | 旧tune均值 | 当前独立均值±标准差 | 当前D/P | 算术／布局 |',
        '|---:|---:|---:|---:|---:|---:|---:|:---|']
    for r in rows:
        text.append(f'| {r["bits"]} | {r["B2"]:,} | {r["normal"]["median_seconds"]:.3f} | {r["carrier"]["median_seconds"]:.3f} | {r["prior"]["seconds"]:.3f} | {r["tuned"]["seconds"]:.3f} ± {r["tuned"]["stddev_seconds"]:.3f} | {r["D"]}/{r["P"]} | '+('承载' if r['carrier_exponent'] else '普通')+f'／{r["groot_balanced"]} |')
    text+=['', '默认筛查每格n=1，明显回退单项补至n=3并保留首轮；表中默认取中位数，当前tune取独立n=3均值，误差带为原始min/max。布局0为连续，1为分块空位。相同Stage1点重放只验证性能，不增加独立ECM覆盖。', '',
        f'![较小B2互斥阶段占比](attachments/{prefix}-phase-shares.png)', '',
        f'堆积高度固定为100%，采用逐调用比例的均值。Baby从init中拆出；F树/初始化余项包括必要自检。当前日志可再从main余项拆出Giant叶准备/同步，其他分项互斥；旧日志缺此细分，仍留在旧余项，跨版本共同口径图见[共同分项对照](attachments/{prefix}-phase-shares-common.png)。NTT/S4嵌套计时不再相加。占比上升不等于绝对秒数增加；异步GPU工作可能在下一处阻塞边界计时，不能把此图当作独立kernel耗时。', '',
        f'![较小B2曲线墙钟阶段占比](attachments/{prefix}-wall-shares.png)', '',
        '曲线墙钟使用curve_done.wall，包含worker启动、选型、读取与发布；不混入采集器退出轮询。选择核心使用runtime_prepare.total_seconds，其余外层成本留在余项，未记录的细分项不猜测。', '',
        '跨版本共同分项的累计值（叶准备仍留在余项；每格先求均值再相加，单位秒）：', '',
        '| 阶段 | 旧tune | 当前较快默认参考 | 当前独立tune |', '|---|---:|---:|---:|']
    for name,label in zip(NAMES,LABELS):
        text.append('| '+label+' | '+' | '.join(f'{summary["phase_totals"][k][name]:.3f}' for k in ('prior','default_best','tuned'))+' |')
    text+=['', '旧文件仅作先验。每格先重测旧最快几何的当前自动布局2次，再解锁D／算术／布局进行有界学习；本轮每格学习合计上限8条，原生停止可能保留1条已启动调用。随后冻结性能文件，用生产入口核对最快已准入重复均值，独立复测不写回。不声称穷尽D表或达到全局最优。', '',
        '代码与复现入口见[有界布局矩阵](../../tools/bench/README_STAGE2_VERSION.md#布局与旧tune的有界验证)。原始证据：`data/experiments/stage2_layout_matrix_20261031/low_warm2/`；默认26格来自同目录上级的`default_screen3/`保留范围快照。汇总、CSV与PNG/SVG位于本次绘图命令指定的忽略目录，公开PNG在attachments。']
    if controls:
        c=summary['layout_controls']
        text+=['',f'同二进制固定D旧布局对照仅重跑布局改变的{c["cells"]}格，旧布局各n=1；自动布局使用现有筛查/补测中位数。完整引擎合计{c["legacy_seconds"]:.3f}→{c["auto_seconds"]:.3f} s（{100*(c["auto_seconds"]/c["legacy_seconds"]-1):+.2f}%），G树分项{c["legacy_gtrees"]:.3f}→{c["auto_gtrees"]:.3f} s（{100*(c["auto_gtrees"]/c["legacy_gtrees"]-1):+.2f}%）。未变化的12格不重复计为独立对照。该筛查不代替每格n=3的布局因果实验；完整回执与根摘要核对留在legacy_controls。']
    text+=['',f'严格重读共{checked["total_calls"]}次完整NF：{summary["learning_calls"]}次学习、39次独立复测、26次默认筛查及{checked["supplemental_calls"]}次补测/布局对照；GMP抽查{checked["all_gmp_checked"]}、自检{checked["all_selftests"]}，错误0。学习引擎总时长{summary["learning_engine_seconds"]:.3f} s，另列成本，不加到独立成绩。普通额度收紧发生的位宽为{summary["reduced_widths"]}。', '',
        f'旧学习观察的曲线墙钟均值之和{totals["parent_prior"]:.3f} s，当前冻结复测为{totals["parent_tuned"]:.3f} s；旧侧仍含学习期选型成本，新侧关闭探索，不能把该差额全部归因于代码或缓存优化。']
    (args.output/'report_fragment.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with (args.output/'cells.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in rows)
    shutil.copy2(__file__,args.output/'plot_snapshot.py')
    print(json.dumps(dict(complete=True,totals=totals,rows=len(rows))))
