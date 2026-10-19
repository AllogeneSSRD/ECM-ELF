"""Audit complete runtime-learning queues and their best measured profile decisions."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics as st
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_runtime_matrix import queue_rows,observation_key,task,stopping_observation,log_rows
from bench_stage2_version import load,validate
from stage2_tune_tables import summaries


def audit(study):
    plan=load(study/'plan.json');baseline=load(Path(plan['baseline'])/'plan.json')
    if plan.get('schema')!=3 or not plan['complete'] or plan['failures']:
        raise ValueError('Need a complete, failure-free queue-learning study')
    if plan['device']!=1 or plan['sample_limit']!=10 or plan['curves_per_condition']<8 or plan['stop_repeats']!=3:
        raise ValueError('Wrong learning protocol')
    if len(plan['cases'])!=13 or len(plan['cells'])!=26 or len(plan['queues'])!=26 or len(plan['decisions'])!=26:
        raise ValueError('Wrong matrix scope')
    cases={c['id']:c for c in plan['cases']};cells={c['id']:c for c in plan['cells']}
    original={c['id']:c for c in baseline['cases']};count=plan['curves_per_condition']
    for case in cases.values():
        old=original[case['id']]
        if any(case[k]!=old[k] for k in ('N_hex','bits','exponent','sigma','b1','known')):
            raise ValueError('Exact input differs from default baseline')
        if Path(case['save']).read_bytes()!=Path(old['save']).read_bytes():raise ValueError('Stage1 point changed')
        validate(Path(case['save']),int(case['N_hex'],16),case['b1'],1,case['sigma'],'ed')
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):
        if (study/'bin'/name).read_bytes()!=(Path(plan['source_exe']).parent/name).read_bytes():
            raise ValueError('Frozen executable/dependency differs from the selected build')
    ids=Counter(r['id'] for r in plan['runs'])
    expected=sum(q['completed_count'] for q in plan['queues'])
    if len(ids)!=expected or any(v!=1 for v in ids.values()):raise ValueError('Missing or duplicate curve')
    if len({q['cell'] for q in plan['queues']})!=26:raise ValueError('Duplicate or missing condition queue')
    imported_calls=0;overshoot=0
    if plan.get('imported_study'):
        origin=Path(plan['imported_study']);source=load(study/'import_source_plan.json')
        if (study/'import_source_plan.json').read_bytes()!=(origin/'plan.json').read_bytes():raise ValueError('Import source changed')
        manual=load(origin/'authorized_manual_stop.json') if (origin/'authorized_manual_stop.json').exists() else None
        if manual:
            if source['failures'] or source.get('active',{}).get('cell')!=manual['cell']:
                raise ValueError('Import does not match the authorized immediate stop')
        else:
            boundary=load(origin/'authorized_protocol_stop.json')
            if source.get('active') or len(source['failures'])!=1 or boundary['boundary_gate'] not in source['failures'][0]['error']:
                raise ValueError('Import was not an authorized queue-boundary protocol change')
        for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):
            if (origin/'bin'/name).read_bytes()!=(study/'bin'/name).read_bytes():raise ValueError('Imported build differs')
    all_runs={r['id']:r for r in plan['runs']};observations=defaultdict(list);checked=selftests=0
    for queue in plan['queues']:
        cell=cells[queue['cell']];case=cases[cell['case']];folder=Path(queue['folder'])
        command=load(folder/'command.json')
        if command!=queue['command'] or any(k in command for k in ('--carrier-exponent','--inferred-carrier-exponent','--tune','--tune-d')):
            raise ValueError('Manual arithmetic choice or standalone tune command found')
        for flag,value in {'--device':'1','--d':'0','--auto-carrier':'1','--tune-runtime':'1',
            '--tune-runtime-samples':'10','--runtime-prepare-cache':'1','--short-calibration':'0',
            '--arena-mb':'6300','--owner-budget-mb':'640','--batch-mb':'256','--relocation':'0'}.items():
            if command[command.index(flag)+1]!=value:raise ValueError('Invocation setting differs')
        planned=queue['planned_count'];executed=queue['completed_count']
        if planned!=count or not 1<=executed<=planned:raise ValueError('Wrong queue size')
        initial=task(case,cell,folder/'replay.save',planned)
        if (folder/'worktodo.initial.txt').read_text(encoding='utf-8-sig')!=initial:
            raise ValueError('Task identity changed')
        point=next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
        if (folder/'replay.save').read_text(encoding='utf-8-sig')!=(point+'\n')*planned:
            raise ValueError('Replayed Stage1 points differ')
        partial=queue['finish_reason']=='retained_user_stopped_queue'
        if partial:
            if not queue.get('telemetry_unavailable') or queue['process_seconds'] is not None:
                raise ValueError('Interrupted collector resources must be marked unavailable')
        elif not load(folder/'telemetry.json'):raise ValueError('Missing device observations')
        for side in ('before','after'):
            if load(folder/(side+'.json'))['foreign']:raise ValueError('Same-GPU foreign process in boundary snapshot')
        restored=queue_rows(folder,case,cell,plan['program_version'],executed,planned,partial)
        raw_records=[json.loads(s) for s in (folder/'result.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        if queue['finish_reason']=='retained_prechange_full_queue':
            if executed!=planned or queue['origin']!=plan.get('imported_study'):raise ValueError('Invalid retained queue')
            original=next(q for q in source['queues'] if q['cell']==queue['cell'])
            if any(queue[k]!=v for k,v in original.items()):raise ValueError('Retained queue metadata changed')
            imported_calls+=executed
        elif partial:
            if queue['origin']!=plan.get('imported_study') or not manual or (queue['cell'],executed)!=(manual['cell'],manual['completed_records']):
                raise ValueError('Partial queue is not the explicitly stopped source')
            counts={k:len(log_rows((folder/'engine.log').read_text(encoding='utf-8'),k)) for k in manual['log_counts']}
            if counts!=manual['log_counts']:raise ValueError('Interrupted raw log counts changed')
            imported_calls+=executed
        elif queue['finish_reason']=='selected_repeats_3':
            stop=load(folder/'stop_request.json')
            if stop!=queue['stop_trigger']:raise ValueError('Stop request changed')
            observed=stop['observed_completed'];expected_stop=stopping_observation(raw_records[:observed],3)
            if expected_stop!={k:v for k,v in stop.items() if k!='observed_completed'}:
                raise ValueError('Stop did not follow the selected repeated observation')
            if not observed<=executed<=observed+1:raise ValueError('Unexpected work after graceful stop request')
            overshoot+=executed-observed
            if '--stage2-stop-handle' not in command:raise ValueError('Missing native stop event')
        elif queue['finish_reason']!='queue_limit' or executed!=planned:
            raise ValueError('Unexplained queue completion')
        remaining=(folder/'worktodo.txt').read_text(encoding='utf-8-sig')
        if executed==planned:
            if remaining.strip():raise ValueError('Full queue did not finish')
        else:
            console=(folder/'console.log').read_text(encoding='utf-8')
            if remaining!=initial or (not partial and f'stage2_stopped: curves_this_run={executed} ' not in console):
                raise ValueError('Missing graceful stop or changed unfinished task')
        if any(r['stage2-engine']['device']!=1 for r in raw_records):
            raise ValueError('Raw result reports a different CUDA device')
        for row in restored:
            if all_runs[row['id']]!=row:raise ValueError('Saved run differs from raw logs/results')
            d=row['D'];phi=d;remaining=d;divisor=2
            while divisor*divisor<=remaining:
                if remaining%divisor==0:
                    phi-=phi//divisor
                    while remaining%divisor==0:remaining//=divisor
                divisor+=1
            if remaining>1:phi-=phi//remaining
            if phi!=2*row['P']:raise ValueError('P differs from phi(D)/2')
            if row['observation']:
                sample=row['observation']
                if sample['benchmark_kind']!='runtime_save' or sample['repeats']!=1 or sample['program_version']!=plan['program_version']:
                    raise ValueError('Unexpected observation identity')
                if not math.isclose(sample['mean_seconds'],row['engine_seconds'],abs_tol=2e-6):
                    raise ValueError('Observation differs from complete engine timing')
                observations[observation_key(sample)].append(row)
            checked+=int(row['s4']['gmp_checked']);selftests+=int(row['s4']['gmp_selftest_cases'])
        if abs(sum(r['engine_seconds'] for r in restored)-queue['engine_sum'])>1e-8 or abs(sum(r['parent_seconds'] for r in restored)-queue['parent_sum'])>1e-8:
            raise ValueError('Queue totals differ')
    frozen=study/'stage2_ecm_tune_frozen.toml'
    if frozen.read_bytes()!=(study/'stage2_ecm_tune.toml').read_bytes():raise ValueError('Frozen and retained active profiles differ')
    samples=[s for s in summaries(tomllib.loads(frozen.read_text(encoding='utf-8-sig'))) if s['source']=='measured']
    groups={observation_key(s):s for s in samples}
    if set(groups)!=set(observations) or len(groups)!=len(samples):raise ValueError('Profile groups differ from raw observations')
    for key,runs in observations.items():
        # Text timing logs are rounded to six decimals. Recompute sufficient
        # statistics from the full-precision observations published to JSONL;
        # each observation is separately compared with its rounded log above.
        sample=groups[key];values=[r['observation']['mean_seconds'] for r in runs]
        if not 1<=len(values)==sample['repeats']<=10:raise ValueError('Retained group count differs')
        deviation=st.stdev(values) if len(values)>1 else 0
        if not math.isclose(st.mean(values),sample['mean_seconds'],rel_tol=1e-10,abs_tol=1e-8) or not math.isclose(deviation,sample['stddev_seconds'],rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('Independent mean/deviation differs')
    if {d['cell'] for d in plan['decisions']}!=set(cells):raise ValueError('Missing decision')
    for decision in plan['decisions']:
        folder=Path(decision['folder']);command=load(folder/'command.json');choice=decision['selection'];sample=decision['sample']
        if '--plan-only' not in command or command[command.index('--tune-runtime')+1]!='0':raise ValueError('Decision query is not read-only planning')
        text=(folder/'console.log').read_text(encoding='utf-8')
        events=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        if choice!=next(s for s in events if s.get('type')=='tune_selection'):raise ValueError('Decision differs from native output')
        eligible=[c for c in choice['runtime_candidates'] if c['samples']>=2]
        if choice['runtime_probe'] or choice['reason']!='runtime_measured_fastest' or choice['mean_seconds']!=min(c['mean_seconds'] for c in eligible):
            raise ValueError('Native decision is not the fastest admitted repeated mean')
        selected=observations[observation_key(sample)]
        if sample!=groups[observation_key(sample)] or decision['run_ids']!=[r['id'] for r in selected] or sample['repeats']<2:
            raise ValueError('Best decision does not match its retained observations')
        if (sample['d'],sample['carrier_exponent'],sample['repeats'])!=(choice['D'],choice['carrier_exponent'],choice['runtime_samples']):
            raise ValueError('Best sample identity differs')
    counts=[d['sample']['repeats'] for d in plan['decisions']]
    if imported_calls!=plan.get('imported_calls',0):raise ValueError('Retained call count differs')
    return dict(complete=True,program_version=plan['program_version'],conditions=26,total_calls=len(plan['runs']),
        planned_curve_cap=count,stop_repeats=3,imported_calls=imported_calls,graceful_stop_extra_calls=overshoot,
        actual_calls_min=min(q['completed_count'] for q in plan['queues']),actual_calls_max=max(q['completed_count'] for q in plan['queues']),
        retained_groups=len(samples),retained_observations=sum(len(v) for v in observations.values()),
        best_sample_count_min=min(counts),best_sample_count_max=max(counts),
        gmp_checked=checked,gmp_selftest_cases=selftests,arithmetic_errors=0,
        scope='Best observed admitted runtime-profile means; actual repeated counts, not independent post-selection validation or global D optimality')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();report=audit(a.study)
    if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
