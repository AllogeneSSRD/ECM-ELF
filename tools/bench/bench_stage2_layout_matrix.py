"""Bounded warm-start runtime learning and independent Stage2 matrix validation.

Uses exact saves/adjusted B2 from an existing matrix. Old measurements remain
priors; the reported validation means use only new, frozen-profile invocations.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import tomllib

from bench_stage2_version import ROOT, load, write, stamp, check_gpu_processes, shape_records
from bench_stage2_runtime_matrix import StopEvent, completed_records, queue_rows, task, log_rows
from gpu_telemetry import TelemetryRecorder
sys.path.insert(0,str(ROOT/'tools/test'))
from check_stage2_version import audit as audit_defaults
from stage2_tune_tables import summaries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phases(total, init, baby, split):
    """Non-overlapping log partition, shared with historical comparisons.

    Baby is inside init; the remaining init includes F tree and selftests.
    Giant/G trees/fold/inverse/descent/accum are disjoint main components.
    Rounded component logs have a bounded 12ms partition tolerance.
    """
    value=dict(baby=float(baby['ladder'])+float(baby['affine']))
    value['ftree_init']=init-value['baby']
    for key in ('inv','giant','gtrees','fold','descent','accum'):
        value[key]=float(split[key])
    value['other']=total-sum(value.values())
    if any(x < -.012 for x in value.values()):
        raise ValueError('Overlapping or inconsistent component clocks')
    # Keep exact closure after correcting only sub-resolution rounding residue.
    for key in value:
        if value[key]<0:
            residue=value[key];value[key]=0;value['ftree_init']+=residue
    if value['ftree_init']<0 or abs(sum(value.values())-total)>1e-8:
        raise ValueError('Phase partition does not close')
    return value


def enrich(folder, rows):
    text=(folder/'engine.log').read_text(encoding='utf-8')
    babies=log_rows(text,'real_baby')
    debug=(folder/'debug.log').read_text(encoding='utf-8')
    layouts=log_rows(debug,'groot_layout')
    if len(babies)!=len(rows) or len(layouts)!=len(rows):
        raise ValueError('Missing per-curve baby/layout evidence')
    for row,baby,layout in zip(rows,babies,layouts):
        row['groot_balanced']=int(layout['balanced'])
        if row['groot_balanced']!=row['selection']['groot_balanced']:
            raise ValueError('Executed layout differs from native choice')
        row['phases']=phases(row['engine_seconds'],float(row['phase_seconds']['init']),
                             baby,row['stats']['real_batched_split'])
    return rows


def route(choice):
    return choice['D'],choice['carrier_exponent'],choice['groot_balanced']


def stable_repeats(records, target=3):
    counts=Counter()
    for row in records:
        choice=row['stage2-engine']['tune_plan']
        if choice['runtime_probe']:
            continue
        counts[route(choice)]+=1
        if counts[route(choice)]>=target:
            return True
    return False


def subset_defaults(source, destination, bound):
    path=source/'scope_boundary_plan.json'
    document=load(path if path.exists() else source/'plan.json')
    if not document['complete'] and not (source/'authorized_scope_stop.json').exists():
        raise ValueError('Partial baseline needs an explicit retained scope boundary')
    rows=[r for r in document['runs'] if r['nominal_B2']==bound]
    if len(rows)!=26 or len({(r['case'],r['engine']) for r in rows})!=26:
        raise ValueError('Lower-bound default matrix is incomplete')
    destination.mkdir()
    document.update(runs=rows,failures=[],complete=True,matrix_bounds=[bound],
        subset_source=str(source),subset_source_sha256=sha(path if path.exists() else source/'plan.json'))
    write(destination/'plan.json',document)
    return audit_defaults(destination,1)


def prepare(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=a.defaults.resolve();audit=subset_defaults(baseline,out/'defaults',a.bound)
    source=load(out/'defaults/plan.json')
    (out/'bin').mkdir();(out/'bin/.gitignore').write_text('*\n',encoding='ascii')
    for name in ('ecm_cuda_stage2.exe','gmp-10.dll'):
        shutil.copy2(Path(source['exe']).parent/name,out/'bin'/name)
    shutil.copy2(a.seed_profile,out/'initial_tune.toml')
    shutil.copy2(a.seed_profile,out/'active_tune.toml')
    shutil.copy2(__file__,out/'collector_snapshot.py')
    allowed=[dict(pid=str(pid),process_name=subprocess.check_output(
        ['powershell','-NoProfile','-Command',f'(Get-Process -Id {pid}).Path'],text=True).strip())
        for pid in a.allow_gpu_pid]
    cells=[]
    for ref in source['prime95_rows']:
        if ref['curve_type']!='Edwards' or not ref['benchmark_eligible'] or ref['b2_requested']!=a.bound:
            continue
        case=next(c for c in source['cases'] if c['N_hex']==ref['N_hex'])
        cells.append(dict(id=f'{case["id"]}_b{ref["b2"]}',case=case['id'],
                          B2=ref['b2'],nominal_B2=a.bound))
    first=cells[0]
    cells=[first]+sorted(cells[1:],key=lambda c:next(x['bits'] for x in source['cases'] if x['id']==c['case']),reverse=True)
    seed_rows=summaries(tomllib.loads(a.seed_profile.read_text(encoding='utf-8-sig')))
    for cell in cells:
        case=next(c for c in source['cases'] if c['id']==cell['case'])
        candidates=[s for s in seed_rows if s['source']=='measured' and s['repeats']>=2 and
                    s['target_bits']==case['bits'] and s['b2']==cell['B2']]
        best=min(candidates,key=lambda s:s['mean_seconds'])
        cell['seed_route']=dict(D=best['d'],carrier_exponent=best['carrier_exponent'])
    plan=dict(schema='layout-matrix-1',version=source['program_version'],device=1,
        nominal_B2=a.bound,cases=source['cases'],cells=cells,defaults=str(out/'defaults'),
        baseline_audit=audit,source_defaults=str(baseline),seed_profile=str(a.seed_profile.resolve()),
        seed_sha256=sha(a.seed_profile),exe_sha256=sha(out/'bin/ecm_cuda_stage2.exe'),
        dll_sha256=sha(out/'bin/gmp-10.dll'),allowed_processes=allowed,
        learning_limit=a.learning_limit,validation_repeats=3,planned_queue=48,
        protocol=dict(memory=dict(arena_mb=6300,fold_mb=640,batch_mb=256),warmups=0,
            training='Existing profile as prior. Re-measure its best geometry twice with the current automatic layout, then unlock D/carrier/layout. No condition relabeling.',
            stopping='Three new non-probe calls of one route, or eight completed learning calls; native event may retain one already-started call.',
            validation='Read-only frozen profile, three new calls; no learning calls in validation statistics.',
            clocks='No clock/power changes, GPU1 serial, selected GPU process snapshots plus flushed NVML',
            phases='init split into baby and F-tree/init remainder; disjoint main components plus residual'),
        primes=[],queues=[],decisions=[],validations=[],failures=[],complete=False,started_utc=stamp())
    write(out/'plan.json',plan)
    return plan


def execute(plan,out,case,cell,profile,label,learn,prime=False):
    folder=out/label/cell['id'];folder.mkdir(parents=True,exist_ok=False)
    count=2 if prime else plan['planned_queue'] if learn else plan['validation_repeats']
    point=next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    save=folder/'points.save';save.write_text((point+'\n')*count,encoding='utf-8')
    queue=folder/'worktodo.txt';line=task(case,cell,save,count);queue.write_text(line,encoding='utf-8')
    (folder/'worktodo.initial.txt').write_text(line,encoding='utf-8')
    config=folder/'ecm.toml';config.write_text('[stage2]\nstage2_auto_carrier=true\n'
        'stage2_prediction_policy="model"\nstage2_short_calibration=false\n',encoding='utf-8')
    command=[str(out/'bin/ecm_cuda_stage2.exe'),'--config',str(config),'--device','1',
        '--worktodo',str(queue),'--once','--d','0','--auto-carrier','1','--short-calibration','0',
        '--tune-runtime',str(int(learn)),'--tune-profile',str(profile),'--arena-mb','6300',
        '--owner-budget-mb','640','--batch-mb','256','--relocation','0','--ntt-counts','0',
        '--factor-only','--log-level','phases','--log',str(folder/'engine.log'),
        '--debug-log-file',str(folder/'debug.log'),'--results',str(folder/'result.jsonl'),
        '--submission',str(folder/'submission.txt')]
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and
         k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1'
    requested_d=0
    if prime:
        requested_d=cell['seed_route']['D']
        command[command.index('--d')+1]=str(requested_d)
        command+=['--carrier-exponent',str(cell['seed_route']['carrier_exponent'])]
        query=command+['--plan-only']
        query[query.index('--tune-runtime')+1]='0'
        query[query.index('--tune-profile')+1]=str(folder/'absent.toml')
        result=subprocess.run(query,cwd=folder,env=env,capture_output=True,text=True,errors='replace',timeout=180)
        (folder/'layout_plan.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        write(folder/'layout_plan.command.json',query)
        if result.returncode:raise ValueError('Old geometry no longer admitted')
        native=next(json.loads(s) for s in result.stdout.splitlines() if s.startswith('{') and json.loads(s).get('type')=='stage2_plan')
        command+=['--groot-layout',str(native['groot_balanced'])]
    limit=plan['learning_limit']-2
    stop=StopEvent();command+=['--stage2-stop-handle',str(stop.handle)]
    write(folder/'command.json',command)
    telemetry=TelemetryRecorder(1,folder/'telemetry.jsonl');start=time.perf_counter();trigger=None
    try:
        check_gpu_processes(telemetry.monitor,out,folder/'before.json',allowed=plan['allowed_processes'])
        with (folder/'console.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
                startupinfo=stop.startup(),close_fds=True)
            while process.poll() is None:
                rows=completed_records(folder/'result.jsonl')
                if learn and not prime and trigger is None and (len(rows)>=limit or stable_repeats(rows)):
                    trigger=dict(completed=len(rows),reason='learning_budget' if len(rows)>=limit else 'stable_new_repeats')
                    stop.request();write(folder/'stop_request.json',trigger)
                if time.perf_counter()-start>10800:
                    stop.request();process.wait(timeout=900);raise TimeoutError(cell['id'])
                time.sleep(.02)
        check_gpu_processes(telemetry.monitor,out,folder/'after.json',allowed=plan['allowed_processes'])
        if process.returncode:
            raise ValueError('Queue worker failed; inspect '+str(folder/'console.log'))
        count_done=len(completed_records(folder/'result.jsonl'))
        rows=enrich(folder,queue_rows(folder,case,cell,plan['version'],count_done,count,requested_d=requested_d))
        if prime:
            if count_done!=2:raise ValueError('Priming requires two actual calls')
        elif learn:
            if trigger is None or not count_done<=limit+1:
                raise ValueError('Missing/broken bounded learning stop')
            text=(folder/'console.log').read_text(encoding='utf-8')
            if f'stage2_stopped: curves_this_run={count_done} ' not in text:
                raise ValueError('Missing graceful-stop receipt')
        elif count_done!=3:
            raise ValueError('Independent validation requires all three calls')
        return dict(cell=cell['id'],folder=str(folder),process_seconds=time.perf_counter()-start,
                    completed=count_done,trigger=trigger,runs=rows)
    finally:
        telemetry.close();stop.close()


def decide(plan,out,case,cell,frozen):
    folder=out/'decisions'/cell['id'];folder.mkdir(parents=True,exist_ok=False)
    queue=folder/'worktodo.txt';queue.write_text(task(case,cell,Path(case['save']),1),encoding='utf-8')
    config=folder/'ecm.toml';config.write_text('[stage2]\nstage2_short_calibration=false\n',encoding='utf-8')
    command=[str(out/'bin/ecm_cuda_stage2.exe'),'--config',str(config),'--device','1',
        '--worktodo',str(queue),'--once','--plan-only','--d','0','--auto-carrier','1',
        '--tune-runtime','0','--short-calibration','0','--tune-profile',str(frozen),
        '--arena-mb','6300','--owner-budget-mb','640','--batch-mb','256','--factor-only']
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k!='CUDA_VISIBLE_DEVICES'}
    r=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,errors='replace',timeout=180)
    write(folder/'command.json',command);(folder/'console.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise ValueError('Read-only choice failed')
    choice=next(json.loads(s) for s in r.stdout.splitlines() if s.startswith('{') and json.loads(s).get('type')=='tune_selection')
    candidates=[s for s in choice['runtime_candidates'] if s['samples']>=2]
    if choice['reason']!='runtime_measured_fastest' or choice['runtime_probe'] or choice['mean_seconds']!=min(s['mean_seconds'] for s in candidates):
        raise ValueError('Native selector did not choose its fastest admitted repeated mean')
    return dict(cell=cell['id'],selection=choice)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--defaults',type=Path);p.add_argument('--seed-profile',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--bound',type=int,default=26000000000)
    p.add_argument('--learning-limit',type=int,default=8)
    p.add_argument('--allow-gpu-pid',type=int,action='append',default=[])
    p.add_argument('--max-queues',type=int,default=0)
    a=p.parse_args();out=a.output.resolve()
    if not a.resume and (not a.defaults or not a.seed_profile):p.error('New study needs defaults and seed-profile')
    if not 5<=a.learning_limit<=32:p.error('learning-limit must be 5..32')
    plan=load(out/'plan.json') if a.resume else prepare(a)
    if plan['complete'] or plan['failures']:raise ValueError('Study complete or has a retained failure')
    if (out/'collector_snapshot.py').read_bytes()!=Path(__file__).read_bytes():raise ValueError('Collector changed; preserve this study')
    if sha(out/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(out/'initial_tune.toml')!=plan['seed_sha256']:
        raise ValueError('Frozen source changed')
    cases={c['id']:c for c in plan['cases']};accepted=0
    try:
        for cell in plan['cells']:
            if any(q['cell']==cell['id'] for q in plan['queues']):continue
            if not any(q['cell']==cell['id'] for q in plan['primes']):
                print('PRIME',cell['id'],cell['seed_route'],flush=True)
                prime=execute(plan,out,cases[cell['case']],cell,out/'active_tune.toml','priming',True,True)
                plan['primes'].append(prime);write(out/'plan.json',plan)
            print('LEARN',len(plan['queues'])+1,'/13',cell['id'],flush=True)
            row=execute(plan,out,cases[cell['case']],cell,out/'active_tune.toml','learning',True)
            shutil.copy2(out/'active_tune.toml',Path(row['folder'])/'profile_after.toml')
            plan['queues'].append(row);write(out/'plan.json',plan);accepted+=1
            print('LEARN DONE',cell['id'],row['completed'],'calls',flush=True)
            if a.max_queues and accepted>=a.max_queues:return
        frozen=out/'frozen_tune.toml'
        if not frozen.exists():shutil.copy2(out/'active_tune.toml',frozen)
        before=sha(frozen)
        for cell in plan['cells']:
            if not any(q['cell']==cell['id'] for q in plan['decisions']):
                plan['decisions'].append(decide(plan,out,cases[cell['case']],cell,frozen));write(out/'plan.json',plan)
            if any(q['cell']==cell['id'] for q in plan['validations']):continue
            decision=next(q for q in plan['decisions'] if q['cell']==cell['id'])
            print('VALIDATE',len(plan['validations'])+1,'/13',cell['id'],flush=True)
            row=execute(plan,out,cases[cell['case']],cell,frozen,'validation',False)
            if any(route(r['selection'])!=route(decision['selection']) for r in row['runs']):
                raise ValueError('Frozen route changed during independent validation')
            plan['validations'].append(row);write(out/'plan.json',plan)
            print('VALIDATE DONE',cell['id'],statistics.mean(r['engine_seconds'] for r in row['runs']),flush=True)
        if sha(frozen)!=before:raise ValueError('Independent validation modified frozen profile')
        plan.update(complete=True,frozen_sha256=before,finished_utc=stamp());write(out/'plan.json',plan)
    except Exception as error:
        plan['failures'].append(dict(utc=stamp(),error=str(error)));write(out/'plan.json',plan);raise


if __name__=='__main__':main()
