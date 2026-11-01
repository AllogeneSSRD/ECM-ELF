"""Reuse a frozen D/layout matrix, learn finite chain lengths, then freeze A/B.

All runs are serial on GPU1. Repeated saved points measure costs, not additional
ECM coverage. Learning and held-out comparisons never share timing samples.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

from bench_stage2_version import load,write,stamp,check_gpu_processes
from bench_stage2_point_routes import sha,validate_clock
from bench_stage2_n_scaling import fields
from bench_stage2_runtime_matrix import StopEvent,completed_records
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result
from stage2_d_catalog import phi
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'test'))
from stage2_tune_tables import summaries


def prepare(a,out):
    source=load(a.study/'plan.json');default=load(Path(source['defaults'])/'plan.json')
    if not source['complete'] or not default['complete']:raise ValueError('Need completed source matrices')
    out.mkdir(parents=True,exist_ok=False);(out/'bin').mkdir();(out/'bin/.gitignore').write_text('*\n',encoding='ascii')
    manifest=load(a.exe.parent/'manifest.json')
    if not manifest['complete'] or manifest['exe_sha256'].lower()!=sha(a.exe):raise ValueError('Unverified build')
    for unit in manifest['reused'].values():
        if sha(unit['object'])!=unit['sha256'].lower():raise ValueError('Changed device object')
    for f in ('ecm_cuda_stage2.exe','gmp-10.dll'):shutil.copy2(a.exe.parent/f,out/'bin'/f)
    shutil.copy2(a.exe.parent/'manifest.json',out/'build_manifest.json')
    shutil.copy2(a.study/'frozen_tune.toml',out/'initial_tune.toml')
    shutil.copy2(out/'initial_tune.toml',out/'active_tune.toml')
    shutil.copy2(__file__,out/'collector_snapshot.py')
    rows=summaries(tomllib.loads((out/'initial_tune.toml').read_text(encoding='utf-8-sig')))
    (out/'inputs').mkdir();cases=[];cells=[]
    for c in source['cases']:
        c=dict(c);save=out/'inputs'/Path(c['save']).name;shutil.copy2(c['save'],save)
        c.update(save=str(save),save_sha256=sha(save));cases.append(c)
        b2=next(x['B2'] for x in source['cells'] if x['case']==c['id'])
        for carrier in (0,c['exponent']):
            eligible=[r for r in rows if r['source']=='measured' and r['repeats']>=2 and
                r['target_bits']==c['bits'] and r['b2']==b2 and r['carrier_exponent']==carrier and
                not r.get('giant_tail_balance',0) and r.get('giant_chain_block',64)==64]
            seed=min(eligible,key=lambda r:r['mean_seconds'])
            cells.append(dict(id=c['id']+('_ordinary' if not carrier else '_carrier'),case=c['id'],
                carrier=carrier,B2=b2,D=seed['d'],P=phi(seed['d'])//2,layout=seed['groot_balanced'],seed=seed))
    plan=dict(complete=False,started_utc=stamp(),source_study=str(a.study.resolve()),
        exe_sha256=sha(out/'bin/ecm_cuda_stage2.exe'),dll_sha256=sha(out/'bin/gmp-10.dll'),
        seed_sha256=sha(out/'initial_tune.toml'),allowed_processes=source['allowed_processes'],
        cases=cases,cells=cells,learning=[],decisions=[],evaluations=[],retests=[],failures=[],
        protocol=dict(min_repeats=1,device=1,batch_mb=256,arena_mb=6300,fold_mb=640,
            warmups=0,training='Fixed prior fastest D/layout; useful 64 and 32 observations, each minimum one.',
            validation='Freeze chain and D/layout; one off/on call, alternating order; never train on these calls.',
            repeats='Recheck changed schedules slower by >3% and >0.05 s, retaining all initial calls.',
            clocks='No GPU setting changes. Exact approved process whitelist and flushed NVML.'))
    write(out/'plan.json',plan);return plan


def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and
         k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    env['NTT_NO_PROGRESS']='1';return env


def command(out,folder,case,cell,profile,chain,balance,learn=False):
    save=folder/'points.save'
    point=next(s for s in Path(case['save']).read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    save.write_text((point+'\n')*(3 if learn else 1),encoding='utf-8')
    config=folder/'ecm.toml';config.write_text('[stage2]\nstage2_short_calibration=false\nstage2_tune_min_repeats=1\n',encoding='utf-8')
    return [str(out/'bin/ecm_cuda_stage2.exe'),'--config',str(config),'--save',str(save),
        '--curves',str(3 if learn else 1),'--device','1','--b2',str(cell['B2']),'--d',str(cell['D']),
        '--carrier-exponent',str(cell['carrier']),'--groot-layout',str(cell['layout']),
        '--giant-chain',str(chain),'--giant-tail-balance',str(int(balance)),'--b2-round-chain','0',
        '--tune-min-repeats','1','--tune-runtime',str(int(learn)),'--tune-profile',str(profile),
        '--short-calibration','0','--short-ntt-batch-tune','0','--arena-mb','6300',
        '--owner-budget-mb','640','--batch-mb','256','--relocation','0','--factor-only',
        '--ntt-counts','1','--log-level','phases','--results',str(folder/'result.jsonl'),
        '--submission',str(folder/'result.txt'),'--log',str(folder/'engine.log'),
        '--debug-log-file',str(folder/'debug.log')]


def invoke(plan,out,folder,cmd,stop_second=False):
    monitor=TelemetryRecorder(1,folder/'telemetry.jsonl');stop=StopEvent() if stop_second else None
    if stop:cmd+=['--stage2-stop-handle',str(stop.handle)]
    write(folder/'command.json',cmd);started=time.perf_counter();requested=False
    try:
        check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=plan['allowed_processes'])
        with (folder/'console.log').open('wb') as log:
            extra=dict(startupinfo=stop.startup(),close_fds=True) if stop else {}
            process=subprocess.Popen(cmd,cwd=folder,env=environment(),stdout=log,stderr=subprocess.STDOUT,**extra)
            while process.poll() is None:
                if stop and not requested and 'curve_start: 2/3 ' in (folder/'console.log').read_text(encoding='utf-8',errors='replace'):
                    stop.request();requested=True;write(folder/'stop_request.json',dict(reason='second_chain_observation_inflight'))
                if time.perf_counter()-started>900:
                    if stop:stop.request()
                    process.wait(timeout=900);raise TimeoutError(str(folder))
                time.sleep(.01 if stop else .2)
        check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=plan['allowed_processes'])
        if process.returncode:raise ValueError('Worker failed: '+str(folder))
    finally:
        monitor.close()
        if stop:stop.close()
    return time.perf_counter()-started


def point_policy_values(engine):
    # Old receipts store permission only. Do not reinterpret them as proof that
    # borrowing occurred; new receipts record both treatment and actual schedule.
    requested=bool(engine.get('giant_tail_balance_requested',engine.get('giant_tail_balance',False)))
    effective=bool(engine['giant_tail_balance']) if 'giant_tail_balance_requested' in engine else None
    if effective and not requested:raise ValueError('Borrowing occurred without permission')
    return requested,effective


def records(plan,folder,case,cell):
    raw=completed_records(folder/'result.jsonl');text=(folder/'engine.log').read_text(encoding='utf-8')
    debug=(folder/'debug.log').read_text(encoding='utf-8');rows=[]
    labels=('stage2_full_wall','s4_multiply_stats','real_baby','real_batched_split','real_batched_wall')
    data={k:[fields(s,k) for s in (text+'\n'+debug).splitlines() if s.startswith(k+':')] for k in labels}
    policies=[fields(s,'giant_chain_policy') for s in debug.splitlines() if s.startswith('giant_chain_policy:')]
    layouts=[fields(s,'groot_layout') for s in debug.splitlines() if s.startswith('groot_layout:')]
    offset=0
    if not raw or any(len(v)!=len(raw) for v in data.values()) or len(layouts)!=len(raw):raise ValueError('Incomplete evidence')
    for i,r in enumerate(raw):
        value=normalize_result(r);engine=r['stage2-engine'];clock=data['stage2_full_wall'][i];check=data['s4_multiply_stats'][i]
        expected=dict(N_hex=case['N_hex'],sigma=case['sigma'],B1=case['b1'],B2=cell['B2'],carrier_exponent=cell['carrier'],curve='ed')
        if any(value[k]!=v for k,v in expected.items()) or r['source-build']!=plan['exe_sha256'] or r['status']!='NF' or value['factors'] or value['bad_factors']:
            raise ValueError('Identity/NF/build differs')
        if r['d']!=cell['D'] or r['poly-size']!=cell['P'] or int(layouts[i]['balanced'])!=cell['layout']:raise ValueError('Frozen geometry changed')
        validate_clock(clock)
        if min(int(check[k]) for k in ('gmp_checked','gmp_selftest_cases'))<=0 or any(int(check[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
            raise ValueError('Missing/failed arithmetic checks')
        count=0;route=[]
        while offset<len(policies) and count<cell['B2']//cell['D']+2:
            p=policies[offset];route.append(p);count+=int(p['npts']);offset+=1
        if count!=cell['B2']//cell['D']+2:raise ValueError('Incomplete point schedule')
        requested,effective=point_policy_values(engine)
        rows.append(dict(seconds=float(clock['total']),clock=clock,s4=check,policies=route,
            chain=engine['giant_chain_block'],balance=requested,effective_balance=effective,
            selection=engine.get('tune_plan'),sample=r.get('tune_observation'),
            stats={k:data[k][i] for k in labels[2:]}))
    if offset!=len(policies):raise ValueError('Unexpected replay in NF timing')
    return rows


def learn(plan,out):
    for cell in plan['cells']:
        if any(x['cell']==cell['id'] for x in plan['learning']):continue
        case=next(c for c in plan['cases'] if c['id']==cell['case']);folder=out/'learning'/cell['id'];folder.mkdir(parents=True,exist_ok=False)
        cmd=command(out,folder,case,cell,out/'active_tune.toml','auto',False,True)
        seconds=invoke(plan,out,folder,cmd,True);rows=records(plan,folder,case,cell)
        # A compatible seed can skip the initial64 probe. Either way the trial
        # must contain 32, and the frozen decision must have both actual means.
        if not any(r['chain']==32 for r in rows) or any(r['selection']['runtime_min_repeats']!=1 for r in rows):raise ValueError('Chain minimum one not exercised')
        plan['learning'].append(dict(cell=cell['id'],folder=str(folder),process_seconds=seconds,rows=rows))
        write(out/'plan.json',plan);print('chain',len(plan['learning']),len(plan['cells']),cell['id'],[r['chain'] for r in rows],flush=True)
    if not (out/'frozen_tune.toml').exists():shutil.copy2(out/'active_tune.toml',out/'frozen_tune.toml')
    plan['frozen_sha256']=sha(out/'frozen_tune.toml');write(out/'plan.json',plan)


def decide(plan,out):
    for cell in plan['cells']:
        if any(x['cell']==cell['id'] for x in plan['decisions']):continue
        case=next(c for c in plan['cases'] if c['id']==cell['case']);folder=out/'decisions'/cell['id'];folder.mkdir(parents=True,exist_ok=False)
        cmd=command(out,folder,case,cell,out/'frozen_tune.toml','auto',False)+['--plan-only']
        invoke(plan,out,folder,cmd)
        events=[json.loads(s) for s in (folder/'console.log').read_text(encoding='utf-8').splitlines() if s.startswith('{')]
        choice=next(r for r in events if r.get('type')=='tune_selection')
        candidates=[r for r in choice['runtime_candidates'] if r['samples']>=1]
        if choice['runtime_probe'] or {r['giant_chain_block'] for r in candidates}!={32,64} or choice['mean_seconds']!=min(r['mean_seconds'] for r in candidates):
            raise ValueError('Not the measured fastest chain')
        plan['decisions'].append(dict(cell=cell['id'],chain=choice['giant_chain_block'],selection=choice))
        write(out/'plan.json',plan)
    if sha(out/'frozen_tune.toml')!=plan['frozen_sha256']:raise ValueError('Read-only profile changed')


def evaluate(plan,out):
    for i,cell in enumerate(plan['cells']):
        case=next(c for c in plan['cases'] if c['id']==cell['case']);chain=next(d['chain'] for d in plan['decisions'] if d['cell']==cell['id'])
        for balance in ([False,True] if i%2==0 else [True,False]):
            label='on' if balance else 'off'
            if any(x['cell']==cell['id'] and x['balance']==balance for x in plan['evaluations']):continue
            folder=out/'evaluation'/cell['id']/label;folder.mkdir(parents=True,exist_ok=False)
            cmd=command(out,folder,case,cell,out/'frozen_tune.toml',chain,balance)
            before=sha(out/'frozen_tune.toml');seconds=invoke(plan,out,folder,cmd);rows=records(plan,folder,case,cell)
            if len(rows)!=1 or rows[0]['chain']!=chain or rows[0]['balance']!=balance or sha(out/'frozen_tune.toml')!=before:
                raise ValueError('Frozen validation path/profile changed')
            plan['evaluations'].append(dict(cell=cell['id'],balance=balance,chain=chain,folder=str(folder),process_seconds=seconds,row=rows[0]))
            write(out/'plan.json',plan);print('heldout',len(plan['evaluations']),2*len(plan['cells']),cell['id'],label,rows[0]['seconds'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path);p.add_argument('--exe',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--phase',choices=('all','learn','evaluate'),default='all')
    a=p.parse_args();out=a.output.resolve()
    if not a.resume and (not a.study or not a.exe):p.error('New study requires source study and verified exe')
    plan=load(out/'plan.json') if a.resume else prepare(a,out)
    if plan['failures'] or plan['complete']:raise ValueError('Completed or retained failed study')
    if (out/'collector_snapshot.py').read_bytes()!=Path(__file__).read_bytes():raise ValueError('Collector changed; preserve evidence')
    if sha(out/'bin/ecm_cuda_stage2.exe')!=plan['exe_sha256'] or sha(out/'initial_tune.toml')!=plan['seed_sha256']:raise ValueError('Frozen source changed')
    try:
        if a.phase in ('all','learn'):learn(plan,out)
        if a.phase in ('all','evaluate'):
            if len(plan['learning'])!=len(plan['cells']):raise ValueError('Finish all chain learning before validation')
            decide(plan,out);evaluate(plan,out);plan.update(complete=True,finished_utc=stamp())
        write(out/'plan.json',plan)
    except Exception as e:
        plan['failures'].append(dict(utc=stamp(),error=str(e)));write(out/'plan.json',plan);raise


if __name__=='__main__':main()
