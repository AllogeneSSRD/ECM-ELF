"""Freeze phase predictions first, then run a new blind B2/D/path validation grid."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import time
from ecm_cost_model import predict,admits,FEATURE_PROFILE
from calibrate_stage2_d import parse
from bench_stage2_budget_scaling import fields


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--save-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--b2',type=int,default=4500000000);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--g1',action='store_true',help='Validate 5/8, P-1, root replay, and chain crossover neighbors for G1 scopes')
    p.add_argument('--shuffle-seed',type=int,default=20261006,help='Frozen reproducible order across widths, D and resident paths')
    p.add_argument('--check-inputs-only',action='store_true',help='Check frozen build/profile/save identity without device query or curves')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Use a fresh directory for blind validation')
    model=json.loads(a.profile.read_text(encoding='utf-8'));profile_sha=sha(a.profile)
    if sha(a.stage2)!=model['identity']['stage2_sha256']:raise ValueError('Binary/profile mismatch')
    if model['schema']!=2 or model['feature_profile']!=FEATURE_PROFILE:raise ValueError('Need exact-tree profile v2')
    if any(s['regime']=='g1' for s in model['stage2']) and not a.g1:raise ValueError('Use --g1 to validate every declared G1 scope')
    chain_min=model['chain_min']
    if a.repeats<1:raise ValueError('At least one repetition is required')
    manifest_path=a.stage2.resolve().parent/'frozen_sources_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['binary_sha256']!=sha(a.stage2):raise ValueError('Frozen build differs from binary')
    def verify_build():
        for name,digest in manifest['sources'].items():
            if sha(a.stage2.resolve().parent/'sources'/name)!=digest.lower():raise ValueError('Frozen build source changed: '+name)
    verify_build()
    study_path=Path(model['source'])
    if sha(study_path)!=model['source_sha256']:raise ValueError('Calibration source changed')
    study=json.loads(study_path.read_text(encoding='utf-8'))
    saves={}
    for bits in {s['bits'] for s in model['stage2']}:
        saved=a.save_dir.resolve()/f'm{bits}.save';digest=study['saves'][str(bits)]['sha256']
        if sha(saved)!=digest:raise ValueError('Validation save differs from verified calibration: '+str(bits))
        saves[bits]=(saved,digest)
    if a.check_inputs_only:
        result=dict(kind='validation_input_preflight',passed=True,executed_gpu_curves=0,device_queried=False,
            binary_sha256=sha(a.stage2),profile_sha256=profile_sha,study_sha256=sha(study_path),
            verified_save_sha256={str(bits):digest for bits,(_,digest) in saves.items()})
        (out/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));return
    cases=[]
    for scope in model['stage2']:
        # Failed holdout scopes remain in independent validation and the audit.
        # Skipping them would make the declared profile coverage incomplete.
        for d in scope['d_values']:
            if scope['regime']=='g1':
                if not a.g1:continue
                from calibrate_stage2_d import phi
                points=phi(d)//2
                targets={d*(5*points//8-2):'blind',d*(points-3):'boundary_blind',d*(points-2):'root_replay'}
                for count in (chain_min-1,chain_min,chain_min+1):
                    if 2<=count<=points:targets.setdefault(d*(count-2),'crossover_blind')
            else:targets={a.b2:'blind'}
            for b2,kind in sorted(targets.items()):
                prediction,f=predict(scope['bits'],d,b2,scope['rates'],chain_min)
                if not admits(scope,f):raise ValueError('Blind point outside exact scope: '+scope['id'])
                for rep in range(a.repeats):
                    cases.append(dict(bits=scope['bits'],D=d,B2=b2,owner_mb=scope['owner_mb'],rep=rep,kind=kind,
                        scope_id=scope['id'],prediction=prediction,features=f,arena_mb=scope['arena_mb'],cold_overhead=scope['cold_overhead_seconds']))
    random.Random(a.shuffle_seed).shuffle(cases)
    if not cases:raise ValueError('No validation cases')
    tool_names=('validate_ecm_costs.py','ecm_cost_model.py','calibrate_stage2_d.py','bench_stage2_budget_scaling.py')
    data=dict(schema=1,profile_sha256=profile_sha,model_code_sha256=sha(Path(__file__).with_name('ecm_cost_model.py')),
        binary_sha256=sha(a.stage2),shuffle_seed=a.shuffle_seed,build_manifest_sha256=sha(manifest_path),
        tools={n:sha(Path(__file__).with_name(n)) for n in tool_names},cases=cases,runs=[])
    (out/'predictions.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    prediction_sha=sha(out/'predictions.json')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',NTT_POINT_MERSENNE='1',NTT_GIANT_CHAIN_MIN=str(chain_min),CUDA_LAUNCH_BLOCKING='0')
    info=json.loads(subprocess.run([str(a.stage2.resolve()),'--cost-device-info','--device','1'],env=env,capture_output=True,check=True,timeout=60).stdout)
    if info['uuid_hex']!=model['device']['uuid'].removeprefix('GPU-').replace('-',''):raise ValueError('Validation GPU differs from calibration')
    for i,case in enumerate(cases):
        name=f'{i:03d}_m{case["bits"]}_d{case["D"]}_b{case["B2"]}_o{case["owner_mb"]}_r{case["rep"]}'
        result=out/(name+'.jsonl');log=out/(name+'.log');save,saved_hash=saves[case['bits']]
        cmd=[str(a.stage2.resolve()),'--save',str(save),'--b2',str(case['B2']),'--d',str(case['D']),
             '--device','1','--arena-mb',str(case['arena_mb']),'--results',str(result),'--log',str(log),'--factor-only']
        start=time.perf_counter();child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env|{'NTT_FOLD_DEVICE_MAX_MB':str(case['owner_mb'])})
        try:stdout,stderr=child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.communicate(timeout=30)
            raise RuntimeError('Own blind process tree timed out')
        elapsed=time.perf_counter()-start;(out/(name+'_driver.log')).write_bytes(stdout+stderr)
        if child.returncode:raise RuntimeError('Child validation failed')
        text=log.read_text(encoding='utf-8');record=json.loads(result.read_text(encoding='utf-8'))
        if record['bad_factors'] or any(not 1<int(f)<(1<<case['bits'])-1 or pow(2,case['bits'],int(f))!=1 for f in record['factors']):raise ValueError('Invalid factor')
        if not all(s in text for s in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Missing arithmetic checks')
        actual=parse(text);leaf=fields(text,'descent_values')['hash']
        for old in data['runs']:
            if all(old['case'][k]==case[k] for k in ('bits','D','B2')):
                if old['leaf_hash']!=leaf or old['factors']!=record['factors']:raise ValueError('Path changed the arithmetic output')
        row=dict(name=name,case=case,actual=actual,process_seconds=elapsed,leaf_hash=leaf,factors=record['factors'],
            engine_error_percent=100*(case['prediction']['full']/actual['full']-1),log=str(log),log_sha256=sha(log))
        if sha(a.profile)!=profile_sha or sha(a.stage2)!=data['binary_sha256'] or sha(out/'predictions.json')!=prediction_sha:
            raise ValueError('Frozen prediction inputs changed')
        if sha(save)!=saved_hash or sha(study_path)!=model['source_sha256']:raise ValueError('Verified calibration/save changed')
        if sha(manifest_path)!=data['build_manifest_sha256']:raise ValueError('Frozen build manifest changed')
        verify_build()
        for n,digest in data['tools'].items():
            if sha(Path(__file__).with_name(n))!=digest:raise ValueError('Validation tool changed')
        data['runs'].append(row);(out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(name,'actual',actual['full'],'error%',row['engine_error_percent'],flush=True)
    data['complete']=True;data['prediction_sha256']=prediction_sha
    (out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


if __name__=='__main__':main()
