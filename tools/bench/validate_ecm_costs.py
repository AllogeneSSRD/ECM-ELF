"""Freeze phase predictions first, then run a new blind B2/D/path validation grid."""
import argparse
import hashlib
import json
import os
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
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Use a fresh directory for blind validation')
    model=json.loads(a.profile.read_text(encoding='utf-8'));profile_sha=sha(a.profile)
    if sha(a.stage2)!=model['identity']['stage2_sha256']:raise ValueError('Binary/profile mismatch')
    if model['schema']!=2 or model['feature_profile']!=FEATURE_PROFILE:raise ValueError('Need exact-tree profile v2')
    chain_min=model['chain_min']
    cases=[]
    for scope in model['stage2']:
        if not scope['usable']: continue
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
    data=dict(schema=1,profile_sha256=profile_sha,model_code_sha256=sha(Path(__file__).with_name('ecm_cost_model.py')),
        binary_sha256=sha(a.stage2),cases=cases,runs=[])
    (out/'predictions.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    prediction_sha=sha(out/'predictions.json')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',NTT_POINT_MERSENNE='1',NTT_GIANT_CHAIN_MIN=str(chain_min))
    for i,case in enumerate(cases):
        name=f'{i:03d}_m{case["bits"]}_d{case["D"]}_b{case["B2"]}_o{case["owner_mb"]}_r{case["rep"]}'
        result=out/(name+'.jsonl');log=out/(name+'.log');save=a.save_dir.resolve()/f'm{case["bits"]}.save'
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
        data['runs'].append(row);(out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(name,'actual',actual['full'],'error%',row['engine_error_percent'],flush=True)
    data['complete']=True;data['prediction_sha256']=prediction_sha
    (out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


if __name__=='__main__':main()
