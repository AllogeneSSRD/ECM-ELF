"""GPU1 wait-mode/config/queue checks and real giant tail-chain boundaries.

The generated profile has synthetic rates: plan-only checks search and binding,
not cost accuracy or production publication. Real curves retain all checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_budget_scaling import fields
from ecm_cost_model import features,giant_work


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh directory')
    exe=a.exe.resolve();binary=sha(exe);study=json.loads(a.study.read_text(encoding='utf-8'))
    saves={bits:Path(study['saves'][str(bits)]['path']) for bits in (2203,8191)}
    for bits,path in saves.items():
        if sha(path)!=study['saves'][str(bits)]['sha256']:raise ValueError('Verified save changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(CUDA_LAUNCH_BLOCKING='0',NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',
        NTT_CARRY_CHECK_FUSED='0',NTT_GIANT_CHAIN_MIN='8192',NTT_FOLD_DEVICE_MAX_MB='0')
    info=json.loads(subprocess.run([str(exe),'--cost-device-info','--device','1'],capture_output=True,env=env,check=True,timeout=60).stdout)
    if info['uuid_hex']!='8a67b1f8ef1c3177a822813a7ac2224d':raise ValueError('Wrong device')
    ini=out/'ecm.ini';ini.write_text('[gpu]\ndevice=1\n[queue]\nworktodo=worktodo.txt\nfinished=finished.txt\ntmp_dir=.\n[stage2]\nstage2_factor_only=1\n',encoding='utf-8')
    width=128;points=2880;capacity=max(points,(256<<20)//(16*width));chunk=points*((capacity+points-1)//points);threshold=chunk+8192
    lo=30030*(threshold-3);hi=30030*(threshold+8192+100-2);expected=30030*(threshold-1)-1
    identity=[binary,info['uuid_hex'],info['major'],info['minor'],info['runtime'],info['driver'],info['fixed_mode'],info['outer_unroll_u'],2,0,7,8192,'0'*64,'0'*64]
    rate=[0]*17;rate[9]=1;rate[11]=100
    scope=[8191,1000,4096,0,lo,hi,points,points,2,1000,0.25,1,30030,*rate,1,1,1]
    profile=out/'synthetic.cprof';profile.write_text('ECM_STAGE2_COST_PROFILE 2\nidentity '+' '.join(map(str,identity))+'\nstage1 8191 1000 1 1 0.1\nscope '+' '.join(map(str,scope))+'\nEND 1 1\n',encoding='utf-8')
    rows=[];curves=[];checks=0
    def check(ok,label):
        nonlocal checks
        if not ok:raise AssertionError(label)
        checks+=1
    def invoke(name,args,mode=0,ok=True,reason=None):
        result=out/(name+'.jsonl');log=out/(name+'.log')
        command=[str(exe),'--ini',str(ini),'--device','1','--results',str(result),'--log',str(log),*map(str,args)]
        r=subprocess.run(command,cwd=out,env=env|{'NTT_CUDA_WAIT_MODE':str(mode)},capture_output=True,timeout=180)
        driver=out/(name+'_driver.log');driver.write_bytes(r.stdout+r.stderr)
        text=(r.stdout+r.stderr).decode(errors='replace')+(log.read_text(encoding='utf-8') if log.exists() else '')
        check((r.returncode==0)==ok,name+': exit')
        if reason:check(reason in text,name+': rejection')
        check(sha(exe)==binary,name+': binary unchanged')
        for bits,saved in saves.items():check(sha(saved)==study['saves'][str(bits)]['sha256'],name+': save unchanged')
        rows.append(dict(name=name,command=command,exit_code=r.returncode,mode=mode,driver_log=str(driver),driver_sha256=sha(driver)))
        return r,result,log,text
    plan_args=['--save',saves[8191],'--auto-b2','--cost-profile',profile,'--plan-only','--arena-mb','4096','--owner-budget-mb','0','--factor-only']
    r,result,_,text=invoke('synthetic_plan',plan_args)
    plan=next(json.loads(line) for line in r.stdout.decode().splitlines() if line.startswith('{'))
    check(plan['B2']==expected and plan['I']==threshold,'real packing selects tail-chain plateau end')
    check(not result.exists() and 'gmp_selftest_cases' not in text,'plan executes no curve')
    _,result,_,_=invoke('unbound_wait_plan',plan_args,mode=4,ok=False,reason='cost profile configuration mismatch: NTT_CUDA_WAIT_MODE')
    check(not result.exists(),'unbound wait writes no result')
    queue=out/'worktodo.txt';queue.write_text(f'ECMSTAGE2=1,2,8191,-1,"{saves[8191]}",53993940,0,1,""\n',encoding='utf-8');original=queue.read_bytes()
    _,result,_,text=invoke('invalid_wait_queue',['--worktodo',queue,'--once'],mode=3,ok=False,reason='NTT_CUDA_WAIT_MODE must be')
    check(queue.read_bytes()==original and not result.exists() and not (out/'finished.txt').exists(),'failed worker preserves queue')
    check('gmp_selftest_cases' not in text,'invalid wait executes no GPU curve')
    reference={}
    def curve(name,bits,b2,mode):
        _,result,log,text=invoke(name,['--save',saves[bits],'--b2',b2,'--d','30030','--arena-mb','4096','--factor-only'],mode=mode)
        r=json.loads(result.read_text(encoding='utf-8'));f=features(30030,b2,bits,8192)
        check(all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')),name+': arithmetic checks')
        check(r['B1']==1000 and r['B2']==b2 and r['sigma']==26 and int(r['N_hex'],16)==(1<<bits)-1,name+': input')
        check(not r['bad_factors'] and all(1<int(x)<(1<<bits)-1 and ((1<<bits)-1)%int(x)==0 for x in r['factors']),name+': proper factors')
        wait=fields(text,'stage2_cuda_wait');check(int(wait['device'])==1 and int(wait['after'])&7==mode,name+': context flags')
        gt=fields(text,'real_batched_gdevice');check([int(gt[k]) for k in ('pairs','groups','copies')]==[f[k] for k in ('g_tree_pairs','gtrees_groups','g_tree_copies')],name+': exact tree')
        seed=fields(text,'real_giant_seed');work=giant_work(f);check(int(seed['chunks'])==work['chain_chunks'],name+': actual giant dispatch')
        value=(fields(text,'descent_values')['hash'],r['factors'],fields(text,'s4_oracle_stats')['signature']);key=(bits,b2)
        if key in reference:check(value==reference[key],name+': wait modes preserve output/check selection')
        reference[key]=value;curves.append(dict(name=name,bits=bits,B2=b2,mode=mode,features=f,giant_seed=seed,result=r,
            leaf_hash=value[0],gmp=fields(text,'s4_multiply_stats'),log=str(log),log_sha256=sha(log)))
    for bits in (2203,8191):
        for mode in ((0,1,2,4) if bits==2203 else (0,4)):curve(f'smoke_m{bits}_wait{mode}',bits,53993940,mode)
    for offset in (-1,0,1):curve(f'tail_boundary_{offset+1}',8191,30030*(threshold+offset-2),0)
    check([int(c['giant_seed']['chunks']) for c in curves[-3:]]==[1,2,2],'chunk tail chain threshold -1/0/+1')
    result=dict(passed=True,binary_sha256=binary,checks=checks,calls=rows,executed_gpu_curves=len(curves),curves=curves,
        synthetic_profile_is_not_calibration=True,expected_plan_B2=expected,chunk_points=chunk,threshold_points=threshold)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('passed','checks','executed_gpu_curves','expected_plan_B2')}))


if __name__=='__main__':main()
