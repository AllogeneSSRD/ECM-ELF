"""GPU1 native Auto B2 acceptance, including private INI/queue transactions.

Requires a freshly calibrated .cprof for the exact binary. Arithmetic trials
use independently verified Stage1 saves; malformed-input fixtures only plan.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from ecm_cost_model import predict,kruppa_value,features,admits
from bench_stage2_budget_scaling import fields


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--save-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-bits',type=int,nargs='+',default=[8191],help='Widths expected to pass independent profile publication')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();profile=a.profile.resolve();binary_sha=sha(exe);profile_sha=sha(profile)
    model=json.loads(a.model.read_text(encoding='utf-8'))
    if model['identity']['stage2_sha256']!=binary_sha:raise ValueError('Binary/model mismatch')
    runtime_bits={int(line.split()[1]) for line in profile.read_text().splitlines() if line.startswith('scope ')}
    if runtime_bits!=set(a.expected_bits):raise ValueError('Published widths do not match the required acceptance scope')
    save=a.save_dir.resolve()/'m8191.save';save_sha=sha(save)
    ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n[stage2]\nstage2_factor_only=1\n',encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_CARRY_CHECK_FUSED='0',NTT_STAGE1_Q_DUMP='1')
    env['NTT_GIANT_CHAIN_MIN']=str(model['chain_min'])
    rows=[];checks=0
    def check(condition,label):
        nonlocal checks
        if not condition:raise AssertionError(label)
        checks+=1
    def invoke(name,args,success=True,extra_env=None,reason=None):
        cmd=[str(exe),'--device','1','--ini',str(ini),*map(str,args)]
        r=subprocess.run(cmd,cwd=out,env=env|(extra_env or {}),capture_output=True,timeout=180)
        log=out/(name+'_driver.log');log.write_bytes(r.stdout+r.stderr)
        text=(r.stdout+r.stderr).decode(errors='replace')
        check((r.returncode==0)==success,name+': exit')
        if reason:check(reason in text,name+': error reason')
        rows.append(dict(name=name,returncode=r.returncode,log=str(log),log_sha256=sha(log)))
        check(sha(exe)==binary_sha and sha(profile)==profile_sha and sha(save)==save_sha,name+': immutable inputs')
        return [json.loads(line) for line in r.stdout.decode().splitlines() if line.startswith('{')]
    def auto(name,extra=(),**kwargs):
        return invoke(name,['--save',save,'--auto-b2','--cost-profile',profile,'--plan-only',*extra],**kwargs)
    plans={}
    for name,extra,adjust,t1 in [
        ('default',[],1,None),('fallback',['--owner-budget-mb',0],1,None),
        ('batch12',['--stage1-batch',12],1,None),
        ('explicit_t1',['--stage1-seconds-per-curve','0.12500000000000003'],1,0.12500000000000003),
        ('adjust2',['--stage2-ratio-adjust',2],2,None),
        ('fixed_d',['--d',120120],1,None),
        ('fixed_interval',['--auto-min-b2',4500000000,'--auto-max-b2',4500000000],1,None)]:
        plan=auto(name,extra)[0];plans[name]=plan
        f=features(plan['D'],plan['B2'],8191,model['chain_min'])
        scope=next(s for s in model['stage2'] if s['usable'] and s['owner_resident']==plan['owner_resident'] and admits(s,f))
        phases,f=predict(8191,plan['D'],plan['B2'],scope['rates'],model['chain_min'])
        check((plan['P'],plan['I'],plan['G'],plan['fold_length'])==(f['P'],f['I'],f['G'],f['fold_ntt']),name+': geometry')
        for key,value in plan['phases'].items():check(math.isclose(value,phases[key],rel_tol=1e-12,abs_tol=1e-12),name+': phase '+key)
        check(math.isclose(plan['T2'],adjust*(phases['full']+scope['cold_overhead_seconds']),rel_tol=1e-12),name+': T2')
        check(math.isclose(plan['K'],kruppa_value(1000,plan['B2']),rel_tol=1e-12),name+': benefit')
        check(math.isclose(plan['score'],plan['K']/(plan['T1']+plan['T2']),rel_tol=1e-12),name+': score')
        check(plan['profile_sha256']==profile_sha and not plan['process_peak_guaranteed'],name+': identity/admission')
        if t1 is not None:check(plan['T1']==t1 and plan['T1_source']=='explicit_seconds',name+': explicit T1')
    check(plans['default']['schema']==2 and plans['default']['feature_profile']==7 and plans['default']['chain_min']==model['chain_min'],'versioned feature/policy')
    check(not plans['fallback']['owner_resident'] and plans['fallback']['owner_runtime_mb']==0,'owner budget zero')
    check(plans['fixed_d']['D']==120120 and plans['fixed_interval']['B2']==4500000000,'overrides')
    failures=[('small_arena',['--arena-mb',512],'no measured'),
        ('outside_b2',['--auto-min-b2',1],'outside measured'),
        ('bad_d',['--d',300],'no measured'),('bad_batch',['--stage1-batch',2],'no matching Stage1'),
        ('nan_ratio',['--stage2-ratio-adjust','nan'],'finite and positive'),
        ('zero_t1',['--stage1-seconds-per-curve',0],'finite and positive')]
    for name,extra,reason in failures:auto(name,extra,success=False,reason=reason)
    auto('changed_kernel',success=False,extra_env={'NTT_GIANT_CHAIN_MIN':'1'},reason='configuration mismatch')
    auto('launch_blocking',success=False,extra_env={'CUDA_LAUNCH_BLOCKING':'1'},reason='configuration mismatch')
    invoke('explicit_conflict',['--save',save,'--auto-b2','--b2',3000000000,'--plan-only'],False,reason='conflicts')
    invoke('missing_profile',['--save',save,'--auto-b2','--plan-only'],False,reason='requires --cost-profile')
    invoke('legacy_zero',['--save',save,'--b2',0,'--plan-only'],False,reason='B2 must exceed')
    for bits in (2203,4423):
        args=['--save',a.save_dir.resolve()/f'm{bits}.save','--auto-b2','--cost-profile',profile,'--plan-only']
        if bits not in runtime_bits:
            invoke('excluded_'+str(bits),args,False,reason='no measured')
        else:
            plan=invoke('included_'+str(bits),args)[0];plans[str(bits)]=plan
            f=features(plan['D'],plan['B2'],bits,model['chain_min'])
            scope=next(s for s in model['stage2'] if s['usable'] and s['owner_resident']==plan['owner_resident'] and admits(s,f))
            phases,f=predict(bits,plan['D'],plan['B2'],scope['rates'],model['chain_min'])
            check((plan['P'],plan['I'],plan['G'],plan['fold_length'])==(f['P'],f['I'],f['G'],f['fold_ntt']),str(bits)+': native geometry')
            for key,value in plan['phases'].items():check(math.isclose(value,phases[key],rel_tol=1e-12,abs_tol=1e-12),str(bits)+': native phase '+key)
    for name,n,b1 in [('bad_b1','(2^8191-1)',1001),('generic_n','(2^8191-3)',1000)]:
        fixture=out/(name+'.save');fixture.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1={b1}; N={n}; X=3;\n')
        invoke(name,['--save',fixture,'--auto-b2','--cost-profile',profile,'--plan-only'],False,
            reason='no measured' if name=='bad_b1' else 'exact Mersenne')
    text=profile.read_text();identity=next(line for line in text.splitlines() if line.startswith('identity '))
    tokens=identity.split();stage1_line=next(line for line in text.splitlines() if line.startswith('stage1 '))
    mutations={
        'old_profile_format':text.replace('ECM_STAGE2_COST_PROFILE 2','ECM_STAGE2_COST_PROFILE 1',1),
        'wrong_binary':text.replace(tokens[1],'0'*64,1),
        'wrong_uuid':text.replace(tokens[2],'0'*32,1),
        'wrong_driver':text.replace(identity,' '.join(tokens[:6]+['1']+tokens[7:]),1),
        'truncated':text[:text.rfind('END ')],
        'duplicate':text.replace('END ',identity+'\nEND ',1),
        'nan_profile':text.replace(stage1_line,' '.join(stage1_line.split()[:-1]+['nan']),1),
    }
    leading=stage1_line.split();leading[1]='0'+leading[1]
    mutations['canonical_duplicate']=text.replace('END ', ' '.join(leading)+'\nEND ',1)
    g1_line=next(line for line in text.splitlines() if line.startswith('scope ') and line.split()[9]=='1')
    parts=g1_line.split();parts[4]='1'
    mutations['g1_owner_conflict']=text.replace(g1_line,' '.join(parts),1)
    for name,value in mutations.items():
        bad=out/(name+'.cprof');bad.write_text(value)
        invoke(name,['--save',save,'--auto-b2','--cost-profile',bad,'--plan-only'],False)
    g1=next(s for s in model['stage2'] if s['bits']==8191 and s['regime']=='g1' and s['usable'])
    middle=(g1['b2_min']+g1['b2_max'])//2
    g1_plan=auto('g1_forced',['--d',g1['d_values'][0],'--auto-min-b2',middle,'--auto-max-b2',middle])[0]
    check(g1_plan['G']==1 and g1_plan['local_inverse_work']>0 and g1_plan['phases']['inv']==0 and not g1_plan['owner_resident'],'G1 local inverse dispatch')
    endpoint=auto('g1_root_boundary',['--d',g1['d_values'][0],'--auto-min-b2',g1['b2_max'],'--auto-max-b2',g1['b2_max']])[0]
    check(endpoint['I']==endpoint['P'] and endpoint['root_reduction_work']>0,'G1 full-root boundary')
    auto('unmeasured_gap',['--auto-min-b2',2000000000,'--auto-max-b2',2000000000],success=False,reason='no feasible')
    # A private queue verifies transactions, without touching production files.
    queue=out/'worktodo.txt';finished=out/'finished.txt';qini=out/'queue.ini'
    qini.write_text('[gpu]\ndevice=1\n[queue]\nworktodo=worktodo.txt\nfinished=finished.txt\ntmp_dir='+
        str(a.save_dir.resolve())+'\n[stage2]\nstage2_auto_b2=1\nstage2_cost_profile='+
        os.path.relpath(profile,out)+'\nstage2_factor_only=1\n',encoding='utf-8')
    line='ECMSTAGE2=auto-test,1,2,8191,-1,"m8191.save",0,0,1,""\n';queue.write_text(line)
    invoke('queue_plan',['--ini',qini,'--worktodo',queue,'--plan-only'])
    check(queue.read_text()==line and not finished.exists(),'plan-only leaves queue unchanged')
    badini=out/'bad_queue.ini';badini.write_text(qini.read_text().replace(os.path.relpath(profile,out),'missing.cprof'))
    failed_result=out/'failed_result.jsonl'
    invoke('queue_failure',['--ini',badini,'--worktodo',queue,'--once','--results',failed_result],False,reason='queue retained')
    check(queue.read_text()==line and not finished.exists() and not failed_result.exists(),'failed worker retains queue/no results')
    actual={};leaves={}
    b2=plans['default']['B2'];d=plans['default']['D'];owner=plans['default']['owner_runtime_mb']
    for name,args,is_auto in [
        ('actual_auto',['--save',save,'--auto-b2','--cost-profile',profile],True),
        ('actual_fixed',['--save',save,'--b2',b2,'--d',d,'--arena-mb',4096],False),
        ('queue_auto',['--ini',qini,'--worktodo',queue,'--once'],True),
        ('queue_fixed',['--ini',qini,'--worktodo',queue,'--once'],False)]:
        if name=='queue_fixed':queue.write_text(line.replace(',0,0,1,',f',{b2},0,1,'))
        result=out/(name+'.jsonl');log=out/(name+'.log')
        # Queue fixed uses the measured D explicitly; global auto ignores its missing profile.
        if name=='queue_fixed':args+=['--d',d,'--cost-profile',out/'unused_missing.cprof','--arena-mb',4096]
        invoke(name,[*args,'--factor-only','--results',result,'--log',log],extra_env={'NTT_FOLD_DEVICE_MAX_MB':str(owner),'NTT_D_MODEL':'0'})
        row=json.loads(result.read_text());actual[name]=row;raw=log.read_text(encoding='utf-8')
        check(row['auto_b2']==is_auto and row['B2']==b2 and row['bad_factors']==0,name+': actual result')
        check(all(t in raw for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')),name+': arithmetic checks')
        check(all(pow(2,8191,int(f))==1 for f in row['factors']),name+': exact divisors')
        leaves[name]=fields(raw,'descent_values')['hash']
        if is_auto:
            check(row['requested_B2']==0 and row['requested_D']==0 and row['auto_planning_seconds']>0,name+': request/selection metadata')
            check(row['auto_plan']['D']==d and row['auto_plan']['B2']==b2,name+': worker selection')
        if name.startswith('queue_'):check('ECMSTAGE2' not in queue.read_text(),name+': queue consumed on success')
    check(len(set(leaves.values()))==1,'same-binary auto/manual/queue leaf equality')
    check(all(row['factors']==actual['actual_auto']['factors'] for row in actual.values()),'same-binary factor equality')
    check(finished.read_text()==line+line.replace(',0,0,1,',f',{b2},0,1,'),'finished keeps original tasks')
    for bits in sorted(runtime_bits-{8191}):
        selected=plans[str(bits)];pair=[];hashes=[]
        for mode in ('auto','fixed'):
            name=f'actual_m{bits}_{mode}';result=out/(name+'.jsonl');log=out/(name+'.log')
            args=['--save',a.save_dir.resolve()/f'm{bits}.save']
            if mode=='auto':args+=['--auto-b2','--cost-profile',profile]
            else:args+=['--b2',selected['B2'],'--d',selected['D'],'--arena-mb',selected['arena_mb']]
            invoke(name,[*args,'--factor-only','--results',result,'--log',log],
                extra_env={'NTT_FOLD_DEVICE_MAX_MB':str(selected['owner_runtime_mb']),'NTT_D_MODEL':'0'})
            row=json.loads(result.read_text());text=log.read_text()
            check(row['bad_factors']==0 and row['B2']==selected['B2'] and row['auto_b2']==(mode=='auto'),name+': arithmetic/result')
            check(all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')),name+': required checks')
            check(all(1<int(f)<(1<<bits)-1 and pow(2,bits,int(f))==1 for f in row['factors']),name+': divisors')
            actual[name]=row;pair.append(row['factors']);hashes.append(fields(text,'descent_values')['hash']);leaves[name]=hashes[-1]
        check(pair[0]==pair[1] and hashes[0]==hashes[1],f'M{bits}: auto/manual equality')
    summary=dict(passed=True,checks=checks,calls=len(rows),binary_sha256=binary_sha,profile_sha256=profile_sha,
        save_sha256=save_sha,plans=plans,actual=actual,leaf_hashes=leaves,cases=rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=checks,calls=len(rows),leaf_hashes=leaves),indent=2))


if __name__=='__main__':main()
