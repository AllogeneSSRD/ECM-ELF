"""Verify measured T1 in native Auto B2, overrides, INI and private queues."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[2]


def queue_task(n,exponent):
    if type(exponent) is not int or not 2<=exponent<=16384 or type(n) is not int or n<=3 or not n&1:
        raise ValueError('invalid target or queue exponent')
    original=(1<<exponent)-1
    if original%n:raise ValueError('queue target must divide the explicitly supplied Mersenne number')
    factor=original//n
    factors='' if factor==1 else str(factor)
    return f'ECMSTAGE2=stage1-tune,1,2,{exponent},-1,"input.save",0,0,2,"{factors}"\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['exe','ecm-profile','stage1-profile','save','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--choose12-profile',type=Path,help='optional matching choose12 cost profile; tests enabled when supplied')
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--queue-exponent',type=int,default=521,help='Explicit Mersenne origin for the private queue; defaults to the M521 fixture')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=[a.exe,a.ecm_profile,a.stage1_profile,a.save]
    if a.choose12_profile:paths.append(a.choose12_profile)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identity={str(path.resolve()):sha(path) for path in paths}
    s1=tomllib.loads(a.stage1_profile.read_text(encoding='utf-8'))
    c12=tomllib.loads(a.choose12_profile.read_text(encoding='utf-8')) if a.choose12_profile else None
    with a.save.open(encoding='utf-8-sig') as stream:line=next(x for x in stream if x.strip())
    fields=dict(re.findall(r'(\w+)\s*=\s*([^;]+)',line));n=int(fields['N'].strip(),0)
    task=queue_task(n,a.queue_exponent)
    assert {s['batch'] for s in s1['stage1'].values()}=={1,8}
    ini=out/'bench.ini';ini.write_text('verbose=false\nexponent=lcm\n',encoding='utf-8')
    common=[str(a.exe.resolve()),'--ini',str(ini),'--device',str(a.device),'--batch-mb','256',
            '--arena-mb','6300','--owner-budget-mb','640','--log-level','quiet']
    calls=[];plans={}
    def run(name,options,ok=True,reason=''):
        proc=subprocess.run(common+list(map(str,options)),cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=300)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert (proc.returncode==0)==ok,(name,proc.stdout,proc.stderr)
        assert not reason or reason in proc.stdout+proc.stderr,(name,reason)
        calls.append(dict(name=name,success=ok))
        return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]
    auto=['--save',a.save.resolve(),'--auto-b2','--tune-profile',a.ecm_profile.resolve(),
          '--stage1-tune-profile',a.stage1_profile.resolve()]
    def select(name,batch=1,extra=(),profile=s1,exponent='lcm'):
        rows=run(name,[*auto,'--stage1-batch',batch,'--plan-only',*extra])
        choice=next(r for r in rows if r.get('type')=='stage2_auto_plan')
        expected=next(s for s in profile['stage1'].values() if s['batch']==batch)
        assert choice['T1_source']=='measured_stage1_profile' and choice['stage1_exponent']==exponent
        assert choice['T1']==expected['median_seconds'] and choice['stage1_batch']==batch
        assert math.isclose(choice['score'],choice['K']/(choice['T1']+choice['ratio_adjust']*choice['guarded_engine_seconds']),rel_tol=1e-12)
        actual=next(r for r in rows if r.get('type')=='stage2_plan')
        assert all(choice[k]==actual[k] for k in ['B2','D','carrier_exponent'])
        plans[name]=choice;return choice
    one=select('batch1');eight=select('batch8',8)
    assert eight['T1']<one['T1']
    if a.choose12_profile:
        select('choose12',8,['--stage1-tune-profile',a.choose12_profile.resolve(),'--stage1-exponent','choose12'],c12,'choose12')
        cfg=out/'choose12.ini';cfg.write_text('verbose=false\nexponent=choose12\n',encoding='utf-8')
        select('ini_choose12',8,['--ini',cfg,'--stage1-tune-profile',a.choose12_profile.resolve()],c12,'choose12')
    rows=run('provided_priority',[*auto,'--stage1-seconds-per-curve','3','--stage1-tune-profile',out/'absent.toml','--plan-only'])
    provided=next(r for r in rows if r.get('type')=='stage2_auto_plan')
    assert provided['T1']==3 and provided['T1_source']=='provided_per_curve' and 'stage1_profile_sha256' not in provided
    for name,extra,reason in [
        ('missing_batch',['--stage1-batch','12'],'no matching Stage1 tune'),
        ('mismatched_exponent',['--stage1-exponent','choose12'],'no matching Stage1 tune'),
        ('missing_file',['--stage1-tune-profile',out/'absent.toml'],'cannot lock Stage1 tune'),
        ('output_conflict',['--log',a.stage1_profile.resolve()],'conflicts with Stage1 tune'),
    ]:run(name,[*auto,'--plan-only',*extra],False,reason)
    broken=out/'device_mismatch.toml'
    broken.write_text(a.stage1_profile.read_text(encoding='utf-8').replace('sm_minor = 9','sm_minor = 6'),encoding='utf-8')
    run('wrong_device',[*auto,'--stage1-tune-profile',broken,'--plan-only'],False,'Stage1 tune device/runtime mismatch')
    run('check_without_gpu',['--device','9999','--check-stage1-tune-profile',a.stage1_profile.resolve()])
    results=out/'actual.jsonl';log=out/'actual_curve.log'
    run('actual',[*auto,'--stage1-batch','8','--curves','1','--results',results,'--log',log])
    row=json.loads(results.read_text(encoding='utf-8'))
    assert row['auto_plan']['T1']==eight['T1'] and row['auto_plan']['T1_source']=='measured_stage1_profile'
    assert row['B2']==eight['B2'] and row['hits']==row['bad_factors']==0 and row['requested_B2']==0
    text=log.read_text(encoding='utf-8');assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
    qdir=out/'queue';qdir.mkdir();qsave=qdir/'input.save';qsave.write_bytes(a.save.read_bytes()*2)
    queue=qdir/'worktodo.txt';queue.write_text(task,encoding='utf-8')
    qini=qdir/'ecm.ini';qini.write_text('verbose=false\nexponent=lcm\ntmp_dir='+str(qdir)+
        '\nstage2_worktodo=worktodo.txt\nstage2_finished=finished.txt\nstage2_auto_b2=1\nstage1_batch=8\n'
        'stage2_tune_profile='+os.path.relpath(a.ecm_profile.resolve(),qdir)+'\nstage1_tune_profile='+os.path.relpath(a.stage1_profile.resolve(),qdir)+
        '\nstage2_results_file=results.jsonl\nstage2_log_file=stage2.log\n',encoding='utf-8')
    run('queue_plan',['--ini',qini,'--worktodo',queue,'--plan-only'])
    assert queue.read_text()==task
    run('queue',['--ini',qini,'--worktodo',queue,'--once'])
    result=qdir/'results.jsonl';rows=[json.loads(x) for x in result.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2 and all(r['auto_plan']['T1_source']=='measured_stage1_profile' and
        r['auto_plan']['T1']==eight['T1'] and r['hits']==r['bad_factors']==0 for r in rows)
    assert not queue.read_text() and task in (qdir/'finished.txt').read_text()
    before=result.read_bytes();run('queue_rerun',['--ini',qini,'--worktodo',queue,'--once'])
    assert result.read_bytes()==before
    assert identity=={str(path.resolve()):sha(path) for path in paths}
    report=dict(passed=True,plans=plans,calls=calls,full_curves=3,arithmetic_bad=0,
                native_stage1_cost=True,provided_cost_priority=True,ini_and_queue=True,source_files_unchanged=True,
                choose12_verified=a.choose12_profile is not None,queue_exponent=a.queue_exponent,target_bits=n.bit_length())
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,plans=len(plans),calls=len(calls),full_curves=3)))


if __name__=='__main__':main()
