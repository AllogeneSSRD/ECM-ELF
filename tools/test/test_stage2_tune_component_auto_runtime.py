"""Exercise embedded NTT costs through production Auto B2 and its INI worker.

Provided T1 is a decision-test input, not a Stage1 performance measurement.
One full curve checks execution; it is not a repeated timing benchmark.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from stage2_tune_route_cost import predict_route
from analyze_stage2_tune_workload import embedded_ntt_samples, one_json, workload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('exe','profile','save','training-plans','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--b2',type=int,default=12500000000)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities={str(path.resolve()):sha(path) for path in (a.exe,a.profile,a.save,Path(__file__))}
    for name in ('stage2_tune_route_cost.py','analyze_stage2_tune_workload.py'):
        path=ROOT/'tools/bench'/name;identities[str(path)]=sha(path)
    profile=tomllib.loads(a.profile.read_text(encoding='utf-8-sig'))
    assert embedded_ntt_samples(profile)
    first=next(iter(profile['ecm'].values()));b1=first['b1']
    assert all(s['b2']!=a.b2 for s in profile['ecm'].values())
    policy=profile['policy'];env=dict(os.environ)
    for k,v in policy['environment'].items():env['NTT_'+k.upper()]=str(v)
    ini=out/'ecm.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\nstage2_b2=0\nstage2_auto_b2=1\n'
        'stage1_seconds_per_curve=3\nstage2_tune_profile='+str(a.profile.resolve())+'\n',encoding='utf-8')
    common=[str(a.exe.resolve()),'--ini',str(ini),'--save',str(a.save.resolve()),'--device',str(a.device),
        '--batch-mb',str(policy['batch_mb']),'--arena-mb',str(policy['arena_mb']),
        '--owner-budget-mb',str(policy['fold_mb']),'--log-level','quiet']
    cases={}

    def run(name,extra=(),fixed=True,receipt_path=None):
        command=common+list(map(str,extra))
        if fixed:command+=['--auto-min-b2',str(a.b2),'--auto-max-b2',str(a.b2)]
        proc=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,errors='replace',timeout=600)
        (out/(name+'.console.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert proc.returncode==0,(name,proc.stderr)
        rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
        choice=(json.loads(receipt_path.read_text(encoding='utf-8'))['auto_plan'] if receipt_path else
                next(row for row in rows if row.get('type')=='stage2_auto_plan'))
        assert choice['source']=='full_ecm_tune' and choice['T1']==3 and choice['T1_source']=='provided_per_curve'
        benefit=.11343+.88657*(math.log10(choice['B2']/b1)/2)**(1.96617-.06781*math.log10(b1))
        assert math.isclose(choice['K'],benefit,rel_tol=1e-12)
        assert math.isclose(choice['score'],benefit/(3+choice['guarded_engine_seconds']),rel_tol=1e-12)
        assert not any(row.get('type')=='tune_selection' for row in rows)
        if fixed:assert choice['B2']==a.b2
        cases[name]=choice
        return choice

    selected=run('ini_fixed_range',['--plan-only'])
    assert selected['model']=='phase_ntt_loop_v1'
    ordinary=run('ordinary_lock',['--plan-only','--carrier-exponent','0'])
    assert ordinary['carrier_exponent']==0 and ordinary['model']=='phase_ntt_loop_v1'
    carrier=selected['carrier_exponent'];d=selected['D']
    locked=run('d_and_carrier_lock',['--plan-only','--d',d,'--carrier-exponent',carrier])
    assert locked['D']==d and locked['carrier_exponent']==carrier and locked['model']=='phase_ntt_loop_v1'
    run('unrestricted',['--plan-only'],False)
    exact=next(s['b2'] for s in profile['ecm'].values() if s['d']==d and s['carrier_exponent']==carrier)
    measured=run('exact_anchor',['--plan-only','--auto-min-b2',exact,'--auto-max-b2',exact,'--d',d,
        '--carrier-exponent',carrier],False)
    assert measured['B2']==exact and measured['model']=='measured_exact_scope_v1'

    # Remove one actual training shape; a partial reference must never rank.
    group=[s for s in profile['ecm'].values() if s['d']==ordinary['D'] and s['carrier_exponent']==0]
    sample=group[0];plans=[]
    for path in a.training_plans.glob('case_*.plan.jsonl'):
        plan=one_json(path)
        if (plan['D'],plan['B2'],plan['carrier_exponent'],plan['bits'])==(sample['d'],sample['b2'],0,sample['arithmetic_bits']):
            plans.append((path,plan))
    assert len(plans)==1
    path,plan=plans[0];identities[str(path.resolve())]=sha(path)
    row=workload(plan)[0];section=f"ntt.length_{row['length']}.slices_{row['slices']}"
    text=a.profile.read_text(encoding='utf-8-sig')
    changed,count=re.subn(r'(?m)^\['+re.escape(section)+r'\]\n.*?(?=^\[|\Z)','',text,flags=re.S)
    assert count==1
    changed,count=re.subn(r'(?m)^ntt_measured = \d+$','ntt_measured = '+str(profile['summary']['ntt_measured']-1),changed)
    assert count==1
    fallback=out/'partial.toml';fallback.write_text(changed,encoding='utf-8')
    choice=run('missing_shape_fallback',['--plan-only','--tune-profile',fallback,'--d',ordinary['D'],'--carrier-exponent','0'])
    expected=predict_route(group,a.b2,profile['profile'].get('prediction_model'))
    assert expected and choice['model']=='giant_route_cost_v2'
    assert math.isclose(choice['engine_seconds'],expected['seconds'],rel_tol=1e-10)

    receipt=out/'curve.jsonl';log=out/'curve.log'
    execution=run('ini_curve',['--curves','1','--results',receipt,'--log',log],receipt_path=receipt)
    result=json.loads(receipt.read_text(encoding='utf-8'))
    assert result['status']=='stage2_completed' and result['hits']==result['bad_factors']==0
    assert result['requested_D']==0 and result['requested_carrier_exponent']==0
    assert result['auto_plan']['model']=='phase_ntt_loop_v1' and result['carrier_exponent']==selected['carrier_exponent']
    assert execution['D']==selected['D']
    text=log.read_text(encoding='utf-8')
    assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text and re.search(r'stage2_full_wall:.*clean=1',text)
    assert all(sha(Path(path))==digest for path,digest in identities.items())
    report=dict(complete=True,plan_cases=6,ini_curve=True,explicit_locks=True,exact_anchor_priority=True,
        missing_shape_route_fallback=True,provided_t1_not_measured=True,cases=cases,identities=identities)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','identities')}))


if __name__=='__main__':main()
