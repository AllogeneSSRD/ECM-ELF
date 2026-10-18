"""Time a real task's ordinary/carrier default references with no implicit warmup.

Reads production inputs, freezes identities and writes all measurements into a
fresh evidence directory. Never consumes the production queue or changes power.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib

from stage2_tune_tables import summaries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--production',type=Path,required=True)
    p.add_argument('--exponent',type=int,default=12011)
    p.add_argument('--d',type=int,default=240240)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true',help='Validate retained measurements without running GPU again.')
    a=p.parse_args(); out=a.output.resolve()
    if not a.analyze_only:out.mkdir(parents=True,exist_ok=False)
    production=a.production.resolve();exe=a.exe.resolve()
    before={name:sha(production/name) for name in ('ecm.ini','stage2_worktodo.txt','stage2_worktodo.txt.progress','stage2_ecm_tune.toml') if (production/name).exists()}
    queue=out/'worktodo.txt';profile=out/'stage2_ecm_tune.toml'
    ini=out/'ecm.ini'
    if not a.analyze_only:
        rows=(production/'stage2_worktodo.txt').read_text(encoding='utf-8-sig').splitlines()
        task=next(s for s in rows if s.startswith('ECMSTAGE2=') and f',2,{a.exponent},-1,' in s)
        queue.write_text(task+'\n',encoding='utf-8')
        shutil.copyfile(production/'stage2_ecm_tune.toml',profile)
        original=(production/'ecm.ini').read_text(encoding='utf-8-sig')
        assert 'stage2_tune_warmup' not in original, 'This gate checks the missing-key default; use an INI without a warmup override.'
        ini.write_text(original+'\nstage2_save_dir='+str(production)+'\nstage2_worktodo='+str(queue)+
                       '\nstage2_tune_profile='+str(profile)+'\n',encoding='utf-8')
    command=[str(exe),'--ini',str(ini),'--tune','ecm','--tune-file',str(profile),
             '--tune-d',str(a.d),'--tune-repeats',str(a.repeats),'--tune-budget-seconds','600']
    assert '--tune-warmup' not in command and '--tune-carrier-exponent' not in command
    if not a.analyze_only:(out/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
    identity={'exe_sha256':sha(exe),'production':str(production),'production_before':before,'command':command,
              'power_clock_changes':False,'repeats_requested':a.repeats}
    if a.analyze_only:
        identity=json.loads((out/'identity.json').read_text(encoding='utf-8'));before=identity['production_before']
    else:
        (out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k not in ('CUDA_VISIBLE_DEVICES',)}
    log=out/'console.log'; started=time.monotonic()
    if not a.analyze_only:
        with log.open('wb') as stream:
            r=subprocess.run(command,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    text=log.read_text(encoding='utf-8',errors='replace')
    assert ('ecm_tune_complete:' in text) if a.analyze_only else r.returncode==0,text[-3000:]
    assert 'source=worktodo' in text and 'auto_carrier=1 warmup=0' in text,text
    runs=list((out/'data/experiments').glob('portable_ecm_tune_*'))
    assert len(runs)==1
    evidence=runs[0]
    receipts=[f for f in evidence.glob('measure_*.jsonl') if not f.name.endswith('.plan.jsonl')]
    assert len(receipts)==2*a.repeats,(len(receipts),receipts)
    assert not list(evidence.glob('measure_*_0.jsonl')) and not list(evidence.glob('measure_*_0.log'))
    data=tomllib.loads(profile.read_text(encoding='utf-8'))
    plans=[json.loads(line) for path in evidence.glob('measure_*.plan.jsonl') for line in path.read_text(encoding='utf-8').splitlines()]
    assert len({s['B2'] for s in plans})==1 and len({s['target_bits'] for s in plans})==1
    target_bits=plans[0]['target_bits'];b2=plans[0]['B2']
    measured=[s for s in summaries(data) if s['target_bits']==target_bits and s['b2']==b2 and s['d']==a.d and s['source']=='measured']
    assert {s['carrier_exponent'] for s in measured}=={0,a.exponent},measured
    for s in measured:
        assert s['warmup_runs']==0 and s['repeats']==a.repeats and s['clean']==1 and s['bad']==0 and s['hits']==0
    assert sha(exe)==identity['exe_sha256']
    assert before=={name:sha(production/name) for name in before}, 'production inputs changed during the gate'
    for path in receipts:
        receipt=json.loads(path.read_text(encoding='utf-8'))
        assert receipt['clean'] and receipt['bad']==0 and receipt['selftest_cases']>0 and receipt['checked']>0
    import re
    budget_elapsed=float(re.search(r'ecm_tune_complete:.*budget_elapsed=([\d.]+)',text)[1])
    report={'passed':True,'identity':identity,'producer_budget_elapsed_seconds':budget_elapsed,
            'warmups_executed':0,'formal_curve_calls':len(receipts),'profile_sha256':sha(profile),'profile':str(profile),
            'summaries':measured,'evidence':str(evidence)}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'formal_curve_calls':len(receipts),'warmups_executed':0,
                      'summaries':[{k:s[k] for k in ('carrier_exponent','mean_seconds','stddev_seconds','repeats')} for s in measured]},indent=2))


if __name__=='__main__':
    main()
