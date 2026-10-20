"""Reuse an older format-8 production profile without resetting its statistics.

Runs one verified NF point on GPU1 in a fresh directory. The source profile,
save and binary are read only; this checks integration, not performance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from stage2_tune_matrix import utc_updated_at
from stage2_tune_tables import summaries

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('exe','save','profile','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identities={str(f.resolve()):sha(f) for f in (a.exe,a.save,a.profile)}
    point=next(s for s in a.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    saved=out/'point.save';saved.write_text(point+'\n',encoding='utf-8')
    profile=out/'tune.toml';shutil.copy2(a.profile,profile)
    original=tomllib.loads(profile.read_text(encoding='utf-8-sig'))
    before=summaries(original)
    ordinary=next(s for s in before if s['benchmark_kind']=='runtime_save' and not s['carrier_exponent'])
    count=ordinary['repeats'];assert 2<=count<1000
    ini=out/'ecm.toml';ini.write_text('user=\ncomputer=\nverbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    common=[str(a.exe.resolve()),'--ini',str(ini),'--device',str(a.device),'--save',str(saved),
        '--curves','1','--b2',str(ordinary['b2']),'--d',str(ordinary['d']),'--carrier-exponent','0',
        '--batch-mb','64','--arena-mb','6300','--owner-budget-mb','640','--short-calibration','0',
        '--factor-only','--tune-profile',str(profile),'--submission',str(out/'results.json.txt')]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    def call(name,args):
        command=common+args
        (out/(name+'.command.json')).write_text(json.dumps(command,indent=2),encoding='utf-8')
        r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,errors='replace',timeout=300)
        text=r.stdout+r.stderr;(out/(name+'.log')).write_text(text,encoding='utf-8')
        assert not r.returncode and 'tune_profile_reset:' not in text,(name,text)
        return [json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
    initial_hash=sha(profile)
    plan=next(s for s in call('read_only',['--tune-runtime','0','--plan-only']) if s.get('type')=='tune_selection')
    assert plan['model']=='runtime_shape_mean_v2' and plan['runtime_samples']==count and not plan['runtime_probe']
    assert sha(profile)==initial_hash
    results=out/'internal.jsonl'
    call('extend',['--tune-runtime','1','--tune-runtime-samples',str(count+1),
        '--results',str(results),'--log',str(out/'engine.log')])
    row=json.loads(results.read_text(encoding='utf-8'))
    assert row['status']=='stage2_completed' and not row['factors'] and not row['bad_factors']
    assert row['tune_plan']['runtime_samples']==count and not row['tune_plan']['runtime_probe']
    sample=row['tune_observation']['sample']
    after_doc=tomllib.loads(profile.read_text(encoding='utf-8'))
    after=summaries(after_doc)
    assert len(after_doc['condition'])==len(original['condition']) and len(after)==len(before)
    updated=next(s for s in after if s['benchmark_kind']=='runtime_save' and not s['carrier_exponent'])
    assert updated['repeats']==count+1
    version=re.search(r'inline constexpr char version\[\] = "([^"]+)"',
        (ROOT/'src/core/ecm_program_version.h').read_text(encoding='utf-8'))[1]
    assert ordinary['program_version']!=version and updated['program_version']==version
    assert updated['updated_at']==utc_updated_at(sample['runtime_completed_ns'])
    delta=sample['mean_seconds']-ordinary['mean_seconds']
    mean=ordinary['mean_seconds']+delta/(count+1)
    deviation=math.sqrt(((count-1)*ordinary['stddev_seconds']**2+delta**2*count/(count+1))/count)
    assert math.isclose(updated['mean_seconds'],mean,rel_tol=1e-12)
    assert math.isclose(updated['stddev_seconds'],deviation,rel_tol=1e-12)
    for old in before:
        if not old['carrier_exponent']:continue
        retained=next(s for s in after if s['carrier_exponent']==old['carrier_exponent'])
        assert retained['repeats']==old['repeats'] and retained['program_version']==old['program_version']
        assert not retained.get('updated_at'),'unknown legacy sampling time was invented'
    assert all('program_version' not in c for c in after_doc['condition'].values())
    final_hash=sha(profile)
    plan=next(s for s in call('read_only_after',['--tune-runtime','0','--plan-only']) if s.get('type')=='tune_selection')
    assert plan['runtime_samples']==count+1 and sha(profile)==final_hash
    assert all(sha(f)==h for f,h in identities.items())
    report=dict(complete=True,device=a.device,identities=identities,source_version=ordinary['program_version'],
        version=version,old_samples=count,new_samples=count+1,read_only_preserves_file=True,
        no_recalibration=True,pooled_statistics=True,untouched_row_source_preserved=True,
        current_row_timestamp=True,CUDA_curves=1,comparison='functional integration; no performance claim')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
