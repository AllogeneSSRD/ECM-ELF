"""Replay a verified NF saved point to check automatic runtime D/carrier choice.

This is a functional selection gate, not new independent ECM coverage. Copies
the first point and a legacy profile into a fresh ignored evidence directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
from stage2_tune_tables import summaries
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('exe','save','profile','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--exponent',type=int,required=True);p.add_argument('--b2',type=int,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--curves',type=int,default=12)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identities={str(f.resolve()):sha(f) for f in (a.exe,a.save,a.profile)}
    point=next(s for s in a.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD='))
    n=int(re.search(r'\bN=0x([\da-fA-F]+);',point)[1],16);whole=(1<<a.exponent)-1
    assert whole%n==0 and whole!=n
    saved=out/'points.save';saved.write_text((point+'\n')*a.curves,encoding='utf-8')
    profile=out/'tune.toml';shutil.copy2(a.profile,profile)
    old_samples=len(summaries(tomllib.loads(profile.read_text(encoding='utf-8-sig'))))
    queue=out/'worktodo.txt';queue.write_text(
        f'ECMSTAGE2=1,2,{a.exponent},-1,points.save,{a.b2},0,{a.curves},"{whole//n}"\n',encoding='utf-8')
    ini=out/'ecm.ini';ini.write_text('user=\ncomputer=\nverbose=false\nstage2_finished=finished.txt\n'
        'stage2_tune_runtime_samples=3\nstage2_debug_log=false\nstage2_save_dir='+str(out)+'\n',encoding='utf-8')
    results=out/'internal.jsonl';exe=a.exe.resolve()
    command=[str(exe),'--ini',str(ini),'--device',str(a.device),'--worktodo',str(queue),'--once',
        '--batch-mb','64','--arena-mb','6400','--owner-budget-mb','960','--short-calibration','0',
        '--factor-only','--results',str(results),'--submission',str(out/'results.json.txt'),
        '--tune-profile',str(profile),'--log',str(out/'engine.log')]
    (out/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    with (out/'console.log').open('wb') as log:
        r=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    assert not r.returncode,'inspect console.log'
    rows=[json.loads(s) for s in results.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==a.curves and all(not r['factors'] and not r['bad_factors'] for r in rows)
    assert [r['carrier_exponent'] for r in rows[:2]]==[a.exponent]*2
    assert [r['tune_plan']['reason'] for r in rows[:2]]==['runtime_probe_carrier_first']*2
    final=rows[-3:];assert all(not r['tune_plan']['runtime_probe'] for r in final)
    assert len({(r['d'],r['carrier_exponent']) for r in final})==1,'winner reverted during stable observations'
    samples=summaries(tomllib.loads(profile.read_text(encoding='utf-8-sig')))
    current=[s for s in samples if s.get('measurement_binary_sha256')==identities[str(exe)] and s.get('benchmark_kind')=='runtime_save']
    groups={}
    for s in current:groups.setdefault((s['d'],s['carrier_exponent']),[]).append(s['median_seconds'])
    assert all(2<=len(v)<=3 for v in groups.values()) and len(samples)>old_samples
    assert all(sha(Path(f))==h for f,h in identities.items())
    report=dict(complete=True,device=a.device,identities=identities,curves=len(rows),
        comparison='functional replay of one saved point; no new independent ECM coverage',
        target_bits=n.bit_length(),b2=a.b2,carrier_first=True,stable_winner=dict(d=final[-1]['d'],carrier=final[-1]['carrier_exponent']),
        groups=[dict(d=d,carrier=c,samples=times) for (d,c),times in groups.items()],legacy_samples_retained=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
