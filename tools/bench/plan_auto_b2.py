"""Plan Auto B2 through the audited native v2 selector; executes no curves.

The JSON model identifies the calibration, and --runtime-profile identifies
its validated .cprof export. The binary owns search, packing and live admission.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--runtime-profile',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True);p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage1-batch',type=int,default=1)
    p.add_argument('--stage1-seconds-per-curve',type=float);p.add_argument('--ratio-adjust',type=float,default=1)
    p.add_argument('--owner-budget-mb',type=int,default=640);p.add_argument('--arena-mb',type=int,default=4096)
    p.add_argument('--min-b2',type=int);p.add_argument('--max-b2',type=int);p.add_argument('--d',type=int)
    a=p.parse_args()
    if a.output.resolve() in {path.resolve() for path in (a.profile,a.runtime_profile,a.save,a.stage2)}:
        raise ValueError('Plan output must differ from input files and executable')
    model=json.loads(a.profile.read_text(encoding='utf-8'))
    if model['schema']!=2 or model['feature_profile']!=7:raise ValueError('Use an exact-tree v2 model')
    if sha(a.stage2)!=model['identity']['stage2_sha256']:raise ValueError('Model/binary mismatch')
    rows=a.runtime_profile.read_text(encoding='utf-8-sig').splitlines()
    identity=next(line.split() for line in rows if line.startswith('identity '))
    if rows[0]!='ECM_STAGE2_COST_PROFILE 2' or len(identity)!=15 or identity[13]!=sha(a.profile):
        raise ValueError('Runtime export does not identify this JSON model')
    for value in (a.ratio_adjust,a.stage1_seconds_per_curve):
        if value is not None and (not math.isfinite(value) or value<=0):raise ValueError('Seconds/ratio must be finite and positive')
    cmd=[str(a.stage2.resolve()),'--save',str(a.save.resolve()),'--device','1','--curves','1',
         '--auto-b2','--cost-profile',str(a.runtime_profile.resolve()),'--plan-only',
         '--stage1-batch',str(a.stage1_batch),'--stage2-ratio-adjust',format(a.ratio_adjust,'.17g'),
         '--owner-budget-mb',str(a.owner_budget_mb),'--arena-mb',str(a.arena_mb)]
    for key,value in [('--stage1-seconds-per-curve',a.stage1_seconds_per_curve),('--auto-min-b2',a.min_b2),
                      ('--auto-max-b2',a.max_b2),('--d',a.d)]:
        if value is not None:cmd.extend([key,format(value,'.17g') if isinstance(value,float) else str(value)])
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    run=subprocess.run(cmd,capture_output=True,env=env,timeout=60)
    if run.returncode:raise RuntimeError(run.stderr.decode(errors='replace'))
    choice=next(json.loads(line) for line in run.stdout.decode().splitlines() if line.startswith('{'))
    if choice['schema']!=2 or choice['profile_sha256']!=sha(a.runtime_profile):raise ValueError('Unexpected native response')
    result=dict(schema=2,scope='native_measured_scope_v2',executed_curves=0,binary_sha256=sha(a.stage2),
        profile_sha256=sha(a.profile),runtime_profile_sha256=sha(a.runtime_profile),chosen=choice,
        range_limited=choice['range_limited'],process_peak_guaranteed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
