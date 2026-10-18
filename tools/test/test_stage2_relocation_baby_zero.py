"""A valid Edwards d=-1 baby infinity must bypass relocation on host and device."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import re
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--expect-old-miss',action='store_true',help='Reproduce the pre-fix classification without calling it a passing gate')
    p.add_argument('--counterexample',type=Path,default=Path('data/experiments/stage2_relocation_feasibility_20261017/gcd_collision.json'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh evidence directory')
    source=json.loads(a.counterexample.read_text());c=source['curves'][0];n=c['prime']
    x=c['weierstrass_point'][0]*pow(2,-1,n)%n
    # Q has prime order29663. j=29663 belongs to J for D=180180, hence Z=0 mod N.
    assert n==1186519 and c['residual_order']==29663
    report=dict(complete=False,expected_old_miss=a.expect_old_miss,probe_sha256=sha(a.probe),
                source_sha256=sha(__file__),fixture_sha256=sha(a.counterexample),runs=[])
    leaves=[]
    for device_baby in (0,1):
        for relocation in (0,1):
            env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
            env.update(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_ALLOC_FAIL=str(1-device_baby),
                       NTT_BABY_DEVICE_CHECK='1',NTT_TARGET_LEAF_HASH='1')
            cmd=list(map(str,[a.probe.resolve(),f'{n:x}',f'{x:x}',f'{pow(2,-1,n):x}',20,6486480,180180,a.device,relocation,0]))
            r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,timeout=300)
            name=f'baby{device_baby}_relocation{relocation}.log';(out/name).write_bytes(r.stdout+r.stderr)
            assert r.returncode==0,name
            text=(r.stdout+r.stderr).decode('utf-8',errors='replace')
            line=text.split('relocation_probe_result: ')[-1].splitlines()[0]
            result=json.loads(line);assert result['factors']==[]
            leaf=re.findall(r'^target_descent_values: (.*)$',text,re.M)[-1];leaves.append(leaf)
            if relocation:
                if a.expect_old_miss:assert result['probe_seconds']>0
                else:assert result['replayed'] and result['probe_seconds']==0 and result['first']==1
            else:assert not result['replayed'] and result['first']==1
            report['runs'].append(dict(device_baby=device_baby,relocation=relocation,result=result,log=name,sha256=sha(out/name)))
            print('REPRODUCED' if a.expect_old_miss else 'PASS',name,flush=True)
    if not a.expect_old_miss:assert len(set(leaves))==1
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
