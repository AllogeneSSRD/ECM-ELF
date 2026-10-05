"""Actual saved Stage2 dispatch/GMP/xADD gates for the optional point fold."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--real-save',type=Path,required=True,help='Previously validated production M4423 save for model scope checks')
    p.add_argument('--calibrated-point-model',action='store_true',help='Expect separate profile6 in the exact real-save scope')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_XADD6_TEST='1')
    runs=[]
    for bits in (61,127,521,607,1279,2203,4423):
        n=2**bits-1
        for mode in (0,1):
            name=f'{bits}_{mode}';save=out/(name+'.save');x=2
            checksum=1000*26*n*x%4294967291
            save.write_text(f'METHOD=ECM; SIGMA=26; B1=1000; N=(2^{bits}-1); X=0x2; CHECKSUM={checksum};\n')
            cmd=[str(exe),'--save',str(save),'--b2','2000','--d','210','--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
            r=subprocess.run(cmd,env=env|{'NTT_POINT_MERSENNE':str(mode)},capture_output=True,timeout=180)
            (out/(name+'_driver.log')).write_bytes(r.stdout+r.stderr)
            assert r.returncode==0,(name,r.stderr[-2000:])
            text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
            for token in (f'point_mersenne_mode: requested={mode} enabled={mode} bits={bits}', 'mont_selftest: cases=2048 mismatches=0','xadd6_selftest: cases=1280','bad=0 first_bad=0','stage1_skipped=1','gmp_check_bad=0','clean=0'):
                assert token in text,(name,token)
            result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1]);assert result['bad_factors']==0
            runs.append(dict(name=name,command=cmd,mode=mode,save_sha256=hashlib.sha256(save.read_bytes()).hexdigest(),result=result))
            print(name,'passed',flush=True)
    # Exact-shape rejection: odd generic N, including a partly filled top limb.
    for bits in (128,130):
        n=2**bits+1;x=2;name=f'generic_{bits}'
        save=out/(name+'.save');save.write_text(f'METHOD=ECM; SIGMA=26; B1=1000; N={n}; X={x}; CHECKSUM={1000*26*n*x%4294967291};\n')
        cmd=[str(exe),'--save',str(save),'--b2','2000','--d','210','--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
        r=subprocess.run(cmd,env=env|{'NTT_POINT_MERSENNE':'1'},capture_output=True,timeout=180);(out/(name+'_driver.log')).write_bytes(r.stdout+r.stderr)
        assert r.returncode==0,name
        text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
        assert 'point_mersenne_mode: requested=1 enabled=0' in text and 'mont_selftest: cases=2048 mismatches=0' in text
        result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1]);assert result['bad_factors']==0
        runs.append(dict(name=name,command=cmd,result=result));print(name,'passed',flush=True)
    real_save=a.real_save.resolve();save_sha=hashlib.sha256(real_save.read_bytes()).hexdigest()
    clean={k:v for k,v in env.items() if k!='NTT_XADD6_TEST'};clean['NTT_ARENA_CAP_KB']='6451200'
    for mode in (0,1):
        name=f'model_scope_{mode}'
        cmd=[str(exe),'--save',str(real_save),'--b2','100000000000','--d','390390','--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
        r=subprocess.run(cmd,env=clean|{'NTT_POINT_MERSENNE':str(mode)},capture_output=True,timeout=180)
        (out/(name+'_driver.log')).write_bytes(r.stdout+r.stderr);assert r.returncode==0,name
        text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
        want='d_model: requested=1 enabled=1 version=resident_fixed_ptx_v1' if mode==0 else \
             'd_model: requested=1 enabled=1 version=resident_point_fold_v1' if a.calibrated_point_model else \
             'd_model: requested=1 enabled=0 version=legacy_56_1'
        assert want in text and f'point_mersenne_mode: requested={mode} enabled={mode}' in text
        assert 'hash=1689529688547722991' in text and 'gmp_check_bad=0' in text and 'pending=0' in text
        result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1]);assert result['bad_factors']==0 and result['factors']==[]
        runs.append(dict(name=name,command=cmd,result=result));print(name,'passed',flush=True)
    assert hashlib.sha256(real_save.read_bytes()).hexdigest()==save_sha
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
    summary=dict(exe=str(exe),sha256=sha,runs=runs,passed=len(runs),failed=0,scope='Synthetic saved X=2 for primitive/dispatch checks, not claimed valid production Stage1 curves; GMP Mont and exact xADD aliases checked in the actual native executable.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(passed=len(runs),failed=0,sha256=sha)))


if __name__=='__main__':main()
