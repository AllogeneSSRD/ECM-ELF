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
    p.add_argument('--outer-unroll-u',type=int,choices=(0,4),default=0,
                   help='Experimental NTT schedule requires legacy D fallback for both point modes')
    p.add_argument('--carry-check-fused',type=int,choices=(0,1),default=0,
                   help='Experimental fused carry diagnostics require legacy D fallback')
    p.add_argument('--payload-accounting-v2',action='store_true',
                   help='Current payload ledger disables legacy cache-rate profiles for both point modes')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    manifest=json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    assert manifest.get('outer_unroll_u',0)==a.outer_unroll_u
    assert not ((a.outer_unroll_u or a.carry_check_fused or a.payload_accounting_v2) and a.calibrated_point_model)
    root=Path(__file__).resolve().parents[2]
    sources={r[1]:r[2].lower() for line in manifest['sources']
             if (r:=re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})',line))}
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha==manifest['sha256'].lower()
        for name,want in sources.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
    verify()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_XADD6_TEST='1',NTT_CARRY_CHECK_FUSED=str(a.carry_check_fused))
    runs=[]
    for bits in (61,127,521,607,1279,2203,4423):
        n=2**bits-1
        for mode in (0,1):
            name=f'{bits}_{mode}';save=out/(name+'.save');x=2
            checksum=1000*26*n*x%4294967291
            save.write_text(f'METHOD=ECM; SIGMA=26; B1=1000; N=(2^{bits}-1); X=0x2; CHECKSUM={checksum};\n')
            cmd=[str(exe),'--save',str(save),'--b2','2000','--d','210','--device',str(a.device),'--results',str(out/(name+'.jsonl')),'--log',str(out/(name+'_engine.log'))]
            verify()
            r=subprocess.run(cmd,env=env|{'NTT_POINT_MERSENNE':str(mode)},capture_output=True,timeout=180)
            verify()
            (out/(name+'_driver.log')).write_bytes(r.stdout+r.stderr)
            assert r.returncode==0,(name,r.stderr[-2000:])
            text=(out/(name+'_engine.log')).read_text(encoding='utf-8',errors='replace')
            if a.outer_unroll_u:assert f'ntt_outer_schedule: unroll_u={a.outer_unroll_u}' in text
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
        want='d_model: requested=1 enabled=0 version=legacy_56_1' if (a.outer_unroll_u or a.carry_check_fused or a.payload_accounting_v2) else \
             'd_model: requested=1 enabled=1 version=resident_fixed_ptx_v1' if mode==0 else \
             'd_model: requested=1 enabled=1 version=resident_point_fold_v1' if a.calibrated_point_model else \
             'd_model: requested=1 enabled=0 version=legacy_56_1'
        assert want in text and f'point_mersenne_mode: requested={mode} enabled={mode}' in text
        if a.carry_check_fused or 'tools/bench/ntt_carry_partial.cuh' in sources:
            carry=dict(re.findall(r'(\w+)=(\d+)',re.search(r'ntt_carry_check_stats: (.*)',text)[1]))
            assert int(carry['requested'])==a.carry_check_fused
            assert int(carry['fused_calls'])>0 if a.carry_check_fused else int(carry['fused_calls'])==0
        if a.outer_unroll_u:assert f'ntt_outer_schedule: unroll_u={a.outer_unroll_u}' in text
        assert 'hash=1689529688547722991' in text and 'gmp_check_bad=0' in text and 'pending=0' in text
        result=json.loads((out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()[-1]);assert result['bad_factors']==0 and result['factors']==[]
        runs.append(dict(name=name,command=cmd,result=result));print(name,'passed',flush=True)
    assert hashlib.sha256(real_save.read_bytes()).hexdigest()==save_sha
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
    verify()
    summary=dict(exe=str(exe),sha256=sha,manifest=manifest,carry_check_fused=a.carry_check_fused,runs=runs,passed=len(runs),failed=0,scope='Synthetic saved X=2 for primitive/dispatch checks, not claimed valid production Stage1 curves; GMP Mont and exact xADD aliases checked in the actual native executable.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(passed=len(runs),failed=0,sha256=sha)))


if __name__=='__main__':main()
