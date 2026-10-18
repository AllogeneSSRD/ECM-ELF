"""Edwards-only relocation gates: independent leaves, replay, INI, and exact GCD=N case."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
import test_edwards_gpu_pipeline as ref


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fields(text,prefix):
    lines=[s for s in text.splitlines() if s.startswith(prefix+':')]
    if not lines:raise ValueError('Missing '+prefix)
    return dict(re.findall(r'([\w]+)=([^\s]+)',lines[-1]))


def save(path,n,sigma,b1):
    a,x,z=ref.stage1(n,sigma,b1)
    if math.gcd(x,n)!=1 or math.gcd(z,n)!=1:raise ValueError('Stage1 point is not unit')
    x=x*pow(z,-1,n)%n
    path.write_text(f'METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA={sigma}; B1={b1}; '
                    f'N=0x{n:x}; X=0x{x:x}; STATUS=POINT; CHECKSUM={ref.checksum(n,x,sigma,b1,"POINT")};\n')
    return dict(save=str(path),n=n,x=x,a24=a,sigma=sigma,b1=b1,save_sha256=sha(path))


def leaf_hash(case,b2,d,relocation):
    n,x,a=case['n'],case['x'],case['a24']
    def affine(k):
        u,z=ref.mont.ladder(k,x,1,a,n)
        return u*pow(z,-1,n)%n
    J=[j for j in range(1,d//2+1) if math.gcd(j,d)==1]
    m=next(v for v in range(2,d+2) if math.gcd(v,d)==1)
    last=b2//d+2;first=max(1,last//m) if relocation else 1
    result=[1]*len(J);babies=list(map(affine,J))
    for i in range(first,last+1):
        u=affine(i*d)
        result=[v*(b-u)%n for b,v in zip(babies,result)]
    h=1469598103934665603
    for v in result:
        for w in range((n.bit_length()+63)//64):h=((h^((v>>(w*64))&((1<<64)-1)))*1099511628211)&((1<<64)-1)
    return str(h)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--counterexample',type=Path,default=ROOT/'data/experiments/stage2_relocation_feasibility_20261017/gcd_collision.json')
    p.add_argument('--wide-reference',type=Path,help='Optional frozen Edwards study for a wide chain/index proof')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh evidence directory')
    report=dict(complete=False,performance_samples=0,engine_sha256=sha(a.exe),probe_sha256=sha(a.probe),
                source_sha256=sha(__file__),device=a.device,checks=[])
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_TARGET_LEAF_HASH='1',NTT_GFINV_SEG_CHECK='1',NTT_SCALED_ROOT_CHECK='1',NTT_SCALED_CHECK='1',
               NTT_GIANT_SEED_CHECK='1',NTT_FOLD_DEVICE_CHECK='1',NTT_SCALED_FRONTIER_CHECK='1')
    def persist():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(tag,cmd,good=True,extra_env=None):
        proc=subprocess.run(list(map(str,cmd)),cwd=out,env=env|(extra_env or {}),capture_output=True,timeout=1200)
        path=out/(tag+'.log');path.write_bytes(proc.stdout+proc.stderr)
        text=path.read_text(encoding='utf-8',errors='replace')
        if (proc.returncode==0)!=good:raise ValueError(f'{tag}: exit {proc.returncode}; inspect {path}')
        report['checks'].append(dict(name=tag,exit=proc.returncode,log=str(path),sha256=sha(path)))
        persist();print('PASS',tag,flush=True);return text
    geometry=run('geometry',[a.probe,'--geometry']);assert 'bad=0' in geometry
    ini=out/'test.ini';ini.write_text('user=\ncomputer=\nstage2_tune_runtime=0\nstage2_relocation=1\n')
    def engine(tag,case,b2=13230,d=210,reloc=None,carrier=0,extra=(),extra_env=None):
        folder=out/tag;folder.mkdir();result=folder/'internal.jsonl';debug=folder/'debug.log'
        cmd=[a.exe,'--ini',ini,'--save',case['save'],'--b2',b2,'--d',d,'--device',a.device,'--curves',1,
             '--carrier-exponent',carrier,'--results',result,'--submission',folder/'results.json.txt',
             '--log',folder/'engine.log','--debug-log-file',debug,'--factor-only',
             '--short-calibration',0,'--tune-runtime',0,*extra]
        if reloc is not None:cmd+=['--relocation',reloc]
        text=run(tag,cmd,extra_env=extra_env)+(folder/'engine.log').read_text()+debug.read_text()
        row=json.loads(result.read_text());assert row['curve']=='ed' and row['param'] is None
        assert row['bad_factors']==0 and all(1<int(f)<case['n'] and case['n']%int(f)==0 for f in row['factors'])
        c=fields(text,'s4_multiply_stats')
        assert int(c['gmp_checked']) and int(c['gmp_selftest_cases']) and int(c['gmp_check_bad'])==int(c['gmp_selftest_bad'])==0
        assert fields(text,'real_batched_projective')['leaves']==fields(text,'real_batched_projective')['gamma_points']
        return row,text
    prime=save(out/'prime.save',(1<<127)-1,26,20)
    for b2 in (4830,5040,13230):
        pair=[]
        for mode in (0,1):
            row,text=engine(f'prime_{b2}_{mode}',prime,b2=b2,reloc=mode)
            assert not row['factors'] and not row['relocation_replayed']
            assert fields(text,'target_descent_values')['hash']==leaf_hash(prime,b2,210,mode)
            pair.append(row)
        assert pair[0]['factors']==pair[1]['factors']
    row,text=engine('ini_enabled',prime)
    assert row['relocation'] and row['giant_first']==5
    row,text=engine('no_reduction',prime,b2=120,d=30)
    assert row['relocation'] and row['giant_first']==1 and fields(text,'stage2_relocation')['reason']=='no_reduction'
    carrier=save(out/'carrier.save',(1<<128)+1,17,20)
    for mode in (0,1):
        for exponent in (0,256):
            row,text=engine(f'carrier_{exponent}_{mode}',carrier,reloc=mode,carrier=exponent)
            assert not row['factors'] and not row['relocation_replayed']
            assert fields(text,'target_descent_values')['hash']==leaf_hash(carrier,13230,210,mode)
    # Atkin-Morain factor curve, same exact point and bounds for both algorithms.
    chosen=None
    for q in (10007,10009,10037,10039,10061,10067,10069,10079):
        try:case=save(out/f'factor_{q}.save',q*((1<<127)-1),17,20)
        except ValueError:continue
        chosen=(q,case);break
    assert chosen
    q,case=chosen
    for b2 in (4830,13230,40000):
        off,_=engine(f'factor_{b2}_off',case,b2=b2,reloc=0)
        on,text=engine(f'factor_{b2}_on',case,b2=b2,reloc=1)
        assert sorted(off['factors'])==sorted(on['factors']) and any(int(f)%q==0 for f in on['factors'])
        assert on['relocation_replayed'] and on['giant_first']==1
        assert fields(text,'target_descent_values')['hash']==fields((out/f'factor_{b2}_off'/'debug.log').read_text(),'target_descent_values')['hash']
    # The exact Edwards d=-1 factor-block collision uses the public engine API.
    counter=json.loads(a.counterexample.read_text());N=counter['n'];p0,p1=[r['prime'] for r in counter['curves']]
    X0,X1=[r['weierstrass_point'][0] for r in counter['curves']]
    X=(X0+p0*((X1-X0)*pow(p0,-1,p1)%p1))*pow(2,-1,N)%N
    result=[]
    for mode in (0,1):
        text=run(f'gcd_N_{mode}',[a.probe,f'{N:x}',f'{X:x}',f'{pow(2,-1,N):x}',20,counter['b2'],2310,a.device,mode,0],
                 extra_env={'NTT_FOLD_DEVICE_CHECK':'0'}) # This diagnostic is explicitly limited to P<=64.
        row=json.loads(text.split('relocation_probe_result: ')[-1].splitlines()[0]);result.append(row)
        assert '1186519' in row['factors']
    assert sorted(result[0]['factors'])==sorted(result[1]['factors']) and result[1]['replayed']
    run('baby_zero',[sys.executable,ROOT/'tools/test/test_stage2_relocation_baby_zero.py',
        '--probe',a.probe.resolve(),'--output',out/'baby_zero','--device',a.device,
        '--counterexample',a.counterexample.resolve()])
    if a.wide_reference:
        study=json.loads(a.wide_reference.read_text());source=study['cases']['10267']
        wide=dict(save=source['saves']['ed'],n=int(source['N_hex'],16))
        assert sha(wide['save'])==source['save_hashes']['ed']
        row,text=engine('wide_chain',wide,b2=600600000,d=30030,reloc=1,
            extra_env=dict(NTT_GIANT_CHAIN_CHECK='1',NTT_SCALED_ROOT_CHECK='0',NTT_SCALED_CHECK='0',
                           NTT_FOLD_DEVICE_CHECK='0',NTT_SCALED_FRONTIER_CHECK='0'))
        assert not row['factors'] and not row['relocation_replayed'] and row['giant_first']>1
        checks=[dict(re.findall(r'([\w]+)=([^\s]+)',s)) for s in text.splitlines() if s.startswith('giant_chain_check:')]
        assert checks and all(int(c['mismatches'])==0 for c in checks)
        # The combined console/debug text repeats some log records; unique checks cover all points.
        unique={json.dumps(c,sort_keys=True):c for c in checks}
        assert sum(int(c['points']) for c in unique.values())==row['giant_points']
    row,text=engine('balanced_chunks',prime,b2=27015978990,d=180180,reloc=1,
        extra_env=dict(NTT_GIANT_POINT_BUDGET_KB='2160',NTT_GIANT_CHAIN_CHECK='1',
                       NTT_SCALED_ROOT_CHECK='0',NTT_SCALED_CHECK='0',
                       NTT_FOLD_DEVICE_CHECK='0',NTT_SCALED_FRONTIER_CHECK='0'))
    chunks=[dict(re.findall(r'([\w]+)=([^\s]+)',s)) for s in text.splitlines() if s.startswith('relocation_chunk:')]
    chunks=list({c['offset']:c for c in chunks}.values())
    assert [int(c['points']) for c in chunks]==[69120,34560,37441]
    assert not row['factors'] and not row['relocation_replayed']
    chains=[dict(re.findall(r'([\w]+)=([^\s]+)',s)) for s in text.splitlines() if s.startswith('giant_chain_check:')]
    chains=list({json.dumps(c,sort_keys=True):c for c in chains}.values())
    assert len(chains)==3 and sum(int(c['points']) for c in chains)==141121 and all(c['mismatches']=='0' for c in chains)
    # Strict config validation and plan identity; no curve completion is possible.
    run('bad_switch',[a.exe,'--ini',ini,'--relocation',2],good=False)
    run('legacy_tune_rejected',[a.exe,'--ini',ini,'--tune','ecm'],good=False)
    text=run('plan',[a.exe,'--ini',ini,'--save',prime['save'],'--b2',13230,'--d',210,
                     '--device',a.device,'--plan-only','--short-calibration',0])
    plans=[json.loads(s) for s in text.splitlines() if s.startswith('{') and '"type":"stage2_plan"' in s]
    assert len(plans)==1 and plans[0]['I']==65 and plans[0]['relocation_points']==61
    report['complete']=True;persist();print('COMPLETE Edwards relocation gates',flush=True)


if __name__=='__main__':main()
