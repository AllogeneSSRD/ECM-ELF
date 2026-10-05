"""Fixed-D same-binary A/B of host/device baby normalization with mandatory checks."""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--d',type=int,default=1381380)
    p.add_argument('--runs',type=int,choices=(4,8),default=8);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8'))
    repo=Path(__file__).resolve().parents[2];sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    assert manifest['sha256']==sha and manifest['exit']==0 and a.d>0
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        for name,want in manifest['sources'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want,name
    m=json.loads(a.calibration.read_text());env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update({k:str(v) for k,v in m['env'].items() if v is not None})
    env.update(NTT_D_MODEL='0',NTT_NO_PROGRESS='1',NTT_BABY_DEVICE_CHECK='0',NTT_BABY_DEVICE_TEST='0',
               NTT_BABY_DEVICE_TEST_BAD='0',NTT_BABY_DEVICE_ALLOC_FAIL='0',NTT_BABY_DEVICE_MAX_MB='512')
    argv=[str(exe),*m['runs'][0]['command'][1:]];argv[argv.index('--d')+1]=str(a.d)
    argv[argv.index('--device')+1]=str(a.device)
    order=[0,1,1,0] if a.runs==4 else [0,1,1,0,1,0,0,1]
    data=dict(exe=str(exe),sha256=sha,sources=manifest['sources'],env={k:v for k,v in env.items() if k.startswith('NTT_')},
              D=a.d,device=a.device,order=order,Q_line=m['Q_line'],runs=[])
    signature=None
    for number,mode in enumerate(order,1):
        verify();name=f'{number}_{mode}';log=out/(name+'.log');start=time.monotonic();print('RUN',name,flush=True)
        with log.open('wb') as f:r=subprocess.run(argv,env=env|{'NTT_BABY_DEVICE':str(mode)},stdout=f,stderr=subprocess.STDOUT,timeout=900)
        verify();text=log.read_text(encoding='utf-8',errors='replace');assert r.returncode==0,(name,r.returncode,text[-1200:])
        for token in (m['Q_line'],f'baby_device: requested={mode} enabled={mode}','point_arithmetic: xadd6=1',
                      f'ntt_gl_reduce_mode: device={a.device} short=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1',
                      'small_prime_reuse: requested=1 available=1 matched=1'):
            assert token in text,(name,token)
        assert '[trace]' not in text
        wall=re.search(r'stage2_full_wall:.*?init=([\d.]+) main=([\d.]+) total=([\d.]+)',text)
        baby=re.search(r'real_baby:.*?ladder=([\d.]+) s affine=([\d.]+) s degenerate=(\d+)',text)
        leaf=re.search(r'descent_values: (.*)',text)
        oracle=re.search(r's4_oracle_stats:.*selected=(\d+) queued=(\d+) compared=(\d+) samples=(\d+) pending=(\d+).*signature=(\S+)',text)
        factors=re.search(r'stage2: algorithm=tree_gpu_batched .*hits=(\d+) bad_factors=(\d+) factors=(.*?) hit_primes=(.*?) elapsed=',text)
        assert wall and baby and leaf and oracle and factors and oracle[1]==oracle[2]==oracle[3] and oracle[5]=='0' and factors[2]=='0'
        current=(leaf[1],oracle[1],oracle[4],oracle[6],factors.group(1,2,3,4),baby[3])
        if signature is None:signature=current
        else:assert current==signature,(name,current,signature)
        row=dict(name=name,mode=mode,command=argv,driver_seconds=time.monotonic()-start,init=float(wall[1]),
                 main=float(wall[2]),full=float(wall[3]),baby_ladder=float(baby[1]),baby_affine=float(baby[2]),
                 leaf=leaf[1],oracle_signature=oracle[6],baby_stats=re.search(r'baby_device: (.*)',text)[1])
        data['runs'].append(row);(out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(name,'full=',row['full'],'affine=',row['baby_affine'],flush=True)
    data['means']={str(mode):{key:statistics.mean(r[key] for r in data['runs'] if r['mode']==mode)
                           for key in ('full','init','main','baby_ladder','baby_affine')} for mode in (0,1)}
    data['gain_percent']=100*(1-data['means']['1']['full']/data['means']['0']['full'])
    (out/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8');print(json.dumps(data['means']),flush=True)


if __name__=='__main__':main()
