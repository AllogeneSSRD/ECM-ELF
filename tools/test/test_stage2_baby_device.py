"""Independent GMP kernel gate and optional real host/factor/cache regression gate."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--stage2-exe',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_XADD6='1',NTT_SMALL_PRIME_REUSE='1')
    rows=[];shas={}
    def run(name,exe,args,controls=None,code=0):
        exe=exe.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest()
        shas[str(exe)]=sha;cmd=[str(exe),*map(str,args)]
        r=subprocess.run(cmd,env=env|(controls or {}),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        text=r.stdout.decode('utf-8',errors='replace');(out/(name+'.log')).write_text(text,encoding='utf-8')
        assert r.returncode==code,(name,r.returncode,text[-1200:])
        rows.append(dict(name=name,command=cmd,controls=controls or {},exit=r.returncode))
        return text
    text=run('kernel_gmp',a.probe,[a.device])
    total=re.search(r'TOTAL cases=(\d+) words=(\d+) bad=0',text);assert total and int(total[1])==42
    kernel_words=int(total[2]);assert len(re.findall(r'^PASS baby_device ',text,re.M))==42
    text=run('kernel_corruption',a.probe,[a.device,'bad'],code=3)
    assert 'FATAL: baby probe affine GMP mismatch' in text
    if a.stage2_exe:
        frozen=['--real','--n-hex',format((1<<128)+1,'x'),'--sigma','26','--b1','1000',
                '--b2','1000000','--d','210','--device',a.device]
        signatures=[]
        for mode in (0,1):
            text=run('frozen_'+str(mode),a.stage2_exe,frozen,
                     dict(NTT_BABY_DEVICE=str(mode),NTT_BABY_DEVICE_CHECK=str(mode)))
            assert f'baby_device: requested={mode} enabled={mode}' in text
            assert 'gmp_check_bad=0' in text and 'factors=59649589127497217 hit_primes=114713' in text
            signatures.append(re.search(r'descent_values: (.*)',text)[1])
        assert signatures[0]==signatures[1]
        for bits in (129,4423):
            args=frozen.copy();args[args.index('--n-hex')+1]=format((1<<128)+1 if bits==129 else (1<<bits)-1,'x')
            args[args.index('--b1')+1]='20';args[args.index('--b2')+1]='1000'
            text=run('host_fixture_'+str(bits),a.stage2_exe,args,
                     dict(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_CHECK='1',NTT_BABY_DEVICE_TEST='1'))
            assert len(re.findall(r'baby_device_fixture:.*bad=0',text))==2
            assert 'baby_device: requested=1 enabled=1' in text
        text=run('host_corruption',a.stage2_exe,frozen,
                 dict(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_TEST='1',NTT_BABY_DEVICE_TEST_BAD='1'),code=3)
        assert 'FATAL: baby device affine GMP mismatch' in text
        text=run('allocation_fallback',a.stage2_exe,frozen,
                 dict(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_ALLOC_FAIL='1'))
        assert 'baby_device: requested=1 enabled=0' in text and re.search(r'descent_values: (.*)',text)[1]==signatures[0]
        assert 'factors=59649589127497217 hit_primes=114713' in text
        text=run('budget_fallback',a.stage2_exe,frozen,dict(NTT_BABY_DEVICE='1',NTT_BABY_DEVICE_MAX_MB='0'))
        assert 'baby_device: requested=1 enabled=0' in text and re.search(r'descent_values: (.*)',text)[1]==signatures[0]
    (out/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,kernel_cases=42,
        kernel_words=kernel_words,binary_sha256=shas,runs=rows),indent=2),encoding='utf-8')
    print(f'TOTAL {len(rows)} gates passed / 0 failed; kernel cases=42 words={kernel_words}',flush=True)


if __name__=='__main__':main()
