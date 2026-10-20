"""Paired full experimental worker path through manifest-verified objects.

GMP startup/scale/products/baby/descent checks cover the seven kernels whose
compiler output changed when their point primitives moved into owned headers.
No timing or performance qualification is inferred from this diagnostic run.
"""
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stat'))
import suyama_mont_ref as reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    exes=[a.baseline.resolve(),a.exe.resolve()];identities=[sha(e) for e in exes];rows=[]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update({key:'1' for key in ('NTT_XADD6_TEST','NTT_BABY_DEVICE_CHECK','NTT_SCALED_CHECK','NTT_S4_DESCENT_CHECK',
        'NTT_GSCALE_DEVICE','NTT_GSCALE_DEVICE_CHECK','NTT_GSCALE_DEVICE_TEST','NTT_S3_PRODUCTS_TEST')})
    for bits in (383,521,1279):
        n=(1<<bits)-1;sigma=26;b1=20;b2=13230;d=210
        q=reference.stage1(sigma,b1,n)
        assert q['gcd']==1 and q['x']
        pair=[]
        for side,exe in zip(('old','new'),exes):
            command=list(map(str,[exe,f'{n:x}',f'{q["x"]:x}',sigma,b1,b2,d,a.device]))
            r=subprocess.run(command,env=env,capture_output=True,timeout=180)
            text=(r.stdout+r.stderr).decode('utf-8','replace')
            (a.output/f'{bits}_{side}.log').write_text(text)
            assert r.returncode==0,(bits,side,r.returncode,text[-1000:])
            assert re.search(r'mont_selftest: .*mismatches=0',text)
            assert re.search(r'xadd6_selftest: cases=1280 .*bad=0',text)
            assert re.search(r'gscale_device_fixture: cases=12 .*bad=0',text)
            assert re.search(r'descent_check: .*mismatching_coefficients=0',text)
            assert not re.search(r'(?:gmp_bad|gmp_check_bad|gmp_selftest_bad)=[1-9]',text)
            leaves=re.findall(r'descent_values: ([^\r\n]+)',text);assert leaves
            result=[json.loads(s) for s in text.splitlines() if s.startswith('{') and 'adapter_result' in s]
            assert len(result)==1 and result[0]['bad_factors']==0
            pair.append(dict(leaves=leaves[-1],result=result[0]))
        assert pair[0]==pair[1],bits
        rows.append(dict(bits=bits,nw=(bits+63)//64,complete_leaves_equal=True,gmp_checks=True))
        print(bits,'PASS',flush=True)
    assert identities==[sha(e) for e in exes]
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True,binaries=identities,cases=rows,
        performance_qualification=False),indent=2)+'\n')


if __name__=='__main__':main()
