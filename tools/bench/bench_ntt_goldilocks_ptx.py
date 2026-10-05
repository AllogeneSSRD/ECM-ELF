"""Independent GMP and dependent-chain A/B for experimental Goldilocks PTX."""
import argparse,hashlib,json,os,re,statistics,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();repo=Path(__file__).resolve().parents[2]
    manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==manifest['sha256'].lower()
        for name,want in manifest['sources'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want.lower(),name
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    data=dict(manifest=manifest,device=a.device,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks={},runs=[])
    def run(name,args,code=0):
        verify();r=subprocess.run([str(exe),str(a.device),*args],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8');verify()
        assert r.returncode==code,(name,r.returncode,text[-1500:])
        assert re.search(r'gl_ptx_memory: peak_bytes=\d+ live_bytes=0',text),name
        return text
    text=run('gmp',[])
    for method in (0,1,2):
        for reduce in (0,1):
            token=f'gl_ptx_check: method={method} reduce={reduce} words=200144 bad=0'
            assert token in text;data['checks'][f'primitive_{method}_{reduce}']=True
        token=f'gl_ptx_chain_check: method={method} words=256 iterations=256 bad=0'
        assert token in text;data['checks'][f'chain_{method}']=True
    assert 'gl_ptx_gate: fault=0 bad=0' in text
    resources=[dict(re.findall(r'(\w+)=(\d+)',s)) for s in re.findall(r'gl_ptx_resources: (.*)',text)]
    assert len(resources)==3 and all(r['local']=='0' for r in resources)
    data['resources']=resources;data['checks']['no_local']=True
    text=run('fault',['--fault'],3)
    assert 'gl_ptx_gate: fault=1 bad=1' in text;data['checks']['fault']=True
    for method in (1,2):
        text=run(f'bench_{method}',['--bench',str(method)])
        rows=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'gl_ptx_bench: (.*)',text)]
        assert [int(r['mode']) for r in rows]==[0,1,1,0,1,0,0,1]
        assert [int(r['run']) for r in rows]==list(range(1,9))
        assert all(r['bad']=='0' and int(r['candidate'])==method and int(r['N'])==1<<20 and r['iterations']=='512' for r in rows)
        means={str(m):statistics.mean(float(r['seconds']) for r in rows if int(r['mode'])==m) for m in (0,1)}
        item=dict(candidate=method,means=means,gain_percent=100*(1-means['1']/means['0']),raw=rows)
        data['runs'].append(item)
        (a.output/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in item.items() if k!='raw'}),flush=True)
    data.update(passed=len(data['checks']),failed=0)
    (a.output/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=data['passed'],failed=0)),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
