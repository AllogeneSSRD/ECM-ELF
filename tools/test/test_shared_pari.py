"""Qualify shared DLL-first diagnostics, GP fallback and explicit-only GMP."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['probe','gp','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();probe=a.probe.resolve();gp=a.gp.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    checks=[];counter=out/'gp-counter.txt';host_counter=out/'host-counter.txt'
    env=os.environ.copy();env['PATH']=str(Path(os.environ['SystemRoot'])/'System32');env['ECM_DIAG_GP_COUNTER']=str(counter);env['ECM_DIAG_HOST_COUNTER']=str(host_counter)
    def run(name,backend='auto',op='group',n='3436193553215096358758503',sigma='1955774944',param=0,gp_path=None,host='default',timeout=5000,binary=probe,mode=None):
        e=env.copy()
        if mode:e['ECM_DIAG_HOST_MODE']=mode
        cmd=[str(binary),backend,str(gp_path or probe),op,str(n),str(sigma),str(param),str(host),str(timeout)]
        start=time.perf_counter();p=subprocess.run(cmd,capture_output=True,env=e,timeout=40);elapsed=time.perf_counter()-start
        (out/(name+'.log')).write_bytes(p.stdout+p.stderr);assert p.returncode==0,(name,p.returncode,p.stderr)
        value=json.loads(p.stdout.decode());checks.append(dict(name=name,result=value,wall_seconds=elapsed));return value,elapsed
    inputs=[('user','3436193553215096358758503','1955774944',0)]
    for bits,n in [(127,(1<<127)-1),(160,(1<<160)-(1<<31)-1),(192,(1<<192)-(1<<64)-1)]:
        for param in [0,-1,1,2,3]:inputs.append((f'p{bits}-{param}',str(n),'26',param))
    for name,n,sigma,param in inputs:
        native,_=run(name,n=n,sigma=sigma,param=param)
        assert native['backend']=='pari' and native['status']=='complete' and native['all_proven'],native
        reference,_=run(name+'-gp',backend='gp',n=n,sigma=sigma,param=param,gp_path=gp)
        assert reference['backend']=='gp' and reference['status']=='complete',reference
        for key in ['group_order','point_order','point_factors']:assert native[key]==reference[key],(name,key,native,reference)
        assert native['factorization']['factors']==reference['factorization']['factors']
    assert not counter.exists(),'DLL-first must not start the GP fixture'
    factor,_=run('factor',op='factor',n=1009*1013);assert factor['backend']=='pari' and factor['status']=='complete'
    missing=out/'absent-host.exe'
    fallback,_=run('gp-fallback',host=missing,gp_path=gp);assert fallback['backend']=='gp' and fallback['status']=='complete' and fallback['pari_failure']
    refused,_=run('native-no-gp',backend='pari',host=missing);assert refused['status']=='resource_limit' and refused['backend']=='pari' and not counter.exists()
    failed,_=run('both-unavailable',host=missing);assert failed['backend']=='pari' and failed['status']=='resource_limit' and failed['gp_failure'] and failed['pari_failure']
    assert len(counter.read_text().splitlines())==1
    for backend in ['gmp','native']:
        value,_=run('explicit-'+backend,backend=backend,n=1009,sigma=26)
        assert value['status']=='complete' and value['backend']==('gmp' if backend=='gmp' else 'pari')
    before=len(counter.read_text().splitlines())
    value,_=run('bad-curve',n=1009,sigma=0,host=probe);assert value['status']=='invalid_input' and not host_counter.exists()
    assert len(counter.read_text().splitlines())==before
    value,elapsed=run('timeout-keeps-factors',backend='pari',op='factor',n=1009*1013,host=probe,timeout=200,mode='timeout')
    assert elapsed<2 and value['status']=='partial' and value['cofactor']=='1013' and value['factors'][0]['prime']=='1009' and not value['all_proven'],value
    invalid,_=run('bad-product',backend='pari',op='factor',n=1009*1013,host=probe,mode='invalid_product');assert invalid['status']=='resource_limit' and not invalid['factors']
    portable=out/'中文 可移动目录';portable.mkdir();shutil.copy2(probe,portable/probe.name);shutil.copy2(probe.parent/'gmp-10.dll',portable/'gmp-10.dll')
    shutil.copy2(probe.parent/'ecm_diagnostics_host.exe',portable/'ecm_diagnostics_host.exe')
    value,_=run('missing-dll-no-gmp',backend='pari',n=1009,sigma=26,binary=portable/probe.name);assert value['status']=='resource_limit' and value['backend']=='pari'
    value,_=run('missing-dll-gp',n=1009,sigma=26,gp_path=gp,binary=portable/probe.name);assert value['status']=='complete' and value['backend']=='gp'
    shutil.copy2(probe.parent/'libpari.dll',portable/'libpari.dll');shutil.copytree(probe.parent/'pari-data',portable/'pari-data')
    value,_=run('unicode-relocation',binary=portable/probe.name);assert value['status']=='complete' and value['backend']=='pari'
    result=dict(passed=True,checks=checks,identities={p.name:sha(p) for p in [probe,probe.parent/'ecm_diagnostics_host.exe',probe.parent/'libpari.dll',gp]},
        scope='16 prime-field DLL/GP equivalence pairs; fallback, no implicit GMP, error isolation, partial timeout, Unicode relocation. No speed comparison.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(f'Shared diagnostics: {len(checks)} checks passed')
if __name__=='__main__':main()
