"""Production construction-factor persistence and degenerate-lane rejection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();exe=args.exe.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),device=args.device,cases=[],complete=False)
    def call(name,param,sigma,n,count,algorithm,good):
        folder=out/name;folder.mkdir()
        ini=folder/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n[stage1]\nauto_stage2=0\n',encoding='utf-8')
        env=dict(os.environ,ECM_GPU_STAGE1_ALGO=algorithm)
        for key in tuple(env):
            if key.startswith('ECM_PRAC_') or key=='ECM_GPU_STAGE1_SAMPLE_SECONDS':env.pop(key)
        env['ECM_PRAC_PLAN_CACHE']=str(folder/'plan_cache')
        command=[str(exe),'--config',str(ini),'--curve','mont','--target','gpu','--gpu-param',str(param),
            '-d',str(args.device),'-sigma',str(sigma),'-gpucurves',str(count),'--ckpt','0',
            '--tmp-dir',str(folder/'tmp'),'-save',str(folder/'points.save'),'2','0']
        proc=subprocess.run(command,input=(str(n)+'\n').encode(),env=env,cwd=folder,capture_output=True,timeout=120)
        (folder/'run.log').write_bytes(proc.stdout+proc.stderr)
        text=(proc.stdout+proc.stderr).decode('utf-8','replace')
        assert (proc.returncode==0)==good,(name,proc.returncode,text[-3000:])
        report['cases'].append(dict(name=name,returncode=proc.returncode))
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS',name,flush=True)
        return folder,text
    prime=(1<<127)-1
    for algorithm in ('ladder','resident','prac'):
        folder,_=call('proper_'+algorithm,0,101,101*prime,1,algorithm,True)
        data=(folder/'points.save').read_bytes()
        value=re.search(rb'(?:^|;\s*)X=([^;]+)',data,re.M)
        assert value and int(value[1],0)==101
        rows=[json.loads(line) for line in (folder/'stage1_results.jsonl').read_text(encoding='utf-8').splitlines()]
        assert rows and any(row.get('factors')==['101'] for row in rows)
    # A restricted 384/1024 build must not silently skip an absent smaller
    # PARAM2 tier. Use an input whose original selection is the compiled 384.
    folder,_=call('proper_param2',2,2,3*((1<<347)-1),1,'ladder',True)
    data=(folder/'points.save').read_bytes()
    assert int(re.search(rb'(?:^|;\s*)X=([^;]+)',data,re.M)[1],0)==3
    for algorithm in ('ladder','resident','prac'):
        folder,text=call('degenerate_'+algorithm,0,101,101,1,algorithm,False)
        assert 'degenerate construction denominator' in text
        assert not (folder/'points.save').exists()
    # sigma 73 has gcd(4*(sigma^2-5)^3*(4*sigma),77)=11; sigma 77 is
    # degenerate. The earlier genuine factor must survive this later failure.
    assert math.gcd(4*(73*73-5)**3*(4*73),77)==11
    folder,text=call('factor_before_degenerate',0,73,77,5,'ladder',False)
    assert 'degenerate construction denominator' in text and not (folder/'points.save').exists()
    rows=[json.loads(line) for line in (folder/'results.json.txt').read_text(encoding='utf-8').splitlines()]
    assert rows and any(row.get('factors')==['11'] for row in rows)
    assert all(row.get('factors') for row in rows),'incomplete batch published an NF record'
    assert not (folder/'stage1_results.jsonl').exists(),'incomplete batch produced a completion ledger'
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
