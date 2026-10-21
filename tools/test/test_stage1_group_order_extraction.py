"""Compare the actual PARI diagnostic with a frozen pre-extraction worker."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','exe','gp','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    for target in ['cpu','gpu']:
        compared=[]
        for label,exe in [('before',a.baseline.resolve()),('after',a.exe.resolve())]:
            work=out/(target+'_'+label);work.mkdir();ini=work/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n')
            cmd=[exe,'--curve','mont','--target',target,'--go','--gp',a.gp.resolve(),'-d',a.device,'-sigma','0:26',
                 '-gpucurves','2','--tmp-dir',work,'--config',ini,'--exp-cache','off','1000','1000']
            if target=='cpu':cmd+=['--impl','scalar','--stage1-threads','1']
            else:cmd+=['--gpu-param','0']
            r=subprocess.run(list(map(str,cmd)),input=str(1009*((1<<127)-1))+'\n',capture_output=True,text=True,errors='replace',cwd=work,timeout=120)
            log=r.stdout+r.stderr;(work/'run.log').write_text(log,encoding='utf8')
            assert r.returncode==0,log
            plain=re.sub(r'^\[\d{4}-[^\]\r\n]+\] ','',log,flags=re.M)
            diagnostics=[v for v in plain.splitlines() if v.startswith(('  go[','  go_factor[','  go_min_bounds['))]
            assert diagnostics and any(v.startswith('  go_min_bounds[') for v in diagnostics),log
            hits=re.findall(r'^factor\[\d+\]=\d+',plain,re.M);assert hits,log
            compared.append((diagnostics,hits))
        assert compared[0]==compared[1],target;rows.append(target)
    result=dict(complete=True,pairs=rows,diagnostic_semantics_unchanged=True,strict_minimum_bounds_claimed=False,
                gp_sha256=hashlib.sha256(a.gp.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),performance_samples=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
