"""Check actual CUDA export/readback and host CSV output for a partial PRAC window."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    ini=out/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n',encoding='utf8')
    env={k:v for k,v in os.environ.items() if not k.startswith('ECM_PRAC_') and k not in ['ECM_STAGE1_TPI','ECM_GPU_STAGE1_SAMPLE_SECONDS']}
    env.update(ECM_GPU_STAGE1_ALGO='prac',ECM_PRAC_WINDOW='prefix',ECM_PRAC_WINDOW_COUNT='1',
               ECM_PRAC_WINDOW_WARMUP='0',ECM_PRAC_WINDOW_DUMP='1',ECM_GPU_STAGE1_SAMPLE_SECONDS='0.01')
    cmd=[exe,'--curve','mont','--target','gpu','--gpu-param','0','-d',a.device,'-sigma','26','-gpucurves','2',
         '--tmp-dir',out,'--config',ini,'-save',out/'partial.save','1000','1000']
    r=subprocess.run(list(map(str,cmd)),input=str((1<<347)-1)+'\n',text=True,capture_output=True,errors='replace',cwd=out,env=env,timeout=120)
    log=r.stdout+r.stderr;(out/'run.log').write_text(log,encoding='utf8')
    assert r.returncode!=0 and 'PRAC_WINDOW_DONE' in log and 'partial product, no Stage1 save or checkpoint' in log,log
    rows=(out/'window_q.csv').read_text().splitlines()
    assert rows[0]=='kind,b1,first,count,work,sigma,x,z' and len(rows)==3,rows
    for i,row in enumerate(rows[1:]):
        fields=row.split(',');assert len(fields)==8 and fields[:4]==['partial_product','1000','0','1'] and fields[5]==str(26+i)
        for field in fields[6:]:assert field.startswith('0x') and 0<=int(field,16)<(1<<384)
    assert not (out/'partial.save').exists() and not (out/'stage1_results.jsonl').exists() and not (out/'results.json.txt').exists()
    result=dict(complete=True,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),curves=2,partial_result_not_published=True,
                actual_csv_readback=True,performance_samples=0,device=a.device)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
