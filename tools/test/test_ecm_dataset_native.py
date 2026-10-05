"""Real production results drive the database's conditional best-bound update."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/ecm_dataset'))
from dataset import connect, import_catalog, ingest, gp_path, digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gp',type=Path,nargs='?',const=None);p.add_argument('--stage1',type=Path,default=ROOT/'build_cuda_cmake/ecm_cuda.exe')
    p.add_argument('--stage2',type=Path,default=ROOT/'build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('Use a fresh test directory')
    gp=gp_path(a.gp);db=connect(out/'database.sqlite');import_catalog(db,ROOT/'.refactor/Mersenne_exponent_factor_1-9999.html')
    before=db.execute('SELECT sigma,b1,b2,group_order FROM factors WHERE exponent=223 AND value="196687"').fetchone()
    assert all(v is None for v in before)
    spec=importlib.util.spec_from_file_location('ref',ROOT/'tools/stat/suyama_mont_ref.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    exe1,exe2=a.stage1.resolve(),a.stage2.resolve();hashes=(digest(exe1),digest(exe2))
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['NTT_NO_PROGRESS']='1'
    snapshots=[]
    for sigma,b1,b2 in [(6,29,563),(9,9,227)]:
        where=out/f'sigma{sigma}';where.mkdir();save=where/'stage1.save'
        ini=where/'ecm.ini';ini.write_text('[gpu]\ndevice=1\n[queue]\nworktodo=unused.txt\n')
        cmd=[str(exe1),'-ini',str(ini),'--method','gpu','--gpu-param','0','--exponent','lcm','--exp-cache','off',
             '-d','1','-sigma',str(sigma),'-gpucurves','1','--ckpt','0','-save',str(save),str(b1)]
        modulus=(1<<223)-1
        for index in range(32):
            _,a24,x0,z0=ref.suyama_curve(sigma,modulus)
            _,z=ref.ladder(ref.lcm_1_to(b1),x0,z0,a24,modulus);g=math.gcd(z,modulus)
            if g==1:break
            assert 1<g<modulus and g%196687!=0
            strip=cmd.copy();strip[strip.index('-save')+1]=str(where/f'strip{index}.save')
            rr=subprocess.run(strip,input=f'{modulus}\n'.encode(),capture_output=True,cwd=where,env=env,timeout=120)
            (where/f'strip{index}.log').write_bytes(rr.stdout+rr.stderr)
            assert rr.returncode==0 and re.findall(r'factor\[\d+\]=(\d+)',rr.stdout.decode(errors='replace'))==[str(g)]
            modulus//=g
        else:raise RuntimeError('Too many Stage1 factors')
        point=ref.stage1(sigma,b1,modulus);assert point['gcd']==1
        r=subprocess.run(cmd,input=f'{modulus}\n'.encode(),capture_output=True,cwd=where,env=env,timeout=120)
        (where/'stage1.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0
        assert int(re.search(r'X=(0x[0-9a-fA-F]+)',save.read_text())[1],16)==point['x']
        results=where/'results.jsonl'
        cmd=[str(exe2),'--save',str(save),'--b2',str(b2),'--d','30','--device','1','--arena-mb','1024',
             '--results',str(results),'--log',str(where/'stage2.log')]
        r=subprocess.run(cmd,capture_output=True,cwd=where,env=env,timeout=120);(where/'stage2_driver.log').write_bytes(r.stdout+r.stderr)
        assert r.returncode==0
        record=json.loads(results.read_text());assert any(int(f)%196687==0 for f in record['factors'])
        outcome=ingest(db,results,gp,exponent=223)
        best=dict(db.execute('SELECT sigma,b1,b2,group_order,point_order FROM factors WHERE exponent=223 AND value="196687"').fetchone())
        assert (best['sigma'],best['b1'],best['b2'])==(str(sigma),str(b1),str(b2))
        snapshots.append(dict(native_result=record,best=best,ingestion=outcome))
    duplicate=ingest(db,results,gp,exponent=223);assert duplicate['updated']==0 and duplicate['unchanged']>0
    assert (digest(exe1),digest(exe2))==hashes
    (out/'summary.json').write_text(json.dumps(dict(passed=3,failed=0,snapshots=snapshots,duplicate=duplicate,
        stage1_sha256=hashes[0],stage2_sha256=hashes[1]),indent=2),encoding='utf-8')
    print(json.dumps({'passed':3,'before':['6','29','563'],'after':['9','9','227'],'duplicate':duplicate}),flush=True)
    db.close()


if __name__=='__main__':main()
