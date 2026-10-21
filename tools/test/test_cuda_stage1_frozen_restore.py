"""Restore pre-refactor production batch snapshots and compare completed points.

Fixtures come from test_stage1_usability's frozen before-fault snapshots and
their completed reference saves. Original files are never altered.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from stage1_checkpoint_decode import batch
from test_edwards_gpu_pipeline import records


def mont_rows(path):
    result=[]
    for line in path.read_bytes().splitlines():
        fields=dict(re.findall(rb'(?:^|;\s*)([A-Z0-9]+)=([^;]+)',line))
        if b'X' in fields and b'SIGMA' in fields:
            result.append((int(fields[b'SIGMA']),int(fields[b'X'],0),int(fields.get(b'PARAM',b'0'))))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--random-sigma',action='store_true',help='resume with random policy, retaining archived seeds')
    parser.add_argument('--edwards-only',action='store_true',help='check an older Edwards text archive independently')
    args=parser.parse_args();exe=args.exe.resolve();source=args.fixtures.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),device=args.device,cases=[],complete=False)
    for snapshot in sorted(source.glob('*_gpu*_checkpoint_before_fault.bin')):
        original=snapshot.read_bytes();identity,lanes=batch(snapshot)
        nh,model,bound,*_=identity.split('|');n=int(nh,16);b1=int(bound);count=len(lanes);sigma=lanes[0][0]
        if args.edwards_only and model!='ed':continue
        if model!='ed':assert all(saved==sigma+i for i,(saved,_) in enumerate(lanes))
        assert (n+1)&n==0,'fixture must have a Mersenne target'
        name=snapshot.name.removesuffix('_checkpoint_before_fault.bin');folder=out/name;folder.mkdir()
        token='e' if model=='ed' else 'm'+model.removeprefix('mont')
        checkpoint=folder/f'.{token}_s1_m{n.bit_length()}_{b1}.ckpt';checkpoint.write_bytes(original)
        save=folder/'completed.save'
        ini=folder/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n[stage1]\nauto_stage2=0\n',encoding='utf-8')
        command=[str(exe),'--config',str(ini),'-v','--curve','ed' if model=='ed' else 'mont','--target','gpu',
            '-d',str(args.device),'-gpucurves',str(count),'--ckpt','0',
            '--tmp-dir',str(folder),'-save',str(save)]
        if not args.random_sigma:command+=['-sigma',str(sigma)]
        if model!='ed':command+=['--gpu-param',model.removeprefix('mont')]
        command += [str(b1),'0']
        env=dict(os.environ,ECM_GPU_STAGE1_ALGO='ladder')
        env.pop('ECM_EDWARDS_STAGE1_ALGO',None);env.pop('ECM_EDWARDS_DBL_ALGO',None)
        proc=subprocess.run(command,input=(str(n)+'\n').encode(),cwd=folder,env=env,capture_output=True,timeout=300)
        text=(proc.stdout+proc.stderr).decode('utf-8','replace');(folder/'run.log').write_bytes(proc.stdout+proc.stderr)
        assert proc.returncode==0,(name,proc.returncode,text[-3000:])
        assert ('GPU Edwards resume:' if model=='ed' else 'Resuming from checkpoint:') in text
        expected=[p for p in (source/name).glob('*.save') if '.bak' not in p.name]
        assert len(expected)==1,(name,expected)
        before=records(expected[0]) if model=='ed' else mont_rows(expected[0])
        after=records(save) if model=='ed' else mont_rows(save)
        assert after==before and len(after)==count,(name,'frozen endpoint mismatch')
        assert not list(folder.glob('*.ckpt'))
        assert snapshot.read_bytes()==original
        report['cases'].append(dict(name=name,curves=count,snapshot_sha256=hashlib.sha256(original).hexdigest(),payload_magic=lanes[0][1][:8].hex()))
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',name,count,flush=True)
    assert len(report['cases'])==(1 if args.edwards_only else 4),'required frozen CUDA curve/parameter fixtures missing'
    report['complete']=True;(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
