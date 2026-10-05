"""Gate --real normalization against independent Python points/F and the CPU dump."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

repo=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,default=repo/'build_cuda_cmake/stage2_tree_gpu.exe')
p.add_argument('--ref',type=Path,default=repo/'build_cuda_cmake/stage2_tree_ref.exe')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',type=int,default=1)
p.add_argument('--baby-device',type=int,choices=(0,1),default=0)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise RuntimeError('Use a fresh directory')
spec=importlib.util.spec_from_file_location('mont_ref',repo/'tools/stat/suyama_mont_ref.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_OLDTAIL='0',NTT_GROOT_DEVICE='1',NTT_SCALED_DESCENT='1',NTT_S4_OUTPUT_WINDOW='1',
            NTT_S4_CHUNK_OUTPUT='1',NTT_NO_PROGRESS='1',NTT_NAME_MAX='0',NTT_STAGE1_Q_DUMP='1',
            NTT_BABY_DEVICE=str(a.baby_device),NTT_BABY_DEVICE_CHECK=str(a.baby_device))
rows=[]

def read(path):
    text=path.read_text(encoding='utf-8');data={};baby=[];f=[]
    assert text.startswith('stage2_tree_F_dump v1\n')
    for line in text.splitlines()[1:]:
        key,_,value=line.partition(' ')
        if key=='baby':
            j,x=value.split();baby.append((int(j),int(x,16)))
        elif key=='F':f.append(int(value,16))
        else:data[key]=value
    assert 'end' in data
    return data,baby,f

for name,n,b1,b2,d,extra in (
    ('frozen',(1<<128)+1,1000,1000000,210,1),
    ('tail',(1<<128)+1,1000,400000,2310,1),
    ('multi_segment',(1<<128)+1,20,114000,30030,1),
    ('choose12',(1<<128)+1,1000,114000,210,12),
    ('production_width',(1<<4423)-1,1000,1000,2310,12),
    ('wide',(1<<5261)-1,20,1000,210,12)):
    q=oracle.stage1(26,b1,n,torsion=extra)['x'];_,a24,_,_=oracle.suyama_curve(26,n)
    baby=[];f=[1]
    for j in range(1,d//2+1):
        if math.gcd(j,d)!=1:continue
        x,z=oracle.ladder(j,q,1,a24,n);x=x*pow(z,-1,n)%n;baby.append((j,x))
        nextf=[0]*(len(f)+1)
        for k,v in enumerate(f):
            nextf[k]=(nextf[k]-v*x)%n;nextf[k+1]=(nextf[k+1]+v)%n
        f=nextf
    if extra==1:
        cpu=out/(name+'_cpu.dump')
        proc=subprocess.run([str(a.ref.resolve()),'--n',str(n),'--sigma','26','--b1',str(b1),'--b2',str(b2),
                             '--d',str(d),'--dump-F',str(cpu)],capture_output=True,text=True,timeout=240)
        (out/(name+'_cpu.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert proc.returncode==0
        data,cb,cf=read(cpu);assert int(data['Q_hex'],16)==q and cb==baby and cf==f
    for batch in ('0','1'):
        dump=out/(name+'_'+batch+'.dump')
        command=[str(a.exe.resolve()),'--real','--n-hex',format(n,'x'),'--sigma','26','--b1',str(b1),
                 '--b2',str(b2),'--d',str(d),'--device',str(a.device)]
        proc=subprocess.run(command,env=base|dict(NTT_STAGE1_EXTRA=str(extra),NTT_GFINV_BATCH=batch,NTT_REAL_F_DUMP=str(dump)),
                            capture_output=True,text=True,timeout=240,cwd=out)
        text=proc.stdout+proc.stderr;(out/(name+'_'+batch+'.log')).write_text(text,encoding='utf-8')
        assert proc.returncode==0 and dump.exists(),(name,batch,proc.returncode)
        if a.baby_device or 'baby_device:' in text:
            assert f'baby_device: requested={a.baby_device} enabled={a.baby_device}' in text
        data,gb,gf=read(dump)
        assert int(data['N_hex'],16)==n and int(data['Q_hex'],16)==q and int(data['a24_hex'],16)==a24
        assert gb==baby and gf==f,(name,batch,'real baby/F differ from independent reference')
        assert re.search(r'stage2_full_wall:.*clean=0',text)
        if name=='frozen':
            assert 'factors=59649589127497217 hit_primes=114713' in text
            assert 'leaves=24 words=72 hash=7706779146789021619' in text
        rows.append(dict(case=name+'_'+batch,babies=len(baby),coefficients=len(f),extra=extra,
                         q_sha256=hashlib.sha256(format(q,'x').encode()).hexdigest()))
        print('PASS',name+'_'+batch,flush=True)
(out/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,runs=rows,
    binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(a.ref.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed',flush=True)
