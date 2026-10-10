"""Measure completed CUDA Stage1 batches and publish reusable T1 performance data.

CPU point validation and warmups are excluded. Each timed process generates a
fresh batch with exponent cache off; partial-run speed projections are rejected.
The performance TOML has no paths or binary hashes. Raw evidence retains both.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PRIMES=[521,2203,4423,9689,1279,3217,11213,607,2281,4253,127,107,9941]
B1_GRID=[20,1000,10000,100000,1000000,10000000,26000000,100000000,260000000]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)


def effort(level):
    if not 1<=level<=10:raise ValueError('tune level must be 1..10')
    return dict(exponents=PRIMES[:min(13,level+3)],b1=B1_GRID[:min(9,level)],
                batches=[1,8,64,256][:min(4,1+(level-1)//3)],repeats=2*level+1)


def profile_text(device,level,repeats,samples):
    tables=[('profile',dict(format=1,algorithm_revision=1,unit='process_seconds_per_curve',
                           effort_level=level,repeats=repeats,warmups=1)),
            ('device',{k:device[k] for k in ['uuid_hex','sm_major','sm_minor','cuda_runtime','cuda_driver']}),
            ('policy',dict(algorithm='ladder',backend='cgbn_montgomery',param=0,requested_tpi=0,exp_cache='off'))]
    tables += [('stage1.sample_'+str(i),s) for i,s in enumerate(samples)]
    tables += [('summary',dict(complete=1,failed=0,measured=len(samples)))]
    return '# Completed Stage1 batch costs. Seconds per curve; no extrapolation.\n'+''.join(
        '\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in fields.items()) for name,fields in tables)


def read_native_reference(text,bits,b1,first,count,mode):
    """Validate a complete scoped oracle response; partial/duplicate points fail."""
    if len(text.encode('utf-8'))>16*1048576:raise ValueError('reference output exceeds 16MiB')
    rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows)!=count+2:raise ValueError('incomplete reference response')
    h,end=rows[0],rows[-1]
    expected=dict(type='stage1_reference',bits=bits,b1=b1,sigma_first=first,curves=count,exponent=mode)
    if any(h.get(k)!=v or type(h.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('reference input scope mismatch')
    if (not isinstance(h.get('n_hex'),str) or not re.fullmatch('[0-9a-f]+',h['n_hex']) or
        int(h['n_hex'],16)!=(1<<bits)-1 or type(h.get('scalar_bits')) is not int or h['scalar_bits']<1):
        raise ValueError('reference modulus/scalar mismatch')
    if (end.get('type')!='complete' or end.get('algorithm')!='plain_gmp_ladder' or
        type(end.get('curves')) is not int or end['curves']!=count or
        type(end.get('seconds')) not in (int,float) or not math.isfinite(end['seconds']) or end['seconds']<=0):
        raise ValueError('reference completion missing')
    points=[]
    for i,row in enumerate(rows[1:-1]):
        if (row.get('type')!='point' or type(row.get('sigma')) is not int or row['sigma']!=first+i or
            not isinstance(row.get('x_hex'),str) or not re.fullmatch('[0-9a-f]+',row['x_hex'])):
            raise ValueError('duplicate/mismatched reference point')
        x=int(row['x_hex'],16)
        if x>=(1<<bits)-1:raise ValueError('reference coordinate outside target N')
        points.append(x)
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--stage2',type=Path,required=True,help='Production executable for device and native-profile validation')
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--tune-level',type=int,default=1)
    p.add_argument('--exponents',type=int,nargs='+')
    p.add_argument('--b1',type=int,nargs='+')
    p.add_argument('--batch',type=int,nargs='+')
    p.add_argument('--exponent',choices=['lcm','choose12'],default='lcm')
    p.add_argument('--repeats',type=int)
    p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('--reference',type=Path,help='Optional independent stage1_gmp_reference.exe for high B1')
    p.add_argument('--reference-timeout',type=float,default=7200,help='Per native reference process; excluded from T1')
    p.add_argument('--output',type=Path,required=True,help='Fresh raw evidence directory under data/experiments')
    p.add_argument('--profile',type=Path,required=True,help='Published Stage1 performance TOML')
    a=p.parse_args();grid=effort(a.tune_level)
    for name,value in [('exponents',a.exponents),('b1',a.b1),('batches',a.batch)]:
        if value is not None:grid[name]=value
    if a.repeats is not None:grid['repeats']=a.repeats
    if (a.device<0 or not 1<=grid['repeats']<=1000 or not math.isfinite(a.timeout) or a.timeout<=0 or
        not math.isfinite(a.reference_timeout) or a.reference_timeout<=0 or
        not set(grid['exponents'])<=set(PRIMES) or any(b<2 or b>260000000 for b in grid['b1']) or
        any(b<1 or b>1048576 for b in grid['batches']) or
        any(len(values)!=len(set(values)) for values in [grid['exponents'],grid['b1'],grid['batches']])):
        p.error('invalid benchmark grid or timeout')
    if a.reference and max(grid['batches'])>4096:p.error('native reference supports at most 4096 curves per scope')
    if len(grid['exponents'])*len(grid['b1'])*len(grid['batches'])>4096:p.error('grid exceeds 4096 scopes')
    out=a.output.resolve();destination=a.profile.resolve();exes=[a.stage1.resolve(),a.stage2.resolve()]
    if destination.suffix.lower()!='.toml':p.error('profile must use .toml')
    if destination in exes:p.error('profile must differ from executable')
    out.mkdir(parents=True,exist_ok=False)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities={str(path):sha(path) for path in exes}
    if a.reference:
        a.reference=a.reference.resolve()
        if destination==a.reference:p.error('profile must differ from reference executable')
        identities[str(a.reference)]=sha(a.reference)
        manifest=a.reference.parent/'manifest.json'
        if manifest.exists():
            frozen=json.loads(manifest.read_text(encoding='utf-8-sig'))
            if frozen['binary_sha256'].lower()!=sha(a.reference):raise RuntimeError('reference build identity mismatch')
            identities[str(manifest)]=sha(manifest)
            for name,want in frozen['sources'].items():
                source=a.reference.parent/'sources'/name
                if sha(source)!=want.lower():raise RuntimeError('reference frozen source mismatch')
                identities[str(source)]=sha(source)
        dll=a.reference.parent/'gmp-10.dll'
        if dll.exists():identities[str(dll)]=sha(dll)
    own=sha(Path(__file__))
    spec=importlib.util.spec_from_file_location('reference',ROOT/'tools/stat/suyama_mont_ref.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    identities[str(ROOT/'tools/stat/suyama_mont_ref.py')]=sha(ROOT/'tools/stat/suyama_mont_ref.py')
    probe=subprocess.run([str(exes[1]),'--cost-device-info','--device',str(a.device)],
                         cwd=out,capture_output=True,text=True,errors='replace',timeout=60)
    (out/'device.log').write_text(probe.stdout+probe.stderr,encoding='utf-8')
    if probe.returncode:raise RuntimeError('device query failed')
    info=next(json.loads(line) for line in probe.stdout.splitlines() if line.startswith('{'))
    device=dict(uuid_hex=info['uuid_hex'],sm_major=info['major'],sm_minor=info['minor'],
                cuda_runtime=info['runtime'],cuda_driver=info['driver'])
    data=dict(complete=False,identity=identities,tool_sha256=own,device=device,grid=grid,exponent=a.exponent,runs=[],references=[])
    def persist():
        (out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    persist();samples=[];reference={};scalars={}
    env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_PRAC_','ECM_GPU_STAGE1_','ECM_STAGE1_'))}
    env.update(ECM_GPU_STAGE1_ALGO='ladder',ECM_STAGE1_TPI='0',ECM_GPU_STAGE1_SAMPLE_SECONDS='0')
    for bits in grid['exponents']:
        n=(1<<bits)-1
        for b1 in grid['b1']:
            for batch in grid['batches']:
                print(f'stage1_tune_prepare: bits={bits} B1={b1} batch={batch} independent_points={batch}',flush=True)
                missing=[sigma for sigma in range(26,26+batch) if (bits,b1,sigma) not in reference]
                if a.reference and missing:
                    first,last=min(missing),max(missing);count=last-first+1
                    command=[str(a.reference),str(bits),str(b1),str(first),str(count),a.exponent]
                    where=out/f'reference_m{bits}_b{b1}_s{first}_c{count}';where.mkdir()
                    try:
                        checked=subprocess.run(command,capture_output=True,timeout=a.reference_timeout)
                    except subprocess.TimeoutExpired as error:
                        (where/'driver.log').write_bytes((error.stdout or b'')+(error.stderr or b''))
                        data['failure']=dict(case=where.name,reason='reference_timeout');persist();raise
                    (where/'points.jsonl').write_bytes(checked.stdout);(where/'driver.log').write_bytes(checked.stderr)
                    if checked.returncode:raise RuntimeError('independent GMP reference failed: '+str(where))
                    values=read_native_reference(checked.stdout.decode('utf-8'),bits,b1,first,count,a.exponent)
                    for sigma,x in zip(range(first,last+1),values):reference[(bits,b1,sigma)]=x
                    data['references'].append(dict(command=command,output=where.name,sha256=sha(where/'points.jsonl')));persist()
                elif missing:
                    if b1 not in scalars:scalars[b1]=ref.lcm_1_to(b1)*(12 if a.exponent=='choose12' else 1)
                    for sigma in missing:
                        _,a24,x,z=ref.suyama_curve(sigma,n)
                        x,z=ref.ladder(scalars[b1],x,z,a24,n)
                        if math.gcd(z,n)!=1:raise RuntimeError('benchmark reference point is not a unit; evidence retained')
                        reference[(bits,b1,sigma)]=x*pow(z,-1,n)%n
                trials=[];gpu=[]
                for repeat in range(grid['repeats']+1):
                    where=out/f'm{bits}_b{b1}_c{batch}_r{repeat}';where.mkdir()
                    ini=where/'ecm.ini';save=where/'stage1.save'
                    ini.write_text('device='+str(a.device)+'\nworktodo=unused.txt\ntmp_dir='+str(where)+
                                   '\nlog_file='+str(where/'screen.log')+'\nverbose=true\n',encoding='utf-8')
                    command=[str(exes[0]),'-ini',str(ini),'-gpu','-d',str(a.device),'--gpu-param','0',
                             '-sigma','0:26','-gpucurves',str(batch),'--ckpt','0','-v','--exponent',a.exponent,
                             '--exp-cache','off','-save',str(save),str(b1),'0']
                    start=time.perf_counter()
                    try:
                        proc=subprocess.run(command,cwd=where,env=env,input=f'{n}\n'.encode(),capture_output=True,timeout=a.timeout)
                    except subprocess.TimeoutExpired as error:
                        (where/'driver.log').write_bytes((error.stdout or b'')+(error.stderr or b''))
                        data['failure']=dict(case=where.name,reason='timeout');persist();raise
                    seconds=time.perf_counter()-start;text=(proc.stdout+proc.stderr).decode('utf-8',errors='replace')
                    (where/'driver.log').write_text(text,encoding='utf-8')
                    if proc.returncode or not save.exists():raise RuntimeError('Stage1 failed or produced no final save: '+str(where))
                    match=re.search(r'GPU stage1 returned: 0 gputime=([\d.]+) ms',text)
                    if not match or 'paused' in text.lower():raise RuntimeError('not a completed no-factor Stage1 batch')
                    geometry=re.search(r'CGBN<(\d+),\s*(\d+)>',text)
                    count=re.search(r'GPU: sigma=26 \(param 0, (\d+) curves\)',text)
                    if not geometry or not count or int(count[1])!=batch:raise RuntimeError('missing or mismatched CGBN geometry')
                    runtime=re.search(r'CUDA runtime (\d+)\.(\d+)',text)
                    selected=re.search(r'GPU: will use device (\d+):',text)
                    if (not runtime or int(runtime[1])*1000+int(runtime[2])*10!=device['cuda_runtime'] or
                        not selected or int(selected[1])!=a.device):
                        raise RuntimeError('Stage1 runtime/device differs from the performance profile')
                    lines=[line for line in save.read_text(encoding='utf-8').splitlines() if line.strip()]
                    if len(lines)!=batch:raise RuntimeError('missing completed Stage1 curves')
                    for index,line in enumerate(lines):
                        fields={k.strip().upper():v.strip() for k,v in (token.split('=',1) for token in line.split(';') if '=' in token)}
                        sigma=int(fields['SIGMA'].removeprefix('0:'));x=int(fields['X'],0)
                        if (fields.get('METHOD')!='ECM' or int(fields['N'],0)!=n or
                            sigma!=26+index or int(fields.get('PARAM','0')) or int(fields['B1'])!=b1 or
                            int(fields.get('Z','1'),0)!=1 or x!=reference[(bits,b1,sigma)] or
                            int(fields['CHECKSUM'])!=b1*sigma*n*x%4294967291):
                            raise RuntimeError('Stage1 save disagrees with independent point/checksum')
                    row=dict(bits=bits,b1=b1,batch=batch,repeat=repeat,warmup=repeat==0,process_seconds=seconds,
                             gpu_seconds=float(match[1])/1000,checked_curves=batch,save_sha256=sha(save),command=command)
                    data['runs'].append(row);persist()
                    if repeat:trials.append(seconds/batch);gpu.append(row['gpu_seconds']/batch)
                    print(f'stage1_tune_progress: bits={bits} B1={b1} batch={batch} repeat={repeat}/{grid["repeats"]} process={seconds/batch:.6f} s/curve',flush=True)
                median=statistics.median(trials)
                samples.append(dict(target_bits=bits,modulus_kind='mersenne',b1=b1,batch=batch,exponent=a.exponent,
                    benchmark_kind='known_mersenne_prime',benchmark_exponent=bits,sigma_first=26,repeats=grid['repeats'],
                    seconds=trials,median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in trials),
                    gpu_seconds=gpu,median_gpu_seconds=statistics.median(gpu),checked_curves=batch,hits=0,bad=0,
                    container_bits=int(geometry[2]),tpi=int(geometry[1])))
    if own!=sha(Path(__file__)) or any(sha(Path(path))!=value for path,value in identities.items()):
        raise RuntimeError('binary or measurement source changed; profile not published')
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+f'.partial.{os.getpid()}')
    partial.write_text(profile_text(device,a.tune_level,grid['repeats'],samples),encoding='utf-8')
    # Native reader validates every scope and its statistics before publication.
    checked=subprocess.run([str(exes[1]),'--check-stage1-tune-profile',str(partial)],
                           cwd=out,capture_output=True,text=True,errors='replace',timeout=60)
    (out/'profile_check.log').write_text(checked.stdout+checked.stderr,encoding='utf-8')
    if checked.returncode:raise RuntimeError('native Stage1 performance profile validation failed; partial retained')
    os.replace(partial,destination);data['complete']=True;data['profile_sha256']=sha(destination);persist()
    print(json.dumps(dict(passed=True,measured=len(samples),completed_batches=len(data['runs']))))


if __name__=='__main__':main()
