"""Serial scoped Stage1 amortization and Stage2 phase calibration on GPU1.

Keeps production arithmetic checks; preparation/reference work is outside timings.
Use a fresh output directory or --resume with identical binaries and source hashes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
from calibrate_stage2_d import features, parse
from bench_stage2_budget_scaling import fields, Nvml

ROOT = Path(__file__).resolve().parents[2]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--stage1',type=Path,default=ROOT/'build_cuda_cmake/ecm_cuda.exe')
    p.add_argument('--save-dir',type=Path,required=True,help='Verified m2203/m4423/m8191.save inputs, B1=1000 sigma26')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',type=int,nargs='+',default=[2203,4423,8191])
    p.add_argument('--d',type=int,nargs='+',default=[30030,60060,120120])
    p.add_argument('--b2',type=int,default=3000000000)
    p.add_argument('--holdout-b2',type=int,default=6000000000)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--stage1-batch',type=int,nargs='+',default=[1,12])
    p.add_argument('--arena-mb',type=int,default=4096)
    p.add_argument('--resident-mb',type=int,default=640)
    p.add_argument('--name-hits',type=int,choices=(0,1),default=0,help='Optional prime-witness naming; default factor-only')
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if (not set(a.bits)<= {2203,4423,8191} or a.repeats<1 or a.b2<=1000 or
        a.holdout_b2<=1000 or a.arena_mb<1 or a.resident_mb<1 or
        any(d<6 or d%2 for d in a.d) or any(b<1 or b>96 for b in a.stage1_batch)):
        p.error('Invalid calibration ranges/configuration')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not a.resume: p.error('Use a fresh output directory')
    exe1,exe2=a.stage1.resolve(),a.stage2.resolve()
    manifest=json.loads((exe2.parent/'frozen_sources_manifest.json').read_text(encoding='utf-8'))
    frozen=exe2.parent/'sources'
    if digest(exe2)!=manifest['binary_sha256']: raise ValueError('Stage2 binary differs from frozen manifest')
    def verify():
        if digest(exe1)!=identity['stage1_sha256'] or digest(exe2)!=identity['stage2_sha256']:
            raise ValueError('Binary changed during study')
        for name,sha in manifest['sources'].items():
            if digest(frozen/name).lower()!=sha.lower(): raise ValueError('Frozen source changed: '+name)
        for name,sha in identity['tools'].items():
            if digest(ROOT/name)!=sha: raise ValueError('Measurement tool changed: '+name)
    identity=dict(stage1_sha256=digest(exe1),stage2_sha256=digest(exe2),
        build_manifest_sha256=digest(exe2.parent/'build_manifest.json'),sources=manifest['sources'],
        tools={name:digest(ROOT/name) for name in ('tools/bench/measure_ecm_costs.py',
          'tools/bench/calibrate_stage2_d.py','tools/bench/bench_stage2_budget_scaling.py','tools/stat/suyama_mont_ref.py')})
    controls=dict(bits=a.bits,d=a.d,b2=a.b2,holdout_b2=a.holdout_b2,repeats=a.repeats,
        stage1_batch=a.stage1_batch,arena_mb=a.arena_mb,resident_mb=a.resident_mb,name_hits=a.name_hits)
    study_path=out/'measurements.json'
    if a.resume:
        data=json.loads(study_path.read_text(encoding='utf-8'))
        if data['identity']!=identity or data['controls']!=controls: raise ValueError('Resume identity/controls changed')
    else:
        data=dict(schema=1,identity=identity,controls=controls,stage1=[],stage2=[],saves={})
    def persist(): study_path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    spec=importlib.util.spec_from_file_location('mont_ref',ROOT/'tools/stat/suyama_mont_ref.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    nvml=Nvml();data['device']=dict(uuid='GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d',name=nvml.name)
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',NTT_POINT_MERSENNE='1',
               NTT_NAME_HITS=str(a.name_hits))
    def run(cmd,stdin,where,log):
        verify();start=time.perf_counter()
        child=subprocess.Popen(cmd,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=where,env=env)
        try:
            stdout,stderr=child.communicate(input=stdin,timeout=900)
        except subprocess.TimeoutExpired:
            # Only terminate the process tree we launched; Stage2 has its own child worker.
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            stdout,stderr=child.communicate(timeout=30)
            log.write_bytes(stdout+stderr)
            raise RuntimeError('Own calibration process tree timed out')
        seconds=time.perf_counter()-start
        log.write_bytes(stdout+stderr);verify()
        if child.returncode: raise RuntimeError('Child failed; inspect '+str(log))
        return seconds,(stdout+stderr).decode('utf-8',errors='replace')
    # Reference validation always happens before timed GPU work for that modulus.
    for bits in a.bits:
        source=a.save_dir.resolve()/f'm{bits}.save';text=source.read_text(encoding='utf-8')
        if 'B1=1000;' not in text or 'SIGMA=26;' not in text or f'N=(2^{bits}-1);' not in text:
            raise ValueError('Save does not match the calibration input')
        x=int(re.search(r'\bX=(0x[0-9a-fA-F]+)',text)[1],16);n=(1<<bits)-1
        if x!=ref.stage1(26,1000,n)['x']: raise ValueError('Save/independent CPU mismatch')
        checksum=int(re.search(r'CHECKSUM=(\d+)',text)[1])
        if checksum!=1000*26*n*x%4294967291: raise ValueError('Save checksum mismatch')
        saved=out/f'm{bits}.save'
        if saved.exists() and saved.read_bytes()!=source.read_bytes(): raise ValueError('Saved input changed')
        if not saved.exists(): saved.write_bytes(source.read_bytes())
        data['saves'][str(bits)]=dict(path=str(saved),sha256=digest(saved),Q_sha256=hashlib.sha256(f'{x:x}'.encode()).hexdigest())
    persist()
    # Stage1: one warmup per size, then batch-specific cold-process measurements.
    for bits in a.bits:
        for batch in a.stage1_batch:
            for rep in range(a.repeats+1):
                name=f's1_m{bits}_batch{batch}_r{rep}'
                if any(r['name']==name for r in data['stage1']): continue
                where=out/name;where.mkdir(exist_ok=True);save=where/'stage1.save'
                if save.exists(): raise ValueError('Unaccounted Stage1 output: '+str(save))
                ini=where/'ecm.ini';ini.write_text('[gpu]\ndevice=1\n[queue]\nworktodo=unused.txt\ntmp_dir=.\n',encoding='utf-8')
                cmd=[str(exe1),'-ini',str(ini),'--method','gpu','--gpu-param','0','--exponent','lcm',
                     '--exp-cache','off','-d','1','-sigma','26','-gpucurves',str(batch),'--ckpt','0','-save',str(save),'1000']
                seconds,log=run(cmd,f'{(1<<bits)-1}\n'.encode(),where,where/'driver.log')
                gputime=float(re.search(r'GPU stage1 returned: 0 gputime=([\d.]+) ms',log)[1])/1000
                lines=save.read_text(encoding='utf-8').splitlines()
                if len(lines)!=batch: raise ValueError('Missing Stage1 curves')
                # Verify every curve, outside GPU/process timing. Batch contains sigma26+i.
                for i,line in enumerate(lines):
                    sigma=int(re.search(r'SIGMA=(\d+)',line)[1]);xx=int(re.search(r'X=(0x[0-9a-fA-F]+)',line)[1],16)
                    cc=int(re.search(r'CHECKSUM=(\d+)',line)[1]);n=(1<<bits)-1
                    if sigma!=26+i or xx!=ref.stage1(sigma,1000,n)['x'] or cc!=1000*sigma*n*xx%4294967291:
                        raise ValueError('Stage1 batch/CPU mismatch')
                data['stage1'].append(dict(name=name,bits=bits,B1=1000,torsion=1,batch=batch,warmup=rep==0,
                    process_seconds=seconds,gpu_seconds=gputime,amortized_process_seconds=seconds/batch,
                    amortized_gpu_seconds=gputime/batch,save_sha256=digest(save),command=cmd))
                persist();print('STAGE1',name,'wall/curve',seconds/batch,'gpu/curve',gputime/batch,flush=True)
    cases=[]
    for bits in a.bits:
        for rep in range(a.repeats):
            for d in a.d:
                for owner in (a.resident_mb,0): cases.append((bits,d,a.b2,owner,rep,'train'))
            for owner in (a.resident_mb,0): cases.append((bits,a.d[len(a.d)//2],a.holdout_b2,owner,rep,'holdout'))
    for bits,d,b2,owner,rep,kind in cases:
        name=f's2_m{bits}_d{d}_b{b2}_o{owner}_r{rep}_{kind}'
        if any(r['name']==name for r in data['stage2']): continue
        where=out/name;where.mkdir(exist_ok=True);result=where/'result.jsonl';log=where/'engine.log'
        if result.exists(): raise ValueError('Unaccounted Stage2 result: '+str(result))
        saved=Path(data['saves'][str(bits)]['path']);saved_hash=digest(saved)
        env['NTT_FOLD_DEVICE_MAX_MB']=str(owner)
        cmd=[str(exe2),'--save',str(saved),'--b2',str(b2),'--d',str(d),'--device','1',
             '--arena-mb',str(a.arena_mb),'--results',str(result),'--log',str(log)]
        if a.name_hits==0: cmd.append('--factor-only')
        seconds,_=run(cmd,None,where,where/'driver.log');text=log.read_text(encoding='utf-8',errors='replace')
        if digest(saved)!=saved_hash: raise ValueError('Stage2 mutated save')
        for token in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1','stage1_skipped=1','fixed=3','point_arithmetic: xadd6=1'):
            if token not in text: raise ValueError('Required configuration/check absent: '+token)
        raw=json.loads(result.read_text(encoding='utf-8'))
        if raw['bad_factors'] or any(not 1<int(f)<(1<<bits)-1 or pow(2,bits,int(f))!=1 for f in raw['factors']):
            raise ValueError('Stage2 reported an invalid proper divisor')
        fold=fields(text,'real_batched_folddevice');account=fields(text,'ntt_arena_accounting')
        if account['version']!='2' or bool(int(fold['enabled']))!=(owner>0): raise ValueError('Unexpected accounting/path')
        if not owner and fold['fallback']!='budget': raise ValueError('Fallback was not forced by owner budget')
        phases=parse(text)
        row=dict(name=name,kind=kind,bits=bits,B1=1000,B2=b2,D=d,owner_mb=owner,rep=rep,process_seconds=seconds,
            phases=phases,features=features(d,b2,bits),fold=fold,accounting=account,
            ntt=fields(text,'ntt_workspace_stats'),split=fields(text,'real_batched_split'),
            oracle=fields(text,'s4_oracle_stats'),result=raw,command=cmd,log=str(log),log_sha256=digest(log))
        data['stage2'].append(row);persist()
        print('STAGE2',name,'full',phases['full'],'fold',phases['fold'],flush=True)
    data['complete']=True;persist()


if __name__=='__main__': main()
