"""Crossed, seeded-order GPU1 cost replay with host/GPU state observations.

Replay diagnostics are not independent validation after a model uses their
measurements. Never changes production queues, binaries, or GPU settings.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
from bench_stage2_budget_scaling import Nvml,fields
from calibrate_stage2_d import parse
from ecm_cost_model import predict,features,admits


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class State:
    def __init__(self):
        self.gpu=Nvml();self.previous=None
    def sample(self):
        out=self.gpu.sample();out['monotonic']=time.perf_counter()
        for key,fn,extra in [('graphics_mhz','nvmlDeviceGetClockInfo',[0]),
                             ('sm_mhz','nvmlDeviceGetClockInfo',[1]),
                             ('mem_mhz','nvmlDeviceGetClockInfo',[2]),
                             ('temperature_c','nvmlDeviceGetTemperature',[0]),
                             ('power_mw','nvmlDeviceGetPowerUsage',[])]:
            value=ctypes.c_uint();code=getattr(self.gpu.lib,fn)(self.gpu.handle,*extra,ctypes.byref(value))
            out[key]=value.value if code==0 else None;out[key+'_status']=code
        idle,kernel,user=ctypes.c_ulonglong(),ctypes.c_ulonglong(),ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)):
            current=(idle.value,kernel.value+user.value)
            if self.previous:
                di,dt=(current[i]-self.previous[i] for i in range(2))
                out['system_cpu_percent']=100*(1-di/dt) if dt>0 else None
            self.previous=current
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',type=int,nargs='+',default=[2203,4423])
    p.add_argument('--d',type=int,nargs='+',default=[60060,120120])
    p.add_argument('--b2',type=int,nargs='+',default=[3000000000,4500000000,6000000000])
    p.add_argument('--owners',type=int,nargs='+',default=[640,0]);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261006);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    if a.repeats<1 or len(set(a.bits))!=len(a.bits) or len(set(a.d))!=len(a.d):raise ValueError('Invalid/duplicate diagnostic matrix')
    study=json.loads(a.study.read_text());model=json.loads(a.profile.read_text())
    exe=a.stage2.resolve();binary=sha(exe);profile_sha=sha(a.profile);study_sha=sha(a.study)
    if model['identity']['stage2_sha256']!=binary or model['source_sha256']!=study_sha:raise ValueError('Binary/study/model mismatch')
    saves={bits:Path(study['saves'][str(bits)]['path']) for bits in a.bits}
    for bits,path in saves.items():
        if sha(path)!=study['saves'][str(bits)]['sha256']:raise ValueError('Verified save changed')
    controls=dict(bits=a.bits,d=a.d,b2=a.b2,owners=a.owners,repeats=a.repeats,seed=a.seed)
    identity=dict(binary_sha256=binary,profile_sha256=profile_sha,study_sha256=study_sha,tool_sha256=sha(__file__))
    destination=out/'measurements.json'
    if a.resume:
        data=json.loads(destination.read_text())
        if data['identity']!=identity or data['controls']!=controls:raise ValueError('Resume inputs changed')
    else:
        cases=[]
        for rep in range(a.repeats):
            block=[dict(bits=bits,D=d,B2=b,owner_mb=owner,rep=rep) for bits in a.bits for d in a.d for b in a.b2 for owner in a.owners]
            random.Random(a.seed+rep).shuffle(block);cases.extend(block)
        for case in cases:
            minimum=model.get('chain_min',32768)
            f=features(case['D'],case['B2'],case['bits'],minimum)
            scope=next(s for s in model['stage2'] if s['owner_mb']==case['owner_mb'] and admits(s,f))
            phases,f=predict(case['bits'],case['D'],case['B2'],scope['rates'],minimum)
            case.update(prediction=phases,features=f,arena_mb=scope['arena_mb'])
        data=dict(schema=1,kind='diagnostic_replay',identity=identity,controls=controls,cases=cases,runs=[])
        (out/'frozen_cases.json').write_text(json.dumps(data,indent=2))
        data['cases_sha256']=sha(out/'frozen_cases.json')
    def persist():destination.write_text(json.dumps(data,indent=2))
    persist();state=State()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_CARRY_CHECK_FUSED='0',NTT_STAGE1_Q_DUMP='1',NTT_POINT_MERSENNE='1')
    env['NTT_GIANT_CHAIN_MIN']=str(model.get('chain_min',32768))
    ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n',encoding='utf-8')
    for index,case in enumerate(data['cases']):
        name=f'{index:02d}_m{case["bits"]}_d{case["D"]}_b{case["B2"]}_o{case["owner_mb"]}_r{case["rep"]}'
        if any(r['name']==name for r in data['runs']):continue
        log=out/(name+'.log');result=out/(name+'.jsonl')
        if log.exists() or result.exists():raise ValueError('Unaccounted run output; inspect before resuming')
        samples=[];errors=[];stop=threading.Event()
        def observe():
            try:
                while not stop.is_set():
                    samples.append(state.sample());stop.wait(0.2)
            except Exception as e:errors.append(str(e))
        sampler=threading.Thread(target=observe);sampler.start()
        cmd=[str(exe),'--ini',str(ini),'--save',str(saves[case['bits']]),'--device','1',
             '--b2',str(case['B2']),'--d',str(case['D']),'--arena-mb',str(case['arena_mb']),
             '--factor-only','--results',str(result),'--log',str(log)]
        child=None;start=time.perf_counter()
        try:
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env|{'NTT_FOLD_DEVICE_MAX_MB':str(case['owner_mb'])})
            stdout,stderr=child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            child.communicate(timeout=30);raise RuntimeError('Own diagnostic child timed out')
        finally:
            elapsed=time.perf_counter()-start;stop.set();sampler.join()
        driver=out/(name+'_driver.log');driver.write_bytes(stdout+stderr)
        if child.returncode:raise RuntimeError('Diagnostic curve failed: '+name)
        text=log.read_text();row=json.loads(result.read_text())
        if row['bad_factors'] or not all(s in text for s in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):
            raise ValueError('Arithmetic failure/missing check')
        n=(1<<case['bits'])-1
        if any(not 1<int(f)<n or n%int(f) for f in row['factors']):raise ValueError('Invalid proper factor')
        fold=fields(text,'real_batched_folddevice')
        if bool(int(fold['enabled']))!=(case['owner_mb']>0):raise ValueError('Unexpected owner route')
        leaf=fields(text,'descent_values')['hash']
        for old in data['runs']:
            if all(old['case'][k]==case[k] for k in ('bits','D','B2')) and (old['leaf_hash']!=leaf or old['factors']!=row['factors']):
                raise ValueError('Cross-path/repetition changed output')
        if sha(exe)!=binary or sha(a.profile)!=profile_sha or sha(a.study)!=study_sha or sha(__file__)!=identity['tool_sha256'] or sha(out/'frozen_cases.json')!=data['cases_sha256']:
            raise ValueError('Frozen inputs changed')
        if sha(saves[case['bits']])!=study['saves'][str(case['bits'])]['sha256']:raise ValueError('Save changed')
        phases=parse(text);summary={}
        for key in ('graphics_mhz','sm_mhz','mem_mhz','temperature_c','power_mw','system_cpu_percent','gpu','used'):
            values=[s[key] for s in samples if s.get(key) is not None]
            summary[key]=dict(min=min(values),max=max(values),mean=statistics.mean(values)) if values else None
        observation=dict(name=name,case=case,actual=phases,process_seconds=elapsed,
            error_percent=100*(case['prediction']['full']/phases['full']-1),leaf_hash=leaf,factors=row['factors'],
            log=str(log),log_sha256=sha(log),samples=samples,state_summary=summary,monitor_errors=errors,
            ntt=fields(text,'ntt_workspace_stats'),gdevice=fields(text,'real_batched_gdevice'),
            multiply=fields(text,'s4_multiply_stats'))
        data['runs'].append(observation);persist()
        print(name,'full',phases['full'],'error%',round(observation['error_percent'],2),'cpu%',round(summary['system_cpu_percent']['mean'],1) if summary['system_cpu_percent'] else None,flush=True)
    data['complete']=True;persist()


if __name__=='__main__':main()
