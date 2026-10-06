"""GPU1 same-binary ABBA comparison of NTT tile/outer or carry readback variants.

No previous binary's seconds are used as a baseline. Stage1 saves keep their
independent verification; all arithmetic checks and raw tail samples remain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
from diagnose_ecm_cost_drift import State
from bench_stage2_budget_scaling import fields
from calibrate_stage2_d import parse
from ecm_cost_model import features


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bits',type=int,nargs='+',default=[2203,4423,8191])
    p.add_argument('--d',type=int,nargs='+',default=[30030])
    p.add_argument('--b2',type=int,nargs='+',default=[86426340,5250000000])
    p.add_argument('--repeats',type=int,default=2,help='Number of ABBA blocks per shape')
    p.add_argument('--owner-mb',type=int,default=0);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--variant',choices=('tile','coop_outer','coop_radix','carry_readback'),default='tile')
    p.add_argument('--values',type=int,nargs=2,default=None,help='Baseline/candidate: tile 12 11 or carry_readback 0 1')
    p.add_argument('--coop-mode',type=int,choices=(0,1,2),default=2,help='Fixed outer mode when comparing tiles')
    p.add_argument('--radix-bits',type=int,choices=(5,6,7,8),default=8,help='Fixed forced-mode outer maximum')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    if a.repeats<1 or a.owner_mb<0 or any(b<=1000 for b in a.b2):raise ValueError('Invalid controls')
    if any(len(set(v))!=len(v) for v in (a.bits,a.d,a.b2)):raise ValueError('Duplicate shape')
    allowed={'tile':set(range(5,14)),'coop_outer':{0,1,2},'coop_radix':set(range(5,9)),'carry_readback':{0,1}}
    defaults={'tile':[12,11],'coop_outer':[2,1],'coop_radix':[8,6],'carry_readback':[0,1]}
    if a.variant=='coop_radix' and a.coop_mode!=1:raise ValueError('Radix comparison requires --coop-mode 1; mode 2 ignores the override')
    values=a.values or defaults[a.variant]
    if len(set(values))!=2 or not set(values)<=allowed[a.variant]:raise ValueError('Invalid variant values')
    study=json.loads(a.study.read_text(encoding='utf-8'));exe=a.exe.resolve();binary=sha(exe)
    if not study['complete'] or any(b not in study['controls']['bits'] for b in a.bits):raise ValueError('Missing verified Stage1 input')
    build=json.loads((exe.parent/'frozen_sources_manifest.json').read_text(encoding='utf-8'))
    if build['binary_sha256']!=binary:raise ValueError('Frozen candidate/binary mismatch')
    saves={b:Path(study['saves'][str(b)]['path']) for b in a.bits}
    for b,path in saves.items():
        if sha(path)!=study['saves'][str(b)]['sha256']:raise ValueError('Verified save changed')
    tools={name:sha(Path(__file__).with_name(name)) for name in (Path(__file__).name,
        'diagnose_ecm_cost_drift.py','bench_stage2_budget_scaling.py','calibrate_stage2_d.py','ecm_cost_model.py')}
    identity=dict(binary_sha256=binary,study_sha256=sha(a.study),sources=build['sources'],tools=tools)
    def verify():
        if sha(exe)!=binary or sha(a.study)!=identity['study_sha256']:raise ValueError('Frozen inputs changed')
        for name,digest in build['sources'].items():
            if sha(exe.parent/'sources'/name)!=digest.lower():raise ValueError('Frozen source changed')
        for name,digest in tools.items():
            if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Measurement tool changed')
        for b,path in saves.items():
            if sha(path)!=study['saves'][str(b)]['sha256']:raise ValueError('Save changed')
    verify()
    info=subprocess.run([str(exe),'--cost-device-info','--device','1'],capture_output=True,timeout=60,check=True)
    device=json.loads(info.stdout)
    if device['uuid_hex']!='8a67b1f8ef1c3177a822813a7ac2224d' or device['fixed_mode']!=3:raise ValueError('Unexpected device/backend')
    blocks=[(b,d,b2,rep) for b in a.bits for d in a.d for b2 in a.b2 for rep in range(a.repeats)]
    random.Random(a.seed).shuffle(blocks)
    sequence=[values[0],values[1],values[1],values[0]]
    cases=[dict(bits=b,D=d,B2=b2,rep=rep,value=mode,position=i,
                features=features(d,b2,b,8192))
           for b,d,b2,rep in blocks for i,mode in enumerate(sequence)]
    data=dict(schema=2,kind='same_binary_stage2_variant_abba',identity=identity,device=device,
        controls=dict(arena_mb=4096,owner_mb=a.owner_mb,chain_min=8192,launch_blocking=0,
                      variant=a.variant,coop_mode=a.coop_mode,radix_bits=a.radix_bits,
                      sequence=sequence,seed=a.seed,repeats=a.repeats),cases=cases,runs=[])
    frozen=out/'frozen_cases.json';frozen.write_text(json.dumps(data,indent=2),encoding='utf-8')
    data['cases_sha256']=sha(frozen)
    destination=out/'measurements.json'
    def persist():destination.write_text(json.dumps(data,indent=2),encoding='utf-8')
    persist();state=State();ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n',encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',
               NTT_GIANT_CHAIN_MIN='8192',NTT_POINT_MERSENNE='1',
               NTT_FOLD_DEVICE_MAX_MB=str(a.owner_mb),NTT_FUSE_COOP_OUTER=str(a.coop_mode),
               NTT_FUSE_COOP_M=str(a.radix_bits),CUDA_LAUNCH_BLOCKING='0')
    for index,c in enumerate(cases):
        name=f'{index:03d}_m{c["bits"]}_d{c["D"]}_b{c["B2"]}_{a.variant}{c["value"]}'
        log=out/(name+'.log');result=out/(name+'.jsonl')
        cmd=[str(exe),'--ini',str(ini),'--save',str(saves[c['bits']]),'--device','1',
             '--b2',str(c['B2']),'--d',str(c['D']),'--arena-mb','4096','--factor-only',
             '--log',str(log),'--results',str(result)]
        samples=[];errors=[];stop=threading.Event()
        def observe():
            try:
                while not stop.is_set():samples.append(state.sample());stop.wait(0.2)
            except Exception as e:errors.append(str(e))
        sampler=threading.Thread(target=observe);sampler.start();start=time.perf_counter();child=None
        try:
            key={'tile':'NTT_FUSE_T','coop_outer':'NTT_FUSE_COOP_OUTER','coop_radix':'NTT_FUSE_COOP_M',
                 'carry_readback':'NTT_CARRY_PINNED_SYNC'}[a.variant]
            control={key:str(c['value'])}
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env|control)
            stdout,stderr=child.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            child.communicate(timeout=30);raise RuntimeError('Own trial process tree timed out')
        finally:elapsed=time.perf_counter()-start;stop.set();sampler.join()
        (out/(name+'_driver.log')).write_bytes(stdout+stderr)
        if child.returncode:raise RuntimeError('Trial failed '+name)
        text=log.read_text(encoding='utf-8');r=json.loads(result.read_text(encoding='utf-8'));n=(1<<c['bits'])-1
        if r['bad_factors'] or any(not 1<int(f)<n or n%int(f) for f in r['factors']):raise ValueError('Invalid proper factor')
        if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Missing arithmetic checks')
        rb=fields(text,'ntt_carry_readback_stats') if a.variant=='carry_readback' else None
        ntt=fields(text,'ntt_workspace_stats')
        if rb and (int(rb['requested'])!=c['value'] or bool(int(rb['pinned_calls']))!=bool(c['value']) or int(rb['host_peak_bytes'])>1048576):raise ValueError('Requested transport not exercised')
        if int(fields(text,'real_batched_breakdown')['arena_overflow']):raise ValueError('Unexpected allocation route')
        leaf=fields(text,'descent_values')['hash']
        for old in data['runs']:
            if all(old['case'][k]==c[k] for k in ('bits','D','B2')):
                if old['leaf_hash']!=leaf or old['factors']!=r['factors']:raise ValueError('Transport changed arithmetic')
                if a.variant=='carry_readback' and old['workspace']['full_peak_bytes']!=ntt['full_peak_bytes']:raise ValueError('NTT payload changed')
        verify()
        if sha(frozen)!=data['cases_sha256']:raise ValueError('Cases changed')
        phases=parse(text);row=dict(name=name,case=c,command=cmd,process_seconds=elapsed,phases=phases,
            log=str(log),log_sha256=sha(log),factors=r['factors'],leaf_hash=leaf,workspace=ntt,
            readback=rb,carry_time=fields(text,'real_batched_carrytime'),samples=samples,monitor_errors=errors)
        data['runs'].append(row);persist();print(name,'full',phases['full'],'NTT peak',ntt['full_peak_bytes'],flush=True)
    comparisons=[]
    for b in a.bits:
        for d in a.d:
            for b2 in a.b2:
                rows=[r for r in data['runs'] if (r['case']['bits'],r['case']['D'],r['case']['B2'])==(b,d,b2)]
                med={str(m):statistics.median(r['phases']['full'] for r in rows if r['case']['value']==m) for m in values}
                ranges={str(m):[min(r['phases']['full'] for r in rows if r['case']['value']==m),
                               max(r['phases']['full'] for r in rows if r['case']['value']==m)] for m in values}
                reduction=100*(1-med[str(values[1])]/med[str(values[0])])
                comparisons.append(dict(bits=b,D=d,B2=b2,median_seconds=med,ranges=ranges,reduction_percent=reduction))
    data['comparisons']=comparisons;data['complete']=True;persist();print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
