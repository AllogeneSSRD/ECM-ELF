"""GPU1 same-binary ABBA of the opt-in short-chain policy.

Reuse verified Stage1 inputs only. Every actual coordinate chunk is audited;
full affine comparisons run after timing. No runtime cost profile is published.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from bench_stage2_budget_scaling import fields
from calibrate_stage2_d import parse
from ecm_cost_model import features


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def chunks(f):
    p=f['P'];w=(f['bits']+63)//64;capacity=max(p,(256<<20)//(16*w))
    size=p*((capacity+p-1)//p);whole,tail=divmod(f['I'],size)
    return [size]*whole+([tail] if tail else [])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bits',type=int,nargs='+',default=[2203,4423,8191])
    p.add_argument('--d',type=int,default=120120);p.add_argument('--points',type=int,nargs='+',default=[4095,4096,4097,8191,8192,8193])
    p.add_argument('--chunk-tail',type=int,nargs='*',default=[],help='Also test one full coordinate chunk plus each requested tail')
    p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
    if (a.repeats<1 or a.d<6 or a.d%2 or a.d>200000000 or len(set(a.bits))!=len(a.bits) or
        any(n<3 or n>5000000 or a.d*(n-2)<=1000 for n in a.points) or any(n<2 or n>=8192 for n in a.chunk_tail)):
        raise ValueError('Invalid D/point/repetition range')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh directory')
    exe=a.exe.resolve();study=read(a.study);manifest=exe.parent/'frozen_sources_manifest.json';build=read(manifest)
    if not study['complete'] or any(b not in study['controls']['bits'] for b in a.bits):raise ValueError('Need verified Stage1 widths')
    saves={b:Path(study['saves'][str(b)]['path']) for b in a.bits}
    names=('bench_stage2_short_policy.py','bench_stage2_budget_scaling.py','calibrate_stage2_d.py','ecm_cost_model.py')
    identity=dict(binary_sha256=sha(exe),calibration_binary_sha256=study['identity']['stage2_sha256'],study_sha256=sha(a.study),
                  build_manifest_sha256=sha(manifest),tools={n:sha(Path(__file__).with_name(n)) for n in names})
    if build['binary_sha256']!=identity['binary_sha256']:raise ValueError('Build receipt/binary mismatch')
    def verify():
        if sha(exe)!=identity['binary_sha256'] or sha(a.study)!=identity['study_sha256'] or sha(manifest)!=identity['build_manifest_sha256']:raise ValueError('Input changed')
        for b,saved in saves.items():
            if sha(saved)!=study['saves'][str(b)]['sha256']:raise ValueError('Verified save changed')
        for n,h in build['sources'].items():
            if sha(exe.parent/'sources'/n)!=h.lower():raise ValueError('Frozen source changed')
        for n,h in identity['tools'].items():
            if sha(Path(__file__).with_name(n))!=h:raise ValueError('Collector changed')
    verify()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',NTT_POINT_MERSENNE='1',
               NTT_FOLD_DEVICE_MAX_MB='0',NTT_GIANT_CHAIN_BLOCK='64',NTT_GIANT_CHAIN_SMALL_MAX='8192',CUDA_LAUNCH_BLOCKING='0')
    info=json.loads(subprocess.run([str(exe),'--cost-device-info','--device','1'],env=env,capture_output=True,check=True,timeout=60).stdout)
    if info['uuid_hex']!='8a67b1f8ef1c3177a822813a7ac2224d':raise ValueError('Wrong GPU')
    cases=[]
    for b in a.bits:
        f=features(a.d,a.d*4094,b,8192)
        w=(b+63)//64;capacity=max(f['P'],(256<<20)//(16*w));size=f['P']*((capacity+f['P']-1)//f['P'])
        for n in sorted(set(a.points+[size+t for t in a.chunk_tail])):cases.append(dict(bits=b,points=n,B2=a.d*(n-2)))
    data=dict(schema=1,kind='short_chain_policy_abba',identity=identity,device=info,
        controls=dict(D=a.d,arena_mb=4096,owner_mb=0,base_block=64,short_block=8,short_max=8192,
                      baseline_min=8192,adaptive_min=4096,sequence=['baseline','adaptive','adaptive','baseline'],repeats=a.repeats),
        cases=cases,runs=[],gates=[])
    dest=out/'measurements.json'
    def persist():dest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n',encoding='utf-8');persist();reference={};factors={}
    def run(case,mode,index,gate=False):
        b=case['bits'];b2=case['B2'];minimum=8192 if mode=='baseline' else 4096
        f=features(a.d,b2,b,minimum);sizes=chunks(f);blocks=[8 if mode=='adaptive' and n<8192 else 64 for n in sizes]
        chain=[(n,c) for n,c in zip(sizes,blocks) if n>=minimum]
        name=f'm{b}_i{case["points"]}_{mode}_{index}'+('_gate' if gate else '')
        log=out/(name+'.log');result=out/(name+'.jsonl')
        cmd=[str(exe),'--ini',str(ini),'--save',str(saves[b]),'--device','1','--b2',str(b2),'--d',str(a.d),
             '--arena-mb','4096','--factor-only','--log',str(log),'--results',str(result)]
        variant=dict(NTT_GIANT_CHAIN_MIN=str(minimum),NTT_GIANT_CHAIN_SMALL_BLOCK='8' if mode=='adaptive' else '0',NTT_GIANT_CHAIN_CHECK='1' if gate else '0')
        begin=time.perf_counter();child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env|variant)
        try:stdout,stderr=child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.communicate(timeout=30);raise RuntimeError('Owned curve timed out')
        elapsed=time.perf_counter()-begin;(out/(name+'_driver.log')).write_bytes(stdout+stderr)
        if child.returncode:raise RuntimeError('Curve failed: '+name)
        text=log.read_text(encoding='utf-8');r=read(result);n=(1<<b)-1
        if not all(s in text for s in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')) or r['bad_factors']:raise ValueError('Missing arithmetic checks')
        if r['B1']!=1000 or r['B2']!=b2 or r['sigma']!=26 or int(r['N_hex'],16)!=n or any(not 1<int(q)<n or n%int(q) for q in r['factors']):raise ValueError('Input/proper factors differ')
        seed=fields(text,'real_giant_seed');expected_seed=sum(2*((k+c-1)//c)+1 for k,c in chain)
        if int(seed['chunks'])!=len(chain) or int(seed['points'])!=expected_seed:raise ValueError('Actual chain schedule differs')
        dispatch=[dict(re.findall(r'(\w+)=([^\s]+)',s)) for s in re.findall(r'giant_chain_policy: (.*)',text)]
        if mode=='adaptive':
            if len(dispatch)!=len(sizes):raise ValueError('Missing per-chunk policy')
            for row,k,c in zip(dispatch,sizes,blocks):
                if (int(row['npts']),row['route'],int(row['block']))!=(k,'chain' if k>=minimum else 'ladder',c if k>=minimum else 0):raise ValueError('Per-chunk dispatch differs')
        elif dispatch:raise ValueError('Disabled policy emitted dispatch')
        gate_rows=[dict(re.findall(r'(\w+)=([^\s]+)',s)) for s in re.findall(r'giant_chain_check: (.*)',text)]
        if gate:
            if len(gate_rows)!=len(chain):raise ValueError('Missing chunk affine comparison')
            for row,(k,c) in zip(gate_rows,chain):
                if (int(row['points']),int(row['per_block']),int(row['mismatches']))!=(k,c,0):raise ValueError('Affine gate failed')
        leaf=fields(text,'descent_values')['hash'];key=(b,b2,mode)
        if key in reference and reference[key]!=(leaf,r['factors']):raise ValueError('Same-policy output changed')
        reference[key]=(leaf,r['factors']);pair=(b,b2)
        if pair in factors and factors[pair]!=r['factors']:raise ValueError('Policy changed proper factor set')
        factors[pair]=r['factors'];verify()
        record=dict(name=name,case=case,mode=mode,gate=gate,process_seconds=elapsed,phases=parse(text),seed=seed,
            dispatch=dispatch,affine_gates=gate_rows,leaf_hash=leaf,factors=r['factors'],features=f,
            log=str(log),log_sha256=sha(log),ntt=fields(text,'ntt_workspace_stats'),multiply=fields(text,'s4_multiply_stats'))
        data['gates' if gate else 'runs'].append(record);persist();print(name,'full',record['phases']['full'],'giant',record['phases']['giant'],flush=True)
    for case in cases:
        for rep in range(a.repeats):
            for i,mode in enumerate(data['controls']['sequence']):run(case,mode,rep*4+i)
    for case in cases:run(case,'adaptive',0,True)
    data['comparisons']=[]
    for case in cases:
        rows=[r for r in data['runs'] if r['case']==case]
        med={m:{k:statistics.median(r['phases'][k] for r in rows if r['mode']==m) for k in ('full','giant')} for m in ('baseline','adaptive')}
        data['comparisons'].append(dict(case=case,medians=med,full_reduction_percent=100*(1-med['adaptive']['full']/med['baseline']['full'])))
    data['complete']=True;persist();print(json.dumps(data['comparisons'],indent=2))


if __name__=='__main__':main()
