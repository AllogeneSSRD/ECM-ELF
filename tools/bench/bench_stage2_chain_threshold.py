"""Same-binary GPU1 giant ladder/chain crossover trials plus affine-point gates.

The two routes have different projective scales: leaf hashes are recorded,
while correctness uses affine-coordinate comparisons and proper factors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from calibrate_stage2_d import parse,features
from bench_stage2_budget_scaling import fields,Nvml


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bits',type=int,nargs='+',default=[2203,4423,8191])
    p.add_argument('--b2',type=int,nargs='+',default=[1000000000,3000000000]);p.add_argument('--d',type=int,default=120120)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();binary=sha(exe);study=json.loads(a.study.read_text());study_sha=sha(a.study)
    if study['identity']['stage2_sha256']!=binary:raise ValueError('Binary/study mismatch')
    if any(bits not in study['controls']['bits'] for bits in a.bits):raise ValueError('Unverified input width')
    gpu=Nvml();saves={bits:Path(study['saves'][str(bits)]['path']) for bits in a.bits}
    for bits,save in saves.items():
        if sha(save)!=study['saves'][str(bits)]['sha256']:raise ValueError('Verified save changed')
    data=dict(schema=1,binary_sha256=binary,study_sha256=study_sha,tool_sha256=sha(__file__),
        controls=dict(bits=a.bits,b2=a.b2,D=a.d,sequence=['ladder','chain','chain','ladder'],chain_block=64,owner_mb=640),
        gpu=gpu.name,runs=[],gates=[],comparisons=[])
    path=out/'measurements.json'
    def persist():path.write_text(json.dumps(data,indent=2))
    persist();ini=out/'manual.ini';ini.write_text('[gpu]\ndevice=1\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1',NTT_D_MODEL='0',NTT_STAGE1_Q_DUMP='1',NTT_CARRY_CHECK_FUSED='0',NTT_FOLD_DEVICE_MAX_MB='640')
    def run(bits,b2,mode,rep,gate=False):
        name=f'm{bits}_b{b2}_{mode}_{rep}'+('_gate' if gate else '')
        log=out/(name+'.log');result=out/(name+'.jsonl')
        cmd=[str(exe),'--ini',str(ini),'--save',str(saves[bits]),'--device','1','--b2',str(b2),'--d',str(a.d),
             '--arena-mb','4096','--factor-only','--results',str(result),'--log',str(log)]
        controls={'NTT_GIANT_CHAIN_MIN':'0' if mode=='chain' else str(2**63-1),
                  'NTT_GIANT_CHAIN_BLOCK':'64','NTT_GIANT_CHAIN_CHECK':'1' if gate else '0'}
        start=time.perf_counter();r=subprocess.Popen(cmd,env=env|controls,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:stdout,stderr=r.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(r.pid),'/T','/F'],capture_output=True)
            r.communicate(timeout=30);raise RuntimeError('Own threshold trial tree timed out')
        elapsed=time.perf_counter()-start;(out/(name+'_driver.log')).write_bytes(stdout+stderr)
        if r.returncode:raise RuntimeError('Own trial failed: '+name)
        text=log.read_text();row=json.loads(result.read_text());n=(1<<bits)-1
        if row['bad_factors'] or any(not 1<int(f)<n or n%int(f) for f in row['factors']):raise ValueError('Invalid proper factor')
        if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Arithmetic checks missing')
        seed=fields(text,'real_giant_seed');f=features(a.d,b2,bits)
        if (int(seed['chunks'])>0)!=(mode=='chain'):raise ValueError('Requested giant route not used')
        gate_info=None
        if gate:
            gate_info=fields(text,'giant_chain_check')
            if int(gate_info['points'])!=f['I'] or int(gate_info['mismatches']):raise ValueError('Affine chain/ladder point gate failed')
        if sha(exe)!=binary or sha(a.study)!=study_sha or sha(__file__)!=data['tool_sha256']:raise ValueError('Frozen inputs changed')
        phases=parse(text);record=dict(name=name,bits=bits,B2=b2,D=a.d,mode=mode,gate=gate,
            process_seconds=elapsed,phases=phases,features=f,factors=row['factors'],
            leaf_hash=fields(text,'descent_values')['hash'],seed=seed,affine_gate=gate_info,
            log=str(log),log_sha256=sha(log),ntt=fields(text,'ntt_workspace_stats'))
        for old in data['runs']+data['gates']:
            if old['bits']==bits and old['B2']==b2 and old['factors']!=record['factors']:raise ValueError('Route changed the raw factor set')
        data['gates' if gate else 'runs'].append(record);persist()
        print(name,'G',f['G'],'full',phases['full'],'giant',phases['giant'],flush=True)
    for bits in a.bits:
        for b2 in a.b2:
            for rep,mode in enumerate(data['controls']['sequence']):run(bits,b2,mode,rep)
    # Full affine comparisons are deliberately outside the performance sequence.
    for bits in a.bits:
        for b2 in a.b2:run(bits,b2,'chain',0,True)
    for bits in a.bits:
        for b2 in a.b2:
            rows=[r for r in data['runs'] if r['bits']==bits and r['B2']==b2]
            med={mode:{key:statistics.median(r['phases'][key] for r in rows if r['mode']==mode)
                       for key in ('full','giant')} for mode in ('ladder','chain')}
            data['comparisons'].append(dict(bits=bits,B2=b2,D=a.d,medians=med,
                full_reduction_percent=100*(1-med['chain']['full']/med['ladder']['full']),
                giant_reduction_percent=100*(1-med['chain']['giant']/med['ladder']['giant'])))
    data['complete']=True;persist();print(json.dumps(data['comparisons'],indent=2))


if __name__=='__main__':main()
