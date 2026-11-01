"""Production-save gates for point policy replay and normalized leaf equivalence.

Small Montgomery points come from the independent Python lcm reference. This
checks factor/nonunit semantics separately from large NF performance fixtures.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_version import write,stamp,check_gpu_processes
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allowed-study',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.exe.resolve();identity=sha(exe)
    allowed=json.loads(a.allowed_study.read_text(encoding='utf-8-sig'))['allowed_processes']
    spec=importlib.util.spec_from_file_location('suyama_ref',ROOT/'tools/stat/suyama_mont_ref.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    config=out/'ecm.toml'
    config.write_text('[stage2]\nstage2_auto_b2=false\nstage2_tune_runtime=false\nstage2_short_calibration=false\n',encoding='utf-8')
    report=dict(complete=False,started_utc=stamp(),exe_sha256=identity,runs=[])
    env={k:v for k,v in os.environ.items() if not k.startswith(('NTT_','ECM_')) and k not in ('CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING')}
    prime127=(1<<127)-1
    basic_modes=[('legacy',[]),('balance',['--giant-tail-balance','1']),('chain32',['--giant-chain','32'])]
    active_modes=[('legacy_active',[]),('balance_active',['--giant-tail-balance','1'])]
    cases=[('prime127',prime127,20,basic_modes,13230),
           ('nonunit_base',103*65537,2,basic_modes,13230),
           ('factor10007',10007*65537,20,basic_modes,13230),
           ('active_prime127',prime127,20,active_modes,16230900),
           ('active_factor10007',10007*prime127,20,active_modes,16230900),
           ('active_nonunit103',103*prime127,2,active_modes,16230900)]
    for name,n,b1,modes,b2 in cases:
        point=ref.stage1(26,b1,n)
        if point['gcd']!=1 or math.gcd(point['x'],n)!=1:raise ValueError('Invalid independent Stage1 fixture')
        if name in ('factor10007','active_factor10007'):
            # Verify the claimed factor fixture before asking the GPU to run.
            baby=[ref.ladder(j,point['x'],1,point['a24'],n) for j in range(1,106) if math.gcd(j,210)==1]
            hits=set()
            for i in range(1,66):
                gx,gz=ref.ladder(i*210,point['x'],1,point['a24'],n)
                for bx,bz in baby:
                    f=math.gcd((gx*bz-bx*gz)%n,n)
                    if 1<f<n:hits.add(f)
            expected={10007,65537} if name=='factor10007' else {10007}
            if hits!=expected or any(math.gcd(bz,n)!=1 for bx,bz in baby):
                raise ValueError('Independent Stage2/nonunit fixture changed')
        save=out/(name+'.save');x=point['x']
        save.write_text(f'METHOD=ECM; SIGMA=26; B1={b1}; N={n}; X=0x{x:x}; CHECKSUM={b1*26*n*x%4294967291};\n',encoding='utf-8')
        reference=None
        for mode,extra in modes:
            folder=out/(name+'_'+mode);folder.mkdir();log=folder/'engine.log';debug=folder/'debug.log';result=folder/'result.jsonl'
            cmd=[str(exe),'--config',str(config),'--save',str(save),'--b2',str(b2),'--d','210','--device','1',
                '--short-calibration','0','--tune-runtime','0','--tune-profile',str(out/'absent.toml'),
                '--factor-only','--results',str(result),'--submission',str(folder/'result.txt'),
                '--log',str(log),'--debug-log-file',str(debug),'--log-level','phases']+extra
            run_env=dict(env)
            if mode.endswith('_active'):
                # Keep the production chain threshold T=32768. With P=24,
                # I=77292 and C=65592, 65592+11700 becomes 44520+32772.
                # Both modes get the same point budget, only permission differs.
                run_env.update(NTT_GIANT_POINT_BUDGET_KB=str(1025*((n.bit_length()+63)//64)),NTT_GIANT_CHUNK_FLOOR='1')
            write(folder/'environment.json',{k:v for k,v in run_env.items() if k.startswith('NTT_')})
            write(folder/'command.json',cmd)
            monitor=TelemetryRecorder(1,folder/'telemetry.jsonl')
            try:
                check_gpu_processes(monitor.monitor,out,folder/'before.json',allowed=allowed)
                with (folder/'console.log').open('wb') as console:
                    proc=subprocess.run(cmd,cwd=folder,env=run_env,stdout=console,stderr=subprocess.STDOUT,timeout=180)
                check_gpu_processes(monitor.monitor,out,folder/'after.json',allowed=allowed)
            finally:monitor.close()
            if proc.returncode:raise ValueError('Worker failed: '+str(folder))
            raw=json.loads(result.read_text(encoding='utf-8-sig').splitlines()[-1]);receipt=normalize_result(raw)
            engine=raw['stage2-engine']
            factors=set(receipt['factors'])
            if receipt['bad_factors'] or any(not 1<int(f)<n or n%int(f) for f in factors):raise ValueError('Bad factor')
            prime_case=name in ('prime127','active_prime127')
            if prime_case and factors:raise ValueError('False factor on prime')
            if not prime_case and not factors:raise ValueError('Lost independent composite fixture factor')
            text=log.read_text(encoding='utf-8')+'\n'+debug.read_text(encoding='utf-8')
            if any(t not in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0')):raise ValueError('Missing arithmetic checks')
            if reference is None:reference=factors
            elif factors!=reference:raise ValueError('Legacy replay changed the factor set')
            if mode=='balance' and (engine['giant_tail_balance'] or engine['point_policy_replayed']):
                raise ValueError('A single-chunk no-op entered point-policy replay')
            if mode=='balance_active' and name=='active_prime127' and not engine['giant_tail_balance']:
                raise ValueError('Active no-factor fixture did not borrow points')
            if mode=='balance_active' and name=='active_factor10007' and not engine['point_policy_replayed']:
                raise ValueError('An actual balanced factor probe did not replay the original path')
            if not prime_case and mode=='chain32' and 'stage2_point_policy: requested=1 replayed=1' not in text:
                raise ValueError('Nonunit/factor did not exercise legacy replay')
            if sha(exe)!=identity:raise ValueError('Binary changed')
            report['runs'].append(dict(case=name,mode=mode,factors=sorted(factors),status=raw['status'],
                actual_balance=engine['giant_tail_balance'],replayed=engine['point_policy_replayed'],log_sha256=sha(log)))
            write(out/'summary.json',report);print(name,mode,'passed',flush=True)
    report.update(complete=True,finished_utc=stamp());write(out/'summary.json',report)


if __name__=='__main__':main()
