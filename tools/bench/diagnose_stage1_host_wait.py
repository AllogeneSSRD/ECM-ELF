"""GPU1 CPU-wait diagnosis, using unchanged production arithmetic templates.

Probe timing excludes context/curve setup and final point export. Process CPU
time includes all its threads; per-API CPU deltas have Windows timer granularity.
No production queues/configuration, GPU clocks or power limits are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_sample():
    command=['nvidia-smi','-i','1','--query-gpu=name,uuid,driver_version,power.draw,power.limit,clocks.sm,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits']
    p=subprocess.run(command,capture_output=True,text=True,timeout=10)
    if p.returncode:
        return dict(error='nvidia-smi query unavailable')
    values=[v.strip() for v in p.stdout.strip().split(',')]
    return dict(zip(('name','uuid','driver','watts','power_limit_watts','sm_mhz','temperature_c','utilization'),values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scalar-bits',type=int,default=131072)
    p.add_argument('--curves',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--exponent',type=int,default=1279)
    p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--b1',type=int,default=0,help='Use production 48*lcm(1..B1) instead of the synthetic scalar')
    p.add_argument('--primary-only',action='store_true',help='Repeat only original and blocking device waits')
    p.add_argument('--adaptive-candidate',action='store_true',help='Include adaptive blocking slices as a third primary control')
    p.add_argument('--timeout',type=float,default=900)
    a=p.parse_args();exe=a.probe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.repeats<2:
        raise ValueError('Need repeated primary controls')
    env={k:v for k,v in os.environ.items() if k not in ('CUDA_LAUNCH_BLOCKING','CUDA_VISIBLE_DEVICES')}
    modes={'baseline':(0,0,0),'context_only':(4,0,0),'event_only':(0,1,0),
           'context_event':(4,1,0),'auto_device_wait':(0,0,2),'blocking_device_wait':(4,0,2)}
    if a.adaptive_candidate:
        modes['blocking_device_adaptive']=(4,0,2)
    report=dict(complete=False,probe=str(exe),probe_sha256=sha(exe),gmp_sha256=sha(exe.parent/'gmp-10.dll'),
        collector_sha256=sha(Path(__file__)),adaptive_candidate=a.adaptive_candidate,
        host=dict(os=platform.platform(),cpu=platform.processor()),device=gpu_sample(),
        target=f'2^{a.exponent}-1',scalar=f'48*lcm(1..{a.b1})' if a.b1 else f'2^{a.scalar_bits}-1',
        curves=a.curves,sigma_start=17,window=6,chunk_columns=a.chunk,cpu_power_settings='unchanged',
        removed_environment={k:os.environ[k] for k in ('CUDA_LAUNCH_BLOCKING','CUDA_VISIBLE_DEVICES') if k in os.environ},
        timing_boundary='production run entry to return; excludes exponent/NAF/curve/context setup and final export',runs=[])
    sequence=[] if a.primary_only else ['context_only','event_only','context_event','auto_device_wait']
    primary=['baseline','blocking_device_wait']+(['blocking_device_adaptive'] if a.adaptive_candidate else [])
    for i in range(a.repeats):
        sequence+=primary if i%2==0 else primary[::-1]
    for index,name in enumerate(sequence):
        flags,event,order=modes[name];prefix=out/f'{index:02}_{name}';points=prefix.with_suffix('.points')
        samples=[];stop=threading.Event()
        def telemetry():
            while not stop.is_set():
                samples.append(dict(monotonic=time.monotonic(),gpu=gpu_sample()))
                stop.wait(.3)
        worker=threading.Thread(target=telemetry);worker.start()
        chunk=0 if name=='blocking_device_adaptive' else a.chunk
        command=list(map(str,[exe,1,flags,event,a.scalar_bits,a.curves,points,order,a.exponent,chunk,a.b1]))
        try:
            result=subprocess.run(command,cwd=out,env=env,capture_output=True,timeout=a.timeout)
        except subprocess.TimeoutExpired as error:
            prefix.with_suffix('.log').write_bytes((error.stdout or b'')+(error.stderr or b''))
            raise
        finally:
            stop.set();worker.join(timeout=15)
        prefix.with_suffix('.log').write_bytes(result.stdout+result.stderr)
        if result.returncode:
            raise ValueError('Probe failed; inspect preserved log')
        row=json.loads(result.stdout)
        if not row['invariant_ok'] or row['event_creations']!=2 or row['context_flags']&7!=flags or row['created_event_flags']!=event:
            raise ValueError('Probe did not execute the requested wait controls')
        if (row['exponent'],row['chunk'],row['b1'])!=(a.exponent,chunk,a.b1):
            raise ValueError('Probe did not execute the requested arithmetic controls')
        row['curve_columns_per_second']=a.curves*row['columns']/row['wall_seconds']
        row['average_slice_gpu_ms']=(row['gpu_seconds']*1000-row['dictionary_ms'])/row['launches']
        row.update(mode=name,command=command,points_sha256=sha(points),telemetry=samples)
        report['runs'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(mode=name,cpu_cores=row['cpu_cores'],wall_seconds=row['wall_seconds'])),flush=True)
    if len({r['points_sha256'] for r in report['runs']})!=1:
        raise ValueError('Final coordinates differ between wait controls')
    summaries={}
    for name in modes:
        rows=[r for r in report['runs'] if r['mode']==name]
        if not rows:
            continue
        summaries[name]=dict(samples=len(rows),median_cpu_cores=statistics.median(r['cpu_cores'] for r in rows),
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in rows),
            median_cpu_seconds=statistics.median(r['cpu_seconds'] for r in rows),
            median_curve_columns_per_second=statistics.median(r['curve_columns_per_second'] for r in rows),
            median_slice_gpu_ms=statistics.median(r['average_slice_gpu_ms'] for r in rows),
            wall_range=[min(r['wall_seconds'] for r in rows),max(r['wall_seconds'] for r in rows)])
    report.update(complete=True,summaries=summaries,identical_final_coordinates=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(summaries=summaries,identical_final_coordinates=True),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries))


if __name__=='__main__':
    main()
