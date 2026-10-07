#!/usr/bin/env python3
"""Compare explicit TPI16/TPI32 at matched submitted grids, not equal occupancy.

TPI32 uses half the curves at TPB128. Report aggregate curves/s from per-curve
projections; batch completion projections do not compare different batch sizes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

from bench_cuda_prac import sample as prefix_sample
from bench_stage1_prac_windows import sample as window_sample

CONFIGS = {
    'shared16': (16,'single-compact',255),
    'shared16cap168': (16,'single-compact',168),
    'shared16cap128': (16,'single-compact',128),
    'shared32': (32,'single-compact',255),
    'shared32cap168': (32,'single-compact',168),
    'baseline32': (32,'baseline',255),
}


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--mode',choices=['windows','prefix'],required=True)
    p.add_argument('--curves16',type=int,nargs='+',default=[384,768,1536])
    p.add_argument('--configs',nargs='+',choices=list(CONFIGS),
                   default=['shared16','shared16cap168','shared32','shared32cap168','baseline32'])
    p.add_argument('--b1',type=int,nargs='+',default=[10000000,260000000])
    p.add_argument('--chunks',type=int,nargs='+',default=[4,32])
    p.add_argument('--target-ms',type=float,default=50)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--warmup',type=float,default=5)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--exp-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(c<8 or c%8 for c in a.curves16) or min(a.b1)<2 or a.repeats<2:
        p.error('curves16 must be positive multiples of 8; B1>=2 and repeats>=2')
    if not 0<a.seconds<=600 or (a.mode=='prefix' and not 0<=a.warmup<a.seconds):
        p.error('Invalid seconds/warmup')
    if any(not 1<=c<=32 for c in a.chunks) or not 10<=a.target_ms<=500:
        p.error('Chunks must be 1..32; target must be 10..500 ms')
    if any(len(v)!=len(set(v)) for v in [a.curves16,a.configs,a.b1,a.chunks]):
        p.error('Duplicate matrix entries')
    exe=a.exe.resolve(strict=True);binary=sha(exe);cache=a.exp_cache.resolve(strict=True)
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.run(['nvidia-smi','-i',str(a.device),'--query-gpu=index,uuid,name',
                        '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    if gpu.returncode:raise RuntimeError('GPU identity unavailable')
    controls={k:v for k,v in vars(a).items() if isinstance(v,(int,float,str,list))}
    report=dict(schema=1,exe=str(exe),binary_sha256=binary,mode=a.mode,bits=4423,tpb=128,
        pairing='TPI32 half curves; same submitted grid, measured residency may differ',
        measurement='partial projections, not complete Stage1 wall time or Auto B2 T1',
        controls=controls,configurations=CONFIGS,gpu_identity=gpu.stdout.strip(),results=[],complete=False)
    controls_list=a.chunks if a.mode=='windows' else [a.target_ms]
    configs=[(name,control) for name in a.configs for control in controls_list]
    signatures={}
    with (root/'gpu.csv').open('wb') as log,(root/'gpu.err').open('wb') as err:
        monitor=subprocess.Popen(['nvidia-smi','-i',str(a.device),
            '--query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used',
            '--format=csv,noheader,nounits','-lms','500'],stdout=log,stderr=err)
        try:
            for repeat in range(a.repeats):
                for b1 in a.b1:
                    for curves16 in (a.curves16 if repeat%2==0 else list(reversed(a.curves16))):
                        for name,control in (configs if repeat%2==0 else list(reversed(configs))):
                            if sha(exe)!=binary:raise RuntimeError('Executable changed')
                            tpi,variant,reg=CONFIGS[name];curves=curves16 if tpi==16 else curves16//2
                            folder=root/f'{name}_b{b1}_c16_{curves16}_control{control}_r{repeat}'
                            if a.mode=='windows':
                                row=window_sample(exe,folder,4423,b1,curves,a.device,tpi,reg,variant,
                                    'tail',32,a.seconds,2,cache,chunk=control)
                                signature=tuple(row[k] for k in ['first','count','p_first','p_last','work','full_work','sigma','exponent'])
                                if signatures.setdefault(b1,signature)!=signature:
                                    raise RuntimeError('Scalar subproducts differ')
                            else:
                                args=SimpleNamespace(device=a.device,curves=curves,tpi=tpi,prac_registers=reg,
                                    prac_variant=variant,prac_target_ms=control,seconds=a.seconds,warmup=a.warmup,
                                    startup_timeout=600,exponent='lcm',exp_cache=cache)
                                row=prefix_sample(exe,folder,4423,b1,'prac',args)
                                if row['terminated'] or row['final_save_records'] or row['exit_code']!=1:
                                    raise RuntimeError('Prefix did not stop normally at sample limit')
                            if sha(exe)!=binary:raise RuntimeError('Executable changed during sample')
                            geom=row['geometry']
                            if (geom['tpi'],geom['container_bits'],geom['curves'],geom['grid_blocks'])!=(tpi,4608,curves,curves16//8):
                                raise RuntimeError('Wrong TPI/container/curves/submitted grid')
                            if not row['exponent_cache_hit'] or not row['prac_cache_hit']:
                                raise RuntimeError('Comparison requires validated cache hits')
                            row.update(config=name,repeat=repeat,curves16=curves16,control=control,binary_sha256=binary)
                            report['results'].append(row)
                            groups={}
                            for x in report['results']:
                                groups.setdefault((x['B1'],x['curves16'],x['config'],x['control']),[]).append(x)
                            report['aggregates']=[]
                            for key,values in groups.items():
                                rates=[v['projected_s_per_curve'] for v in values];median=statistics.median(rates)
                                group=dict(B1=key[0],curves16=key[1],config=key[2],control=key[3],repeats=len(values),
                                    projected_s_per_curve=median,projected_curves_per_second=1/median,min=min(rates),max=max(rates))
                                if a.mode=='windows':
                                    wall=statistics.median(v['wall_projected_s_per_curve'] for v in values)
                                    group.update(wall_projected_s_per_curve=wall,wall_projected_curves_per_second=1/wall)
                                report['aggregates'].append(group)
                            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            report['complete']=True
        finally:
            monitor.terminate();monitor.wait(timeout=15)
            report.update(telemetry_exit_code=monitor.returncode,telemetry_terminated_by_tool=True)
            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(root/'summary.json',flush=True)


if __name__=='__main__':main()
