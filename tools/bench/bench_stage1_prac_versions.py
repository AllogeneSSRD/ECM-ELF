#!/usr/bin/env python3
"""Pair two explicit PRAC candidate binaries and an unchanged baseline.

Identical option names can denote different candidates across versions. Record
both executable hashes and reverse the three-way order on alternate repeats.
Measurements are projections; this script never certifies a production T1.
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


def sha(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--mode',choices=['windows','prefix'],required=True)
    p.add_argument('--variant',choices=['single-add','single-compact'],default='single-compact')
    p.add_argument('--curves',type=int,nargs='+',default=[384,768,1536])
    p.add_argument('--registers',type=int,nargs='+',choices=[168,255],default=[168])
    p.add_argument('--b1',type=int,nargs='+',default=[260000000])
    p.add_argument('--chunks',type=int,nargs='+',default=[4,32])
    p.add_argument('--target-ms',type=float,nargs='+',default=[50,100])
    p.add_argument('--seconds',type=float,default=6);p.add_argument('--warmup',type=float,default=5)
    p.add_argument('--window',choices=['prefix','middle','tail'],default='tail')
    p.add_argument('--window-count',type=int,default=32);p.add_argument('--window-warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--device',type=int,default=1)
    p.add_argument('--exp-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.curves)<1 or min(a.b1)<2 or a.repeats<2 or not 0<a.seconds<=600:
        p.error('Positive curves, B1>=2, repeats>=2 and seconds in (0,600] required')
    if a.mode=='prefix' and not 0<=a.warmup<a.seconds:p.error('Prefix warmup must be below duration')
    if not 1<=a.window_count<=32 or any(not 1<=c<=a.window_count for c in a.chunks):
        p.error('Require 1 <= chunks <= window-count <= 32')
    if not 0<=a.window_warmup<=32 or any(not 10<=t<=500 for t in a.target_ms):p.error('Invalid warmup/target')
    before=a.before.resolve(strict=True);after=a.after.resolve(strict=True)
    if before==after:p.error('Before and after must be distinct executable paths')
    hashes={str(exe):sha(exe) for exe in (before,after)}
    if len(set(hashes.values()))!=2:p.error('Before and after must have different executable hashes')
    cache=a.exp_cache.resolve(strict=True);root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    identities=subprocess.run(['nvidia-smi','-i',str(a.device),'--query-gpu=index,uuid,name',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    report=dict(schema=1,mode=a.mode,measurement='paired partial projections, not completed Stage1 wall time',
        bits=4423,tpi=16,device=a.device,gpu_identity=identities.stdout.strip(),
        gpu_identity_exit_code=identities.returncode,binaries=hashes,
        controls={k:v for k,v in vars(a).items() if isinstance(v,(int,float,str,list))},
        exp_cache=str(cache),results=[],complete=False)
    controls=a.chunks if a.mode=='windows' else a.target_ms
    configs=[(label,exe,variant,reg,control) for reg in a.registers for control in controls
        for label,exe,variant in [('before',before,a.variant),('after',after,a.variant),('baseline',after,'baseline')]]
    signatures={}
    with (root/'gpu.csv').open('wb') as log,(root/'gpu.err').open('wb') as err:
        monitor=subprocess.Popen(['nvidia-smi','-i',str(a.device),
            '--query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used',
            '--format=csv,noheader,nounits','-lms','500'],stdout=log,stderr=err)
        try:
            for repeat in range(a.repeats):
                for b1 in a.b1:
                    for curves in (a.curves if repeat%2==0 else list(reversed(a.curves))):
                        for label,exe,variant,reg,control in (configs if repeat%2==0 else list(reversed(configs))):
                            if sha(exe)!=hashes[str(exe)]:raise RuntimeError('Executable changed during measurement')
                            folder=root/f'{label}_b{b1}_c{curves}_reg{reg}_control{control}_r{repeat}'
                            if a.mode=='windows':
                                row=window_sample(exe,folder,4423,b1,curves,a.device,16,reg,variant,
                                    a.window,a.window_count,a.seconds,a.window_warmup,cache,chunk=control)
                                signature=tuple(row[k] for k in ('first','count','p_first','p_last','work','full_work','sigma','exponent'))
                                if signatures.setdefault((b1,curves),signature)!=signature:
                                    raise RuntimeError('Compared scalar subproducts differ')
                            else:
                                args=SimpleNamespace(device=a.device,curves=curves,tpi=16,prac_registers=reg,
                                    prac_variant=variant,prac_target_ms=control,seconds=a.seconds,warmup=a.warmup,
                                    startup_timeout=600,exponent='lcm',exp_cache=cache)
                                row=prefix_sample(exe,folder,4423,b1,'prac',args)
                                if row['terminated'] or row['final_save_records'] or row['exit_code']!=1:
                                    raise RuntimeError('Prefix did not stop normally at its sampling limit')
                            if sha(exe)!=hashes[str(exe)]:raise RuntimeError('Executable changed during sample')
                            if not row['exponent_cache_hit'] or not row['prac_cache_hit']:
                                raise RuntimeError('Paired measurements require prebuilt validated caches')
                            if row['geometry']['container_bits']!=4608 or row['geometry']['tpi']!=16:
                                raise RuntimeError('Wrong arithmetic geometry')
                            row.update(version=label,repeat=repeat,exe=str(exe),binary_sha256=hashes[str(exe)],
                                       control=control,register_policy=reg)
                            report['results'].append(row)
                            groups={}
                            for x in report['results']:
                                groups.setdefault((x['B1'],x['curves'],x['register_policy'],x['control'],x['version']),[]).append(x)
                            report['aggregates']=[dict(B1=k[0],curves=k[1],registers=k[2],control=k[3],version=k[4],
                                repeats=len(v),projected_s_per_curve=statistics.median(x['projected_s_per_curve'] for x in v),
                                min=min(x['projected_s_per_curve'] for x in v),max=max(x['projected_s_per_curve'] for x in v),
                                **(dict(wall_projected_s_per_curve=statistics.median(x['wall_projected_s_per_curve'] for x in v))
                                   if a.mode=='windows' else {})) for k,v in groups.items()]
                            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            report['complete']=True
        finally:
            monitor.terminate();monitor.wait(timeout=15)
            report.update(telemetry_exit_code=monitor.returncode,telemetry_terminated_by_tool=True)
            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(root/'summary.json',flush=True)


if __name__=='__main__':main()
