#!/usr/bin/env python3
"""Compare slicing of identical PRAC subproducts at several curve counts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from bench_stage1_prac_windows import sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    p.add_argument('--b1',type=int,nargs='+',default=[260000000])
    p.add_argument('--curves',type=int,nargs='+',default=[384,768,1536])
    p.add_argument('--chunks',type=int,nargs='+',default=[4,8,16,32])
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--window',choices=['prefix','middle','tail'],default='tail')
    p.add_argument('--registers',type=int,nargs='+',choices=[168,255],default=[255,168])
    p.add_argument('--variants',nargs='+',choices=['baseline','compact','outline-add'],default=['baseline'])
    p.add_argument('--seconds',type=float,default=6)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--exp-cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.count<=32 or any(not 1<=c<=a.count for c in a.chunks):p.error('Require 1 <= chunk <= count <= 32')
    if min(a.curves)<1 or min(a.b1)<2 or not 0<a.seconds<=600 or not 0<=a.warmup<=32 or a.repeats<1:p.error('Invalid curves, B1, duration, warmup or repeats')
    exe=a.exe.resolve(strict=True);cache=(a.exp_cache or exe.parent).resolve()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),exe=str(exe),
        measurement='same reset point and contiguous subproduct; partial projections',results=[])
    gpu_info=subprocess.run(['nvidia-smi','-i',str(a.device),'--query-gpu=index,uuid,name',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    report['gpu_identity']=gpu_info.stdout.strip();report['gpu_identity_exit_code']=gpu_info.returncode
    configs=[(v,r,c) for v in a.variants for r in a.registers for c in a.chunks]
    signatures={}
    with (root/'gpu.csv').open('wb') as log,(root/'gpu.err').open('wb') as err:
        monitor=subprocess.Popen(['nvidia-smi','-i',str(a.device),
            '--query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used',
            '--format=csv,noheader,nounits','-lms','500'],stdout=log,stderr=err)
        try:
            for repeat in range(a.repeats):
                for b1 in a.b1:
                    for curves in (a.curves if repeat%2==0 else list(reversed(a.curves))):
                        for variant,registers,chunk in (configs if repeat%2==0 else list(reversed(configs))):
                            folder=root/f'b{b1}_c{curves}_{variant}_reg{registers}_chunk{chunk}_r{repeat}'
                            row=sample(exe,folder,4423,b1,curves,a.device,16,registers,variant,
                                a.window,a.count,a.seconds,a.warmup,cache,chunk=chunk)
                            signature=tuple(row[k] for k in ('first','count','p_first','p_last','work','full_work','sigma','exponent'))
                            key=(b1,curves)
                            if signatures.setdefault(key,signature)!=signature:raise RuntimeError('Compared subproducts differ')
                            row['repeat']=repeat;report['results'].append(row)
                            groups={}
                            for x in report['results']:groups.setdefault((x['B1'],x['curves'],x['registers'],x['chunk'],x['variant']),[]).append(x)
                            report['aggregates']=[dict(B1=k[0],curves=k[1],registers=k[2],chunk=k[3],variant=k[4],repeats=len(v),
                                projected_s_per_curve=statistics.median(x['projected_s_per_curve'] for x in v),
                                wall_projected_s_per_curve=statistics.median(x['wall_projected_s_per_curve'] for x in v),
                                min=min(x['projected_s_per_curve'] for x in v),max=max(x['projected_s_per_curve'] for x in v)) for k,v in groups.items()]
                            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        finally:
            monitor.terminate();monitor.wait(timeout=15)
    report['telemetry_exit_code']=monitor.returncode
    report['telemetry_terminated_by_tool']=True
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(root/'summary.json',flush=True)


if __name__=='__main__':main()
