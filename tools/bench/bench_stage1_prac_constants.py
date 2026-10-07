#!/usr/bin/env python3
"""Same-binary PRAC constants comparisons; partial projections, not complete curves."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace
from bench_cuda_prac import sample as prefix_sample
from bench_stage1_prac_windows import sample as window_sample


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--mode',choices=['windows','prefix'],required=True)
    p.add_argument('--configs',nargs='+',choices=['none','runtime','np0','m4423'],default=['none','runtime','np0','m4423'])
    p.add_argument('--curves',type=int,nargs='+',default=[768,1536,2304])
    p.add_argument('--b1',type=int,nargs='+',default=[10000000,260000000])
    p.add_argument('--chunks',type=int,nargs='+',default=[4,32])
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--warmup',type=float,default=5)
    p.add_argument('--target-ms',type=float,default=50)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--exp-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(c<8 or c%8 for c in a.curves) or a.repeats<2 or min(a.b1)<2:p.error('Require curves multiples of 8, B1>=2 and repeats>=2')
    if any(len(v)!=len(set(v)) for v in [a.configs,a.curves,a.b1,a.chunks]):p.error('Duplicate matrix entries')
    if not 0<a.seconds<=600 or not 10<=a.target_ms<=500 or any(not 1<=c<=32 for c in a.chunks):p.error('Invalid seconds/target/chunks')
    if a.mode=='prefix' and not 0<=a.warmup<a.seconds:p.error('Warmup must be below seconds')
    exe=a.exe.resolve(strict=True);cache=a.exp_cache.resolve(strict=True);binary=sha(exe)
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.run(['nvidia-smi','-i',str(a.device),'--query-gpu=index,uuid,name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
    report=dict(schema=1,bits=4423,tpi=16,tpb=128,exe=str(exe),binary_sha256=binary,gpu_identity=gpu.stdout.strip(),
        controls={k:v for k,v in vars(a).items() if isinstance(v,(str,int,float,list))},results=[],complete=False,
        measurement='Partial fixed subproduct or ordinary-prefix projections; not complete Stage1/Auto B2 T1')
    controls=a.chunks if a.mode=='windows' else [a.target_ms]
    configs=[(config,control) for config in a.configs for control in controls]
    with (root/'gpu.csv').open('wb') as out,(root/'gpu.err').open('wb') as err:
        monitor=subprocess.Popen(['nvidia-smi','-i',str(a.device),'--query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used','--format=csv,noheader,nounits','-lms','500'],stdout=out,stderr=err)
        try:
            for repeat in range(a.repeats):
                for b1 in a.b1:
                    for c in (a.curves if repeat%2==0 else list(reversed(a.curves))):
                        for config,control in (configs if repeat%2==0 else list(reversed(configs))):
                            assert sha(exe)==binary,'Executable changed'
                            folder=root/f'{config}_b{b1}_c{c}_control{control}_r{repeat}'
                            if a.mode=='windows':
                                row=window_sample(exe,folder,4423,b1,c,a.device,16,168,'single-compact','tail',32,a.seconds,2,cache,chunk=control,constants=config)
                            else:
                                args=SimpleNamespace(device=a.device,curves=c,tpi=16,prac_registers=168,prac_variant='single-compact',prac_constants=config,prac_target_ms=control,seconds=a.seconds,warmup=a.warmup,startup_timeout=600,exponent='lcm',exp_cache=cache)
                                row=prefix_sample(exe,folder,4423,b1,'prac',args)
                                assert not row['terminated'] and row['final_save_records']==0 and row['exit_code']==1
                            assert sha(exe)==binary and row['exponent_cache_hit'] and row['prac_cache_hit']
                            assert row['constants']==config
                            text=Path(row['log']).read_text(encoding='utf-8')
                            assert f'PRAC constants policy={config}; np0=1' in text
                            g=row['geometry'];assert (g['tpi'],g['container_bits'],g['curves'],g['grid_blocks'])==(16,4608,c,c//8)
                            assert g['resident_blocks_per_sm']==3
                            if a.mode=='windows':
                                assert row['boundary_logical_bytes_per_round']==(5 if config=='m4423' else 6)*c*576*row['launches_per_round']
                            row.update(config=config,repeat=repeat,control=control,binary_sha256=binary)
                            report['results'].append(row);groups={}
                            for r in report['results']:groups.setdefault((r['B1'],r['curves'],r['config'],r['control']),[]).append(r)
                            report['aggregates']=[]
                            for key,rs in groups.items():
                                values=[r['projected_s_per_curve'] for r in rs]
                                agg=dict(B1=key[0],curves=key[1],config=key[2],control=key[3],repeats=len(rs),projected_s_per_curve=statistics.median(values),min=min(values),max=max(values))
                                if a.mode=='windows':agg['wall_projected_s_per_curve']=statistics.median(r['wall_projected_s_per_curve'] for r in rs)
                                report['aggregates'].append(agg)
                            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            report['complete']=True
        finally:
            monitor.terminate();monitor.wait(timeout=15)
            (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(root/'summary.json',flush=True)


if __name__=='__main__':main()
