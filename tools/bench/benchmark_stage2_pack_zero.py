"""Independent pack/zero gates and same-binary, interleaved device timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--iterations', type=int, default=8)
    a=p.parse_args()
    if not 1<=a.repeats<=31 or not 1<=a.iterations<=128 or a.device<0:
        p.error('invalid device/repeats/iterations')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=a.exe.resolve();build=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe)!=build['sha256'].lower():raise ValueError('binary mismatch')
    for name,want in build['sources'].items():
        source=ROOT/name
        if sha(source)!=want.lower():raise ValueError('source mismatch: '+name)
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env['CUDA_LAUNCH_BLOCKING']='0'
    report=dict(complete=False,build=build,binary_sha256=sha(exe),tool_sha256=sha(__file__),
        device=a.device,timing='Device events per operand; includes memset+pack or complete fused pack/zero; no discarded warmup.',runs=[],summaries=[])
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,args,expected=0):
        if sha(exe)!=report['binary_sha256']:raise ValueError('binary changed')
        cmd=[str(exe),str(a.device),*map(str,args)]
        with (out/(name+'.log')).open('w') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        report['runs'].append(dict(name=name,command=cmd,returncode=result.returncode,log_sha256=sha(out/(name+'.log'))));save()
        if result.returncode!=expected:raise ValueError(f'{name}: {result.returncode} != {expected}')
        return (out/(name+'.log')).read_text()
    telemetry_file=(out/'gpu.csv').open('w')
    telemetry=subprocess.Popen(['nvidia-smi','-i',str(a.device),'--query-gpu=timestamp,index,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu','--format=csv','-lms','500'],
        stdout=telemetry_file,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        gate=run('gate',['--check']);fault=run('fault',['--fault'],3)
        for text,bad in [(gate,0),(fault,1)]:
            if 'pack_zero_division: cases=1048576 bad=0' not in text or not re.search(r'pack_zero_gate: cases=663 words=\d+ bad='+str(bad)+r' fault='+str(bad)+r' injected='+str(bad),text):
                raise ValueError('GMP full-buffer/fault gate did not pass')
        for bits,P,batch in [(318,3,512),(2203,32,64),(4423,1024,4),(4423,5761,1),(7995,5761,1),(12011,1024,1),(2203,77761,1)]:
            for path in [0,1,2]:
                name=f's{bits}_p{P}_b{batch}_path{path}'
                text=run(name,['--bench',bits,P,batch,path,a.repeats,a.iterations])
                rows=[dict(re.findall(r'(\w+)=(\S+)',line)) for line in re.findall(r'^pack_zero_sample: (.*)$',text,re.M)]
                if len(rows)!=2*a.repeats or not re.search(r'pack_zero_bench_done: words=\d+ bad=0',text):raise ValueError('incomplete or different output')
                item=dict(bits=bits,P=P,batch=batch,path=path,N=int(rows[0]['N']),sw=int(rows[0]['sw']),bpw=int(rows[0]['bpw']),samples=rows)
                for fused in [0,1]:
                    values=[float(r['seconds']) for r in rows if int(r['fused'])==fused]
                    median=statistics.median(values)
                    item['on' if fused else 'off']=dict(median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in values))
                item['gain_percent']=100*(1-item['on']['median_seconds']/item['off']['median_seconds'])
                report['summaries'].append(item);save()
                print(name+': '+str(round(item['gain_percent'],2))+'%',flush=True)
        report['complete']=True;save()
    finally:
        telemetry.terminate();telemetry.wait(timeout=10);telemetry_file.close()
if __name__=='__main__':main()
