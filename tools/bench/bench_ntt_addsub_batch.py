"""GMP hot-batch gate and fixed-order cross-build timing for canonical subtraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing'),required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True);exes=[a.baseline.resolve(),a.candidate.resolve()];identity=[]
    for mask,exe in enumerate(exes):
        build=read(exe.parent/'manifest.json')
        if build['arithmetic_mask']!=mask or build['gl_fixed_mode']!=3 or build['compiled_outer_u']!=0 or sha(exe)!=build['sha256']:
            raise ValueError('binary/settings mismatch')
        identity.append(dict(binary_sha256=sha(exe),manifest_sha256=sha(exe.parent/'manifest.json'),sources=build['sources']))
    report=dict(complete=False,mode=a.mode,identity=identity,tool_sha256=sha(__file__),runs=[])
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    def verify():
        if sha(__file__)!=report['tool_sha256']:raise ValueError('collector changed')
        for exe,want in zip(exes,identity):
            if sha(exe)!=want['binary_sha256'] or sha(exe.parent/'manifest.json')!=want['manifest_sha256']:raise ValueError('build changed')
            for name,digest in want['sources'].items():
                if sha(exe.parent/'sources'/name)!=digest:raise ValueError('frozen source changed')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['CUDA_LAUNCH_BLOCKING']='0'
    def run(mask,option,name,exit_code=0):
        verify();dest=out/(name+'.log');command=[str(exes[mask]),option]
        with dest.open('wb') as f:proc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        text=dest.read_text();rows=[{k:float(v) if k=='seconds' else int(v) for k,v in re.findall(r'(\w+)=(\S+)',s)} for s in re.findall(r'^ntt_batch_sample: (.*)$',text,re.M)]
        if proc.returncode!=exit_code or f'ntt_addsub_mask: value={mask}' not in text or not re.search(r'ntt_batch_done: mask='+str(mask)+r' bad=\d+ live=0',text):raise ValueError('process/output mismatch')
        if not rows or any(r['mask']!=mask or r['N']!=2048 or r['words']!=r['batch']*r['stride'] for r in rows):raise ValueError('actual shape differs')
        if exit_code==0 and any(r['bad'] for r in rows):raise ValueError('GMP output mismatch')
        if exit_code==3 and not any(r['bad'] for r in rows):raise ValueError('poison not detected')
        expected=[(nb,pad,phase,repeat) for nb,pad in ([(1,0),(1,17),(3,0),(3,17),(990,0),(990,17)] if option=='--gate' else [(990,17 if option=='--fault' else 0)]) for phase in range(3) for repeat in range(25 if option=='--timing' else 1)]
        if [(r['batch'],r['stride']-2048,r['phase'],r['repeat']) for r in rows]!=expected:raise ValueError('coverage/order mismatch')
        resources=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',s)) for s in re.findall(r'^ntt_batch_resource: (.*)$',text,re.M)]
        if len(resources)!=2*(6 if option=='--gate' else 1) or any(r['mask']!=mask or r['regs']!=40 or r['local'] or r['dynamic']!=16384 or r['blocks']!=3 for r in resources):raise ValueError('resource mismatch')
        entry=dict(name=name,mask=mask,command=command,exit=proc.returncode,log_sha256=sha(dest),samples=rows,resources=resources)
        if option=='--timing':entry['seconds']={str(phase):statistics.mean(r['seconds'] for r in rows if r['phase']==phase and r['repeat']>0) for phase in range(3)}
        report['runs'].append(entry);persist();print(name,entry.get('seconds',len(rows)),flush=True)
    if a.mode=='gate':
        for mask in range(2):run(mask,'--gate',f'gate_m{mask}');run(mask,'--fault',f'fault_m{mask}',3)
    else:
        if not a.gate:raise ValueError('--gate required')
        gate=read(a.gate)
        if not gate['complete'] or gate['identity']!=identity or gate['mode']!='gate':raise ValueError('gate identity mismatch')
        report['gate_sha256']=sha(a.gate)
        for mask in range(2):run(mask,'--timing',f'warmup_m{mask}')
        for index,mask in enumerate((0,1,1,0,1,0,0,1)):run(mask,'--timing',f'timing_{index}_m{mask}')
        selected=[r for r in report['runs'] if r['name'].startswith('timing_')]
        report['statistics']={}
        for phase in range(3):
            means={str(mask):statistics.mean(r['seconds'][str(phase)] for r in selected if r['mask']==mask) for mask in range(2)}
            report['statistics'][str(phase)]=dict(mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']))
    verify();report['complete']=True;persist();print(json.dumps(dict(complete=True,mode=a.mode,runs=len(report['runs']))))


if __name__=='__main__':main()
