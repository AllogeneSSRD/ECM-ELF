"""Freeze isolated add/sub probes, gate, then cross builds in balanced order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing'),required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();study=a.study.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True)
    identity={}
    for mask in range(4):
        folder=study/f'ntt_m{mask}';project=study/f'project_m{mask}';build=read(folder/'manifest.json');gen=read(project/'generation.json')
        if not gen['complete'] or gen['mask']!=mask or build['gl_fixed_mode']!=3 or build['compiled_outer_u']!=0:
            raise ValueError('build/generation settings differ')
        if build['sources']!=gen['generated_sources']:raise ValueError('compiled source closure differs')
        identity[str(mask)]=dict(binary_sha256=sha(folder/'ntt_outer_v_probe.exe'),build_sha256=sha(folder/'manifest.json'),generation_sha256=sha(project/'generation.json'),sources=build['sources'])
        if identity[str(mask)]['binary_sha256']!=build['sha256']:raise ValueError('binary changed')
        for name,want in build['sources'].items():
            if sha(folder/'sources'/name)!=want or sha(project/name)!=want:raise ValueError('source snapshot changed')
    report=dict(complete=False,identity=identity,mode=a.mode,tool_sha256=sha(__file__),runs=[])
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    def persist():(out/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
    def verify():
        if sha(__file__)!=report['tool_sha256']:raise ValueError('collector changed')
        for key,want in identity.items():
            folder=study/('ntt_m'+key)
            if sha(folder/'ntt_outer_v_probe.exe')!=want['binary_sha256']:raise ValueError('binary changed')
            for name,digest in want['sources'].items():
                if sha(folder/'sources'/name)!=digest:raise ValueError('source changed')
    if a.mode=='gate':
        gate_tool=ROOT/'tools/bench/bench_ntt_outer_v.py';report['gate_tool_sha256']=sha(gate_tool)
        for mask in range(4):
            verify();folder=study/f'ntt_m{mask}';target=out/f'mask{mask}'
            command=['--exe',str(folder/'ntt_outer_v_probe.exe'),'--mode','gate','--output',str(target)]
            proc=subprocess.run([os.sys.executable,'-X','utf8',str(gate_tool),*command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
            (out/f'mask{mask}_collector.log').write_bytes(proc.stdout)
            if proc.returncode:raise ValueError('gate failed: '+str(mask))
            g=read(target/'summary.json')
            if not g['complete'] or g['binary_sha256']!=identity[str(mask)]['binary_sha256']:raise ValueError('gate identity')
            for row in g['runs']:
                text=(target/(row['name']+'.log')).read_text()
                if f'ntt_addsub_mask: value={mask}' not in text:raise ValueError('actual compiled mask differs')
            report['runs'].append(dict(mask=mask,gate_sha256=sha(target/'summary.json')));persist()
    else:
        if not a.gate:raise ValueError('--gate required')
        g=read(a.gate)
        if not g['complete'] or g['identity']!=identity:raise ValueError('gate differs')
        report['gate_sha256']=sha(a.gate);report['statistics']={}
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env['CUDA_LAUNCH_BLOCKING']='0'
        def run(mask,k,category,index):
            verify();name=f'k{k}_{category}_{index}_m{mask}';dest=out/(name+'.log')
            command=[str(study/f'ntt_m{mask}/ntt_outer_v_probe.exe'),'1','--vbench',str(k)]
            with dest.open('wb') as f:proc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            if proc.returncode:raise ValueError(name+' failed')
            text=dest.read_text()
            if f'ntt_addsub_mask: value={mask}' not in text or 'ntt_v_done: live=0' not in text:raise ValueError('mask/completion mismatch')
            rows=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^ntt_v_bench: (.*)$',text,re.M)]
            if len(rows)!=32:raise ValueError('expected8 run x4 repeats')
            for i,r in enumerate(rows):
                if (int(r['run']),int(r['repeat']),int(r['mask']),int(r['N']),int(r['bad']))!=(i//4,i%4,0,1<<k,0):raise ValueError('shape/order/result differs')
            mean=statistics.mean(float(r['seconds']) for r in rows if int(r['repeat'])>0)
            report['runs'].append(dict(name=name,mask=mask,k=k,category=category,command=command,log_sha256=sha(dest),seconds=mean));persist()
            print(name,mean,flush=True)
        for index,mask in enumerate((0,1,2,3)):run(mask,27,'warmup',index)
        order=[0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1]
        for k in range(23,28):
            for index,mask in enumerate(order):run(mask,k,'timing',index)
            selected=[r for r in report['runs'] if r['k']==k and r['category']=='timing']
            means={str(mask):statistics.mean(r['seconds'] for r in selected if r['mask']==mask) for mask in range(4)}
            report['statistics'][str(k)]=dict(mean_seconds=means,reduction_percent={str(mask):100*(1-means[str(mask)]/means['0']) for mask in (1,2,3)});persist()
    verify();report['complete']=True;persist();print(json.dumps(dict(complete=True,mode=a.mode,runs=len(report['runs']))))


if __name__=='__main__':main()
