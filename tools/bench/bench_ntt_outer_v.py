"""Serial GMP/resource gate and same-binary offset-width convolution study."""
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
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('gate','timing'),required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('use a fresh output directory')
    out.mkdir(parents=True)
    build=read(exe.parent/'manifest.json')
    if sha(exe)!=build['sha256'] or build['gl_fixed_mode']!=3 or build['compiled_outer_u']!=0:
        raise ValueError('require fixed PTX3/original unroll build')
    for name,want in build['sources'].items():
        if sha(exe.parent/'sources'/name)!=want:raise ValueError('frozen source changed: '+name)
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(CUDA_LAUNCH_BLOCKING='0')
    report=dict(complete=False,binary_sha256=sha(exe),build_sha256=sha(exe.parent/'manifest.json'),
                tool_sha256=sha(__file__),mode=a.mode,runs=[],checks={},resources=[])
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,args,extra=None,code=0):
        if sha(exe)!=report['binary_sha256']:raise ValueError('binary changed')
        r=subprocess.run([str(exe),'1',*args],env=env|(extra or {}),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        text=r.stdout.decode('utf-8',errors='replace');dest=out/(name+'.log');dest.write_text(text,encoding='utf-8')
        report['runs'].append(dict(name=name,args=args,environment=extra or {},exit=r.returncode,log_sha256=sha(dest)))
        save()
        if r.returncode!=code:raise ValueError(f'{name}: unexpected exit {r.returncode}')
        return text
    if a.mode=='gate':
        for mask in range(4):
            text=run('gmp_'+str(mask),[],{'NTT_OUTER_NARROW':str(mask)})
            for token in ('ntt_fuse_coop_check: cases=96 words=27131904 bad=0',
                          'ntt_fuse_coop_switch_check: calls=4 words=3145728 bad=0',
                          'ntt_coop_probe: device=1 bad=0'):
                if token not in text:raise ValueError('missing GMP/cached-plan coverage')
            report['checks']['gmp_mask_'+str(mask)]=True
            text=run('fault_'+str(mask),[],{'NTT_OUTER_NARROW':str(mask),'NTT_FUSE_COOP_BAD':'1'},3)
            if not re.search(r'ntt_fuse_coop_check: .*bad=[1-9]',text):raise ValueError('fault not compared')
        text=run('dense',['--dense'])
        if not re.search(r'ntt_v_dense: cases=24 words=2951568 bad=0 live=0',text):raise ValueError('dense coverage mismatch')
        report['checks']['dense_GMP_and_strided_padding']=True
        text=run('dense_fault',['--dense'],{'NTT_V_DENSE_BAD':'1'},3)
        if not re.search(r'ntt_v_dense: .*bad=[1-9]',text):raise ValueError('dense fault not compared')
        for invalid in ('-1','4','bad','1x'):
            run('invalid_'+invalid,[],{'NTT_OUTER_NARROW':invalid},3)
        report['checks']['invalid_masks_rejected']=True
        text=run('resources',['--resources'])
        rows=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',line)) for line in re.findall(r'ntt_v_resources: (.*)',text)]
        if len(rows)!=8 or any(r['local'] or r['max_blocks']<1 for r in rows):raise ValueError('resource gate')
        report['resources']=rows
        run('policy',['--policy']);run('gl_selftest',['--gl-selftest'])
        report['checks']['policy_and_GL_exit0']=True
    else:
        if not a.gate:raise ValueError('--gate required before timing')
        gate=read(a.gate)
        if not gate['complete'] or gate['binary_sha256']!=report['binary_sha256']:raise ValueError('gate identity mismatch')
        report['gate_sha256']=sha(a.gate);report['statistics']={}
        for k in range(23,28):
            text=run('k'+str(k),['--vbench',str(k)])
            rows=[dict(re.findall(r'(\w+)=(\S+)',line)) for line in re.findall(r'ntt_v_bench: (.*)',text)]
            order=[0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1]
            if len(rows)!=64 or 'ntt_v_done: live=0' not in text:raise ValueError('incomplete convolution study')
            for idx,r in enumerate(rows):
                if (int(r['run']),int(r['repeat']),int(r['mask']),int(r['k']),int(r['N']),int(r['bad']))!=(idx//4,idx%4,order[idx//4],k,1<<k,0):
                    raise ValueError('order/shape/independent full output mismatch')
            stats={}
            for mask in range(4):
                runs=[statistics.mean(float(r['seconds']) for r in rows if int(r['run'])==i and int(r['repeat'])>0) for i in range(16) if order[i]==mask]
                stats[str(mask)]=dict(run_means=runs,mean=statistics.mean(runs))
            report['statistics'][str(k)]=stats
            save()
    for name,want in build['sources'].items():
        if sha(exe.parent/'sources'/name)!=want:raise ValueError('frozen source changed after runs')
    report['complete']=True;save();print(json.dumps(dict(complete=True,mode=a.mode,runs=len(report['runs']))))


if __name__=='__main__':main()
