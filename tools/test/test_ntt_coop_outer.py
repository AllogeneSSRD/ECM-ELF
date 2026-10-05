"""Serial independent GMP gate for the experimental cooperative outer passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--short-reduce',type=int,choices=(0,1),default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env['NTT_GL_SHORT_REDUCE']=str(a.short_reduce)
    checks={};resources=[]
    for name,poison,code in (('gmp','0',0),('fault','1',3)):
        r=subprocess.run([str(a.exe.resolve()),str(a.device)],env=env|{'NTT_FUSE_COOP_BAD':poison},
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8')
        checks[name+'_exit']=r.returncode==code
        if name=='gmp':
            checks['short_reduce_configured']=f'ntt_gl_reduce_mode: device={a.device} short={a.short_reduce}' in text
            checks['GMP_DFT_DIT_all_widths_cache_eviction']=bool(re.search(r'ntt_fuse_coop_check: cases=96 words=27131904 bad=0',text))
            checks['same_arena_mode_switch']=bool(re.search(r'ntt_fuse_coop_switch_check: calls=4 words=3145728 bad=0',text))
            checks['leak_and_completion']=f'ntt_coop_probe: device={a.device} bad=0' in text
            resources=[dict(re.findall(r'(\w+)=(\d+)',line)) for line in re.findall(r'ntt_coop_resources: (.*)',text)]
            checks['all_kernel_resources']=len(resources)==8
            checks['no_local_spill']=len(resources)==8 and all(r['local']=='0' and int(r['max_blocks'])>0 for r in resources)
        else:checks['fault_compared']=bool(re.search(r'ntt_fuse_coop_check: .*bad=[1-9]',text))
    r=subprocess.run([str(a.exe.resolve()),str(a.device),'--policy'],env=env,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    text=r.stdout.decode('utf-8',errors='replace');(a.output/'policy.log').write_text(text,encoding='utf-8')
    checks['shape_policy_exit']=r.returncode==0
    checks['shape_policy_boundaries_overrides']=bool(re.search(r'ntt_shape_policy_check: calls=88 supported=[01] bad=0',text))
    r=subprocess.run([str(a.exe.resolve()),str(a.device),'--gl-selftest'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    text=r.stdout.decode('utf-8',errors='replace');(a.output/'gl_selftest.log').write_text(text,encoding='utf-8')
    checks['gl_selftest_exit']=r.returncode==0
    checks['gl_selftest_selected_mode']=f'ntt_gl_reduce_mode: device={a.device} short={a.short_reduce}' in text and '200000 device cases' in text
    result=dict(exe=str(a.exe.resolve()),sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),
                device=a.device,short_reduce=a.short_reduce,checks=checks,resources=resources,passed=sum(checks.values()),failed=sum(not v for v in checks.values()))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    return int(result['failed']!=0)
if __name__=='__main__':raise SystemExit(main())
