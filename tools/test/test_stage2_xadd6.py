"""Serial GPU primitive gate for six-Montgomery xADD and exact coordinate scale."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('Use a fresh output directory')
    base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    base.update(NTT_NO_PROGRESS='1',NTT_NAME_MAX='1',NTT_XADD6_TEST='1',NTT_FUSE_WARP_TAIL='1',NTT_S4_OLDTAIL='0')
    rows=[]
    # Dispatch ceilings and near-full radix odd moduli cover the modular half carry.
    for bits in (63,64,65,129,257,513,1025,2049,4097,4423,8192):
        n=(1<<bits)-1
        # Keep the curve's small denominators units; the modulus need not be prime.
        while any(n%v==0 for v in (2,3,5,7,11,13)):n-=2
        for flag in ('0','1'):
            cmd=[str(exe),'--real','--n-hex',format(n,'x'),'--sigma','26',
                 '--b1','20','--b2','1000','--d','210','--device',str(a.device)]
            r=subprocess.run(cmd,env=base|{'NTT_XADD6':flag},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
            text=r.stdout.decode('utf-8',errors='replace');name=f'{bits}_{flag}'
            (out/(name+'.log')).write_text(text,encoding='utf-8')
            checks={'exit':r.returncode==0,
                    'selector':f'point_arithmetic: xadd6={flag} xadd_mont_muls={6 if flag=="1" else 8} coordinate_scale=legacy_exact' in text,
                    'gmp_coordinates_alias_domains_half':bool(re.search(r'xadd6_selftest: cases=1280 aliases=5 domains=2 modes=2 coordinates=2560 halves=1280 bad=0 ',text))}
            rows.append({'name':name,'modulus_hex':format(n,'x'),'checks':checks,'returncode':r.returncode})
            print(name,checks,flush=True)
    cmd=[str(exe),'--real','--n-hex','ffffffffffffffc5','--sigma','26','--b1','20','--b2','1000','--d','210','--device',str(a.device)]
    r=subprocess.run(cmd,env=base|{'NTT_XADD6':'1','NTT_XADD6_TEST_BAD':'1'},capture_output=True,timeout=180)
    text=(r.stdout+r.stderr).decode('utf-8',errors='replace');(out/'fault.log').write_text(text,encoding='utf-8')
    rows.append({'name':'fault','returncode':r.returncode,'checks':{'reject':r.returncode!=0 and bool(re.search(r'xadd6_selftest:.*bad=[1-9]',text))}})
    result={'exe':str(exe),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'device':a.device,'runs':rows,
            'passed':sum(sum(x['checks'].values()) for x in rows),'failed':sum(sum(not v for v in x['checks'].values()) for x in rows)}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TOTAL',result['passed'],'passed /',result['failed'],'failed',flush=True)
    return int(result['failed']!=0)

if __name__=='__main__':raise SystemExit(main())
