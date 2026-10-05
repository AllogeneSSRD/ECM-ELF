"""Check reducer-specific empirical D rates and unsupported-mode fallback.

Uses the observed fixed-D A/B environment, selects one device, and executes only
the planner. No curve, save, ini or work queue is modified.
"""
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
    p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--short-calibrated',action='store_true')
    p.add_argument('--default-short',type=int,choices=(0,1),default=0)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    argv=list(map(str,prov['args']))
    assert int(argv[argv.index('--n-hex')+1],16)==(1<<4423)-1
    for key,value in {'--d':'1231230','--device':str(a.device)}.items():argv[argv.index(key)+1]=value
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update({k:str(v) for k,v in prov['env'].items() if v is not None})
    control=next(r for r in prov['mode_controls'] if r['NTT_XADD6']=='1' and r['NTT_FUSE_COOP_OUTER']=='2')
    env.update({k:str(v) for k,v in control.items() if k!='mode'})
    env.pop('NTT_FUSE_TRACE',None);rows=[]
    short_enabled=int(a.short_calibrated)
    short_version='resident_short_v1' if a.short_calibrated else 'legacy_56_1'
    default_enabled=short_enabled if a.default_short else 1
    default_version=short_version if a.default_short else 'resident_shape_v1'
    for flag,requested,outer,enabled,version in (
        (0,1,2,1,'resident_shape_v1'),(1,1,2,short_enabled,short_version),
        (None,1,2,default_enabled,default_version),(1,0,2,0,'legacy_56_1'),
        (0,1,0,1,'resident_xadd6_v1'),(1,1,0,0,'legacy_56_1')):
        ee=env|{'NTT_D_MODEL':str(requested),'NTT_FUSE_COOP_OUTER':str(outer)}
        if flag is None:ee.pop('NTT_GL_SHORT_REDUCE',None)
        else:ee['NTT_GL_SHORT_REDUCE']=str(flag)
        name=f'{flag}_{requested}_{outer}';cmd=[str(exe),*argv,'--d-plan-only']
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        r=subprocess.run(cmd,env=ee,capture_output=True,timeout=120)
        text=(r.stdout+r.stderr).decode('utf-8',errors='replace')
        (out/(name+'.log')).write_text(text,encoding='utf-8')
        line=re.search(r'd_model: requested=(\d+) enabled=(\d+) version=(\S+).*gl_short=(\d+)',text)
        checks={'exit':r.returncode==0,
            'selector':bool(line) and (line[1],line[2])==(str(requested),str(enabled)),
            'version':bool(line) and line[3]==version,
            'gl_short':bool(line) and line[4]==str(a.default_short if flag is None else flag),
            'explicit_D':bool(re.search(r'd_plan_only: D=1231230 P=115200 curves_executed=0',text)),
            'no_curve':'stage2_full_wall:' not in text and 'real_baby:' not in text}
        rows.append(dict(name=name,command=cmd,checks=checks));print(name,checks,flush=True)
    assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
    result=dict(exe=str(exe),sha256=sha,device=a.device,provenance=str(a.provenance.resolve()),runs=rows,
        passed=sum(sum(r['checks'].values()) for r in rows),failed=sum(sum(not v for v in r['checks'].values()) for r in rows))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
