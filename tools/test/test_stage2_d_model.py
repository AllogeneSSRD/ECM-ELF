"""Check the calibrated selector against offline features without executing curves.

Uses an immutable experiment binary and a verified A/B environment. GPU contexts
are created serially on the explicitly selected device. Logs and a summary are
saved in a fresh output directory; no ini, work queue or production files change.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import hashlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from calibrate_stage2_d import features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--large-plan',type=Path,required=True)
    p.add_argument('--small-plan',type=Path,required=True)
    p.add_argument('--budget-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update({k:str(v) for k,v in prov['env'].items() if v is not None})
    env.update({k:str(v) for k,v in next(r for r in prov['mode_controls'] if r['mode']=='6_mont').items() if k!='mode'})
    env.pop('NTT_FUSE_TRACE',None)
    env.update(NTT_D_MODEL='1',NTT_XADD6_TEST='0',NTT_XADD6_TEST_BAD='0')
    argv=[str(a.exe.resolve()),*map(str,prov['args']),'--d-plan-only']
    sha=hashlib.sha256(a.exe.read_bytes()).hexdigest();checks=[]

    def run(name,d=0,b2=2011326186870,overrides=None,arguments=None,enabled=True,code=0):
        ee=env|dict(overrides or {});cmd=argv.copy()
        for key,val in {'--d':d,'--b2':b2,'--device':a.device,**(arguments or {})}.items():
            cmd[cmd.index(key)+1]=str(val)
        if d==0:cmd.append('--choose-d')
        assert hashlib.sha256(a.exe.read_bytes()).hexdigest()==sha
        r=subprocess.run(cmd,env=ee,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8')
        assert r.returncode==code,(name,r.returncode,text[-1200:])
        line=re.search(r'd_model: requested=(\d+) enabled=(\d+) version=(\S+)',text)
        assert line and line[2]==str(int(enabled)),(name,line)
        assert 'real_baby:' not in text and 'prime_powers=' not in text and 'stage2_full_wall:' not in text
        wall=re.search(r'd_scan_wall: seconds=([\d.]+)',text)
        if code==0:
            decision=re.search(r'd_plan_only: D=(\d+) P=(\d+) curves_executed=0',text)
            assert decision and wall,(name,text[-500:])
            if d:assert int(decision[1])==d
        else:decision=None
        checks.append(dict(name=name,exit=r.returncode,enabled=enabled,D=int(decision[1]) if decision else None,scan_seconds=float(wall[1]) if wall else None))
        return text,decision

    for name,path,b2,overrides in (
        ('large',a.large_plan,2011326186870,{}),
        ('small',a.small_plan,100000000000,{}),
        ('fold512',a.budget_plan,2011326186870,{'NTT_FOLD_DEVICE_MAX_MB':'512'})):
        plan=json.loads(path.read_text());winner=plan['top'][0]['features']['D']
        text,decision=run(name,b2=b2,overrides=overrides)
        assert int(decision[1])==winner,(name,decision[1],winner)
    for d in (570570,1231230,1381380,1411410):
        text,_=run('features_'+str(d),d=d)
        lines=re.findall(r'd_model_features: (.*)',text)
        row=next(dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in lines if s.startswith('D='+str(d)+' '))
        f=features(d,2011326186870)
        for key,field in (('P','P'),('n_fold','fold_ntt'),('n_tree','tree_ntt'),('tree_work','ftree'),('inverse_work','inverse'),('owner_bytes','owner_bytes')):
            assert float(row[key])==f[field],(d,key,row[key],f[field])
    for name,over in (('legacy',{'NTT_D_MODEL':'0'}),('old_xadd',{'NTT_XADD6':'0'}),
        ('old_tile',{'NTT_FUSE_WARP_TAIL':'0'}),('oracle_pack',{'NTT_S4_ORACLE_PACK':'0'}),
        ('no_pool',{'NTT_ARENA_WORKSPACE_POOL':'0'}),('sample',{'NTT_S4_SAMPLE':'95'}),
        ('carry',{'NTT_S4_CARRY_BATCH':'0'}),('chain',{'NTT_GIANT_CHAIN_BLOCK':'32'}),
        ('coop',{'NTT_FUSE_COOP_OUTER':'1'})):
        run('fallback_'+name,overrides=over,enabled=False)
    run('fallback_B1',arguments={'--b1':1001},enabled=False)
    run('fallback_N',arguments={'--n-hex':hex((1<<257)-1)[2:]},enabled=False)
    run('fallback_bound',b2=99999999999,enabled=False)
    run('fallback_G1',d=1231230,b2=100000000000,enabled=False)
    run('arena_refused',overrides={'NTT_ARENA_CAP_KB':'8'},code=3)
    result=dict(exe=str(a.exe.resolve()),sha256=sha,device=a.device,passed=len(checks),failed=0,checks=checks)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
