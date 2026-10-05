"""Observe the native save worker's actual D ranking, then stop it before curve completion.

This gate checks the production call site and budgets. It is not an arithmetic
or completed-curve test, and writes only fresh logs with a frozen Stage1 save.
"""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from calibrate_stage2_d import features,load_fixed_ptx_weights
from fit_stage2_d import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--large-plan',type=Path,required=True)
    p.add_argument('--small-plan',type=Path,required=True);p.add_argument('--budget-plan',type=Path,required=True)
    p.add_argument('--baby-budget-plan',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();save=a.save.resolve();repo=Path(__file__).resolve().parents[2]
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    digest=sha(exe);save_digest=sha(save);fingerprint=14695981039346656037
    for byte in save.read_bytes().split(b'\n',1)[0]:fingerprint=((fingerprint^byte)*1099511628211)&((1<<64)-1)
    manifest=json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    deps={m[1]:m[2].lower() for line in manifest['sources'] if (m:=re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})',line))}
    assert digest==manifest['sha256'].lower() and len(deps)>=17
    fit=json.loads(a.fit.read_text(encoding='utf-8'));profile=fit['feature_profile'];assert profile in (4,5,6)
    if profile in (5,6):
        assert len(deps)==(19 if profile==6 else 18) and manifest['gl_fixed_mode']==3
        load_fixed_ptx_weights(fit['ntt_weights'])
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_ARENA_CAP_KB='6451200',NTT_NO_PROGRESS='1',NTT_BABY_DEVICE='1')
    env['NTT_POINT_MERSENNE']='1' if profile==6 else '0'
    rows=[]
    cases=[('large',0,2011326186870,{},a.large_plan),('small',0,100000000000,{},a.small_plan),
           ('fold512',0,2011326186870,{'NTT_FOLD_DEVICE_MAX_MB':'512'},a.budget_plan)]
    if a.baby_budget_plan:
        cases.append(('baby128',0,2011326186870,{'NTT_BABY_DEVICE_MAX_MB':'128'},a.baby_budget_plan))
    cases += [('features_'+str(d),d,2011326186870,{},None) for d in (570570,1141140,1231230,1381380,1411410)]
    def verify():
        assert sha(exe)==digest and sha(save)==save_digest
        for rel,want in deps.items():assert sha(repo/rel)==want,rel
    for name,d,b2,overrides,plan_path in cases:
        verify();log=out/(name+'.log');result=out/(name+'.jsonl')
        cmd=[str(exe),'--curve-worker','--save',str(save),'--record-offset','0','--record-hash',str(fingerprint),
             '--record-index','1','--b2',str(b2),'--d',str(d),'--device',str(a.device),'--results',str(result)]
        start=time.monotonic()
        with log.open('wb') as output:
            child=subprocess.Popen(cmd,env=env|overrides,stdout=output,stderr=subprocess.STDOUT)
            try:
                while time.monotonic()-start<60:
                    text=log.read_text(encoding='utf-8',errors='replace')
                    if 'd_scan_wall:' in text or child.poll() is not None:break
                    time.sleep(.01)
            finally:
                if child.poll() is None:child.terminate()
                child.wait(timeout=10)
        verify();text=log.read_text(encoding='utf-8',errors='replace')
        version='resident_point_fold_v1' if profile==6 else 'resident_fixed_ptx_v1' if profile==5 else 'resident_baby_v1'
        assert f'd_model: requested=1 enabled=1 version={version}' in text,(name,text[-1500:])
        assert 'stage2_full_wall:' not in text and not result.exists(),(name,'worker completed unexpectedly')
        decision=re.search(r'd_scan_wall: seconds=([\d.]+) selected_D=(\d+)',text);assert decision,name
        selected=int(decision[2]);want=d
        if plan_path:want=json.loads(plan_path.read_text(encoding='utf-8'))['top'][0]['features']['D']
        assert selected==want,(name,selected,want)
        f=features(selected,b2,4423,profile);pred=predict(f,fit['rates'])
        candidates=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in re.findall(r'd_model_features: (.*)',text)]
        row=next(r for r in candidates if int(r['D'])==selected)
        for key,field in (('P','P'),('n_fold','fold_ntt'),('n_tree','tree_ntt'),('owner_bytes','owner_bytes')):
            assert int(row[key])==f[field],(name,key)
        for key,field in (('tree_work','ftree'),('inverse_work','inverse')):
            assert abs(float(row[key])-f[field])<=.501,(name,key)
        for key,field in (('init','init'),('giant','giant'),('gtrees','gtrees'),('fold','fold'),
                          ('descent','descent'),('inv','inv'),('accum','accum'),('glue','residual'),('total','full')):
            assert abs(float(row[key])-pred[field])<=.000001,(name,key,row[key],pred[field])
        rows.append(dict(name=name,D=selected,scan_seconds=float(decision[1]),command=cmd,controlled_stop=True))
        print(name,'D=',selected,'OK',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(exe=str(exe),sha256=digest,save_sha256=save_digest,
        scope='Native save worker planner only; no completed curves',passed=len(rows),failed=0,runs=rows),indent=2))


if __name__=='__main__':main()
