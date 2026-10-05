"""Compare the production D class/real NTT shape backend with independent integer features.

CPU only; no curve or GPU allocation is performed. Scope/automatic selection is
separately accepted through the production save driver.
"""
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from calibrate_stage2_d import features,load_fixed_ptx_weights
from fit_stage2_d import predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--old-fit',type=Path,required=True)
    p.add_argument('--shape-fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--short-fit',type=Path)
    p.add_argument('--baby-fit',type=Path)
    p.add_argument('--fixed-ptx-fit',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    fits={0:json.loads(a.old_fit.read_text(encoding='utf-8')),2:json.loads(a.shape_fit.read_text(encoding='utf-8'))}
    assert fits[2]['feature_profile']==2
    if a.short_fit:
        fits[3]=json.loads(a.short_fit.read_text(encoding='utf-8'))
        assert fits[3]['feature_profile']==3
    if a.baby_fit:
        fits[4]=json.loads(a.baby_fit.read_text(encoding='utf-8'))
        assert fits[4]['feature_profile']==4
    if a.fixed_ptx_fit:
        fits[5]=json.loads(a.fixed_ptx_fit.read_text(encoding='utf-8'))
        assert fits[5]['feature_profile']==5
        load_fixed_ptx_weights(fits[5]['ntt_weights'])
    sha=hashlib.sha256(a.exe.read_bytes()).hexdigest();checks=[]
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    for profile in fits:
        for b2 in (100000000000,2011326186870):
            for d in (210,330330,510510,570570,1141140,1231230,1381380,1411410):
                f=features(d,b2,4423,profile);prediction=predict(f,fits[profile]['rates'])
                name=f'{profile}_{b2}_{d}'
                r=subprocess.run([str(a.exe.resolve()),'4423',str(b2),str(d),str(f['P']),str(profile)],
                                 env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
                text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8')
                assert r.returncode==0 and f'd_model_probe: profile={profile} curves_executed=0' in text,(name,text)
                row=dict(re.findall(r'(\w+)=([^\s]+)',text.splitlines()[0]))
                for key,field in (('P','P'),('n_fold','fold_ntt'),('n_tree','tree_ntt'),('owner_bytes','owner_bytes')):
                    assert int(row[key])==f[field],(name,key,row[key],f[field])
                for key,field in (('tree_work','ftree'),('inverse_work','inverse')):
                    assert abs(float(row[key])-f[field])<=.501,(name,key,row[key],f[field])
                for key,field in (('init','init'),('giant','giant'),('gtrees','gtrees'),('fold','fold'),
                                  ('descent','descent'),('inv','inv'),('accum','accum'),('glue','residual'),('total','full')):
                    assert abs(float(row[key])-prediction[field])<=.000001,(name,key,row[key],prediction[field])
                if profile in (4,5):
                    count=f['P'];nodes=0
                    for _ in range(8):count=(count+1)//2;nodes+=count
                    expected=8*((3*f['P']+5)*70+f['P']+nodes*70)+count
                    assert f'd_model_baby_payload: bytes={expected}' in text,(name,'baby budget')
                assert hashlib.sha256(a.exe.read_bytes()).hexdigest()==sha
                checks.append({'name':name,'total':float(row['total'])})
    result=dict(exe=str(a.exe.resolve()),sha256=sha,passed=len(checks),failed=0,checks=checks)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
