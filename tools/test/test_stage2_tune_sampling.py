"""Compare sampling diagnostics with native scoped predictions, without a GPU."""
import argparse,copy,json,math,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from analyze_stage2_tune_sampling import SCOPE

def literal(value):
    if isinstance(value,str):return json.dumps(value)
    if isinstance(value,bool):return str(value).lower()
    if isinstance(value,list):return '['+', '.join(map(literal,value))+']'
    return repr(value)

def write(path,data,samples):
    tables=[('profile',data['profile']),('device',data['device']),
        ('policy',{k:v for k,v in data['policy'].items() if k!='environment'}),
        ('policy.environment',data['policy']['environment'])]
    tables += [('ecm.sample_'+str(i),s) for i,s in enumerate(samples)]
    summary=dict(data['summary'],measured=len(samples));tables += [('summary',summary)]
    path.write_text(''.join('\n['+name+']\n'+''.join(k+' = '+literal(v)+'\n' for k,v in fields.items())
        for name,fields in tables),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--profile',type=Path,nargs='+',required=True)
    p.add_argument('--b2',type=int,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    tool=ROOT/'tools/bench/analyze_stage2_tune_sampling.py';fixture=a.fixture.resolve()
    command=[sys.executable,str(tool),'--profile',*[str(path.resolve()) for path in a.profile],
        '--b2',*map(str,a.b2),'--output',str(out/'diagnosis')]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (out/'diagnosis.console.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode==0,proc.stderr
    report=json.loads((out/'diagnosis/result.json').read_text(encoding='utf-8'))
    assert report['complete'] and 'not independent' in report['scope']
    checked=eligible=0;first=None
    for path in a.profile:
        data=tomllib.loads(path.read_text(encoding='utf-8-sig'));groups={}
        for sample in data['ecm'].values():groups.setdefault(tuple(sample[k] for k in SCOPE),[]).append(sample)
        for key,samples in groups.items():
            dest=out/f'scope_{checked}.toml';write(dest,data,samples)
            if first is None:first=(data,samples)
            diagnostic=next(g for g in report['groups'] if tuple(g['scope'][k] for k in SCOPE)==key
                and g['profile_sha256']==next(x['sha256'] for x in report['profiles'] if x['path']==str(path.resolve())))
            for query in diagnostic['queries']:
                proc=subprocess.run([str(fixture),'--predict',str(dest),str(query['b2'])],capture_output=True,text=True,timeout=60)
                assert proc.returncode==0,proc.stderr
                native=json.loads(proc.stdout);reference=query['reference_prediction']
                assert native['eligible']==bool(reference),(key,query,native)
                if reference:
                    assert math.isclose(native['seconds'],reference['seconds'],rel_tol=1e-9,abs_tol=1e-9)
                    assert math.isclose(native['max_relative_error'],reference['fit_relative_error'],rel_tol=1e-9,abs_tol=1e-9)
                    eligible+=1
                checked+=1
    rejected=0
    for name in ['route_mismatch','nonpositive_time','duplicate_I','wrong_unit']:
        data,samples=copy.deepcopy(first);dest=out/(name+'.toml')
        if name=='route_mismatch':samples[0]['giant_ladder_steps']+=1
        elif name=='nonpositive_time':samples[0]['median_seconds']=0
        elif name=='duplicate_I':samples.append(copy.deepcopy(samples[0]))
        else:data['profile']['unit']='field_convolution'
        write(dest,data,samples)
        proc=subprocess.run([sys.executable,str(tool),'--profile',str(dest),'--output',str(out/name)],
            capture_output=True,text=True,timeout=60)
        (out/(name+'.console.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert proc.returncode and not (out/name/'result.json').exists(),name
        rejected+=1
    result=dict(complete=True,native_queries=checked,eligible=eligible,ineligible=checked-eligible,
        bad_profiles_rejected=rejected,exploratory_model_not_used_in_production=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
