"""Verify full-ECM tune Auto B2 candidates against independent benefit costs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tomllib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from stage2_tune_route_cost import MODEL, annotate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--valid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    template=tomllib.loads(args.valid.read_text(encoding='utf-8'))
    template['profile']['prediction_model']=MODEL
    template['profile']['max_batches']=0
    bounds=[26000000000,52000000000,104000000000]
    sample=template['ecm']['sample_0']
    samples=[]
    for b2 in bounds:
        value=json.loads(json.dumps(sample));points=b2//value['d']+2;seconds=.1+points*.000001
        value.update(b2=b2,giant_points=points,seconds=[seconds-.001,seconds,seconds+.001],
                     median_seconds=seconds,mad_seconds=.001)
        for key,fraction in {'init_seconds':.2,'main_seconds':.8,'giant_seconds':.12,'gtrees_seconds':.4,
                             'fold_seconds':.12,'descent_seconds':.05,'inverse_seconds':.05,'accum_seconds':.01}.items():
            value[key]=seconds*fraction
        samples.append(annotate(value,172800000))
    counts={'accepted':0,'rejected':0}
    def write(name,values=samples,opted=True):
        data=json.loads(json.dumps(template))
        if not opted:data['profile'].pop('prediction_model')
        data['summary']['measured']=len(values)
        sections=[('profile',data['profile']),('device',data['device']),
                  ('policy',{k:v for k,v in data['policy'].items() if k!='environment'}),
                  ('policy.environment',data['policy']['environment'])]
        sections += [('ecm.sample_'+str(i),v) for i,v in enumerate(values)]
        sections += [('summary',data['summary'])]
        path=out/(name+'.toml')
        path.write_text(''.join('\n['+section+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in fields.items())
                                for section,fields in sections),encoding='utf-8')
        return path
    base=write('linear')
    def invoke(name,path=base,t1=3,lo=0,hi=0,ratio=1,success=True):
        proc=subprocess.run([str(args.fixture.resolve()),'--auto',str(path),str(t1),str(lo),str(hi),str(ratio)],
                            capture_output=True,text=True,errors='replace',timeout=30)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        assert (proc.returncode==0)==success,(name,proc.stdout,proc.stderr)
        counts['accepted' if success else 'rejected']+=1
        return json.loads(proc.stdout) if success else None
    def benefit(b2):
        return .11343+.88657*(math.log10(b2/20)/2)**(1.96617-.06781*math.log10(20))
    low=invoke('low_t1',t1=.001)
    high=invoke('high_t1',t1=100)
    assert low['B2']//sample['d']==bounds[0]//sample['d']
    assert high['B2']==bounds[-1] and low['limited'] and high['limited']
    middle=invoke('interior')
    assert bounds[0]<middle['B2']<bounds[-1] and middle['predicted'] and not middle['limited']
    for choice,t1 in [(low,.001),(middle,3),(high,100)]:
        seconds=.1+(choice['B2']//sample['d']+2)*.000001
        assert math.isclose(choice['seconds'],seconds,rel_tol=1e-10)
        assert math.isclose(choice['K'],benefit(choice['B2']),rel_tol=1e-12)
        assert math.isclose(choice['score'],benefit(choice['B2'])/(t1+choice['rank']),rel_tol=1e-12)
    best_dense=0
    for index in range(20001):
        b2=round(bounds[0]*4**(index/20000))
        score=benefit(b2)/(3+.1+(b2//sample['d']+2)*.000001+.002)
        best_dense=max(best_dense,score)
    assert middle['score']/best_dense>=.9999
    adjusted=invoke('adjusted',ratio=2)
    assert adjusted['B2']<middle['B2']
    fixed=invoke('fixed_interval',lo=78000000000,hi=78000000000)
    assert fixed['B2']==78000000000 and fixed['predicted'] and fixed['limited']
    bounded=invoke('bounded',lo=70000000000,hi=80000000000)
    assert 70000000000<=bounded['B2']<=80000000000
    untagged=write('untagged',opted=False)
    discrete=invoke('discrete',path=untagged)
    assert discrete['B2'] in bounds and not discrete['predicted']
    invoke('no_discrete_in_interval',path=untagged,lo=70000000000,hi=80000000000,success=False)
    one=write('one',samples[:1])
    assert invoke('one_exact',path=one)['B2']==bounds[0]
    nonresident=[dict(s,fold_resident=0) for s in samples]
    invoke('no_resident',path=write('nonresident',nonresident),success=False)
    for name,t1,ratio in [('zero_t1',0,1),('negative_t1',-1,1),('nan_t1','nan',1),('inf_t1','inf',1),
                          ('zero_ratio',3,0),('negative_ratio',3,-1),('nan_ratio',3,'nan')]:
        invoke(name,t1=t1,ratio=ratio,success=False)
    invoke('below',lo=bounds[0]-1,success=False)
    invoke('above',hi=bounds[-1]+1,success=False)
    invoke('reversed',lo=80000000000,hi=70000000000,success=False)
    # Qualification is per group, eligibility per query. An unseen ladder at
    # the group's midpoint must not disable every other chain-only grid point.
    chain_values=[]
    for i in [100000,230000,275600]:
        value=dict(samples[0]);seconds=.1+i*.000001
        value.update(b2=(i-2)*value['d'],giant_points=i,seconds=[seconds-.001,seconds,seconds+.001],
                     median_seconds=seconds,mad_seconds=.001)
        chain_values.append(annotate(value,172800))
    path=write('midpoint_unseen_ladder',chain_values)
    grid=json.loads(subprocess.check_output([str(args.fixture.resolve()),'--auto-grid',str(path),
        '3','0','0','1'],text=True))
    assert len(grid)>len(chain_values) and any(c['predicted'] for c in grid)
    report=dict(counts,independent_benefit=True,dense_reference_score_ratio=middle['score']/best_dense,
                stage1_and_ratio_change_selection=True,no_extrapolation=True,old_profile_exact_only=True,
                unseen_midpoint_keeps_eligible_grid=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
