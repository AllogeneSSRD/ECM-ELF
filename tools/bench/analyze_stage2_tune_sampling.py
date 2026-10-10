"""Diagnose anchor coverage with independent nonnegative least squares (CPU only).

Exploratory models do not change production eligibility. The report retains all
anchors and compares leave-one-out errors; it is not an independent GPU holdout.
"""
import argparse,hashlib,itertools,json,math,sys,tomllib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from stage2_tune_route_cost import route_work,predict_route
SCOPE=('target_bits','arithmetic_bits','carrier_exponent','modulus_kind','b1','d',
       'giant_chunk_points','giant_chain_min','giant_force_ladder')

def diagnose(samples,chain_feature=False):
    names=['constant','giant_points','giant_ladder_steps','giant_ladder_chunks']
    if chain_feature:names+=['giant_chain_chunks']
    mixed=any(s['giant_ladder_steps'] for s in samples)
    if not mixed:names=['constant','giant_points']+(['giant_chain_chunks'] if chain_feature else [])
    x=np.array([[1. if name=='constant' else s[name] for name in names] for s in samples],dtype=float)
    y=np.array([s['median_seconds'] for s in samples],dtype=float);scale=x.max(axis=0)
    def fit(omit=None):
        rows=np.ones(len(x),dtype=bool)
        if omit is not None:rows[omit]=False
        best=None;best_error=math.inf
        subsets=[(0,1)] if not mixed and not chain_feature else (cols for n in range(1,len(names)+1) for cols in itertools.combinations(range(len(names)),n))
        for cols in subsets:
            if any(scale[j]<=0 for j in cols):continue
            design=x[rows][:,cols]/scale[list(cols)]
            values,_,rank,_=np.linalg.lstsq(design,y[rows],rcond=1e-10)
            if rank!=len(cols) or np.any(values < -1e-10):continue
            c=np.zeros(len(names));c[list(cols)]=np.maximum(values,0)/scale[list(cols)]
            residual=float(np.sum((x[rows]@c-y[rows])**2))
            if residual<best_error:best,best_error=c,residual
        return best
    full=fit();rows=[]
    for i,s in enumerate(samples):
        c=fit(i);predicted=None if c is None else float(x[i]@c)
        rows.append(dict(b2=s['b2'],source=s.get('sampling_source','base'),I=s['giant_points'],
            chain_chunks=s['giant_chain_chunks'],ladder_steps=s['giant_ladder_steps'],
            measured_seconds=s['median_seconds'],mad_seconds=s['mad_seconds'],
            leave_one_seconds=predicted,relative_error=None if predicted is None else abs(predicted-y[i])/y[i]))
    errors=[r['relative_error'] for r in rows]
    return dict(features=names,coefficients=None if full is None else full.tolist(),
        max_relative_error=None if any(e is None for e in errors) else max(errors),rows=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--b2',type=int,nargs='*',default=[],help='Check current model eligibility at queries; no GPU timing')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if any(b<3 or b>(1<<63)-8193 for b in a.b2):p.error('B2 must be 3..INT64_MAX-8192')
    report=dict(complete=False,scope='offline anchor diagnosis; not independent time/rank qualification',profiles=[],groups=[])
    for path in a.profile:
        raw=path.read_bytes();assert len(raw)<=64*1048576
        data=tomllib.loads(raw.decode('utf-8-sig'));groups={}
        assert data['profile']['format'] in (2,3) and data['profile']['unit']=='full_stage2'
        assert 0<len(data['ecm'])<=4096 and data['summary']['measured']==len(data['ecm'])
        for s in data['ecm'].values():
            assert s.get('giant_work_model')=='chunk_routes_v1','route-work metadata required'
            assert 0<s['giant_points']<=2**53 and 0<=s['giant_ladder_steps']<=2**53
            assert math.isfinite(s['median_seconds']) and s['median_seconds']>0
            assert math.isfinite(s['mad_seconds']) and s['mad_seconds']>=0
            work=route_work(s['giant_points'],s['d'],s['giant_chunk_points'],s['giant_chain_min'],bool(s['giant_force_ladder']))
            assert all(s[k]==v for k,v in work.items()),'profile work differs from independent count'
            groups.setdefault(tuple(s[k] for k in SCOPE),[]).append(s)
        identity=dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),prediction_model=data['profile'].get('prediction_model'))
        report['profiles'].append(identity)
        for key,samples in groups.items():
            assert len(samples)<=128,'group exceeds production model anchor limit'
            samples.sort(key=lambda s:s['b2']);assert all(a['giant_points']<b['giant_points'] for a,b in zip(samples,samples[1:])), 'duplicate I'
            report['groups'].append(dict(profile_sha256=identity['sha256'],scope=dict(zip(SCOPE,key)),samples=len(samples),
                chain_anchors=sum(not s['giant_ladder_steps'] for s in samples),ladder_anchors=sum(bool(s['giant_ladder_steps']) for s in samples),
                route_v2=diagnose(samples),exploratory_chain_chunks=diagnose(samples,True),
                queries=[dict(b2=b,reference_prediction=predict_route(samples,b,identity['prediction_model'])) for b in a.b2]))
    report['complete']=True;(out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for g in report['groups']:print(g['scope']['d'],g['scope']['carrier_exponent'],g['samples'],g['route_v2']['max_relative_error'],g['exploratory_chain_chunks']['max_relative_error'])
if __name__=='__main__':main()
