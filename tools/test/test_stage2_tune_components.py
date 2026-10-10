"""CPU gates for paired fixed costs, nonnegative component fits and qualification."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
import analyze_stage2_tune_components as components
from stage2_tune_route_cost import route_work


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    records = []
    rates = [.5,2.,.01,.05]
    for points in [20,41,44,47,80,84,120,160]:
        route = route_work(points,6,40,10)
        reference = .001*points+.000002*points*points
        row = [1,reference,route['giant_ladder_steps'],route['giant_ladder_chunks']]
        loop = sum(x*y for x,y in zip(row,rates))
        sample = dict(target_bits=16,arithmetic_bits=16,carrier_exponent=0,modulus_kind='generic',
            b1=20,b2=points*6,d=6,giant_chunk_points=40,giant_chain_min=10,giant_force_ladder=0,
            fold_resident=1,frontier_resident=1,bad=0,hits=0,clean=1,checked=1,selftest_cases=1,
            giant_points=points,median_seconds=1+loop,mad_seconds=0,seconds=[1+loop]*3,
            phase_accounting='exclusive_engine_v1',**route)
        for name in components.ENGINE_PHASES:
            sample['phase_'+name+'_samples'] = [loop if name=='giant_loop' else 1 if name=='setup' else 0]*3
        records.append(dict(sample=sample,loop_reference_seconds=reference))
    model = components.train(records)
    assert model['qualified'] and model['max_loo_relative_error'] < 1e-9
    assert abs(model['fixed_seconds']-1) < 1e-12
    assert max(abs(a-b) for a,b in zip(model['coefficients'],rates)) < 1e-9
    rejected = 0

    def refuse(fn):
        nonlocal rejected
        try:
            fn()
        except (ValueError,KeyError):
            rejected += 1
        else:
            raise AssertionError('invalid component evidence accepted')

    for key,value in [('phase_accounting','unknown'),('phase_setup_samples',[.5]*3),
                      ('phase_giant_loop_samples',[-1]*3),('seconds',[1]),('seconds',[True]*3)]:
        sample = copy.deepcopy(records[0]['sample'])
        sample[key] = value
        refuse(lambda:components.paired_fixed(sample))
    for key,value in [('carrier_exponent',17),('fold_resident',0),('bad',1),('checked',0),
                      ('median_seconds',float('nan')),('giant_points',1),('giant_ladder_steps',-1)]:
        broken = copy.deepcopy(records)
        broken[3]['sample'][key] = value
        refuse(lambda:components.train(broken))
    refuse(lambda:components.train(records[:6]))
    broken = copy.deepcopy(records)
    broken[2]['sample']['seconds'] = [100.]*3
    broken[2]['sample']['median_seconds'] = 100.
    broken[2]['sample']['phase_giant_loop_samples'] = [99.]*3
    assert not components.train(broken)['qualified']
    for rows,targets in [([[1,2]],[3]),([[1,2,3,4]],[-1]),([[0,0,0,0]],[1]),
                          ([[1,float('nan'),0,0]],[1])]:
        refuse(lambda:components.nonnegative_fit(rows,targets))
    minimal = dict(target_bits=16,bits=16,carrier_exponent=0,B1=20,D=6,B2=120)
    refuse(lambda:components.predict(model,minimal,{}))  # Exact endpoint is not interpolation.
    minimal['B2'] = 400
    minimal['bits'] = 17
    refuse(lambda:components.predict(model,minimal,{}))
    result = dict(complete=True,synthetic_anchors=8,coefficients_recovered=True,
        paired_cost_conservation=True,rejected=rejected,noisy_group_ineligible=True,
        production_ranking_qualified=False)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
