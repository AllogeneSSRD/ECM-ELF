"""Fit an experimental NTT/paired-phase model; never publish production ranking data."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import tomllib

import numpy as np
from analyze_stage2_tune_workload import (ENGINE_PHASES, integer, ntt_phase_references,
    ntt_profile_set, ntt_workload_features, one_json, positive, table, workload)
from stage2_tune_route_cost import route_work

MODEL = 'phase_ntt_loop_v1'
SCOPE = ('target_bits','arithmetic_bits','carrier_exponent','modulus_kind','b1','d',
         'giant_chunk_points','giant_chain_min','giant_force_ladder')
LIMIT = .08


def nonnegative_fit(rows, targets):
    x, y = np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 4 or len(x) != len(y) or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            (x < 0).any() or (y <= 0).any()):
        raise ValueError('invalid component fit input')
    scales = x.max(axis=0)
    best, error = None, math.inf
    for size in range(1,5):
        for columns in itertools.combinations(range(4),size):
            if any(scales[c] <= 0 for c in columns):
                continue
            coef, _, rank, _ = np.linalg.lstsq(x[:,columns]/scales[list(columns)], y, rcond=1e-10)
            if rank != size or np.any(coef < -1e-10):
                continue
            rates = np.zeros(4)
            rates[list(columns)] = np.maximum(coef,0)/scales[list(columns)]
            residual = float(np.sum((x@rates-y)**2))
            if residual < error:
                best, error = rates, residual
    if best is None:
        raise ValueError('component fit has no nonnegative active set')
    return best


def paired_fixed(sample):
    if sample['phase_accounting'] != 'exclusive_engine_v1':
        raise ValueError('exclusive paired phases required')
    arrays = [sample['phase_'+name+'_samples'] for name in ENGINE_PHASES]
    total = sample['seconds']
    if any(len(array) != len(total) for array in arrays) or not total:
        raise ValueError('incomplete paired phase arrays')
    for i, actual in enumerate(total):
        values = [a[i] for a in arrays]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid exclusive phase time')
        if not math.isclose(math.fsum(values),positive(actual),rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('exclusive phases do not conserve full engine time')
    loop = sample['phase_giant_loop_samples']
    return [positive(t)-v for t,v in zip(total,loop)]


def loop_reference(rows, measurements):
    coverage, annotated = ntt_workload_features(rows, measurements)
    if coverage['ntt_missing_batch_bins']:
        raise ValueError('missing exact query NTT shape')
    phases = ntt_phase_references(annotated)
    return positive(phases['gtrees']['reference_seconds']+phases['fold']['reference_seconds'])


def train(records):
    records = sorted(records,key=lambda r:r['sample']['b2'])
    samples = [r['sample'] for r in records]
    if not 7 <= len(records) <= 128:
        raise ValueError('component group requires 7..128 anchors')
    if any(any(s[k] != samples[0][k] for k in SCOPE) for s in samples):
        raise ValueError('different component training scopes')
    if any(not s['fold_resident'] or not s['frontier_resident'] or s['bad'] or s['hits'] or
           not s['clean'] or not s['checked'] or not s['selftest_cases'] for s in samples):
        raise ValueError('unchecked/nonresident component anchor')
    points = [integer(s['giant_points'],1) for s in samples]
    if points[-1] < 2*points[0] or points[-1] > 2**53 or any(a >= b for a,b in zip(points,points[1:])):
        raise ValueError('invalid component span/order')
    ladder = [integer(s['giant_ladder_steps']) for s in samples]
    if ladder.count(0) < 3 or len(ladder)-ladder.count(0) < 3:
        raise ValueError('component model requires measured chain and ladder branches')
    rows = [[1,positive(r['loop_reference_seconds']),s['giant_ladder_steps'],s['giant_ladder_chunks']]
            for r,s in zip(records,samples)]
    totals = [positive(s['median_seconds']) for s in samples]
    if any(not math.isclose(t,statistics.median(s['seconds']),rel_tol=1e-9) for t,s in zip(totals,samples)):
        raise ValueError('invalid component anchor median')
    fixed = [paired_fixed(s) for s in samples]

    def fitted(indices):
        base = statistics.median(v for i in indices for v in fixed[i])
        if not math.isfinite(base) or base < 0:
            raise ValueError('invalid paired fixed cost')
        coef = nonnegative_fit([rows[i] for i in indices],[totals[i]-base for i in indices])
        return base,coef

    relative, absolute = [],[]
    for i in range(len(rows)):
        base,coef = fitted([j for j in range(len(rows)) if j != i])
        seconds = base+float(np.asarray(rows[i])@coef)
        absolute.append(abs(seconds-totals[i]))
        relative.append(absolute[-1]/totals[i])
    base,coef = fitted(list(range(len(rows))))
    positive_ladder = [v for v in ladder if v]
    return dict(model=MODEL,scope={k:samples[0][k] for k in SCOPE},anchors=len(rows),
        fixed_seconds=base,coefficients=coef.tolist(),
        coefficient_units=['seconds','engine_seconds_per_reference_second','seconds_per_ladder_step','seconds_per_ladder_chunk'],
        b2_min=samples[0]['b2'],b2_max=samples[-1]['b2'],
        ladder_min=min(positive_ladder),ladder_max=max(positive_ladder),
        max_loo_relative_error=max(relative),max_loo_absolute_error=max(absolute),
        mad_seconds=max(s['mad_seconds'] for s in samples),
        loo_errors=relative,qualified=max(relative) <= LIMIT)


def predict(model,plan,measurements):
    s = model['scope']
    if tuple(plan[k] for k in ('target_bits','bits','carrier_exponent','B1','D')) != tuple(
            s[k] for k in ('target_bits','arithmetic_bits','carrier_exponent','b1','d')):
        raise ValueError('component query scope mismatch')
    if not model['qualified'] or not model['b2_min'] < plan['B2'] < model['b2_max']:
        raise ValueError('component query outside qualified anchors')
    policy = plan['giant_memory']['policy']
    if (plan['giant_memory']['chunk_points'] != s['giant_chunk_points'] or
            policy['chain_min'] != s['giant_chain_min'] or policy['force_ladder'] != bool(s['giant_force_ladder'])):
        raise ValueError('component query giant policy mismatch')
    route = route_work(plan['I'],s['d'],s['giant_chunk_points'],s['giant_chain_min'],bool(s['giant_force_ladder']))
    ladder = route['giant_ladder_steps']
    if ladder and not model['ladder_min'] <= ladder <= model['ladder_max']:
        raise ValueError('component query ladder branch not covered')
    reference = loop_reference(workload(plan),measurements)
    row = np.asarray([1,reference,ladder,route['giant_ladder_chunks']])
    seconds = positive(model['fixed_seconds']+float(row@model['coefficients']))
    return dict(seconds=seconds,rank_seconds=seconds+model['max_loo_absolute_error']+2*model['mad_seconds'],
                loop_reference_seconds=reference,**route)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('profile','workload','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ntt-profile',type=Path,action='append',default=[])
    p.add_argument('--holdout-result',type=Path)
    p.add_argument('--query-plans',type=Path)
    a = p.parse_args()
    if bool(a.holdout_result) != bool(a.query_plans):
        p.error('holdout-result and query-plans must be supplied together')
    a.output.mkdir(parents=True,exist_ok=False)
    import hashlib
    sources = {}

    def source(path):
        data = path.read_bytes()
        if len(data) > 64*1048576:
            raise ValueError('component input exceeds 64 MiB')
        digest = hashlib.sha256(data).hexdigest()
        sources[str(path.resolve())] = digest
        frozen = a.output/'inputs'/digest/path.name
        frozen.parent.mkdir(parents=True,exist_ok=True)
        frozen.write_bytes(data)
        return data

    result = dict(complete=False,ranking_qualified=False,groups=[],holdouts=[],sources=sources)
    try:
        source(Path(__file__))
        profile = tomllib.loads(source(a.profile).decode('utf-8-sig'))
        features = tomllib.loads(source(a.workload).decode('utf-8-sig'))
        if (not profile['summary']['complete'] or profile['summary']['failed'] or
            not features['summary']['complete'] or features['summary']['failed'] or
            not features['profile']['ntt_policy_qualified'] or features['device'] != profile['device'] or
            features['policy'] != profile['policy'] or set(features['ecm']) != set(profile['ecm'])):
            raise ValueError('complete matching full ECM/workload inputs required')
        measurements,qualified = ntt_profile_set(profile,[tomllib.loads(source(p).decode('utf-8-sig')) for p in a.ntt_profile])
        if not qualified:
            raise ValueError('declared NTT policies required')
        groups = {}
        for key,s in profile['ecm'].items():
            if features['ecm'][key]['total_seconds'] != s['seconds']:
                raise ValueError('different paired workload costs')
            rows = list(features['workload'][key].values())
            reference = loop_reference(rows,measurements)
            scope = tuple(s[k] for k in SCOPE)
            groups.setdefault(scope,[]).append(dict(sample=s,loop_reference_seconds=reference))
        models = [train(records) for records in groups.values()]
        result['groups'] = models
        result['all_groups_qualified'] = all(m['qualified'] for m in models)
        if a.holdout_result:
            holdouts = json.loads(source(a.holdout_result))
            if not holdouts['complete'] or any(not h['complete'] or len(h['candidates']) < 4 for h in holdouts['holdouts']):
                raise ValueError('complete four-candidate held-out receipts required')
            for i,holdout in enumerate(holdouts['holdouts']):
                cases = []
                for candidate in holdout['candidates']:
                    d,carrier = candidate['d'],candidate['carrier']
                    path = a.query_plans/f'holdout_{i}_d{d}_c{carrier}_plan.jsonl'
                    source(path)
                    plan = one_json(path)
                    matches = [m for m in models if m['scope']['d'] == d and m['scope']['carrier_exponent'] == carrier]
                    if len(matches) != 1 or plan['B2'] != holdout['b2']:
                        raise ValueError('missing/ambiguous component holdout scope')
                    predicted = predict(matches[0],plan,measurements)
                    actual = statistics.median([positive(t) for t in candidate['seconds']])
                    if not math.isclose(actual,candidate['actual_median'],rel_tol=1e-9):
                        raise ValueError('inconsistent holdout median')
                    cases.append(dict(d=d,carrier=carrier,actual_seconds=actual,
                        relative_error=abs(predicted['seconds']-actual)/actual,**predicted))
                selected = min(cases,key=lambda c:c['rank_seconds'])
                loss = selected['actual_seconds']/min(c['actual_seconds'] for c in cases)-1
                result['holdouts'].append(dict(b2=holdout['b2'],candidates=cases,rank_loss=loss,
                    qualified=loss <= .05 and all(c['relative_error'] <= LIMIT for c in cases)))
        text = '# Experimental component model; not accepted by production selection.\n'
        text += table('profile',dict(format=1,unit='full_stage2_prediction',model=MODEL,
            ranking_qualified=False,timing_boundary='engine_init_plus_main',
            fixed_cost='median_of_paired_total_minus_giant_loop',holdout_error_limit=LIMIT))
        for i,model in enumerate(models):
            text += table('component.group_'+str(i),{k:v for k,v in model.items() if k != 'scope'})
            text += table('component.group_'+str(i)+'.scope',model['scope'])
        for path,digest in sources.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('component input changed during analysis')
        tomllib.loads(text)
        (a.output/'models.toml').write_text(text,encoding='utf-8')
        result['complete'] = True
        print(json.dumps(dict(complete=True,groups=len(models),all_groups_qualified=result['all_groups_qualified'],
            max_loo=max(m['max_loo_relative_error'] for m in models),
            max_retrospective_error=max((c['relative_error'] for h in result['holdouts'] for c in h['candidates']),default=None))))
    except Exception as error:
        result['failure'] = repr(error)
        raise
    finally:
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
