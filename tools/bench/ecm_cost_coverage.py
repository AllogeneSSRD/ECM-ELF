"""Check complete calibration/validation schedules without relying on flags."""
from collections import Counter
from ecm_cost_cases import low_cases


MEASURE_KEYS=('bits','D','B2','owner_mb','rep','kind','regime')


def measured_key(row):return tuple(row[key] for key in MEASURE_KEYS)


def expected_measurements(controls):
    out=[];anchors=controls.get('train_b2') or [controls['b2']]
    def add(bits,d,b2,owner,rep,kind,regime):
        out.append(dict(zip(MEASURE_KEYS,(bits,d,b2,owner,rep,kind,regime))))
    for bits in controls['bits']:
        for rep in range(controls['repeats']):
            for d in controls['d']:
                for b2 in anchors:
                    for owner in (controls['resident_mb'],0):add(bits,d,b2,owner,rep,'train','multiple')
            held=controls['d'] if controls.get('holdout_all_d') else [controls['d'][len(controls['d'])//2]]
            for d in held:
                for owner in (controls['resident_mb'],0):add(bits,d,controls['holdout_b2'],owner,rep,'holdout','multiple')
            if controls.get('g1') or controls.get('g2') or controls.get('bridge'):
                for d in controls['d']:
                    for regime,b2,kind in low_cases(d,min(anchors),controls['chain_min'],
                            controls.get('g1',False),controls.get('g2',False),controls.get('bridge',False)):
                        for owner in ([0] if regime=='g1' else [controls['resident_mb'],0]):add(bits,d,b2,owner,rep,kind,regime)
    return out


def check_study_coverage(study):
    expected=Counter(measured_key(row) for row in expected_measurements(study['controls']))
    actual=Counter(measured_key(row) for row in study['stage2'])
    if any(count!=1 for count in expected.values()):raise ValueError('Calibration schedule has duplicate inputs')
    if actual!=expected:raise ValueError('Calibration observations are missing, duplicated, or outside the declared schedule')
    if 'planned_stage2_cases' in study:
        planned=Counter(measured_key(row) for row in study['planned_stage2_cases'])
        if planned!=expected:raise ValueError('Stored calibration plan differs from its controls')
    expected_s1=Counter((bits,batch,f's1_m{bits}_batch{batch}_r{rep}',rep==0) for bits in study['controls']['bits']
        for batch in study['controls']['stage1_batch'] for rep in range(study['controls']['repeats']+1))
    actual_s1=Counter((row['bits'],row['batch'],row['name'],row['warmup']) for row in study['stage1'])
    if actual_s1!=expected_s1:raise ValueError('Stage1 amortization observations do not cover every declared batch and repetition')
    return dict(stage2_observations=sum(expected.values()),stage1_batches=sum(expected_s1.values()))


def check_blind_coverage(frozen,runs):
    def key(c):return (c['scope_id'],c['bits'],c['D'],c['B2'],c['owner_mb'],c['rep'],c['kind'])
    expected=Counter(key(c) for c in frozen['cases']);actual=Counter(key(r['case']) for r in runs)
    if not expected or any(n!=1 for n in expected.values()):raise ValueError('Frozen validation has duplicate case identities')
    if expected!=actual:raise ValueError('Frozen validation case missing or duplicated')
    return sum(expected.values())


def check_replay_labels(study,cases):
    training={(r['bits'],r['D'],r['owner_mb'],r['features']['I']) for r in study['stage2'] if r['kind']=='train'}
    for c in cases:
        geometry=(c['bits'],c['D'],c['owner_mb'],c['features']['I'])
        if (geometry in training)!=c['kind'].endswith('_replay'):
            raise ValueError('Validation replay label disagrees with training point geometry')


def independent_scope_counts(cases):
    return Counter(c['scope_id'] for c in cases if not c['kind'].endswith('_replay'))
