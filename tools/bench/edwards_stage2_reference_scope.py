"""Validate an explicit reporting scope without rewriting the original plan."""
from collections import Counter


def measurement_scope(plan, scope='all', allow_partial=False):
    engines = plan['engines'] if scope == 'all' else ['gpu', 'gpu-carrier']
    if scope not in ('all', 'gpu') or not set(engines) <= set(plan['engines']):
        raise ValueError('Reporting scope is not declared in the study')
    expected = set()
    for ref in plan['prime95_rows']:
        if ref['curve_type'] != 'Edwards' or not ref['benchmark_eligible']:
            continue
        matches = [c for c in plan['cases'] if c['N_hex'] == ref['N_hex']
                   and c['sigma'] == int(ref['s']) and c['b1'] == ref['b1']]
        if len(matches) != 1:
            raise ValueError('Ambiguous reference workload')
        for engine in engines:
            for rep in range(1, plan['repeats'] + 1):
                expected.add(f'{matches[0]["id"]}_b{ref["b2"]}_{engine}_r{rep}')
    selected = [r for r in plan['runs'] if r['engine'] in engines]
    seen = {r['id'] for r in selected}
    if len(seen) != len(selected) or not seen <= expected:
        raise ValueError('Unexpected or duplicate invocation in reporting scope')
    complete = bool(expected) and seen == expected
    if scope == 'all':
        if plan['complete'] and not complete:
            raise ValueError('Completed study lacks declared samples')
        complete = complete and plan['complete']
    if not complete and not allow_partial:
        raise ValueError('Measurement scope incomplete: ' + scope)
    return dict(scope=scope, complete=complete, original_matrix_complete=plan['complete'],
                engines=engines, expected_invocations=len(expected),
                accepted_invocations=len(seen), missing_invocations=len(expected - seen),
                engine_counts=dict(Counter(r['engine'] for r in selected)),
                cpu_reference_invocations=sum(r['engine'].startswith('cpu-') for r in plan['runs']))
