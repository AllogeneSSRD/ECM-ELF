"""Audit a complete numbered-version Stage2 matrix against its original files."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
from bench_stage2_version import foreign_gpu_processes, load, validate
from bench_stage2_n_scaling import fields, number
from result_format import normalize_result


def audit(study, expected_repeats=3):
    plan = load(study/'plan.json')
    if expected_repeats < 1 or not isinstance(expected_repeats, int):
        raise ValueError('Expected repeats must be a positive integer')
    if not plan['complete'] or plan['failures'] or plan['repeats'] != expected_repeats:
        raise ValueError('Need a complete, failure-free matrix with the explicitly expected repeat count')
    if plan['device'] != 1 or len(plan['cases']) != 13:
        raise ValueError('Wrong device or input scope')
    cases = {c['id']: c for c in plan['cases']}
    original_cases = {c['id']:c for c in load(Path(plan['reference'])/'plan.json')['cases']}
    refs = [r for r in plan['prime95_rows'] if r['curve_type']=='Edwards' and r['benchmark_eligible']]
    if len(refs) != 26:
        raise ValueError('Need 26 exact Edwards reference conditions')
    expected = set()
    for case in cases.values():
        original_case = original_cases[case['id']]
        if any(case[k] != original_case[k] for k in ('N_hex','sigma','b1','known','exponent','bits')):
            raise ValueError('Reference input identity changed')
        if Path(case['save']).read_bytes() != Path(original_case['save']).read_bytes():
            raise ValueError('Frozen Stage1 point differs from its reference')
        n = int(case['N_hex'], 16)
        original = (1 << case['exponent'])-1
        for factor in case['known']:
            f = int(factor)
            if f <= 1 or original % f:
                raise ValueError('Invalid known factor')
            original //= f
        if original != n or n.bit_length() != case['bits']:
            raise ValueError('Wrong exact cofactor')
        points = validate(Path(case['save']), n, case['b1'], 1, case['sigma'], 'ed')
        if points[0][1] != 'POINT':
            raise ValueError('Stage1 input is incomplete')
        matched = [r for r in refs if r['N_hex']==case['N_hex'] and int(r['s'])==case['sigma'] and r['b1']==case['b1']]
        if len(matched) != 2 or len({r['b2_requested'] for r in matched}) != 2:
            raise ValueError('Missing or duplicate reference bound')
        for ref in matched:
            for mode in ('gpu', 'gpu-carrier'):
                for repeat in range(1,expected_repeats+1):
                    expected.add(f'{case["id"]}_b{ref["b2"]}_{mode}_r{repeat}')
    ids = Counter(r['id'] for r in plan['runs'])
    if set(ids) != expected or any(n != 1 for n in ids.values()) or len(expected) != 52*expected_repeats:
        raise ValueError('Missing, duplicate or extra invocation')
    checked = selftests = 0
    geometries = {}; process_snapshots = 0; selected_uuid = None
    for run in plan['runs']:
        case = cases[run['case']]
        if run['id'] != f'{case["id"]}_b{run["B2"]}_{run["engine"]}_r{run["repeat"]}':
            raise ValueError('Invocation metadata differs from its unique ID')
        ref = next(r for r in refs if r['N_hex']==case['N_hex'] and r['b2']==run['B2'])
        if run['nominal_B2'] != ref['b2_requested']:
            raise ValueError('Nominal bound differs from its exact reference')
        raw_result = load(Path(run['prefix']).with_suffix('.jsonl'))
        result = normalize_result(raw_result)
        carrier = case['exponent'] if run['engine']=='gpu-carrier' else 0
        identity = dict(N_hex=case['N_hex'], B1=case['b1'], B2=run['B2'],
                        sigma=case['sigma'], curve='ed', carrier_exponent=carrier)
        if any(result[k] != v for k,v in identity.items()):
            raise ValueError('Raw result has a different input: '+run['id'])
        if result['program']['version'] != plan['program_version'] or run['version'] != plan['program_version']:
            raise ValueError('Program version differs')
        if raw_result['status'] != 'NF' or result['status'] != 'stage2_completed' or result['bad_factors'] or result['factors'] or not run['full_sample']:
            raise ValueError('Full NF timing cannot contain early factor samples')
        prefix = Path(run['prefix'])
        text = prefix.with_suffix('.engine.log').read_text(encoding='utf-8')
        text += '\n'+prefix.with_suffix('.debug.log').read_text(encoding='utf-8')
        phase = fields(text, 'stage2_full_wall')
        s4 = fields(text, 's4_multiply_stats')
        shape = fields(text, 'real_shape')
        if phase != run['phase_seconds'] or shape != run['shape'] or s4 != run['s4']:
            raise ValueError('Recorded metrics differ from raw logs')
        if phase['clean'] != '1' or abs(float(phase['total'])-float(phase['init'])-float(phase['main'])) > .003:
            raise ValueError('Timing does not close')
        if float(phase['total']) != run['engine_seconds'] or run['engine_seconds'] <= 0:
            raise ValueError('Invalid engine time')
        if any(int(s4[k]) for k in ('gmp_selftest_bad','gmp_check_bad')):
            raise ValueError('Arithmetic check failed')
        if min(int(s4['gmp_selftest_cases']), int(s4['gmp_checked'])) <= 0:
            raise ValueError('Mandatory arithmetic checks missing')
        if (number(shape['B1']),number(shape['B2'])) != (case['b1'],run['B2']):
            raise ValueError('Coverage differs')
        if number(shape['S_bits']) != (carrier or case['bits']):
            raise ValueError('Arithmetic width differs from the declared mode')
        if (number(shape['D']),number(shape['P'])) != (run['D'],run['P']):
            raise ValueError('Geometry differs')
        d, p = run['D'], run['P']
        if d <= 0 or p <= 0:
            raise ValueError('Nonpositive geometry')
        # Independent integer coverage check for the original, non-relocated path.
        count = run['B2']//d+2
        if (number(shape['giant_points']),number(shape['num_poly_g'])) != (count,(count+p-1)//p):
            raise ValueError('Original interval coverage differs')
        remaining, phi, divisor = d, d, 2
        while divisor*divisor <= remaining:
            if remaining % divisor == 0:
                phi -= phi//divisor
                while remaining % divisor == 0:
                    remaining //= divisor
            divisor += 1
        if remaining > 1:
            phi -= phi//remaining
        if phi != 2*p or number(shape['baby_points']) != p:
            raise ValueError('Baby polynomial degree differs from phi(D)/2')
        for label, saved in run['stats'].items():
            if fields(text, label, False) != saved:
                raise ValueError('Component metric differs: '+label)
        # A repeated cell must describe one deterministic plan.
        key = (run['case'],run['B2'],run['engine'])
        geometry = tuple(shape[k] for k in ('D','P','giant_points','num_poly_g'))
        if geometries.setdefault(key, geometry) != geometry:
            raise ValueError('Geometry changed within a repeated cell')
        command = run['command']
        for flag,value in {'--device':'1','--relocation':'0','--short-calibration':'0',
                           '--tune-runtime':'0','--carrier-exponent':str(carrier)}.items():
            if command[command.index(flag)+1] != value:
                raise ValueError('Invocation setting differs')
        if not load(prefix.with_suffix('.telemetry.json')):
            raise ValueError('Missing device observations')
        if plan.get('protocol', {}).get('gpu_process_checks'):
            for suffix in ('.processes_before.json', '.processes_after.json'):
                snapshot = load(prefix.with_suffix(suffix))
                uuid = snapshot['uuid']
                if selected_uuid is not None and uuid != selected_uuid:
                    raise ValueError('Measured GPU UUID changed')
                selected_uuid = uuid
                if snapshot['shared_allowed'] != plan['protocol']['shared_device_allowed']:
                    raise ValueError('GPU sharing protocol differs')
                foreign = foreign_gpu_processes(snapshot['processes'], uuid, study)
                if foreign != snapshot['foreign'] or (foreign and not snapshot['shared_allowed']):
                    raise ValueError('Foreign GPU context in a retained process snapshot')
                process_snapshots += 1
        checked += int(s4['gmp_checked'])
        selftests += int(s4['gmp_selftest_cases'])
    return dict(complete=True, program_version=plan['program_version'], invocations=len(ids),
                conditions=len(geometries), repeats=expected_repeats, input_saves=len(cases), gmp_checked=checked,
                gmp_selftest_cases=selftests, arithmetic_errors=0,
                gpu_process_snapshots=process_snapshots,
                scope='Full NF measurements; sampled GMP arithmetic checks, not an independent full-leaf oracle')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--expected-repeats',type=int,default=3,
                   help='Explicit expected samples per cell; default 3, use 1 for an authorized screening matrix')
    a=p.parse_args(); report=audit(a.study,a.expected_repeats)
    if a.output:
        a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
