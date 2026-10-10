"""Audit completed Auto B2 benefit evidence without altering original samples.

Checks frozen sources, measured T1, every receipt/log, repeated medians, finite
benefit ranking, actual loaded modules and read-only NVML. No CUDA calls or
compilation. Current source equality is required only with --require-current.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tomllib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_positive(value):
    assert not isinstance(value, bool) and isinstance(value, (int, float))
    assert math.isfinite(value) and value > 0
    return value


def benefit(b1, b2):
    return .11343 + .88657 * (math.log10(b2 / b1) / 2) ** (1.96617 - .06781 * math.log10(b1))


def audit(evidence, require_current):
    report = json.loads((evidence / 'result.json').read_text(encoding='utf-8-sig'))
    assert report['complete'] and not report.get('failure'), 'coordinator has not completed successfully'
    assert report['time_error_limit'] == .08 and report['rank_loss_limit'] == .05
    repeats = report['repeats']
    assert 2 <= repeats <= 1000 and report['warmups'] == 1
    frozen, changed = {}, []
    for original, expected in report['source_identities'].items():
        path = Path(original)
        copy = evidence / 'inputs' / expected / path.name
        assert copy.is_file() and digest(copy) == expected, ('frozen input mismatch', original)
        frozen[original] = copy
        if not path.is_file() or digest(path) != expected:
            changed.append(original)
    assert not require_current or not changed, ('current source changed', changed)
    profile_paths = []
    stage1_paths = []
    for original, path in frozen.items():
        if path.suffix.lower() != '.toml':
            continue
        data = tomllib.loads(path.read_text(encoding='utf-8-sig'))
        if 'ecm' in data:
            profile_paths.append((original, data))
        if 'stage1' in data:
            stage1_paths.append((original, data))
    assert len(profile_paths) == len(stage1_paths) == 1
    profile_original, profile = profile_paths[0]
    stage1_original, stage1 = stage1_paths[0]
    assert profile['summary']['complete'] == stage1['summary']['complete'] == 1
    assert profile['summary']['failed'] == stage1['summary']['failed'] == 0
    assert all(stage1['device'][key] == profile['device'][key] for key in stage1['device'])
    b1, bits, t1, ratio = report['b1'], report['target_bits'], report['t1_seconds'], report['ratio']
    saves = [p for p in frozen.values() if p.suffix.lower() == '.save']
    assert len(saves) == 1
    line = next(s for s in saves[0].read_text(encoding='utf-8-sig').splitlines()
                if s.strip() and not s.lstrip().startswith(('#', ';')))
    fields = dict(re.findall(r'(\w+)\s*=\s*([^;]+)', line))
    target = int(fields['N'].strip(), 0)
    sigma = int(fields['SIGMA'])
    assert target > 3 and target % 2 and target.bit_length() == bits
    assert int(fields['B1']) == b1 and int(fields.get('PARAM', '0')) == 0
    assert int(fields.get('Z', '1'), 0) == 1
    kind = 'mersenne' if (target + 1) & target == 0 else 'generic'
    finite_positive(t1)
    finite_positive(ratio)
    measured_t1 = [s for s in stage1['stage1'].values() if s['target_bits'] == bits and s['b1'] == b1 and
                   s['batch'] == report['stage1_batch'] and s['exponent'] == report['exponent'] and
                   s['modulus_kind'] == kind]
    assert len(measured_t1) == 1
    for value in measured_t1[0]['seconds']:
        finite_positive(value)
    assert t1 == statistics.median(measured_t1[0]['seconds']) == measured_t1[0]['median_seconds']
    manifest_paths = [p for p in frozen.values() if p.name == 'build_manifest.json']
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding='utf-8-sig'))
    exe_hash = manifest['sha256'].lower()
    executable_copies = [p for p in frozen.values() if p.name == 'ecm_cuda_stage2.exe']
    assert len(executable_copies) == 1 and digest(executable_copies[0]) == exe_hash
    source_hashes = manifest['source_hashes']
    assert len(source_hashes) >= 49, 'incomplete production source closure'
    root = Path(__file__).resolve().parents[2]
    for name, expected in source_hashes.items():
        original = str((root / name).resolve())
        assert original in frozen and digest(frozen[original]) == expected.lower(), ('source closure mismatch', name)
    dll_hashes = {report['source_identities'][k] for k, p in frozen.items() if p.name == 'gmp-10.dll'}
    assert len(dll_hashes) == 1
    modules = json.loads((evidence / 'loaded_modules.json').read_text(encoding='utf-8-sig'))
    if isinstance(modules, dict):
        modules = [modules]
    assert modules
    for process in modules:
        actual = {m['name']: m['sha256'] for m in process['modules']}
        assert actual['ecm_cuda_stage2.exe'] == exe_hash and actual['gmp-10.dll'] in dll_hashes

    scores, errors, selected, devices = [], [], [], set()
    counts = dict(curves=0, mandatory_cases=0, gmp_checked=0, arithmetic_bad=0)
    for candidate in report['candidates']:
        seconds = []
        for repeat in range(repeats + 1):
            stem = f"candidate_{candidate['index']}_repeat{repeat}"
            receipt = json.loads((evidence / (stem + '.jsonl')).read_text(encoding='utf-8-sig'))
            text = (evidence / (stem + '.log')).read_text(encoding='utf-8-sig')
            assert receipt['status'] == 'stage2_completed' and receipt['hits'] == receipt['bad_factors'] == 0
            assert receipt['B1'] == b1 and receipt['B2'] == candidate['b2']
            assert int(receipt['N_hex'], 16) == target and receipt['sigma'] == sigma
            devices.add(receipt['device'])
            assert receipt['carrier_exponent'] == candidate['carrier']
            assert not candidate['carrier'] or ((1 << candidate['carrier']) - 1) % target == 0
            pick = receipt['auto_plan'] if candidate['selected'] else receipt['tune_plan']
            assert pick['D'] == candidate['d'] and pick['model'] == candidate['prediction']['model']
            if candidate['selected']:
                assert receipt['requested_D'] == receipt['requested_carrier_exponent'] == 0
                assert pick['T1'] == t1 and pick['T1_source'] == 'measured_stage1_profile'
            checks = re.findall(r'gmp_selftest_cases=(\d+) gmp_selftest_bad=(\d+) gmp_checked=(\d+) gmp_check_bad=(\d+)', text)
            assert len(checks) == 1
            cases, bad, checked, check_bad = map(int, checks[0])
            assert cases > 0 and bad == check_bad == 0
            counts['curves'] += 1
            counts['mandatory_cases'] += cases
            counts['gmp_checked'] += checked
            counts['arithmetic_bad'] += bad + check_bad
            assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b', text)
            assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b', text)
            walls = [line for line in text.splitlines() if line.startswith('stage2_full_wall:')]
            assert len(walls) == 1 and 'clean=1' in walls[0]
            total = finite_positive(float(re.search(r'\btotal=([0-9.]+)', walls[0])[1]))
            if repeat:
                seconds.append(total)
            else:
                assert total == candidate['warmup_seconds']
        assert seconds == candidate['seconds']
        total = statistics.median(seconds)
        assert total == candidate['actual_median_seconds']
        assert len(candidate['process_seconds']) == repeats
        assert all(finite_positive(p) >= s for p, s in zip(candidate['process_seconds'], seconds))
        assert statistics.median(candidate['process_seconds']) == candidate['process_median_seconds']
        error = abs(total - candidate['prediction']['seconds']) / total
        score = benefit(b1, candidate['b2']) / (t1 + ratio * total)
        assert math.isclose(error, candidate['relative_error'], rel_tol=1e-12, abs_tol=1e-14)
        assert math.isclose(score, candidate['actual_score'], rel_tol=1e-12)
        assert error <= .08
        scores.append(score)
        errors.append(error)
        if candidate['selected']:
            selected.append(candidate)
    assert len(selected) == 1
    assert len(devices) == 1, 'curves used multiple devices'
    device = next(iter(devices))
    selected = selected[0]
    choice = report['auto_plan']
    assert (choice['B2'], choice['D'], choice['carrier_exponent']) == (selected['b2'], selected['d'], selected['carrier'])
    assert choice['T1'] == t1 and choice['T1_source'] == 'measured_stage1_profile'
    assert choice['profile_sha256'] == report['source_identities'][profile_original]
    assert choice['stage1_profile_sha256'] == report['source_identities'][stage1_original]
    assert choice['model'] == selected['prediction']['model']
    assert math.isclose(choice['engine_seconds'], selected['prediction']['seconds'], rel_tol=1e-10)
    assert math.isclose(choice['guarded_engine_seconds'], selected['prediction']['rank'], rel_tol=1e-10)
    assert math.isclose(choice['K'], benefit(b1, selected['b2']), rel_tol=1e-12)
    assert math.isclose(choice['score'], choice['K'] / (t1 + ratio * choice['guarded_engine_seconds']), rel_tol=1e-12)
    loss = 1 - selected['actual_score'] / max(scores)
    assert math.isclose(loss, report['rank_loss'], rel_tol=1e-12, abs_tol=1e-14) and loss <= .05
    assert counts['curves'] == len(report['candidates']) * (repeats + 1)
    loaded, gpu_samples, malformed = [], 0, 0
    expected_uuid = profile['device']['uuid_hex']
    with (evidence / 'telemetry.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if len(row) != 8:
                malformed += 1
                continue
            if row[1].strip() != str(device):
                continue
            assert row[2].strip().removeprefix('GPU-').replace('-', '') == expected_uuid
            gpu_samples += 1
            if float(row[3].strip().split()[0]) > 50:
                loaded.append((float(row[4].strip().split()[0]), float(row[6].strip().split()[0])))
    assert loaded, 'no loaded NVML samples'
    telemetry = dict(device_samples=gpu_samples, util_over_50_samples=len(loaded), malformed_rows=malformed)
    for i, name in enumerate(('sm_mhz', 'power_w')):
        values = [point[i] for point in loaded]
        telemetry[name] = dict(min=min(values), median=statistics.median(values), max=max(values))
    return dict(complete=True, candidates=len(report['candidates']), counts=counts,
                device=device, frozen_inputs=len(frozen), binary_source_closure=len(source_hashes),
                current_source_changes=changed, actual_modules_match=True,
                measured_t1=t1, maximum_time_error=max(errors), rank_loss=loss, telemetry=telemetry,
                selected={k: selected[k] for k in ('b2', 'd', 'carrier', 'actual_median_seconds',
                                                'process_median_seconds', 'actual_score')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-current', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = audit(a.evidence.resolve(), a.require_current)
    result['auditor_sha256'] = digest(Path(__file__))
    result['coordinator_result_sha256'] = digest(a.evidence / 'result.json')
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
