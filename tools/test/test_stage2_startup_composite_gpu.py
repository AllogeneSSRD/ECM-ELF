"""Regression: valid composite save survives factor-producing startup calibration.

Use M2003, sigma=27, B1=20 from the frozen width scan. A sigma=26
replacement is invalid for this target. Never changes hardware settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]

def point_fields(path):
    line = next(line for line in path.read_text(encoding='utf-8').splitlines()
                if re.search(r'\bMETHOD\s*=\s*ECM\b', line))
    return {k: int(v, 16 if v.lower().startswith('0x') else 10)
            for k, v in re.findall(r'(N|X|SIGMA|B1)=([^;\s]+)', line)}

def verify(out, save, log_name, results_name, curve_name):
    log = (out / log_name).read_text(encoding='utf-8', errors='replace')
    assert 'stage2_short_calibration_skip: reason=verified_factor' in log
    assert 'stage2_complete: curves_this_run=1 invalid_tasks=0' in log
    assert 'benchmark Stage1 point is infinity' not in log
    receipt_path = Path(re.search(r'reason=verified_factor evidence=(.+)', log).group(1).strip())
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['status'] == 'verified_factor_not_timing_sample'
    assert receipt['clean'] == 1 and receipt['bad'] == 0
    assert receipt['selftest_cases'] > 0 and receipt['checked'] > 0
    source = dict(re.findall(r'(N|X|SIGMA|B1)=([^;\s]+)', save.read_text(encoding='utf-8')))
    anchor = dict(re.findall(r'(N|X|SIGMA|B1)=([^;\s]+)',
                            (receipt_path.parent / 'target.save').read_text(encoding='utf-8')))
    integer = lambda value: int(value, 16 if value.lower().startswith('0x') else 10)
    expected = {k: integer(v) for k, v in source.items()}
    expected['B1'] = min(expected['B1'], 20)
    assert expected == {k: integer(v) for k, v in anchor.items()}, \
        'Calibration must reuse the validated point and bounds'
    results = [json.loads(x) for x in (out / results_name).read_text().splitlines()]
    assert len(results) == 1
    result = results[0]
    assert result['status'] == 'stage2_completed' and result['bad_factors'] == 0
    assert result['sigma'] == 27 and result['B1'] == 20
    assert '4007' in result['factors']
    curve = (out / curve_name).read_text(encoding='utf-8', errors='replace')
    s4 = re.findall(r'^s4_multiply_stats: (.+)$', curve, re.M)
    assert s4 and 'gmp_selftest_bad=0' in s4[-1] and 'gmp_check_bad=0' in s4[-1]
    return dict(status='passed', factor_receipt=str(receipt_path),
                save_sha256=hashlib.sha256(save.read_bytes()).hexdigest(),
                factor='4007', reused_saved_point=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--save', type=Path, required=True)
    ap.add_argument('--stage2', type=Path)
    ap.add_argument('--ntt-profile', type=Path)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inspect', action='store_true', help='Audit retained manual regression logs')
    ap.add_argument('--production-save', type=Path, help='Also check a validated large-B1 save with plan-only')
    a = ap.parse_args()
    out, save = a.output.resolve(), a.save.resolve()
    if a.inspect:
        names = ('driver_fixed3.log', 'results_fixed3.jsonl', 'curve_fixed3.log')
    else:
        if not a.stage2 or not a.ntt_profile:
            ap.error('--stage2 and --ntt-profile required when running')
        out.mkdir(parents=True, exist_ok=False)
        ini = out / 'test.ini'
        ini.write_text('stage2_arena_mb=6300\nstage2_fold_mb=640\nstage2_batch_mb=256\n')
        shutil.copyfile(a.ntt_profile, out / 'tune.toml')
        names = ('driver.log', 'results.jsonl', 'curve.log')
        command = [str(a.stage2.resolve()), '--ini', str(ini), '--device', str(a.device),
                   '--save', str(save), '--b2', '26000000000', '--d', '0', '--curves', '1',
                   '--factor-only', '--tune-profile', str(out / 'tune.toml'),
                   '--results', str(out / names[1]), '--log', str(out / names[2]),
                   '--debug-log-file', str(out / 'debug.log')]
        (out / 'command.json').write_text(json.dumps(command, indent=2))
        original = save.read_bytes()
        with (out / names[0]).open('w', encoding='utf-8') as log:
            p = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert original == save.read_bytes(), 'User save changed'
        assert p.returncode == 0, 'Inspect retained driver.log'
    report = verify(out, save, *names)
    if a.stage2:
        report['binary_sha256'] = hashlib.sha256(a.stage2.read_bytes()).hexdigest()
    if a.production_save:
        assert not a.inspect and a.stage2 and a.ntt_profile
        high = out / 'production_b1'
        high.mkdir()
        shutil.copyfile(a.ntt_profile, high / 'tune.toml')
        saved = a.production_save.resolve()
        original = saved.read_bytes()
        command = [str(a.stage2.resolve()), '--ini', str(out / 'test.ini'), '--device', str(a.device),
                   '--save', str(saved), '--b2', '26000000000', '--d', '0', '--curves', '1',
                   '--factor-only', '--plan-only', '--tune-profile', str(high / 'tune.toml')]
        (high / 'command.json').write_text(json.dumps(command, indent=2))
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors='replace', timeout=60)
        log = p.stdout + p.stderr
        (high / 'driver.log').write_text(log, encoding='utf-8')
        assert p.returncode == 0 and saved.read_bytes() == original
        receipt = re.search(r'ecm_tune_progress: N=\d+ D=210 B2=42020', log)
        assert receipt, 'Production B1 must not enlarge the short calibration work'
        origins = sorted((ROOT / 'data/experiments').glob('stage2_short_calibration_*/target_origin.jsonl'),
                         key=lambda f: f.stat().st_mtime_ns)
        origin = json.loads(origins[-1].read_text())
        assert origin['saved_b1'] > 20 and origin['probe_b1'] == 20 and origin['point_reused']
        expected = point_fields(saved)
        expected['B1'] = 20
        assert point_fields(origins[-1].parent / 'target.save') == expected
        report['production_b1'] = dict(status='passed', origin=origin, evidence=str(origins[-1]),
                                      point_independently_compared=True,
                                      source_save_sha256=hashlib.sha256(original).hexdigest())
    (out / 'regression_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
