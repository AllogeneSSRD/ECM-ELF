"""Compare no-tune D/P with all cells of the frozen full-width GPU baseline.

Plans execute no curves. This proves parameter preservation, not performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'reference', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = args.reference.resolve() / 'plan.json'
    reference = json.loads(source.read_text(encoding='utf-8-sig'))
    assert reference['complete'] and not reference['failures']
    groups = {}
    for run in reference['runs']:
        key = (run['case'], run['B2'], run['carrier_exponent'])
        if key in groups:
            assert (run['D'], run['P']) == (groups[key]['D'], groups[key]['P'])
        groups[key] = run
    assert len(groups) == 52
    cases = {c['id']: c for c in reference['cases']}
    ini = out / 'ecm.ini'; ini.write_text('user=\ncomputer=\nstage2_tune_runtime=0\n', encoding='utf-8')
    report = dict(complete=False, planning_only=True, reference=str(source),
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(), cells=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    for index, ((case_id, b2, carrier), prior) in enumerate(groups.items()):
        case = cases[case_id]
        command = [str(args.exe.resolve()), '--ini', str(ini), '--device', '1', '--save', case['save'],
            '--b2', str(b2), '--d', '0', '--carrier-exponent', str(carrier), '--auto-carrier', '0',
            '--tune-runtime', '0', '--short-calibration', '0', '--tune-profile', str(out / 'absent.toml'),
            '--batch-mb', '256', '--arena-mb', '6300', '--owner-budget-mb', '640', '--relocation', '0', '--plan-only']
        (out / f'{index:02d}.command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, errors='replace', timeout=90)
        (out / f'{index:02d}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Planning failed; inspect ' + str(index))
        plans = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        plan = next(p for p in plans if p.get('type') == 'stage2_plan')
        assert plan['curves_executed'] == 0
        assert (plan['D'], plan['P']) == (prior['D'], prior['P']), (case_id, b2, carrier, plan['D'], prior['D'])
        report['cells'].append(dict(case=case_id, B2=b2, carrier_exponent=carrier, D=plan['D'], P=plan['P']))
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['complete'] = True
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert not (out / 'absent.toml').exists()
    print(json.dumps({k: v for k, v in report.items() if k != 'cells'} | {'cells': len(report['cells'])}))


if __name__ == '__main__':
    main()
