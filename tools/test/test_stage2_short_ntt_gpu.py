"""Replay frozen short-NTT A/B inputs; check native plans, ordered signatures and leaves."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from benchmark_stage2_short_ntt import fields, parse, sha
from test_stage2_request_program import native_records, verify_topology


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    a = p.parse_args()
    source, out, exe = a.experiment.resolve(), a.output.resolve(), a.exe.resolve()
    identity = json.loads((source / 'identity.json').read_text())
    assert sha(exe) == identity['binary_sha256']
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['NTT_REQUEST_AUDIT'] = '1'
    env['NTT_TARGET_LEAF_HASH'] = '1'
    rows = []
    for case in a.cases:
        leaves = None
        for enabled in (0, 1):
            name = f'{case}_{"on" if enabled else "off"}'
            old = json.loads((source / f'{name}_0.command.json').read_text())['command']
            command = old.copy()
            assert Path(command[0]) == exe
            assert '--short-ntt-batch-tune' in command
            for option in ('--results', '--log'):
                i = command.index(option)
                command[i+1] = str(out / (name + ('.result.jsonl' if option == '--results' else '.driver.log')))
            command += ['--debug-log-file', str(out / (name + '.debug.log'))]
            plan_command = command + ['--plan-only']
            run = subprocess.run(plan_command, env=env, cwd=ROOT, capture_output=True, timeout=60)
            (out / (name + '.plan.log')).write_bytes(run.stdout + run.stderr)
            assert run.returncode == 0, name
            plans = [json.loads(line) for line in run.stdout.decode('utf-8').splitlines() if line.startswith('{')]
            plan = next(row for row in plans if row.get('type') == 'stage2_plan')
            request = verify_topology(plan)
            assert plan['s4_memory']['valid'], plan['s4_memory']
            run = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, timeout=600)
            (out / (name + '.stdout.log')).write_bytes(run.stdout + run.stderr)
            assert run.returncode == 0, name
            log = out / (name + '.driver.log')
            row = parse(log, out / (name + '.result.jsonl'), leaf_fixture=True)
            text = log.read_text(encoding='utf-8') + '\n' + (out / (name + '.debug.log')).read_text(encoding='utf-8')
            memories = native_records(text, 's4_phase_memory')
            signatures = native_records(text, 'stage2_request_audit')
            for phase in request['phases']:
                actual = memories[phase['phase']]
                for key in ('groups','pairs','chunks','request_big_peak_bytes','request_output_peak_bytes'):
                    assert int(actual[key]) == phase[key], (name, phase['phase'], key, actual[key], phase[key])
                for key in ('multiplier','addend'):
                    assert int(signatures[phase['phase']][key]) == phase[key], (name, phase['phase'], key)
            for key in ('multiplier','addend'):
                assert int(signatures['all'][key]) == request[key], (name, 'all', key)
                assert int(fields(text, 'stage2_s4_program_audit')[key]) == plan['s4_memory'][key], (name, 's4', key)
            leaf = fields(text, 'target_descent_values')
            if leaves is None:
                leaves = leaf
            assert leaf == leaves, (name, 'leaf fingerprint differs')
            assert row['full_peak_bytes'] <= request['ntt_retained_bytes'], (name, 'retained NTT upper bound')
            row.update(case=case, enabled=enabled, leaf=leaf, native_plan=plan)
            rows.append(row)
            print(name, 'GMP, leaves, phase counters, request/S4 signatures PASS', flush=True)
    (out / 'checks.json').write_text(json.dumps(dict(complete=True, rows=rows,
        binary_sha256=sha(exe), checker_sha256=sha(Path(__file__))), indent=2))


if __name__ == '__main__':
    main()
