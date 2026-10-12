"""Native INI/CLI/child priority checks for worktodo carrier candidates.

Profiles are real short calibration outputs in a fresh ignored directory.
This functional check makes no performance claim or hardware changes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from stage2_tune_tables import summaries

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('ref', ROOT/'tools/stat/suyama_mont_ref.py')
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    exponent, removed = 67, 193707721
    n = ((1 << exponent)-1)//removed; point = ref.stage1(26, 20, n)
    assert point['gcd'] == 1 and ((1 << exponent)-1) % n == 0
    save = out/'stage1.save'
    save.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1=20; N={n}; X=0x{point["x"]:x}; '
                    f'CHECKSUM={20*26*n*point["x"]%4294967291};\n', encoding='ascii')
    queue = out/'worktodo.txt'
    original = f'ECMSTAGE2=1,2,67,-1,"{save}",260000,0,1,"{removed}"\n'
    queue.write_text(original, encoding='ascii')
    report = dict(binary_sha256=__import__('hashlib').sha256(a.stage2.read_bytes()).hexdigest(), cases=[])
    def run(name, setting, overrides, expected_modes, success=True, line=None, source='queue', expected_error=None):
        folder = out/name; folder.mkdir()
        ini = folder/'ecm.ini'
        ini.write_text('stage2_log_level=quiet\nstage2_log_file=\n' +
                       ('' if setting is None else f'stage2_auto_carrier={setting}\n'), encoding='ascii')
        profile = folder/'tune.toml'
        task = original if line is None else line
        queue.write_text(task, encoding='ascii')
        command = [str(a.stage2.resolve()), '--ini', str(ini), '--device', str(a.device),
                   '--plan-only', '--d', '210',
                   '--tune-profile', str(profile), '--short-calibration-seconds', '3',
                   '--arena-mb', '256', *map(str, overrides)]
        command += ['--save', str(save), '--b2', '260000'] if source == 'save' else ['--worktodo', str(queue), '--once']
        (folder/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        r = subprocess.run(command, cwd=folder, capture_output=True, text=True, errors='replace', timeout=60)
        (folder/'screen.log').write_text(r.stdout+r.stderr, encoding='utf-8')
        assert (r.returncode == 0) == success, (name, r.returncode)
        assert queue.read_text(encoding='ascii') == task
        if expected_error:
            assert expected_error in r.stdout+r.stderr, (name, r.stdout+r.stderr)
        if success:
            modes = {s['carrier_exponent'] for s in summaries(tomllib.loads(profile.read_text(encoding='utf-8')))}
            assert modes == expected_modes, (name, modes, expected_modes)
            if '--carrier-exponent' in overrides:
                requested = overrides[overrides.index('--carrier-exponent')+1]
                plans = [json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
                actual = next(x for x in plans if x.get('type') == 'stage2_plan')
                assert actual['carrier_exponent'] == requested, (name, actual['carrier_exponent'])
        report['cases'].append(dict(name=name, passed=True, exit_code=r.returncode))
    run('default_enabled', None, [], {0, 67})
    run('ini_disabled', '0', [], {0})
    run('cli_disabled', '1', ['--auto-carrier', 0], {0})
    run('cli_enabled', '0', ['--auto-carrier', 1], {0, 67})
    run('explicit_ordinary', '1', ['--carrier-exponent', 0], {0})
    # Calibration retains an ordinary reference even when execution is locked.
    run('explicit_carrier', '0', ['--carrier-exponent', 67], {0, 67})
    run('invalid_ini', 'maybe', [], set(), False)
    run('invalid_cli', '1', ['--auto-carrier', 2], set(), False)
    run('non_mersenne_expression', '1', [], {0},
        line=f'ECMSTAGE2={(n+1)//2},2,1,-1,"{save}",260000,0,1\n')
    p_above = 67*245
    removed_above = ((1 << p_above)-1)//n
    run('above_carrier_limit', '1', [], set(), False,
        line=f'ECMSTAGE2=1,2,{p_above},-1,"{save}",260000,0,1,"{removed_above}"\n',
        expected_error='worktodo exponent exceeds supported input range')
    run('save_does_not_guess', '1', [], {0}, source='save')
    run('save_N_mismatch', '1', [], set(), False,
        line=f'ECMSTAGE2=1,2,67,-1,"{save}",260000,0,1\n', expected_error='worktodo N differs from save N')
    run('invalid_known_divisor', '1', [], set(), False,
        line=f'ECMSTAGE2=1,2,67,-1,"{save}",260000,0,1,"2"\n', expected_error='does not divide')
    run('invalid_explicit_carrier', '0', ['--carrier-exponent', 66], set(), False,
        expected_error='divide')
    report['passed'] = True
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
