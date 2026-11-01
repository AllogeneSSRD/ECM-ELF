"""Check production point-arithmetic defaults, validation and queue children.

Use an independent sigma=26, B1=20 M127 save. Each valid case runs a complete
small Stage2; this is a correctness test, not a performance measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from bench_stage2_version import check_gpu_processes, write
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--allow-gpu-pid', type=int, action='append', default=[])
    p.add_argument('--check', choices=('square', 'giant-coop', 'baby-coop'), default='square')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'bin'
    binary.mkdir()
    (binary / '.gitignore').write_text('*\n')
    for source in (a.exe.resolve(), a.exe.resolve().parent / 'gmp-10.dll',
                   a.exe.resolve().parent / 'build_manifest.json'):
        shutil.copy2(source, binary / source.name)
    exe = binary / a.exe.name
    shutil.copy2(__file__, out / 'test_snapshot.py')
    identity = sha(exe)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    allowed = []
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    for pid in a.allow_gpu_pid:
        names = {line.split(',', 2)[2].strip() for line in processes.splitlines()
                 if line.split(',')[1].strip() == str(pid)}
        if len(names) != 1 or not Path(next(iter(names))).is_absolute():
            raise ValueError('Cannot bind allowed process identity')
        allowed.append(dict(pid=pid, process_name=next(iter(names))))
    report = dict(complete=False, exe_sha256=identity, dll_sha256=sha(binary / 'gmp-10.dll'),
                  allowed_processes=allowed, cases=[], failures=[])
    write(out / 'summary.json', report)
    telemetry = TelemetryRecorder(a.device, out / 'telemetry.jsonl', interval=.2)
    cases = [('default', None, 1, False), ('off', '0', 0, False), ('on', '1', 1, False),
             ('invalid_two', '2', None, False), ('invalid_negative', '-1', None, False),
             ('invalid_float', '1.0', None, False), ('queue_default', None, 1, True)]
    key = 'NTT_POINT_SQUARE'
    if a.check == 'giant-coop':
        key = 'NTT_GIANT_COOP'
        cases = [('default', None, 0, False), ('off', '0', 0, False), ('on', '16', 16, False),
                 ('on32', '32', 32, False), ('invalid_one', '1', None, False), ('invalid_tpi', '64', None, False),
                 ('invalid_negative', '-1', None, False), ('invalid_float', '16.0', None, False),
                 ('queue_default', None, 0, True), ('queue_on', '16', 16, True), ('queue_on32', '32', 32, True)]
    if a.check == 'baby-coop':
        key='NTT_BABY_COOP'
        cases=[('default',None,0,False),('off','0',0,False),('on','32',32,False),
               ('invalid_one','1',None,False),('invalid_tpi','16',None,False),
               ('invalid_negative','-1',None,False),('invalid_float','32.0',None,False),
               ('queue_default',None,0,True),('queue_on','32',32,True)]
    try:
        for name, value, expected, queued in cases:
            directory = out / name
            directory.mkdir()
            shutil.copy2(a.save, directory / 'prime127.save')
            config = directory / 'ecm.toml'
            queue = directory / 'worktodo.txt'
            queue.write_text('ECMSTAGE2=1,2,127,-1,"prime127.save",13230,0,1\n', encoding='utf-8')
            config.write_text('[shared]\nworktodo=' + json.dumps(str(queue)) + '\n[stage2]\n'
                              'stage2_tune_runtime=false\nstage2_short_calibration=false\n', encoding='utf-8')
            command = [str(exe), '--config', str(config), '--device', str(a.device), '--d', '210',
                       '--tune-profile', str(directory / 'absent_tune.toml'),
                       '--results', str(directory / 'results.jsonl'),
                       '--submission', str(directory / 'submission.txt'),
                       '--log', str(directory / 'engine.log'),
                       '--debug-log-file', str(directory / 'debug.log')]
            command += ['--worker', '1', '--once'] if queued else [
                '--save', str(directory / 'prime127.save'), '--curves', '1', '--b2', '13230', '--factor-only']
            runenv = env.copy()
            if value is not None:
                runenv[key] = value
            write(directory / 'command.json', dict(argv=command, environment_key=key, value=value))
            check_gpu_processes(telemetry.monitor, out, directory / 'before.json', allowed=allowed)
            assert sha(exe) == identity, 'Frozen binary changed'
            result = subprocess.run(command, cwd=directory, env=runenv, capture_output=True, timeout=60)
            text = (result.stdout + result.stderr).decode('utf-8', errors='replace')
            (directory / 'console.log').write_text(text, encoding='utf-8')
            check_gpu_processes(telemetry.monitor, out, directory / 'after.json', allowed=allowed)
            if expected is None:
                assert result.returncode != 0 and key in text, text
            else:
                assert result.returncode == 0, text
                trace = text + '\n' + (directory / 'debug.log').read_text(encoding='utf-8')
                modes = re.findall(r'point_square_mode: enabled=([01])', trace)
                expected_square = expected if a.check == 'square' else 1
                assert modes and set(modes) == {str(expected_square)}, modes
                squares = re.findall(r'square_selftest: cases=(\d+) mismatches=(\d+)', trace)
                assert (bool(squares) == bool(expected_square)) and all(x == ('128', '0') for x in squares), squares
                if a.check in ('giant-coop','baby-coop'):
                    coop = re.findall(('baby' if a.check=='baby-coop' else 'giant')+r'_coop_mode: requested=(\d+) eligible=(\d+)', trace)
                    assert coop and set(coop) == {(str(expected), '0')}, coop
                    assert 'coop_selftest:' not in trace  # M127 retains the scalar path.
                assert re.search(r'mont_selftest:.*mismatches=0', trace), trace
                records = [json.loads(line) for line in (directory / 'results.jsonl').read_text(
                    encoding='utf-8-sig').splitlines() if line.strip()]
                assert len(records) == 1 and records[0]['status'] == 'NF', records
                assert records[0]['source-build'].lower() == identity
                normalized = normalize_result(records[0])
                assert int(normalized['N_hex'], 16) == (1 << 127) - 1
                assert (normalized['B1'], normalized['B2'], normalized['sigma']) == (20, 13230, 26)
                assert not normalized['factors'] and not normalized['bad_factors']
            report['cases'].append(dict(name=name, returncode=result.returncode, enabled=expected))
            write(out / 'summary.json', report)
            print(f'PASS: {name}', flush=True)
        report['complete'] = True
    except Exception as error:
        report['failures'].append(str(error))
        raise
    finally:
        telemetry.close()
        write(out / 'summary.json', report)


if __name__ == '__main__':
    main()
