"""Exercise real worker TOML readers with empty queues in an isolated portable copy.

No GUI is started, no work assignment is created, and no clock API is called.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', default='0.49.10')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    portable = out / 'portable'
    portable.mkdir()
    workers = {}
    for role, source in [('stage1', args.stage1), ('stage2', args.stage2)]:
        destination = portable / source.name
        shutil.copy2(source, destination)
        workers[role] = destination
    shutil.copy2(args.runtime, portable / 'gmp-10.dll')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    results = {}

    def run(name, worker, arguments):
        process = subprocess.run([str(worker), *arguments], cwd=portable,
                                 startupinfo=startup, capture_output=True, timeout=20)
        # Windows console encoding varies; all contract markers are ASCII.
        text = (process.stdout + process.stderr).decode('utf-8', errors='replace')
        (out / (name + '.log')).write_text(text, encoding='utf-8')
        (out / (name + '.exit.json')).write_text(json.dumps(
            dict(binary=str(worker), sha256=sha256(worker), arguments=arguments,
                 returncode=process.returncode), indent=2), encoding='utf-8')
        return process.returncode, text

    for role, worker in workers.items():
        code, text = run(role + '-protocol', worker, ['--gui-queue-protocol'])
        assert code == 0 and text.strip() == 'ECM_QUEUE_OWNER=1', (role, text)
        results[role + '_queue_protocol'] = True
    code, text = run('stage1-version', workers['stage1'], ['--version'])
    assert code == 0 and 'ECM-ELF ' + args.version in text, text
    results['stage1_version'] = True
    profiles = {
        'absent': '',
        'manual': 'gpu_clock_ranges="GPU-11111111-1111-1111-1111-111111111111:2600-2600"\n',
        'automatic': 'gpu_clock_ranges="GPU-11111111-1111-1111-1111-111111111111:210-3105:auto;GPU-22222222-2222-2222-2222-222222222222:1000-2000"\n',
        'unknown-gui': 'unexpected_clock_key="future GUI extension"\n',
        'gui-type': 'gpu_clock_ranges=123\n',
        'routing-count': 'NumWorkers=3\n',
        'unknown-worker': '[workers.1]\nfuture_label="GUI-only"\n',
        'routing-gui-extension': 'NumWorkers=3\nfuture_label="GUI-only"\n'
                                 '[stage1]\nauto_stage2=1\nauto_stage2_workers="auto"\n',
        'unknown-compute': '[stage1]\nfuture_compute_key=1\n',
        'compute-type': '[stage1]\ndevice="0"\n',
        'unknown-table': '[future_compute]\nvalue=1\n',
        'malformed-gui': 'future_label="unterminated\n',
        'duplicate-gui': 'future_label=true\nfuture_label=false\n',
        'routing-type': 'NumWorkers="3"\n',
    }
    errors = {
        'unknown-compute': 'unknown TOML key or invalid table: future_compute_key',
        'compute-type': 'wrong TOML value type for device',
        'unknown-table': 'unknown or duplicate TOML table: future_compute',
        'malformed-gui': 'unterminated TOML string',
        'duplicate-gui': 'duplicate TOML key: future_label',
        'routing-type': 'wrong TOML value type for NumWorkers',
    }
    for name, profile in profiles.items():
        case = portable / name
        case.mkdir()
        queue = case / 'worktodo.txt'
        queue.write_text('[Worker #1]\n# Empty compatibility fixture\n[Worker #2]\n', encoding='utf-8')
        original = queue.read_bytes()
        config = case / 'ecm.toml'
        # Stage1 defaults already disable automatic handoff. Keep the section
        # available for the negative fixtures without creating duplicate tables.
        config.write_text('[shared]\nworktodo="worktodo.txt"\n'
                          '[stage2]\nstage2_log_level=1\n'
                          '[gui]\n' + profile, encoding='utf-8')
        config_hash = sha256(config)
        for role, worker in workers.items():
            code, text = run(role + '-' + name, worker,
                             ['--config', str(config), '--worker', '1'])
            if name in errors:
                assert code != 0 and errors[name] in text, (role, name, code, text)
            else:
                assert code == 0, (role, name, code, text)
                if name in {'unknown-gui', 'unknown-worker', 'routing-gui-extension'}:
                    assert text.count('WARNING: ignoring unknown GUI-only TOML key:') == 1, (role, name, text)
                if role == 'stage1':
                    assert '0 task(s) processed' in text, text
                    if name == 'routing-gui-extension':
                        assert 'stage2_transfer: ready source_worker=1 receiver=auto' in text, text
                else:
                    assert 'ECM-ELF ' + args.version + ' (Stage2)' in text, text
                    assert 'stage2_complete: curves_this_run=0 invalid_tasks=0' in text, text
            assert queue.read_bytes() == original, (role, name, 'queue changed')
            assert sha256(config) == config_hash, (role, name, 'config changed')
            results[role + '_' + name] = True
    report = dict(passed=True, version=args.version, cases=results,
                  binaries={role: dict(source=str(source.resolve()), sha256=sha256(source))
                            for role, source in [('stage1', args.stage1), ('stage2', args.stage2)]},
                  runtime_sha256=sha256(args.runtime),
                  boundary='Real production workers, isolated portable copies, empty queues only; '
                           'no compute assignment, GUI, GPU clock write or user configuration used')
    (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
