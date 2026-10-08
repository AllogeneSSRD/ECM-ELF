"""Production/development build isolation and Stage2 logging entry contracts.

CPU mode never starts a curve; GPU mode uses independent wide Stage1 fixtures.
Run GPU mode after timing matrices, so these invocations cannot perturb timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--phase', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('use a fresh evidence directory')
    exes = {'production': args.production.resolve(), 'development': args.development.resolve()}
    identity = {}
    for engine, exe in exes.items():
        manifest = exe.parent / 'build_manifest.json'
        build = read(manifest)
        if build['engine'] != engine or sha(exe) != build['sha256'].lower():
            raise ValueError('wrong build identity')
        identity[engine] = {'exe_sha256': sha(exe), 'manifest_sha256': sha(manifest)}
    prepared = read(args.fixtures)
    if not prepared['complete']:
        raise ValueError('fixtures incomplete')
    case = next(c for c in prepared['fixtures'] if c['name'] == 'generic16384')
    identity['fixtures_sha256'] = sha(args.fixtures)
    identity['tool_sha256'] = sha(__file__)
    report = dict(identity=identity, phase=args.phase, runs=[], complete=False,
                  formal_performance_samples=0)
    (out / 'collector.py').write_bytes(Path(__file__).read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0', CUDA_LAUNCH_BLOCKING='0')
    ini = out / 'manual.ini'
    ini.write_text('device=1\n', encoding='utf-8')

    def verify():
        if sha(__file__) != identity['tool_sha256'] or sha(args.fixtures) != identity['fixtures_sha256']:
            raise ValueError('collector/fixtures changed')
        if sha(case['save']) != case['save_sha256']:
            raise ValueError('save changed')
        for engine, exe in exes.items():
            if sha(exe) != identity[engine]['exe_sha256'] or sha(exe.parent / 'build_manifest.json') != identity[engine]['manifest_sha256']:
                raise ValueError('build changed')

    def persist():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    def call(name, command, code=0, token=None):
        verify()
        child_env = env.copy()
        # Windows PowerShell 5.1 cannot load the inherited bundled PS7 Utility
        # module. Use its own system modules for build-script rejection checks.
        if str(command[0]).lower() == 'powershell':
            child_env['PSModulePath'] = str(Path(os.environ['WINDIR']) /
                'System32/WindowsPowerShell/v1.0/Modules')
        process = subprocess.run([str(c) for c in command], env=child_env, cwd=ROOT,
                                 capture_output=True, timeout=300)
        raw = process.stdout + process.stderr
        log = out / (name + '_driver.log')
        log.write_bytes(raw)
        verify()
        text = raw.decode('utf-8', 'replace')
        if (code == 0 and process.returncode != 0) or (code != 0 and process.returncode == 0):
            raise ValueError(name + ': unexpected exit; inspect raw log')
        if token and token not in text:
            raise ValueError(name + ': expected diagnostic absent')
        row = dict(name=name, command=[str(c) for c in command], exit=process.returncode,
                   driver_log_sha256=sha(log))
        if child_env.get('PSModulePath') != env.get('PSModulePath'):
            row['PSModulePath'] = child_env['PSModulePath']
        report['runs'].append(row)
        persist()
        print(name, 'passed', flush=True)
        return row, text

    prod, dev = exes['production'], exes['development']
    common = ['--ini', ini, '--save', case['save'], '--b2', '13230', '--d', '210', '--device', '1']
    persist()
    if args.phase == 'cpu':
        oversized = out / 'oversized.save'
        oversized.write_text('METHOD=ECM; SIGMA=26; B1=20; N=(2^16385-15); X=2;\n')
        result = out / 'oversized.jsonl'
        call('oversized_save', [prod, '--ini', ini, '--save', oversized, '--b2', '13230',
                               '--results', result, '--log-level', 'quiet'], 2, 'at most 16384 bits')
        if result.exists():
            raise ValueError('oversized save published result')
        queue = out / 'worktodo.txt'
        queue.write_text('ECMSTAGE2=1,2,16385,-15,"oversized.save",13230,0,1\n')
        before = queue.read_bytes()
        qini = out / 'queue.ini'
        qini.write_text(f'worktodo={queue}\nfinished={out / "finished.txt"}\ndevice=1\n')
        call('oversized_queue', [prod, '--ini', qini, '--once', '--results', result,
                                '--log-level', 'quiet'], 2, 'exceeds supported input range')
        if queue.read_bytes() != before or result.exists() or (out / 'finished.txt').exists():
            raise ValueError('oversized queue transaction changed')
        call('development_quiet', [dev, *common, '--dry-run', '--log-level', 'quiet'], 2,
             'development requires debug')
        builder = ROOT / 'tools/build/build_stage2_local.ps1'
        for name, options in [('backend', ['-GlBackend', 'fold']), ('outer', ['-OuterUnrollU', '4'])]:
            destination = out / name
            call('production_reject_' + name, ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                 '-File', builder, '-Engine', 'production', '-Build', destination, *options], 1,
                 'Production requires')
            if destination.exists():
                raise ValueError('invalid build created output')
        call('hostonly_cross_engine', ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
             '-File', builder, '-Engine', 'development', '-Build', prod.parent, '-SplitCompile', '6',
             '-HostOnly'], 1, 'HostOnly requires identical')
    else:
        for engine in ('production', 'development'):
            log, result = out / (engine + '.log'), out / (engine + '.jsonl')
            row, _ = call(engine + '_default_logs', [exes[engine], *common, '--factor-only',
                                                    '--log', log, '--results', result])
            text = log.read_text(encoding='utf-8')
            records = [json.loads(s) for s in result.read_text().splitlines()]
            if len(records) != 1 or records[0]['bad_factors'] or records[0]['factors']:
                raise ValueError('default result mismatch')
            if 'batched_progress:' not in text or 'real_batched_split:' not in text:
                raise ValueError('default major logs missing')
            internal = bool(re.search(r'^(tree_level|mont_selftest):', text, re.M))
            if internal != (engine == 'development'):
                raise ValueError('engine default verbosity wrong')
            if engine == 'development' and 'hash=' + case['expected_leaf_hash']['65'] not in text:
                raise ValueError('development complete monic fingerprint changed')
            row.update(log_sha256=sha(log), result_sha256=sha(result))
        queue, finished = out / 'worktodo.txt', out / 'finished.txt'
        saves = out / 'three.save'
        saves.write_text(Path(case['save']).read_text() * 3)
        task = 'ECMSTAGE2=1,2,16384,-15,"three.save",13230,1,1'
        queue.write_text(task + '\n')
        qini = out / 'queue.ini'
        qini.write_text(f'worktodo={queue}\nfinished={finished}\ntmp_dir={out}\ndevice=1\n'
                       'stage2_log_level=quiet\n[Worker #1]\nstage2_log_level=quiet\n')
        log, result = out / 'override.log', out / 'override.jsonl'
        row, text = call('cli_overrides_worker', [prod, '--ini', qini, '--once', '--factor-only', '--d', '210',
                         '--log-level', 'debug', '--log', log, '--results', result])
        actual = log.read_text(encoding='utf-8')
        if 'stage2_complete:' not in text or 'mont_selftest: cases=2048 mismatches=0' not in actual:
            raise ValueError('CLI debug did not reach driver/worker')
        if 'hash=' + case['expected_leaf_hash']['65'] not in actual:
            raise ValueError('queue complete monic fingerprint changed')
        if read(result)['record'] != 2 or queue.read_text().strip() or finished.read_text().strip() != task:
            raise ValueError('CLI override queue selection/transaction changed')
        row.update(log_sha256=sha(log), result_sha256=sha(result))
        tune = out / 'tune.jsonl'
        row, text = call('quiet_tune', [prod, '--ini', ini, '--tune', 'ntt', '--device', '1',
                         '--length-log2', '16', '--tune-repeats', '2', '--tune-memory-mb', '64',
                         '--tune-file', tune, '--log-level', 'quiet'])
        rows = [json.loads(s) for s in tune.read_text().splitlines()]
        sample = next(r for r in rows if r['type'] == 'sample')
        if (rows[0]['binary_sha256'] != sha(prod) or sample['status'] != 'measured' or
                sample['bad'] or sample['verified_words_per_sample'] != 65536 or
                len(sample['seconds']) != 2 or rows[-1]['failed'] or rows[-1]['measured'] != 1):
            raise ValueError('quiet tune data/verification contract changed')
        if '"type":"sample"' not in text or '"type":"complete"' not in text:
            raise ValueError('quiet suppressed command JSON')
        row['profile_sha256'] = sha(tune)
    verify()
    report.update(complete=True, passed=len(report['runs']), failed=0)
    persist()
    print(json.dumps(dict(passed=report['passed'], failed=0)), flush=True)


if __name__ == '__main__':
    main()
