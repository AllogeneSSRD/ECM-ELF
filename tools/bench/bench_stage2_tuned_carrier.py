"""Tune and time the same cofactor saves with ordinary and Mersenne arithmetic.

Native tune does the D search. Training is finished before either timing arm;
the measured profile is frozen, startup calibration is disabled, and automatic
D remains enabled. Raw invocations, hashes and read-only telemetry are retained.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from bench_stage2_n_scaling import Monitor, collect, read, sha, stamp, write
from bench_stage2_production import freeze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--exponents', type=int, nargs='+')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--tune-budget-seconds', type=int, default=180)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-training', action='store_true', help='Use an already frozen, externally trained profile for holdout timing')
    a = p.parse_args()
    if min(a.repeats, a.tune_budget_seconds, a.timeout) <= 0:
        p.error('counts and budgets must be positive')
    out = a.output.resolve()
    mon = Monitor(a.device)
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    env.update(NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    try:
        if a.resume:
            data = read(out / 'measurements.json')
            if data['complete'] or data['collector_sha256'] != sha(__file__) or data['device'] != mon.identity:
                raise ValueError('resume needs unfinished study with unchanged collector/device')
        else:
            if out.exists() and any(out.iterdir()):
                raise ValueError('use a fresh output directory')
            out.mkdir(parents=True, exist_ok=True)
            previous = read(a.reference_study / 'measurements.json')
            if not previous['complete'] or previous['failures'] or previous['device'] != mon.identity:
                raise ValueError('need verified completed study on this GPU')
            exe = a.exe.resolve()
            identity = freeze(exe)
            build = read(exe.parent / 'build_manifest.json')
            if build.get('engine') != 'production':
                raise ValueError('need production engine')
            (out / 'bin').mkdir()
            for name in (exe.name, 'gmp-10.dll', 'build_manifest.json', 'frozen_sources_manifest.json'):
                shutil.copy2(exe.parent / name, out / 'bin' / name)
            shutil.copytree(exe.parent / 'sources', out / 'bin' / 'sources')
            cases = []
            for c in previous['cases']:
                if c['variant'] != 'cofactor' or (a.exponents and c['exponent'] not in a.exponents):
                    continue
                if sha(c['save']) != c['save_sha256'] or ((1 << c['exponent']) - 1) % int(c['N_hex'], 16):
                    raise ValueError('changed save or target does not divide Mersenne number')
                folder = out / 'inputs' / c['id']
                folder.mkdir(parents=True)
                shutil.copy2(c['save'], folder / 'stage1.save')
                cases.append(dict(c, save=str(folder / 'stage1.save')))
            if not cases:
                raise ValueError('no cases')
            shutil.copy2(a.profile, out / 'training.toml')
            shutil.copy2(a.profile, out / 'seed.toml')
            ini = out / 'bench.ini'
            ini.write_text(f'device={a.device}\nstage2_device={a.device}\nstage2_debug_log=true\n', encoding='ascii')
            data = dict(complete=False, created_utc=stamp(), device=mon.identity, identity=identity,
                        executable=str(out / 'bin' / exe.name), executable_sha256=sha(exe),
                        dll_sha256=sha(exe.parent / 'gmp-10.dll'), ini_sha256=sha(ini),
                        seed_sha256=sha(out / 'seed.toml'), collector_sha256=sha(__file__),
                        profile_source=str(a.profile.resolve()), skip_training=a.skip_training,
                        memory=previous['memory'], bounds=previous['bounds'], cases=cases,
                        reference_manifest_sha256=sha(a.reference_study / 'measurements.json'),
                        repetitions=a.repeats, tune_repeats=1, tune_budget_seconds=a.tune_budget_seconds,
                        environment={k: env[k] for k in ('NTT_NO_PROGRESS', 'CUDA_LAUNCH_BLOCKING')},
                        training=[], runs=[], failures=[])
            shutil.copy2(__file__, out / 'collector.py')
            write(out / 'measurements.json', data)

        def persist():
            write(out / 'measurements.json', data)

        def verify(c):
            if (sha(data['executable']) != data['executable_sha256'] or
                    sha(Path(data['executable']).parent / 'gmp-10.dll') != data['dll_sha256'] or
                    sha(out / 'bench.ini') != data['ini_sha256'] or sha(c['save']) != c['save_sha256']):
                raise ValueError('frozen binary/DLL/config/save changed')
            if 'profile_sha256' in data and sha(out / 'measured.toml') != data['profile_sha256']:
                raise ValueError('holdout timing profile changed')

        common = [data['executable'], '--ini', str(out / 'bench.ini'), '--device', str(a.device),
                  '--arena-mb', str(data['memory']['arena_mb']), '--batch-mb', str(data['memory']['batch_mb']),
                  '--owner-budget-mb', str(data['memory']['fold_mb'])]
        (out / 'training').mkdir(exist_ok=True)
        for c in ([] if data.get('skip_training', False) else data['cases']):
            for b in data['bounds']:
                key = f'{c["id"]}_b{b}'
                if any(r['key'] == key for r in data['training']):
                    continue
                verify(c)
                prefix = out / 'training' / key
                cmd = common + ['--tune', 'ecm', '--tune-file', str(out / 'training.toml'),
                                '--tune-save', c['save'], '--tune-carrier-exponent', str(c['exponent']),
                                '--tune-level', '10', '--tune-b2', str(b), '--tune-repeats', '1',
                                '--tune-budget-seconds', str(data['tune_budget_seconds'])]
                if prefix.with_suffix('.log').exists():
                    raise ValueError('unrecorded training log; retain and investigate before resuming')
                write(prefix.with_suffix('.command.json'), cmd)
                before = sha(out / 'training.toml')
                begin = time.perf_counter()
                print('TUNE', key, flush=True)
                with prefix.with_suffix('.log').open('wb') as f:
                    r = subprocess.run(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout)
                text = prefix.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
                entry = dict(key=key, case=c['id'], B2=b, command=cmd, returncode=r.returncode,
                             seconds=time.perf_counter()-begin, before_sha256=before,
                             after_sha256=sha(out / 'training.toml'), log_sha256=sha(prefix.with_suffix('.log')))
                data['training'].append(entry)
                persist()
                if r.returncode or 'ecm_tune_complete:' not in text:
                    raise ValueError('native tune failed: ' + key)
                print(text.splitlines()[-1], flush=True)
        if 'profile_sha256' not in data:
            shutil.copy2(out / 'training.toml', out / 'measured.toml')
            data['profile_sha256'] = sha(out / 'measured.toml')
            persist()

        sequence = []
        for c in data['cases']:
            for arm in ('generic', 'carrier'):
                sequence.append((c, min(data['bounds']), 'warmup', 0, arm))
        for repeat in range(1, data['repetitions']+1):
            for b in data['bounds']:
                for i, c in enumerate(data['cases']):
                    for arm in (('generic', 'carrier') if (i+repeat)%2 else ('carrier', 'generic')):
                        sequence.append((c, b, 'timing', repeat, arm))
        (out / 'runs').mkdir(exist_ok=True)
        for index, (c, b, category, repeat, arm) in enumerate(sequence):
            if any(r['sequence_index'] == index for r in data['runs']):
                continue
            verify(c)
            prefix = out / 'runs' / f'{index:03d}_{c["id"]}_b{b}_{category}_{arm}_r{repeat}'
            if prefix.with_suffix('.driver.log').exists():
                raise ValueError('unrecorded invocation; retain raw evidence before resuming')
            carrier = c['exponent'] if arm == 'carrier' else 0
            cmd = common + ['--save', c['save'], '--b2', str(b), '--d', '0', '--curves', '1',
                            '--carrier-exponent', str(carrier), '--tune-profile', str(out / 'measured.toml'),
                            '--short-calibration', '0', '--factor-only', '--log-level', 'curve',
                            '--log', str(prefix.with_suffix('.log')), '--results', str(prefix.with_suffix('.result.jsonl')),
                            '--debug-log-file', str(prefix.with_suffix('.debug.log'))]
            write(prefix.with_suffix('.command.json'), cmd)
            samples = [mon.sample()]
            begin = time.perf_counter()
            print(f'RUN {index+1}/{len(sequence)} {c["id"]} {b} {category} {arm}', flush=True)
            with prefix.with_suffix('.driver.log').open('wb') as f:
                proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = proc.wait(timeout=.5)
                        break
                    except subprocess.TimeoutExpired:
                        samples.append(mon.sample())
                        if time.perf_counter()-begin > a.timeout:
                            proc.kill()
                            proc.wait()
                            raise TimeoutError('timing invocation exceeded limit')
            elapsed = time.perf_counter()-begin
            samples.append(mon.sample())
            write(prefix.with_suffix('.telemetry.json'), samples)
            if code:
                raise ValueError('timing invocation failed: ' + str(prefix))
            verify(c)
            row = collect(data, c, dict(case=c['id'], B2=b, category=category, repeat=repeat, arm=arm), prefix, elapsed, samples)
            if row['result']['carrier_exponent'] != carrier:
                raise ValueError('locked arithmetic path was not used')
            row.update(sequence_index=index, command=cmd, debug_sha256=sha(prefix.with_suffix('.debug.log')),
                       telemetry=str(prefix.with_suffix('.telemetry.json')),
                       telemetry_sha256=sha(prefix.with_suffix('.telemetry.json')))
            data['runs'].append(row)
            persist()
            print(f'  full={row["wall"]["total"]} D={row["D"]} P={row["P"]}', flush=True)
        data.update(complete=True, completed_utc=stamp())
        persist()
        print('COMPLETE', len(data['runs']), 'accepted runs', flush=True)
    except Exception as exc:
        if 'data' in locals():
            data['failures'].append(dict(utc=stamp(), error=str(exc)))
            write(out / 'measurements.json', data)
        raise
    finally:
        mon.close()


if __name__ == '__main__':
    main()
