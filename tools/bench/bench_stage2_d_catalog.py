"""Interleave two frozen full-width Stage2 matrices using the existing collector.

Inputs are prepared study directories, not mutable live executables. Default and
frozen-profile modes disable startup calibration and use ordinary cofactor
arithmetic. Outputs remain compatible with analyze_stage2_n_scaling.py.
"""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time

import bench_stage2_n_scaling as bench


def prepare(source, out, device, repeats, exponents, bounds, profile):
    original = bench.read(source / 'measurements.json')
    out.mkdir(parents=True, exist_ok=False)
    data = copy.deepcopy(original)
    selected = [c for c in data['cases'] if c['variant'] == 'cofactor'
                and (not exponents or c['exponent'] in exponents)]
    if not selected or (exponents and set(exponents) != {c['exponent'] for c in selected}):
        raise ValueError('requested exponent missing from frozen study')
    bounds = bounds or data['bounds']
    if not set(bounds) <= set(data['bounds']):
        raise ValueError('bound absent from reference matrix')
    if bench.sha(data['executable']) != data['executable_sha256']:
        raise ValueError('frozen input executable changed')
    binary = out / 'bin' / Path(data['executable']).name
    binary.parent.mkdir()
    for name in (binary.name, 'gmp-10.dll', 'build_manifest.json'):
        shutil.copy2(Path(data['executable']).parent / name, binary.parent / name)
    if bench.sha(binary.parent / 'gmp-10.dll') != data['dll_sha256']:
        raise ValueError('frozen input DLL changed')
    build = bench.read(binary.parent / 'build_manifest.json')
    if build['sha256'].lower() != data['executable_sha256']:
        raise ValueError('build identity differs')
    for name, expected in build['source_hashes'].items():
        src = source / 'sources' / name
        if bench.sha(src) != expected.lower():
            raise ValueError('frozen source changed: ' + name)
        dest = out / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    data['cases'] = []
    for case in selected:
        src = Path(case['save'])
        if bench.sha(src) != case['save_sha256']:
            raise ValueError('frozen input changed: ' + case['id'])
        dst = out / 'inputs' / case['id'] / src.name
        dst.parent.mkdir(parents=True)
        shutil.copy2(src, dst)
        data['cases'].append(dict(case, save=str(dst)))
    shutil.copy2(source / 'database_rows.json', out / 'database_rows.json')
    if bench.sha(out / 'database_rows.json') != data['database_rows_sha256']:
        raise ValueError('database evidence changed')
    ini = out / 'bench.ini'
    ini.write_text(f'device={device}\nstage2_device={device}\nstage2_debug_log=true\n'
                   'stage2_short_calibration=0\nstage2_carrier_exponent=0\n'
                   'stage2_short_ntt_batch_tune=false\n', encoding='ascii')
    if profile:
        shutil.copy2(profile, out / 'stage2_ecm_tune.toml')
        data['tune_profile_seed'] = dict(sha256=bench.sha(profile), startup_calibration_enabled=False)
    else:
        data.pop('tune_profile_seed', None)
    data.update(complete=False, created_utc=bench.stamp(), repetitions=repeats,
                full_controls=False, exponents=[c['exponent'] for c in data['cases']],
                bounds=bounds, executable=str(binary), ini_sha256=bench.sha(ini),
                reference_manifest_sha256=bench.sha(source / 'measurements.json'),
                reference_study=str(source), sequence=[], runs=[], failures=[],
                purpose='interleaved catalogue holdout; startup calibration disabled')
    data.pop('completed_utc', None)
    for case in data['cases']:
        data['sequence'].append(dict(case=case['id'], B2=min(bounds), category='warmup', repeat=0))
    for repeat in range(1, repeats + 1):
        for b2 in bounds:
            for case in data['cases']:
                data['sequence'].append(dict(case=case['id'], B2=b2, category='timing', repeat=repeat))
    bench.write(out / 'measurements.json', data)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-study', type=Path, required=True)
    p.add_argument('--candidate-study', type=Path, required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--exponents', type=int, nargs='+')
    p.add_argument('--b2', type=int, nargs='+')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.repeats < 1 or a.device < 0 or a.timeout < 1:
        p.error('counts must be positive and device nonnegative')
    out = a.output.resolve()
    monitor = bench.Monitor(a.device)
    try:
        if a.resume:
            if bench.sha(out / 'collector.py') != bench.sha(__file__):
                raise ValueError('collector changed; use the frozen collector or a new study')
            data = {arm: bench.read(out / arm / 'measurements.json') for arm in ('baseline', 'candidate')}
        else:
            out.mkdir(parents=True, exist_ok=False)
            data = {arm: prepare(src.resolve(), out / arm, a.device, a.repeats,
                                 a.exponents, a.b2, a.profile)
                    for arm, src in [('baseline', a.baseline_study), ('candidate', a.candidate_study)]}
            shutil.copy2(__file__, out / 'collector.py')
        before, after = data['baseline'], data['candidate']
        for s in data.values():
            if s['device'] != monitor.identity:
                raise ValueError('GPU identity differs from prepared study')
        for key in ('B1', 'memory', 'bounds', 'dll_sha256', 'database_rows_sha256', 'sequence'):
            if before[key] != after[key]:
                raise ValueError('paired scope differs: ' + key)
        if len(before['cases']) != len(after['cases']):
            raise ValueError('paired case count differs')
        for b, c in zip(before['cases'], after['cases']):
            for key in ('id', 'N_hex', 'Q_sha256', 'save_sha256', 'B1', 'sigma'):
                if b[key] != c[key]:
                    raise ValueError('paired input differs: ' + key)
        env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
        env.update(NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
        for index in range(len(before['sequence'])):
            for arm in (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')):
                plan, folder = data[arm], out / arm
                if any(r['sequence_index'] == index for r in plan['runs']):
                    continue
                item = plan['sequence'][index]
                case = next(c for c in plan['cases'] if c['id'] == item['case'])
                for file, expected in [(plan['executable'], plan['executable_sha256']),
                                       (Path(plan['executable']).parent / 'gmp-10.dll', plan['dll_sha256']),
                                       (folder / 'bench.ini', plan['ini_sha256']), (case['save'], case['save_sha256'])]:
                    if bench.sha(file) != expected:
                        raise ValueError('frozen paired input changed: ' + str(file))
                if 'tune_profile_seed' in plan and bench.sha(folder / 'stage2_ecm_tune.toml') != plan['tune_profile_seed']['sha256']:
                    raise ValueError('frozen profile changed')
                runs = folder / 'runs'
                runs.mkdir(exist_ok=True)
                attempt = 1 + sum(f['sequence_index'] == index for f in plan['failures'])
                prefix = runs / f'{index:03d}_{case["id"]}_b{item["B2"]}_{item["category"]}_r{item["repeat"]}_a{attempt}'
                if prefix.with_suffix('.driver.log').exists():
                    raise ValueError('unrecorded call exists; retain and inspect')
                command = bench.command(plan, folder, case, item, prefix)
                bench.write(prefix.with_suffix('.command.json'), command)
                print(f'[{index+1}/{len(before["sequence"])}] {arm} {case["id"]} B2={item["B2"]} {item["category"]}', flush=True)
                samples, started = [monitor.sample()], bench.stamp()
                begin = time.perf_counter()
                with prefix.with_suffix('.driver.log').open('wb') as f:
                    process = subprocess.Popen(command, cwd=folder, env=env, stdout=f, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        try:
                            process.wait(timeout=.25)
                        except subprocess.TimeoutExpired:
                            samples.append(dict(monitor.sample(), elapsed=time.perf_counter() - begin))
                            if time.perf_counter() - begin > a.timeout:
                                process.kill()
                                process.wait()
                                break
                seconds = time.perf_counter() - begin
                samples.append(monitor.sample())
                bench.write(prefix.with_suffix('.telemetry.json'), samples)
                try:
                    if process.returncode:
                        raise RuntimeError('Stage2 exit ' + str(process.returncode))
                    row = bench.collect(plan, case, item, prefix, seconds, samples)
                except Exception as exc:
                    plan['failures'].append(dict(sequence_index=index, attempt=attempt, error=str(exc),
                                                 command=command, prefix=str(prefix)))
                    bench.write(folder / 'measurements.json', plan)
                    raise
                row.update(sequence_index=index, command=command, collector_sha256=bench.sha(__file__),
                           started_utc=started, finished_utc=bench.stamp(),
                           telemetry=str(prefix.with_suffix('.telemetry.json')))
                plan['runs'].append(row)
                bench.write(folder / 'measurements.json', plan)
                print(f'  full={row["wall"]["total"]} s D={row["D"]} P={row["P"]}', flush=True)
        for arm, plan in data.items():
            plan.update(complete=True, completed_utc=bench.stamp())
            bench.write(out / arm / 'measurements.json', plan)
        print('COMPLETE: paired matrix retained; compare engine totals, review regressions separately', flush=True)
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
