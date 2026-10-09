"""Measure Stage2 versus modulus bit length, using read-only corpus factors.

The primary matrix removes known factors and repeats each (N, B2) three times.
Optional intact-Mersenne controls run once per cell. Preparation (including two
independent Stage1 implementations) is outside all Stage2 timing intervals.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXPONENTS = [503, 1009, 2003, 3001, 4001, 5003, 6011, 7001, 8011]
BOUNDS = [26_000_000_000, 260_000_000_000, 2_600_000_000_000]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def fields(text, label, required=True):
    rows = re.findall(r'^' + re.escape(label) + r': (.*)$', text, re.M)
    if not rows:
        if required:
            raise ValueError('Missing log record: ' + label)
        return {}
    return dict(re.findall(r'(\w+)=([^\s]+)', rows[-1]))


def number(value):
    # P=phi(D)/2=... was used by older engines; preserve the original too.
    return int(str(value).split('=')[-1])


def cpu_reference():
    path = ROOT / 'tools/stat/suyama_mont_ref.py'
    spec = importlib.util.spec_from_file_location('mont_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Memory(ctypes.Structure):
    _fields_ = [(key, ctypes.c_ulonglong) for key in ('total', 'free', 'used')]


class Utilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class Monitor:
    """Lightweight NVML observations; values describe the whole selected GPU."""
    def __init__(self, device):
        self.lib = ctypes.WinDLL('nvml.dll') if os.name == 'nt' else ctypes.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2():
            raise RuntimeError('NVML initialization failed')
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(device, ctypes.byref(self.handle)):
            raise RuntimeError('NVML device not found')
        self.identity = {'index': device}
        for key, function in [('name', 'nvmlDeviceGetName'), ('uuid', 'nvmlDeviceGetUUID')]:
            value = ctypes.create_string_buffer(128)
            if getattr(self.lib, function)(self.handle, value, len(value)):
                raise RuntimeError('NVML identity unavailable')
            self.identity[key] = value.value.decode()

    def sample(self):
        mem, util = Memory(), Utilization()
        ms = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
        us = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
        row = dict(utc=stamp(), memory_status=ms, utilization_status=us,
                   used_bytes=mem.used if ms == 0 else None,
                   total_bytes=mem.total if ms == 0 else None,
                   gpu_percent=util.gpu if us == 0 else None)
        for key, fn, extra in [('sm_clock_mhz', 'nvmlDeviceGetClockInfo', [1]),
                               ('temperature_c', 'nvmlDeviceGetTemperature', [0]),
                               ('power_mw', 'nvmlDeviceGetPowerUsage', [])]:
            value = ctypes.c_uint()
            status = getattr(self.lib, fn)(self.handle, *extra, ctypes.byref(value))
            row[key] = value.value if status == 0 else None
        return row

    def close(self):
        self.lib.nvmlShutdown()


def prepare(args, out, monitor):
    if any(out.iterdir()):
        raise ValueError('Preparation requires a fresh output directory')
    exe, gmp = args.exe.resolve(), args.gmp_ecm.resolve()
    manifest = exe.parent / 'build_manifest.json'
    build = read(manifest)
    if build['sha256'].lower() != sha(exe) or build.get('engine') != 'production':
        raise ValueError('Need a matching production build manifest')
    if build.get('architecture') != 'sm_89':
        raise ValueError('This study targets the RTX 4060 Laptop (sm_89)')
    for name, want in build['source_hashes'].items():
        source = ROOT / name
        if sha(source) != want.lower():
            raise ValueError('Compiled source changed: ' + name)
        target = out / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    binary = out / 'bin' / exe.name
    binary.parent.mkdir()
    for path in (exe, exe.parent / 'gmp-10.dll', manifest):
        shutil.copy2(path, binary.parent / path.name)
    if not gmp.is_file():
        raise FileNotFoundError(gmp)
    plan = dict(schema=1, created_utc=stamp(), complete=False, device=monitor.identity,
                B1=args.b1, bounds=args.b2, repetitions=args.repeats,
                full_controls=args.full_controls, exponents=args.exponents,
                memory=dict(arena_mb=args.arena_mb, fold_mb=args.fold_mb, batch_mb=args.batch_mb),
                executable=str(binary), executable_sha256=sha(binary),
                dll_sha256=sha(binary.parent / 'gmp-10.dll'), build=build,
                gmp_ecm=str(gmp), gmp_ecm_sha256=sha(gmp),
                stage1_reference_sha256=sha(ROOT / 'tools/stat/suyama_mont_ref.py'),
                cases=[], preparation=[], sequence=[], runs=[], failures=[])
    # Do not use dataset.connect(): it initializes schema and enables WAL writes.
    db = sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    records = {}
    try:
        for exponent in args.exponents:
            m = db.execute('SELECT * FROM mersennes WHERE exponent=?', (exponent,)).fetchone()
            if m is None:
                raise ValueError('Missing database exponent ' + str(exponent))
            factors = [dict(r) for r in db.execute('SELECT * FROM factors WHERE exponent=?', (exponent,))]
            records[str(exponent)] = dict(mersenne=dict(m), factors=factors)
    finally:
        db.close()
    write(out / 'database_rows.json', records)
    plan['database_rows_sha256'] = sha(out / 'database_rows.json')
    ref = cpu_reference()
    for exponent in args.exponents:
        full = (1 << exponent) - 1
        modulus, removed = full, []
        for row in sorted(records[str(exponent)]['factors'], key=lambda r: int(r['value'])):
            factor = int(row['value'])
            if factor <= 1 or full % factor:
                raise ValueError('Invalid catalog divisor')
            if modulus % factor:
                raise ValueError('Overlapping catalog factors; resolve the corpus first')
            power = 0
            while modulus % factor == 0:
                modulus //= factor
                power += 1
            removed.append(dict(value=str(factor), multiplicity=power))
        if modulus <= 1:
            raise ValueError('No remaining cofactor for M' + str(exponent))
        # Use the same surviving sigma for both members of each control pair.
        rejected = []
        for sigma in range(args.sigma, args.sigma + 1000):
            try:
                target = full if args.full_controls else modulus
                point = ref.stage1(sigma, args.b1, target)
                if point['gcd'] != 1:
                    raise ValueError('nonunit Stage1 point')
            except ValueError as exc:
                rejected.append(dict(sigma=sigma, reason=str(exc)))
                continue
            break
        else:
            raise ValueError('No Stage1 survivor; try a smaller --b1')
        variants = [('cofactor', modulus)]
        if args.full_controls:
            variants.append(('mersenne', full))
        for variant, n in variants:
            case_id = f'm{exponent}_{variant}'
            folder = out / 'inputs' / case_id
            folder.mkdir(parents=True)
            point = ref.stage1(sigma, args.b1, n)
            if point['gcd'] != 1:
                raise ValueError('Invalid reduced Stage1 point')
            raw = folder / 'gmp.save'
            cmd = [str(gmp), '-param', '0', '-sigma', str(sigma), '-c', '1',
                   '-save', str(raw), str(args.b1), '0']
            begin = time.perf_counter()
            proc = subprocess.run(cmd, input=(str(n) + '\n').encode(), cwd=folder,
                                  capture_output=True, timeout=120)
            (folder / 'gmp.log').write_bytes(proc.stdout + proc.stderr)
            if proc.returncode != 0 or not raw.exists():
                raise ValueError('GMP-ECM Stage1 generation failed: ' + case_id)
            saved_text = raw.read_text()
            match = re.search(r'\bX=(0x[0-9a-fA-F]+)', saved_text)
            saved_n = re.search(r'\bN=(\d+)', saved_text)
            if not match or int(match[1], 16) != point['x'] or not saved_n or int(saved_n[1]) != n:
                raise ValueError('Independent Stage1 X mismatch: ' + case_id)
            if sha(gmp) != plan['gmp_ecm_sha256']:
                raise ValueError('GMP-ECM changed')
            save = folder / 'stage1.save'
            checksum = args.b1 * sigma * n * point['x'] % 4294967291
            save.write_text(f'METHOD=ECM; PARAM=0; SIGMA={sigma}; B1={args.b1}; N={n}; '
                            f'X=0x{point["x"]:x}; CHECKSUM={checksum}; PROGRAM=N-scaling;\n', encoding='ascii')
            plan['cases'].append(dict(id=case_id, exponent=exponent, variant=variant,
                                      bits=n.bit_length(), words=(n.bit_length() + 63) // 64,
                                      N_hex=f'{n:x}', B1=args.b1, sigma=sigma,
                                      removed_factors=removed if variant == 'cofactor' else [],
                                      save=str(save), save_sha256=sha(save),
                                      Q_sha256=hashlib.sha256(f'{point["x"]:x}'.encode()).hexdigest()))
            plan['preparation'].append(dict(case=case_id, command=cmd, gmp_exit=proc.returncode,
                                            gmp_seconds=time.perf_counter() - begin,
                                            independent_X_equal=True, rejected_sigmas=rejected))
            print(f'prepared {case_id}: {n.bit_length()} bits B1={args.b1} sigma={sigma}', flush=True)
    # Nine width-specific warmups are retained separately, never averaged in.
    primary = [c for c in plan['cases'] if c['variant'] == 'cofactor']
    for c in primary:
        plan['sequence'].append(dict(case=c['id'], B2=min(args.b2), category='warmup', repeat=0))
    for repeat in range(1, args.repeats + 1):
        cases = primary if repeat % 2 else list(reversed(primary))
        bounds = args.b2 if repeat % 2 else list(reversed(args.b2))
        for b2 in bounds:
            for c in cases:
                plan['sequence'].append(dict(case=c['id'], B2=b2, category='timing', repeat=repeat))
                if args.full_controls and repeat == 2:
                    plan['sequence'].append(dict(case=f'm{c["exponent"]}_mersenne', B2=b2,
                                                 category='control', repeat=1))
    ini = out / 'bench.ini'
    ini.write_text(f'device={args.device}\nstage2_device={args.device}\nstage2_arena_mb={args.arena_mb}\n'
                   f'stage2_fold_mb={args.fold_mb}\nstage2_batch_mb={args.batch_mb}\n'
                   'stage2_d=0\nstage2_debug_log=true\n', encoding='ascii')
    plan['ini_sha256'] = sha(ini)
    write(out / 'measurements.json', plan)
    commands = []
    for index, item in enumerate(plan['sequence']):
        c = next(c for c in plan['cases'] if c['id'] == item['case'])
        commands.append(dict(**item, command=command(plan, out, c, item, out / 'manual' / f'{index:03d}')))
    write(out / 'commands.json', commands)
    quote = lambda s: "'" + str(s).replace("'", "''") + "'"
    ps = ["$ErrorActionPreference='Stop'", f"New-Item -ItemType Directory -Force -Path {quote(out / 'manual')} | Out-Null"]
    for item in commands:
        ps.extend(['& ' + ' '.join(quote(s) for s in item['command']),
                   'if ($LASTEXITCODE -ne 0) { throw "Stage2 failed" }'])
    (out / 'stage2_commands.ps1').write_text('\n'.join(ps) + '\n', encoding='utf-8-sig')
    return plan


def command(plan, out, case, item, prefix):
    return [plan['executable'], '--ini', str(out / 'bench.ini'), '--save', case['save'],
            '--b2', str(item['B2']), '--d', '0', '--curves', '1', '--device', str(plan['device']['index']),
            '--arena-mb', str(plan['memory']['arena_mb']), '--batch-mb', str(plan['memory']['batch_mb']),
            '--owner-budget-mb', str(plan['memory']['fold_mb']), '--factor-only', '--log-level', 'curve',
            '--results', str(prefix.with_suffix('.result.jsonl')), '--log', str(prefix.with_suffix('.log')),
            '--debug-log-file', str(prefix.with_suffix('.debug.log'))]


def collect(plan, case, item, prefix, process_seconds, samples):
    results = [json.loads(s) for s in prefix.with_suffix('.result.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(results) != 1:
        raise ValueError('Expected one curve result')
    result = results[0]
    if (result['N_hex'].lower() != case['N_hex'] or result['B1'] != case['B1']
            or result['B2'] != item['B2'] or result['sigma'] != case['sigma']
            or result['device'] != plan['device']['index'] or result['bad_factors']):
        raise ValueError('Result/input mismatch')
    n = int(case['N_hex'], 16)
    if any(not 1 < int(f) < n or n % int(f) for f in result['factors']):
        raise ValueError('Invalid factor')
    normal = prefix.with_suffix('.log').read_text(encoding='utf-8')
    debug_file = prefix.with_suffix('.debug.log')
    debug = debug_file.read_text(encoding='utf-8') if debug_file.exists() else ''
    text = normal + '\n' + debug
    wall, shape = fields(text, 'stage2_full_wall'), fields(text, 'real_shape')
    if abs(float(wall['total']) - float(wall['init']) - float(wall['main'])) > 0.003:
        raise ValueError('Stage2 wall ledger does not close')
    if wall.get('clean') != '1':
        raise ValueError('Incomplete arithmetic finalization')
    s4 = fields(text, 's4_multiply_stats')
    if any(int(s4.get(key, -1)) != 0 for key in ('gmp_selftest_bad', 'gmp_check_bad')):
        raise ValueError('S4 arithmetic check failure')
    labels = ['real_batched_split', 'real_batched_prepare', 'real_batched_wall', 'ntt_workspace_stats',
              'real_batched_folddevice', 'scaled_frontier_device', 'scaled_root_device',
              'descent_values', 'real_giant_seed_pair', 'real_giant_base', 'real_baby',
              'real_batched_gdevice', 'real_batched_device_gleaf', 'real_batched_outputwindow']
    stats = {key: fields(text, key, False) for key in labels}
    observed = {}
    for key in ('used_bytes', 'gpu_percent', 'sm_clock_mhz', 'temperature_c', 'power_mw'):
        values = [s[key] for s in samples if s[key] is not None]
        observed[key] = dict(min=min(values), max=max(values), mean=sum(values) / len(values)) if values else None
    return dict(**item, exponent=case['exponent'], variant=case['variant'], bits=case['bits'], words=case['words'],
                sigma=case['sigma'], B1=case['B1'], D=number(shape['D']), P=number(shape['P']),
                shape=shape, wall=wall, stats=stats, s4=s4, result=result,
                process_seconds=process_seconds, gpu_observations=observed,
                log=str(prefix.with_suffix('.log')), debug_log=str(debug_file),
                log_sha256=sha(prefix.with_suffix('.log')),
                result_sha256=sha(prefix.with_suffix('.result.jsonl')))


def run(args, out, monitor, plan):
    if monitor.identity != plan['device']:
        raise ValueError('GPU identity changed')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    env.update(NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    collector_sha = sha(__file__)
    shutil.copy2(__file__, out / f'collector_{collector_sha}.py')
    (out / 'runs').mkdir(exist_ok=True)
    cases = {c['id']: c for c in plan['cases']}
    def verify(case):
        if sha(plan['executable']) != plan['executable_sha256']:
            raise ValueError('Frozen binary changed')
        if sha(Path(plan['executable']).parent / 'gmp-10.dll') != plan['dll_sha256']:
            raise ValueError('Frozen DLL changed')
        if sha(out / 'bench.ini') != plan['ini_sha256'] or sha(case['save']) != case['save_sha256']:
            raise ValueError('Frozen configuration/save changed')
    for index, item in enumerate(plan['sequence']):
        if any(r['sequence_index'] == index for r in plan['runs']):
            continue
        case = cases[item['case']]
        verify(case)
        attempt = 1 + sum(f['sequence_index'] == index for f in plan['failures'])
        prefix = out / 'runs' / f'{index:03d}_{case["id"]}_b{item["B2"]}_{item["category"]}_r{item["repeat"]}_a{attempt}'
        if prefix.with_suffix('.driver.log').exists():
            raise ValueError('Unrecorded invocation exists; preserve it before resuming: ' + str(prefix))
        cmd = command(plan, out, case, item, prefix)
        started = stamp()
        samples = [monitor.sample()]
        print(f'[{index+1}/{len(plan["sequence"])}] {case["id"]} B2={item["B2"]} {item["category"]} r{item["repeat"]}', flush=True)
        begin = time.perf_counter()
        with prefix.with_suffix('.driver.log').open('wb') as log:
            proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    exit_code = proc.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    sample = monitor.sample()
                    sample['elapsed'] = time.perf_counter() - begin
                    samples.append(sample)
                    if time.perf_counter() - begin > args.timeout:
                        # Terminating the parent closes its Windows job and the curve child.
                        proc.kill()
                        proc.wait()
                        raise TimeoutError('Stage2 invocation timed out')
        process_seconds = time.perf_counter() - begin
        samples.append(monitor.sample())
        write(prefix.with_suffix('.telemetry.json'), samples)
        try:
            if exit_code:
                raise RuntimeError('Stage2 exit ' + str(exit_code))
            verify(case)
            row = collect(plan, case, item, prefix, process_seconds, samples)
        except Exception as exc:
            plan['failures'].append(dict(sequence_index=index, attempt=attempt, error=str(exc),
                                         started_utc=started, command=cmd, prefix=str(prefix), exit_code=exit_code))
            write(out / 'measurements.json', plan)
            raise
        row.update(sequence_index=index, started_utc=started, finished_utc=stamp(),
                   command=cmd, collector_sha256=collector_sha, telemetry=str(prefix.with_suffix('.telemetry.json')))
        plan['runs'].append(row)
        write(out / 'measurements.json', plan)
        print(f'  full={float(row["wall"]["total"]):.6f}s process={process_seconds:.3f}s '
              f'D={row["D"]} P={row["P"]} factors={len(row["result"]["factors"])}', flush=True)
    plan.update(complete=True, completed_utc=stamp())
    write(out / 'measurements.json', plan)
    print('COMPLETE:', len(plan['runs']), 'accepted runs;', len(plan['failures']), 'recorded failures', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=ROOT / 'tools/ecm_dataset/ecm_stage2_dataset.sqlite')
    p.add_argument('--exe', type=Path, default=ROOT / 'build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe')
    p.add_argument('--gmp-ecm', type=Path, default=Path('D:/code/GIMPS/gmp-ecm/ecm-7.0.5-znver3/ecm.exe'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exponents', type=int, nargs='+', default=EXPONENTS)
    p.add_argument('--b2', type=int, nargs='+', default=BOUNDS)
    p.add_argument('--b1', type=int, default=20)
    p.add_argument('--sigma', type=int, default=26)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--full-controls', action='store_true')
    p.add_argument('--device', type=int, choices=(1,), default=1)
    p.add_argument('--arena-mb', type=int, default=6300)
    p.add_argument('--fold-mb', type=int, default=640)
    p.add_argument('--batch-mb', type=int, default=256)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if (a.b1 < 2 or a.sigma < 6 or a.repeats < 2 or min(a.b2) <= a.b1
            or not a.exponents or len(set(a.exponents)) != len(a.exponents)
            or len(set(a.b2)) != len(a.b2)):
        p.error('Need unique exponents/B2, B1>=2, sigma>=6, repeats>=2, B2>B1')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(a.device)
    try:
        if 'RTX 4060 Laptop' not in monitor.identity['name']:
            raise ValueError('Expected RTX 4060 Laptop GPU1')
        plan = read(out / 'measurements.json') if a.resume else prepare(a, out, monitor)
        if not a.prepare_only:
            run(a, out, monitor, plan)
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
