"""Check full production CUDA param0 Q values and lcm/choose12 configuration paths."""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import struct
from datetime import datetime
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, default=repo / 'build_cuda_cmake/ecm_cuda.exe')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--output', type=Path)
parser.add_argument('--normalization-only', action='store_true', help='Check the original lcm path without choose12 integration')
args = parser.parse_args()
exe = args.exe.resolve()
output = (args.output or repo / 'build_cuda_cmake' / ('_gpu_exponent_' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
output.mkdir(parents=True, exist_ok=True)
if any(output.iterdir()):
    raise RuntimeError('Use a fresh output directory to preserve previous evidence')
spec = importlib.util.spec_from_file_location('mont_ref', repo / 'tools/stat/suyama_mont_ref.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
rows = []

def reference_x(param, sigma, bound, n, torsion):
    if param == 0:
        return oracle.stage1(sigma, bound, n, torsion=torsion)['x']
    if param == 3:
        a24 = sigma * pow(1 << 32, -1, n) % n
    else:
        # Affine Weierstrass group law, independent of the production Jacobian setup.
        def add(p, q):
            if p is None: return q
            if q is None: return p
            x, y = p
            u, v = q
            if x == u and (y + v) % n == 0: return None
            slope = ((3 * x * x) * pow(2 * y, -1, n) if p == q else
                     (v - y) * pow((u - x) % n, -1, n)) % n
            rx = (slope * slope - x - u) % n
            return rx, (slope * (x - rx) - y) % n
        point = None
        base = (n - 3, 3)
        for bit in bin(sigma)[2:]:
            point = add(point, point)
            if bit == '1': point = add(point, base)
        x, y = point
        x3 = (3 * x + y + 6) * pow(2 * (y - 3) % n, -1, n) % n
        a = -(3 * x3**4 + 6 * x3**2 - 1) * pow(4 * x3**3 % n, -1, n) % n
        a24 = (a + 2) * pow(4, -1, n) % n
    x, z = oracle.ladder(torsion * oracle.lcm_1_to(bound), 2, 1, a24, n)
    return x * pow(z, -1, n) % n

def check(case, bits, bound, sigma=26, curves=1, convention='lcm', ini=False, default=False, checkpoint=None, gpu_param=0):
    directory = output / case
    directory.mkdir()
    n = (1 << bits) - 1
    torsion = 12 if convention == 'choose12' else 1
    if checkpoint:
        # Real CUDA checkpoint ABI: 72-byte header followed by seven plain values.
        # Build both adjacent prefix points independently; their difference is P.
        version, saved_torsion = checkpoint
        assert bits == 4423 and curves == 1 and not ini
        scalar = saved_torsion * oracle.lcm_1_to(bound)
        partial = 7
        prefix = int(bin(scalar)[2:2 + partial], 2)
        _, a24, x0, z0 = oracle.suyama_curve(sigma, n)
        x, z = oracle.ladder(prefix, x0, z0, a24, n)
        bx, bz = oracle.ladder(prefix + 1, x0, z0, a24, n)
        values = (n, a24, x0 * pow(z0, -1, n) % n, x, z, bx, bz)
        if version == 4: values = (n, a24, values[2], 0, 0, 0, 0)
        payload = b''.join(v.to_bytes(4608 // 8, 'little') for v in values)
        header = struct.pack('<IIQQiIQIIIIQq', 0x45555047, version, partial, scalar.bit_length(),
                             1, curves, sigma, 4608, 16, 0, 0, len(payload), 0)
        assert len(header) == 72
        nh = format(n, 'x')
        (directory / f'.ecm_ckpt_{bits}_{nh[:8]}_{nh[-8:]}.dat').write_bytes(header + payload)
    if ini:
        saves = directory / 'saves'
        saves.mkdir()
        todo = directory / 'worktodo.txt'
        # ECMSTAGE2 tasks explicitly request a save; plain ECM tasks do not.
        save_path = saves / f'm{bits}_{bound}.save'
        todo.write_text(f'ECMSTAGE2=N/A,1,2,{bits},-1,"{save_path.as_posix()}",0,0,{curves}\n', encoding='ascii')
        config = directory / 'ecm.ini'
        exponent = 'lcm' if case == 'ini_worker_choose12' else convention
        text = (f'method = gpu\ngpu_param = 0\ndevice = {args.device}\nexponent = {exponent}\nsigma = {sigma}\n'
                f'worktodo = {todo.as_posix()}\nfinished = {(directory / "finished.txt").as_posix()}\n'
                f'tmp_dir = {saves.as_posix()}\nexp_cache = off\nckpt_seconds = 0\nlog_file =\n'
                'save_sync_dir_1 =\nsave_sync_dir_2 =\np95_worktodo_path =\n')
        if case == 'ini_worker_choose12': text += '[Worker #1]\nexponent = choose12\n'
        config.write_text(text, encoding='ascii')
        command = [str(exe), '-ini', str(config), '--worker', '1']
        data = None
    else:
        saves = directory
        command = [str(exe), '-v', '--method', 'gpu', '--gpu-param', str(gpu_param), '-d', str(args.device),
                   '-sigma', str(sigma), '-gpucurves', str(curves), '--exp-cache', 'off', '--ckpt', '0',
                   '-save', str(directory / 'result.save')]
        if not default: command += ['--exponent', convention]
        command += [str(bound)]
        data = str(n) + '\n'
    run = subprocess.run(command, input=data, text=True, cwd=directory,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (directory / 'driver.log').write_text(run.stdout, encoding='utf-8')
    actual = []
    for path in saves.glob('*.save'):
        for line in path.read_text().splitlines():
            if 'SIGMA=' in line:
                actual.append((int(re.search(r'SIGMA=(\d+);', line)[1]),
                               int(re.search(r'\bX=0x([0-9a-fA-F]+);', line)[1], 16)))
    expected = [(sigma + i, reference_x(gpu_param, sigma + i, bound, n, torsion)) for i in range(curves)]
    ok = (run.returncode == 0 and sorted(actual) == expected and
          bool(re.search(r'GPU: will use device ' + str(args.device) + r':', run.stdout)) and
          not re.search(r'ERROR|FATAL', run.stdout))
    if checkpoint == (5, torsion):
        ok = ok and 'Resuming from checkpoint:' in run.stdout
    else:
        ok = ok and 'Resuming from checkpoint:' not in run.stdout
    if checkpoint and checkpoint[0] == 4:
        ok = ok and 'Checkpoint version mismatch (expected 5, got 4)' in run.stdout
    if checkpoint and checkpoint[0] == 5 and checkpoint[1] != torsion:
        ok = ok and 'Checkpoint parameters mismatch' in run.stdout
    row = {'case': case, 'passed': ok, 'bits': bits, 'B1': bound, 'sigma': sigma,
           'curves': curves, 'gpu_param': gpu_param, 'torsion': torsion, 'checkpoint_fixture': checkpoint, 'returncode': run.returncode,
           'expected_q_sha256_hex': [hashlib.sha256(format(x, 'x').encode()).hexdigest() for _, x in expected],
           'actual_q_sha256_hex': [hashlib.sha256(format(x, 'x').encode()).hexdigest() for _, x in sorted(actual)],
           'command': command, 'log': str(directory / 'driver.log')}
    rows.append(row)
    print(f'{"PASS" if ok else "FAIL"} {case}: {len(actual)}/{curves} full Q values', flush=True)

# Short scalars expose noncanonical Montgomery products that longer ladders can mask.
for bits in (61, 1277, 4001, 4423, 5261, 8191, 9689):
    for bound in (2, 4, 20):
        check(f'lcm_m{bits}_b{bound}', bits, bound)
check('lcm_m4423_b4_curves8', 4423, 4, curves=8)
check('lcm_m4423_b4_sigma64', 4423, 4, sigma=(1 << 40) + 26)
check('cli_default', 4423, 1000, default=True)
if not args.normalization_only:
    for bits in (61, 1277, 4001, 4423, 5261, 8191, 9689):
        check(f'choose12_m{bits}_b2', bits, 2, convention='choose12')
    check('cli_choose12', 4423, 1000, convention='choose12')
    check('choose12_m4423_b2_curves8', 4423, 2, curves=8, convention='choose12')
    for case, convention in (('ini_lcm', 'lcm'), ('ini_choose12', 'choose12'), ('ini_worker_choose12', 'choose12')):
        check(case, 4423, 1000, convention=convention, ini=True)
    check('choose12_b1_below2', 4423, 1, convention='choose12')
    check('checkpoint_v5_choose12', 4423, 1000, convention='choose12', checkpoint=(5, 12))
    check('checkpoint_v4_rejected', 4423, 1000, convention='choose12', checkpoint=(4, 12))
    check('checkpoint_lcm_to_choose12', 4423, 1000, convention='choose12', checkpoint=(5, 1))
    for param in (2, 3):
        for bits in (61, 4423, 9689):
            for bound in (2, 4, 20):
                check(f'param{param}_m{bits}_b{bound}', bits, bound, gpu_param=param)
            check(f'param{param}_choose12_m{bits}', bits, 1000, convention='choose12', gpu_param=param)
        check(f'param{param}_m4423_b4_curves8', 4423, 4, curves=8, gpu_param=param)
summary = {'passed': sum(r['passed'] for r in rows), 'failed': sum(not r['passed'] for r in rows),
           'checked_q_values': sum(r['curves'] for r in rows), 'device': args.device,
           'normalization_only': args.normalization_only,
           'tested_parametrizations': sorted({r['gpu_param'] for r in rows}),
           'exe': str(exe), 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'runs': rows,
           'scope': ('Production CUDA param0 lcm CLI/default against the independent integer oracle.'
                     if args.normalization_only else
                    'Production CUDA param0 CLI/default/ini/worker override plus param2/3; every full affine Q '
                    'checked against integer ladders with independent affine curve setup. Synthetic CUDA v5 prefix resume, '
                    'v4 invalidation and changed exponent length are covered; no timed save/kill or speed test.')}
(output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"TOTAL {summary['passed']} passed / {summary['failed']} failed", flush=True)
raise SystemExit(1 if summary['failed'] else 0)
