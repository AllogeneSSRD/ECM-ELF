#!/usr/bin/env python3
"""Small-B1 full-Q gates for opt-in CUDA param0 PRAC/resident ladder.

Uses the CPU Montgomery/GMP path in the same executable as an independent field
backend. No production-size Stage1 completion is required for these gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def rows(folder):
    result = {}
    for path in folder.glob('*.save'):
        for line in path.read_text(encoding='utf-8').splitlines():
            fields = dict(re.findall(r'(?:^|;\s*)([A-Z0-9]+)=([^;]+)', line))
            if 'SIGMA' in fields and 'X' in fields:
                result[fields['SIGMA']] = {k: fields.get(k) for k in ('SIGMA', 'X', 'N', 'B1', 'PARAM', 'CHECKSUM')}
    return result


def run(exe, folder, expr, b1, curves, sigma, exponent, algo, device, sample=0, expect_records=True, tpi=0, registers=None):
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ECM_GPU_STAGE1_ALGO=algo, ECM_GPU_STAGE1_SAMPLE_SECONDS=str(sample))
    # Exercise native cold planning without changing the user's shared plan cache.
    env['ECM_PRAC_PLAN_CACHE'] = str(folder / 'plan_cache')
    env['ECM_STAGE1_TPI'] = str(tpi)
    if registers is not None: env['ECM_PRAC_REG_TARGET'] = str(registers)
    env.pop('ECM_GPU_DUMP', None)
    args = [str(exe), '-sigma', f'0:{sigma}', '-gpucurves', str(curves), '--ckpt', '0',
            '--exponent', exponent, '-v']
    if algo == 'cpu':
        env['ECM_GPU_STAGE1_ALGO'] = 'ladder'
        args += ['--method', 'mont', '--backend', 'gmp', '--tmp-dir', str(folder)]
    else:
        args += ['-gpu', '-d', str(device), '--gpu-param', '0', '-savea', 'completed.save']
    args += [str(b1), '0']
    proc = subprocess.run(args, input=(expr + '\n').encode('ascii'), env=env, cwd=folder,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    text = proc.stdout.decode('utf-8', errors='replace')
    (folder / ('sample.log' if sample else 'run.log')).write_text(text, encoding='utf-8')
    if not sample and expect_records and not rows(folder): raise RuntimeError(f'No Stage1 Q records: {folder}\n{text[-1500:]}')
    return text


def result_edges(exe, root, device):
    """Proper factors retain u64 sigma; a whole-modulus gcd cannot be a factor/save."""
    sigma = 4611686018427511360
    settings = ('(1009*(2^127-1))', 1000, 8, sigma, 'lcm')
    expected_dir = root / 'proper_factor' / 'cpu'
    text = run(exe, expected_dir, *settings, 'cpu', device)
    pattern = re.compile(r'factor\[(\d+)\]=(\d+) curve=\d+ sigma=(\d+)')
    expected_hits = pattern.findall(text)
    assert expected_hits and all(int(f) == 1009 and int(s) == sigma + int(i)
                                 for i, f, s in expected_hits)
    expected = rows(expected_dir)
    results = []
    for algo in ('ladder', 'resident', 'prac'):
        folder = root / 'proper_factor' / algo
        text = run(exe, folder, *settings, algo, device, registers=0)
        assert rows(folder) == expected, f'{algo}: proper-factor saves mismatch'
        assert pattern.findall(text) == expected_hits, f'{algo}: factor metadata lost 64-bit sigma'
        results.append(dict(case='proper_factor', algorithm=algo, Q=len(expected), passed=True))
        # This small cofactor is also annihilated in some curves: gcd(Z,N)=N.
        folder = root / 'degenerate' / algo
        text = run(exe, folder, '(1009*1000003)', 1000, 8, sigma, 'lcm', algo, device,
                   expect_records=False, registers=0)
        assert 'degenerate' in text and 'GPU stage1 returned: -1' in text, f'{algo}: missing degenerate error'
        assert not rows(folder), f'{algo}: published non-affine/whole-modulus results'
        assert not pattern.findall(text), f'{algo}: error batch still published factors'
        results.append(dict(case='degenerate', algorithm=algo, Q=0, passed=True))
        print(f'PASS results/{algo}: proper factors, u64 sigma, degenerate rejection', flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', type=Path, default=Path('build_cuda_cmake/prac/ecm_cuda.exe'))
    ap.add_argument('--output', type=Path, default=Path('docs/data/prac_cuda_q_gate'))
    ap.add_argument('--device', type=int, default=1)
    ap.add_argument('--tpi', type=int, choices=[0, 16, 32], default=0,
                    help='Force TPI for the 2203/4423-bit cases and checkpoint gates')
    ap.add_argument('--bits', type=int, nargs='+', choices=[2203, 4423, 8191], default=[2203, 4423, 8191])
    ap.add_argument('--registers', type=int, choices=[0, 168, 255],
                    default=int(os.environ.get('ECM_PRAC_REG_TARGET', '0')))
    args = ap.parse_args(); exe = args.exe.resolve(strict=True); root = args.output.resolve()
    if args.registers == 168 and (args.bits != [4423] or args.tpi == 32):
        ap.error('168-register variant requires --bits 4423 and TPI16/default')
    root.mkdir(parents=True, exist_ok=False)
    results = []
    cases = [(f'M{n}', f'(2^{n}-1)', 1000, 8, 26, t) for n in args.bits for t in ('lcm', 'choose12')]
    if 2203 in args.bits:
        cases += [('sigma64', '(2^2203-1)', 10000, 8, 9007199254740881, 'choose12')]
        cases += [(f'boundary{b}', '(2^2203-1)', b, 4, 26, t) for b in (2, 3, 5) for t in ('lcm', 'choose12')]
    if 4423 in args.bits:
        cases += [('sigma62', '(2^4423-1)', 1000, 8, 4611686018427511360, 'lcm')]
    if args.registers != 168:
        cases += [('composite', '((2^127-1)*(2^521-1))', 1000, 8, 26, 'lcm')]
    for label, expr, b1, curves, sigma, exponent in cases:
        reference = root / f'{label}_{exponent}' / 'cpu'
        run(exe, reference, expr, b1, curves, sigma, exponent, 'cpu', args.device)
        expected = rows(reference)
        assert len(expected) == curves, f'{label}: CPU did not return all Q records'
        for algo in ('ladder', 'resident', 'prac'):
            folder = root / f'{label}_{exponent}' / algo
            forced = args.tpi if expr in ('(2^2203-1)', '(2^4423-1)') else 0
            text = run(exe, folder, expr, b1, curves, sigma, exponent, algo, args.device,
                       tpi=forced, registers=args.registers if algo == 'prac' else 0)
            if forced and algo != 'ladder': assert f'CGBN<{forced},' in text, 'Wrong TPI selected'
            actual = rows(folder)
            assert actual == expected, f'{label}/{algo}: full-Q save mismatch; inspect {folder}'
            results.append(dict(case=label, exponent=exponent, algorithm=algo, Q=len(actual), passed=True))
            print(f'PASS {label}/{exponent}/{algo}: {len(actual)} full Q records', flush=True)
    # Independent directories for checkpoint resume and corruption recovery.
    expected_dir = root / 'checkpoint' / 'reference'
    checkpoint_bits = 2203 if 2203 in args.bits else args.bits[0]
    settings = (f'(2^{checkpoint_bits}-1)', 10000, 16, 9007199254740881, 'choose12')
    run(exe, expected_dir, *settings, 'ladder', args.device)
    expected = rows(expected_dir)
    for algo in ('resident', 'prac'):
        for corrupt in (False, True):
            folder = root / 'checkpoint' / f'{algo}_corrupt{int(corrupt)}'
            text = run(exe, folder, *settings, algo, args.device, sample=0.000001, tpi=args.tpi,
                       registers=args.registers if algo == 'prac' else 0)
            assert 'sample limit reached' in text and not rows(folder), 'Partial Stage1 was published'
            files = list(folder.glob('.ecm_ckpt_*'))
            assert len(files) == 1, f'Missing checkpoint: {folder}'
            if corrupt:
                payload = bytearray(files[0].read_bytes()); payload[-1] ^= 1; files[0].write_bytes(payload)
                if algo == 'prac':
                    cache = next((folder / 'plan_cache').glob('*.bin'))
                    payload = bytearray(cache.read_bytes()); payload[-1] ^= 1; cache.write_bytes(payload)
            text = run(exe, folder, *settings, algo, args.device, tpi=args.tpi,
                       registers=args.registers if algo == 'prac' else 0)
            assert ('mismatch/corruption' if corrupt else 'checkpoint resumed') in text
            if corrupt and algo == 'prac': assert 'PRAC plan built' in text, 'Corrupt plan cache was accepted'
            assert rows(folder) == expected, f'{algo}: resume/corruption full Q mismatch'
            assert not list(folder.glob('.ecm_ckpt_*')), 'Completed checkpoint was not removed'
            results.append(dict(case='checkpoint', algorithm=algo, corruption=corrupt, Q=len(expected), passed=True))
            print(f'PASS checkpoint/{algo}/corrupt={corrupt}: full Q matches', flush=True)
    results.extend(result_edges(exe, root / 'results', args.device))
    metadata = dict(exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    prac_registers=args.registers, bits=args.bits, device=args.device,
                    requested_tpi=args.tpi,
                    Q_comparisons=sum(r['Q'] for r in results), results=results)
    (root / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
