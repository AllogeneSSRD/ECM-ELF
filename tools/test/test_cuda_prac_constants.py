#!/usr/bin/env python3
"""Gate opt-in PRAC constants with native windows, checkpoints and strict guards.

Production B1 checks use independent partial-product oracles, not full curves.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from bench_stage1_prac_windows import sample
from test_cuda_prac import run, rows
from test_cuda_prac_windows import multiply, verify

POLICIES = ('none', 'runtime', 'np0', 'm4423')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(exe, folder, cache, device, settings, message, expr='(2^4423-1)'):
    folder.mkdir()
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('ECM_PRAC_WINDOW'):
            env.pop(key)
    env.update(ECM_GPU_STAGE1_ALGO='prac', ECM_PRAC_REG_TARGET='168',
               ECM_PRAC_VARIANT='single-compact', ECM_STAGE1_TPI='16',
               ECM_PRAC_CONSTANTS='np0', ECM_GPU_STAGE1_SAMPLE_SECONDS='0.001',
               ECM_PRAC_TARGET_MS='50', ECM_PRAC_PLAN_CACHE=str(cache))
    env.pop('ECM_GPU_DUMP', None)
    env.update(settings)
    command = [str(exe), '-gpu', '-d', str(device), '--gpu-param', '0',
               '-sigma', '0:26', '-gpucurves', '8', '--ckpt', '0',
               '--exp-cache', str(cache), '-savea', 'completed.save', '1000', '0']
    proc = subprocess.run(command, input=(expr+'\n').encode('ascii'), cwd=folder,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=120)
    text = proc.stdout.decode('utf-8', errors='replace')
    (folder/'run.log').write_text(text, encoding='utf-8')
    assert proc.returncode == 1 and message in text, (folder, proc.returncode, text[-2000:])
    assert not rows(folder) and not list(folder.glob('.ecm_ckpt_*'))
    assert 'PRAC plan ' not in text, 'Invalid policy reached plan/curve execution'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    exe = a.exe.resolve(strict=True); cache = a.cache.resolve(strict=True)
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    binary = sha(exe); results = []; signatures = {}; byte_checks = 0
    cases = [(1000, exponent, 4, 0, position)
             for exponent in ('lcm', 'choose12')
             for position in ('prefix', 'middle', 'tail')]
    cases += [(b1, 'lcm', 32, chunk, position)
              for b1 in (10_000_000, 260_000_000)
              for chunk in (7, 32) for position in ('prefix', 'middle', 'tail')]
    for b1, exponent, count, chunk, position in cases:
        for policy in POLICIES:
            assert sha(exe) == binary
            folder = root/f'{policy}_b{b1}_{exponent}_{position}_chunk{chunk}'
            r = sample(exe, folder, 4423, b1, 8, a.device, 16, 168,
                       'single-compact', position, count, .001, 1, cache,
                       dump=True, exponent=exponent, chunk=chunk, constants=policy)
            r['Q_compared'] = verify(cache, folder, r)
            signature = sha(folder/'window_q.csv')
            key = (b1, exponent, count, position)
            if key in signatures:
                assert signature == signatures[key], 'Policy/slicing changed complete XZ CSV'
                byte_checks += 1
            else:
                signatures[key] = signature
            bn_count = 5 if policy == 'm4423' else 6
            assert r['boundary_logical_bytes_per_round'] == bn_count*8*576*r['launches_per_round']
            g = r['geometry']
            assert (g['tpi'], g['container_bits'], g['grid_blocks'],
                    g['resident_blocks_per_sm']) == (16, 4608, 1, 3)
            r.update(passed=True, csv_sha256=signature)
            results.append(r)

    # The policy is intentionally absent from the checkpoint ABI. Switch it
    # during a reset window, prove the checkpoint bytes unchanged, then resume.
    settings = ('(2^4423-1)', 10000, 8, 4611686018427511360, 'choose12')
    reference = root/'checkpoint_cpu'
    run(exe, reference, *settings, 'cpu', a.device)
    expected = rows(reference); assert len(expected) == 8
    resumed_q = 0
    for i, source in enumerate(POLICIES):
        destination = POLICIES[(i+1) % len(POLICIES)]
        folder = root/f'checkpoint_{source}_to_{destination}'
        text = run(exe, folder, *settings, 'prac', a.device, sample=.000001,
                   tpi=16, registers=168, variant='single-compact', constants=source)
        files = list(folder.glob('.ecm_ckpt_*'))
        assert 'sample limit reached' in text and len(files) == 1 and not rows(folder)
        before = files[0].read_bytes()
        r = sample(exe, folder, 4423, settings[1], 8, a.device, 16, 168,
                   'single-compact', 'tail', 4, .001, 1, cache, dump=True,
                   sigma=settings[3], exponent=settings[4], constants=destination)
        r['Q_compared'] = verify(cache, folder, r)
        assert files[0].read_bytes() == before
        text = run(exe, folder, *settings, 'prac', a.device, tpi=16,
                   registers=168, variant='single-compact', constants=destination)
        assert 'checkpoint resumed' in text and rows(folder) == expected
        assert not list(folder.glob('.ecm_ckpt_*'))
        r.update(passed=True, checkpoint_source=source, checkpoint_destination=destination)
        results.append(r); resumed_q += len(expected)

    failures = [({'ECM_PRAC_CONSTANTS': 'bad'}, 'ECM_PRAC_CONSTANTS must be'),
                ({'ECM_STAGE1_TPI': '32'}, 'selected 4608/TPI16 tier')]
    for registers in ('0', '128', '255'):
        failures.append(({'ECM_PRAC_REG_TARGET': registers},
                         'register policy 255 or 168' if registers == '0'
                         else 'single-compact variant and register policy 168'))
    for variant in ('baseline', 'compact', 'outline-add', 'single-add'):
        failures.append(({'ECM_PRAC_VARIANT': variant},
                         'single-compact variant and register policy 168'))
    failures.append(({'ECM_GPU_STAGE1_ALGO': 'resident'},
                     'PRAC window/variant requires ECM_GPU_STAGE1_ALGO=prac'))
    for i, (settings_bad, message) in enumerate(failures):
        rejected(exe, root/f'reject_{i}', cache, a.device, settings_bad, message)
    for bits in (2203, 8191):
        rejected(exe, root/f'reject_n{bits}', cache, a.device, {},
                 'selected 4608/TPI16 tier', f'(2^{bits}-1)')
    rejected(exe, root/'reject_wrong_np0', cache, a.device, {},
             'N mod 2^32 = 0xffffffff', '(2^4423-3)')
    rejected(exe, root/'reject_non_mersenne', cache, a.device,
             {'ECM_PRAC_CONSTANTS': 'm4423'}, 'N exactly 2^4423-1',
             '(2^4423-1-7*2^32)')

    # Exercise the broader np0 guard on an actual non-Mersenne modulus with
    # low word FFFFFFFF. Select a witness whose eight B1=2 outputs normalize.
    base = 2**4423-1
    for delta in range(1, 1000):
        n = base-(delta << 32)
        try:
            if all(math.gcd(multiply(s, 2, n)[1], n) == 1 for s in range(26, 34)):
                break
        except ValueError:
            continue
    else:
        raise AssertionError('No valid non-Mersenne witness')
    expr = f'(2^4423-1-{delta}*2^32)'
    settings_non_m = (expr, 2, 8, 26, 'lcm')
    reference = root/'non_mersenne_cpu'
    run(exe, reference, *settings_non_m, 'cpu', a.device)
    expected = rows(reference); assert len(expected) == 8
    non_m_q = 0
    for policy in ('none', 'runtime', 'np0'):
        folder = root/f'non_mersenne_{policy}'
        text = run(exe, folder, *settings_non_m, 'prac', a.device, tpi=16,
                   registers=168, variant='single-compact', constants=policy)
        assert rows(folder) == expected and 'CGBN<16,4608>' in text
        non_m_q += len(expected)
    assert sha(exe) == binary
    report = dict(schema=1, passed=True, exe=str(exe), binary_sha256=binary,
                  device=a.device, window_results=len(results),
                  window_Q=sum(r['Q_compared'] for r in results),
                  complete_csv_byte_checks=byte_checks, checkpoint_resume_Q=resumed_q,
                  guard_rejections=len(failures)+4, non_mersenne_Q=non_m_q,
                  non_mersenne_delta=delta, cpu_oracle_cache=multiply.cache_info()._asdict(),
                  results=results)
    (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
