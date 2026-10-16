"""Complete large-bound Edwards curves, historical Prime95 point and random restart.

GPU Stage1 is compared with the existing scalar CPU Edwards/NAF implementation.
Stage2 is compared with direct ordinary-integer products at every monic leaf.
These are correctness runs; process elapsed times are not performance samples.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time
from test_edwards_gpu_pipeline import ROOT, checksum, curve, values


def cpu_midstage(path):
    """Independently decode the documented Prime95 v6 MIDSTAGE field/checksum layout."""
    raw = path.read_bytes()
    assert struct.unpack_from('<II', raw) == (0x1725bcd9, 6)
    k, b, exponent, c = struct.unpack_from('<dIIi', raw, 8)
    expected_sum, = struct.unpack_from('<I', raw, 48)
    curve_id, average_b2, state, sigma, b1, b2, sigma_type, montg_stage1 = struct.unpack_from('<IQIQQQII', raw, 52)
    assert (state, sigma_type, montg_stage1) == (2, 0, 1)
    total = curve_id+state+(b1 & 0xffffffff)+(b1 >> 32)+(b2 & 0xffffffff)+(b2 >> 32)+sigma_type+montg_stage1
    offset = 100
    def giant():
        nonlocal offset, total
        size, = struct.unpack_from('<I', raw, offset); offset += 4
        assert size and size <= (len(raw)-offset)//4
        limbs = struct.unpack_from('<'+'I'*size, raw, offset)
        value = int.from_bytes(raw[offset:offset+4*size], 'little'); offset += 4*size
        total += 2*size+sum(limbs)
        return value
    x = giant()
    z_flag, = struct.unpack_from('<I', raw, offset); offset += 4; total += z_flag
    assert z_flag == 1
    z = giant()
    gg_flag, = struct.unpack_from('<I', raw, offset); offset += 4; total += gg_flag
    assert gg_flag == 0 and offset == len(raw) and total & 0xffffffff == expected_sum
    assert k == 1 and b == 2 and c == -1 and average_b2 == 0
    return dict(n=(1 << exponent)-1, sigma=sigma, b1=b1, x=x, z=z)


def save_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = dict(token.strip().split('=', 1) for token in line.split(';') if token.strip())
        n, x, sigma, b1 = (int(fields[k], 0) for k in ('N', 'X', 'SIGMA', 'B1'))
        assert fields['METHOD'] == 'ECM-EDWARDS-AM' and fields['VERSION'] == '1' and fields['SCALAR'] == '48LCM'
        status = fields['STATUS']; assert status in ('POINT', 'FACTOR')
        assert int(fields['CHECKSUM']) == checksum(n, x, sigma, b1, status)
        if status == 'FACTOR': assert 1 < x < n and n % x == 0
        else: assert 0 <= x < n
        rows.append(dict(n=n, x=x, sigma=sigma, b1=b1, status=status))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); out = a.output.resolve(); s1 = a.stage1.resolve(); s2 = a.stage2.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh evidence directory')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frozen_source = ROOT/'tools/test/ecm_edwards_save_test.cpp'
    fixture_source = frozen_source.read_text(encoding='utf-8-sig')
    historical = {name: int(re.search(r'mpz_set_str\('+name+r', "(\d+)"', fixture_source)[1]) for name in ('Qx', 'Qz')}
    report = dict(identity=dict(stage1=sha(s1), stage2=sha(s2), test=sha(__file__),
                  oracle=sha(ROOT/'tools/test/test_edwards_gpu_pipeline.py'), prime95_fixture_source=sha(frozen_source)),
                  device=a.device, checks=[], arithmetic=[], complete=False, formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    (out/'prime95_frozen_source.cpp').write_bytes(frozen_source.read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_S4_DESCENT_CHECK='1', NTT_GIANT_SEED_CHECK='1', NTT_GFINV_SEG_CHECK='1')
    def persist(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def verify():
        assert sha(s1) == report['identity']['stage1'] and sha(s2) == report['identity']['stage2']
        assert sha(__file__) == report['identity']['test']
    def call(name, command, input=None):
        verify()
        r = subprocess.run([str(x) for x in command], input=input, env=env, capture_output=True, cwd=out, timeout=1200)
        raw = r.stdout+r.stderr; log = out/(name+'.log'); log.write_bytes(raw)
        assert r.returncode == 0, (name, r.returncode, raw[-2000:])
        verify(); report['checks'].append(dict(name=name, exit=r.returncode, log_sha256=sha(log)))
        persist(); print('PASS', name, flush=True)
        return raw.decode('utf-8', 'replace')
    def gpu(name, bits, b1, sigma=20260922):
        save = out/(name+'.save')
        call(name, [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device, '-sigma', sigma,
                    '-gpucurves', 1, '--tmp-dir', out/(name+'_tmp'), '-save', save, '--ckpt', 0, b1, b1],
             str((1 << bits)-1).encode())
        rows = save_rows(save); assert len(rows) == 1
        return save, rows[0]
    def cpu(name, bits, b1, sigma):
        scratch = out/(name+'_tmp')
        call(name, [s1, '--curve', 'ed', '--target', 'cpu', '--impl', 'scalar', '--stage1-threads', 1,
                    '-sigma', sigma, '-gpucurves', 1, '--tmp-dir', scratch, '--ckpt', 0, b1, b1],
             str((1 << bits)-1).encode())
        files = list(scratch.glob('*.tmp')); assert len(files) == 1
        ref = cpu_midstage(files[0]); assert ref['sigma'] == sigma and ref['b1'] == b1
        return ref
    def compare(name, got, ref):
        n = got['n']; assert n == ref['n'] and got['sigma'] == ref['sigma'] and got['b1'] == ref['b1']
        g = math.gcd(ref['z'], n)
        if got['status'] == 'FACTOR': assert g == got['x']
        else: assert g == 1 and got['x']*ref['z'] % n == ref['x'] % n
        report['arithmetic'].append(dict(name=name, n_bits=n.bit_length(), b1=got['b1'], sigma=got['sigma'], cpu_match=True))
        persist(); print('PASS', name, 'CPU/NAF projection and GCD', flush=True)
    ini2 = out/'stage2.ini'; ini2.write_text(f'device={a.device}\nstage2_tune_runtime=0\n')
    def stage2(name, save, got):
        result = out/(name+'.jsonl'); log = out/(name+'_engine.log'); debug = out/(name+'_debug.log')
        call(name, [s2, '--ini', ini2, '--save', save, '--b2', 100000000, '--d', 30030, '--device', a.device,
                    '--short-calibration', 0, '--tune-runtime', 0, '--factor-only', '--results', result,
                    '--log', log, '--debug-log-file', debug])
        row = json.loads(result.read_text().splitlines()[-1]); n = got['n']
        assert not row['bad_factors'] and row['B1'] == got['b1'] and row['B2'] == 100000000 and row['curve'] == 'ed'
        assert all(1 < int(f) < n and n % int(f) == 0 for f in row['factors'])
        if got['status'] == 'POINT':
            a24, _ = curve(n, got['sigma'])
            expected, gcd = values(n, got['x'], a24, 100000000, 30030, with_gcd=True)
            found = re.findall(r'descent_values:.*?hash=(\d+)', debug.read_text(encoding='utf-8'))
            assert found and found[-1] == expected
            assert math.gcd(math.prod(int(f) for f in row['factors']), n) == gcd
            report['arithmetic'].append(dict(name=name, independent_leaf_hash=expected, leaves=2880,
                                            b2=100000000, d=30030, gcd_hex=f'{gcd:x}'))
            persist(); print('PASS', name, 'all 2880 monic leaves', flush=True)
    persist()
    for bits, b1 in ((347, 1000000), (677, 1000000), (347, 10000000)):
        name = f'm{bits}_b{b1}'; save, got = gpu('gpu_'+name, bits, b1)
        ref = cpu('cpu_'+name, bits, b1, got['sigma']); compare(name, got, ref)
        if bits == 347 and b1 == 1000000:
            assert got['status'] == 'POINT'
            assert got['x']*historical['Qz'] % got['n'] == historical['Qx'] % got['n']
            report['arithmetic'].append(dict(name='historical_prime95_m347', n_bits=347, b1=b1, sigma=got['sigma'], match=True))
            persist(); print('PASS historical Prime95 M347 saved point', flush=True)
        if bits == 347: stage2('stage2_'+name, save, got)
    # An unfixed task generates fresh random sigmas on startup; disk restore must
    # adopt the original sigmas before any further curve arithmetic.
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
    event = kernel.CreateEventW(ctypes.byref(sa), True, False, None); assert event
    restart = out/'random_restart'; restart.mkdir()
    queue = restart/'worktodo.txt'; finished = restart/'finished.txt'; ini = restart/'ecm.ini'
    task = 'ECM2=1,2,521,-1,1000000,0,2\n'; queue.write_text(task)
    ini.write_text(f'curve=ed\ntarget=gpu\ndevice={a.device}\nckpt_seconds=0.05\nworktodo={queue}\nfinished={finished}\ntmp_dir={restart}\nlog_file=\n')
    with (out/'random_stopped.log').open('wb') as log:
        proc = subprocess.Popen([str(s1), '-ini', str(ini), '--stage1-stop-handle', str(event)], env=env, cwd=out,
                                stdout=log, stderr=subprocess.STDOUT, close_fds=False, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic()+120
            while not list(restart.glob('*.gpu.ckpt')) and time.monotonic() < deadline:
                assert proc.poll() is None, 'random worker exited before checkpoint'
                time.sleep(0.01)
            assert list(restart.glob('*.gpu.ckpt')) and kernel.SetEvent(event)
            assert proc.wait(timeout=120) != 0
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            kernel.CloseHandle(event)
    assert queue.read_text() == task and not finished.exists()
    raw = next(restart.glob('*.gpu.ckpt')).read_bytes(); (out/'random_frozen_checkpoint.txt').write_bytes(raw)
    body, tail = raw.rsplit(b'CHECKSUM ', 1); h = 14695981039346656037
    for byte in body: h = ((h ^ byte)*1099511628211) & ((1 << 64)-1)
    assert int(tail) == h
    lines = body.decode().splitlines(); bound, count, done = map(int, lines[3].split())
    assert (bound, count) == (1000000, 2) and done > 0
    sigmas = [int(line.split()[0]) for line in lines[4:]]; assert len(sigmas) == 2
    text = call('random_resume', [s1, '-ini', ini])
    assert 'GPU Edwards resume:' in text and not queue.read_text().strip() and finished.exists()
    rows = save_rows(next(restart.glob('ed_*.save'))); assert [row['sigma'] for row in rows] == sigmas
    for i, row in enumerate(rows): compare('random_curve_'+str(i), row, cpu('cpu_random_'+str(i), 521, 1000000, row['sigma']))
    assert not list(restart.glob('*.ckpt'))
    report['arithmetic'].append(dict(name='random_sigma_restore', sigmas=sigmas, checkpoint_bit=done, identity_preserved=True))
    report['complete'] = True; persist(); print('All large-bound and random-restart gates passed', flush=True)


if __name__ == '__main__': main()
