"""GPU Edwards worker, restart, save identity and complete Stage2 polynomial gates.

The independent Python oracle maps the base point to Montgomery and performs
the full 48*lcm scalar with an x-only ladder. Stage2 checks every monic leaf
against direct integer products, independently of the CUDA polynomial trees.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mont_reference', ROOT/'tools/stat/suyama_mont_ref.py')
mont = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mont)


def curve(n, sigma):
    def add(p, q):
        if p is None: return q
        if q is None: return p
        x, y = p; u, v = q
        if x == u:
            if (y+v) % n == 0: return None
            slope = (3*x*x-8)*pow(2*y, -1, n) % n
        else:
            slope = (v-y)*pow(u-x, -1, n) % n
        z = (slope*slope-x-u) % n
        return z, (slope*(x-z)-y) % n
    point = None
    for bit in bin(sigma)[2:]:
        point = add(point, point)
        if bit == '1': point = add(point, (12, 40))
    s, t = point
    alpha = (s-9)*pow(t+s+16, -1, n) % n
    beta = 2*alpha*(4*alpha+1)*pow(8*alpha*alpha-1, -1, n) % n
    h = (2*beta-1) % n
    d = (2*h*h-1)*pow(pow(h, 4, n), -1, n) % n
    x = h*(4*beta-3)*pow(6*beta-5, -1, n) % n
    y = h*(t*t+50*t-2*s**3+27*s*s-104)*pow((t+3*s-2)*(t+s+16), -1, n) % n
    assert (x*x+y*y-1-d*x*x*y*y) % n == 0
    return pow(1-d, -1, n), (1+y)*pow(1-y, -1, n) % n


def stage1(n, sigma, b1):
    a24, p = curve(n, sigma)
    x, z = mont.ladder(48*mont.lcm_1_to(b1), p, 1, a24, n)
    return a24, x, z


def checksum(n, x, sigma, b1, status):
    h = 14695981039346656037
    text = f'ECM-EDWARDS-AM|1|48LCM|{n:x}|{x:x}|{sigma}|{b1}|{status}'
    for byte in text.encode(): h = ((h ^ byte)*1099511628211) & ((1 << 64)-1)
    return h


def records(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = dict(token.strip().split('=', 1) for token in line.split(';') if token.strip())
        n, x, sigma, b1 = (int(fields[k], 0) for k in ('N', 'X', 'SIGMA', 'B1'))
        assert fields['METHOD'] == 'ECM-EDWARDS-AM' and fields['VERSION'] == '1' and fields['SCALAR'] == '48LCM'
        assert int(fields['CHECKSUM']) == checksum(n, x, sigma, b1, fields['STATUS'])
        if fields['STATUS'] == 'FACTOR':
            assert 1 < x < n and n % x == 0
        else:
            _, rx, rz = stage1(n, sigma, b1)
            assert math.gcd(rz, n) == 1 and x*rz % n == rx
        rows.append((n, x, sigma, b1, fields['STATUS']))
    return rows


def checkpoint(path):
    """Decode both GPU formats without using production C++ checkpoint code."""
    raw = path.read_bytes(); body, tail = raw.rsplit(b'CHECKSUM ', 1)
    h = 14695981039346656037
    for byte in body: h = ((h ^ byte)*1099511628211) & ((1 << 64)-1)
    assert int(tail) == h
    lines = body.decode().splitlines(); tag = lines.pop(0)
    if tag == 'ECM_GPU_EDWARDS_NAF_V2':
        window, doubling = map(int, lines.pop(0).split())
        assert 2 <= window <= 12 and doubling in (4, 5)
    else:
        assert tag == 'ECM_GPU_EDWARDS_BINARY_V1'; window = doubling = 0
    n, scalar = int(lines[0], 16), int(lines[1], 16)
    b1, count, done = map(int, lines[2].split())
    assert n > 3 and n % 2 and scalar == 48*mont.lcm_1_to(b1) and len(lines) == count+3
    total = scalar.bit_length()
    if window:
        bits = bin(scalar)[2:][::-1]; i = carry = total = 0
        while i < len(bits) or carry:
            low = (int(bits[i]) if i < len(bits) else 0)+carry
            if low % 2 == 0: carry = low//2; i += 1; continue
            value = int(bits[i:i+window][::-1] or '0', 2)+carry
            carry = int(value >= 1 << (window-1)); total = i+1; i += window
    assert 0 < done <= total
    rows = []
    for line in lines[3:]:
        tokens = line.split(); assert len(tokens) == 12
        sigma, hit = map(int, tokens[:2]); factor = int(tokens[2], 16)
        state = [int(x, 16) for x in tokens[3:]]
        assert 0 < sigma < 1 << 64 and hit in (0, 1)
        assert (1 < factor < n and n % factor == 0) if hit else factor == 0
        assert all(0 <= x < n for x in state)
        d = state[0]
        for x, y, z, t in (state[1:5], state[5:9]):
            assert any((x, y, z, t)) and (x*y-z*t) % n == 0
            assert (x*x+y*y-z*z-d*t*t) % n == 0
        rows.append(dict(sigma=sigma, hit=hit, factor=factor, state=state))
    return dict(tag=tag, window=window, doubling=doubling, n=n, scalar=scalar,
                b1=b1, count=count, done=done, total=total, rows=rows)


def values(n, x, a24, b2, d=210, with_gcd=False):
    def affine(k):
        u, z = mont.ladder(k, x, 1, a24, n)
        return u*pow(z, -1, n) % n
    baby = [affine(j) for j in range(1, d//2+1) if math.gcd(j, d) == 1]
    result = [1]*len(baby)
    for i in range(1, b2//d+3):
        u = affine(i*d)
        result = [v*(b-u) % n for b, v in zip(baby, result)]
    h = 1469598103934665603
    for v in result:
        for j in range((n.bit_length()+63)//64):
            h = ((h ^ ((v >> (64*j)) & ((1 << 64)-1)))*1099511628211) & ((1 << 64)-1)
    return (str(h), math.gcd(math.prod(result), n)) if with_gcd else str(h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    args = p.parse_args()
    s1, s2, out = args.stage1.resolve(), args.stage2.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh evidence directory')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_S4_DESCENT_CHECK='1', NTT_GIANT_SEED_CHECK='1',
               NTT_GFINV_SEG_CHECK='1', NTT_SCALED_ROOT_CHECK='1', NTT_SCALED_CHECK='1',
               NTT_TARGET_LEAF_HASH='1')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(identity={'stage1': sha(s1), 'stage2': sha(s2), 'test': sha(__file__)},
                  device=args.device, checks=[], complete=False, formal_performance_samples=0)
    def persist(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def call(name, command, input=None, good=True, extra_env=None):
        proc = subprocess.run([str(v) for v in command], input=input, capture_output=True,
                              env=env | (extra_env or {}), timeout=600, cwd=out)
        raw = proc.stdout+proc.stderr; (out/(name+'.log')).write_bytes(raw)
        assert (proc.returncode == 0) == good, (name, proc.returncode, raw[-2000:])
        report['checks'].append(dict(name=name, exit=proc.returncode, log_sha256=sha(out/(name+'.log'))))
        persist(); print('PASS', name, flush=True)
        return raw.decode('utf-8', 'replace')
    def worker(name, n, b1=1000, count=4, sigma=20260922, tmp=None, extra=None):
        save = out/(name+'.save'); scratch = tmp or out/(name+'_scratch')
        text = call(name, [s1, '--curve', 'ed', '--target', 'gpu', '-d', args.device,
                    '-sigma', sigma, '-gpucurves', count, '--tmp-dir', scratch,
                    '-save', save, *(extra or []), b1, b1], str(n).encode())
        rows = records(save); assert len(rows) == count
        return save, rows, text
    ini = out/'stage2.ini'; ini.write_text(f'device={args.device}\nstage2_tune_runtime=0\n')
    def stage2(name, save, b2=13230, extra=None):
        result = out/(name+'.jsonl'); log = out/(name+'_engine.log')
        debug = out/(name+'_debug.log')
        call(name, [s2, '--ini', ini, '--save', save, '--curves', 1, '--b2', b2, '--d', 210,
                    '--short-calibration', 0, '--tune-runtime', 0,
                    '--device', args.device, '--results', result, '--log', log,
                    '--log-level', 'debug', '--debug-log-file', debug, '--factor-only', *(extra or [])])
        row = json.loads(result.read_text().splitlines()[-1]); assert row['curve'] == 'ed' and row['param'] is None and not row['bad_factors']
        return row, log.read_text(encoding='utf-8')+(debug.read_text(encoding='utf-8') if debug.exists() else '')
    persist()
    fixtures = []
    for bits in (127, 347, 677, 991, 2203, 4003):
        save, rows, _ = worker('worker_m'+str(bits), (1 << bits)-1)
        fixtures.append(save)
        row, log = stage2('stage2_m'+str(bits), save)
        n, x, sigma, b1, status = rows[0]; assert status == 'POINT'
        a24, _ = curve(n, sigma)
        expected, expected_gcd = values(n, x, a24, 13230, with_gcd=True)
        hashes = re.findall(r'descent_values:.*?hash=(\d+)', log)
        assert hashes and hashes[-1] == expected, ('complete monic leaf mismatch', bits, hashes, expected)
        assert math.gcd(math.prod(int(f) for f in row['factors']), n) == expected_gcd
        assert all(1 < int(f) < n and n % int(f) == 0 for f in row['factors'])
        assert 'gmp_check_bad=0' in log
        report['checks'][-1]['independent_leaf_hash'] = expected; persist()
    # The same Edwards point/constants must survive a legitimate Mersenne lift.
    save, rows, _ = worker('carrier_input', (1 << 128)+1, 20, 1, 17)
    n, x, sigma, _, _ = rows[0]; a24, _ = curve(n, sigma)
    expected = values(n, x, a24, 13230)
    for name, extra in [('generic', []), ('carrier', ['--carrier-exponent', 256])]:
        row, log = stage2('edwards_'+name, save, extra=extra)
        hashes = re.findall(r'target_descent_values:.*?hash=(\d+)', log)
        assert hashes and hashes[-1] == expected and not row['factors']
        report['checks'][-1]['independent_target_leaf_hash'] = expected; persist()
    # Preserve the existing Suyama route and its omitted-argument API behavior.
    n = (1 << 127)-1; point = mont.stage1(26, 20, n)
    legacy = out/'mont_regression.save'
    legacy.write_text(f'METHOD=ECM; SIGMA=26; B1=20; N=0x{n:x}; X=0x{point["x"]:x};\n')
    result = out/'mont_regression.jsonl'; log = out/'mont_regression_engine.log'; debug = out/'mont_regression_debug.log'
    call('mont_regression', [s2, '--ini', ini, '--save', legacy, '--b2', 13230, '--d', 210,
        '--short-calibration', 0, '--tune-runtime', 0,
        '--device', args.device, '--results', result, '--log', log, '--log-level', 'debug', '--debug-log-file', debug, '--factor-only'])
    row = json.loads(result.read_text().splitlines()[-1])
    expected = values(n, point['x'], point['a24'], 13230)
    hashes = re.findall(r'descent_values:.*?hash=(\d+)', debug.read_text(encoding='utf-8'))
    assert row['curve'] == 'mont' and row['param'] == 0 and not row['factors'] and not row['bad_factors']
    assert hashes and hashes[-1] == expected
    # Nontrivial factor discovered by checked curve construction or Stage1 GCD.
    factor_save, factor_rows, _ = worker('stage1_factor', 1009*((1 << 127)-1), 1000, 4, 17)
    assert any(status == 'FACTOR' for *_, status in factor_rows)
    factor_index = next(i for i, row in enumerate(factor_rows) if row[-1] == 'FACTOR')
    single = out/'single_factor.save'; single.write_text(factor_save.read_text().splitlines()[factor_index]+'\n')
    row, _ = stage2('saved_factor', single); assert row['factors'] and row['status'] == 'factor_in_saved_X'
    # A point with a prime-order component beyond B1 must be found by Stage2.
    chosen = None
    for prime in (10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079):
        n = prime*((1 << 127)-1)
        try:
            _, x, z = stage1(n, 17, 20)
            if math.gcd(z, n) == 1: chosen = (n, prime); break
        except ValueError: pass
    assert chosen
    save, rows, _ = worker('stage2_factor_input', chosen[0], 20, 1, 17)
    row, _ = stage2('stage2_factor', save, 40000)
    assert any(int(f) % chosen[1] == 0 for f in row['factors']), ('Stage2 factor absent', chosen, row)
    for f in row['factors']: assert 1 < int(f) < chosen[0] and chosen[0] % int(f) == 0
    # Model/scalar/point/checksum corruption must fail before arithmetic/publication.
    original = fixtures[0].read_text().splitlines()[0]+'\n'
    for name, old, new in [('model', 'ECM-EDWARDS-AM', 'ECM'), ('scalar', '48LCM', 'LCM'),
                           ('checksum', 'CHECKSUM=', 'CHECKSUM=9'), ('sigma', 'SIGMA=20260922', 'SIGMA=20260923')]:
        bad = out/('bad_'+name+'.save'); bad.write_text(original.replace(old, new, 1))
        call('reject_'+name, [s2, '--ini', ini, '--save', bad, '--b2', 13230, '--dry-run'], good=False)
    # Real process stop + disk restoration, including a corrupt-primary fallback.
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
    event = kernel.CreateEventW(ctypes.byref(sa), True, False, None); assert event
    restart = out/'restart'; restart.mkdir()
    queue = restart/'worktodo.txt'; finished = restart/'finished.txt'
    task = 'ECM2=1,2,347,-1,100000,0,4,20260922\n'
    queue.write_text(task); ini1 = restart/'ecm.ini'
    ini1.write_text(f'curve=ed\ntarget=gpu\ndevice={args.device}\nckpt_seconds=0.001\nckpt_backups=2\nworktodo={queue}\nfinished={finished}\ntmp_dir={restart}\nlog_file=\n')
    with (out/'stopped_worker.log').open('wb') as log:
        proc = subprocess.Popen([str(s1), '-ini', str(ini1), '--stage1-stop-handle', str(event)],
                                stdout=log, stderr=subprocess.STDOUT, close_fds=False,
                                creationflags=subprocess.CREATE_NO_WINDOW, env=env, cwd=out)
        try:
            deadline = time.monotonic()+90
            while not list(restart.glob('*.gpu.ckpt')) and time.monotonic() < deadline:
                assert proc.poll() is None, 'worker exited before checkpoint'
                time.sleep(0.05)
            assert kernel.SetEvent(event)
            assert proc.wait(timeout=120) != 0
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
            kernel.CloseHandle(event)
    assert queue.read_text() == task and (not finished.exists() or not finished.read_text().strip())
    checkpoints = list(restart.glob('*.gpu.ckpt')); assert len(checkpoints) == 1
    primary = checkpoints[0]; backup = primary.with_name(primary.stem+'.bak'+primary.suffix)
    assert backup.exists(); (out/'frozen_checkpoint.txt').write_bytes(primary.read_bytes())
    primary.write_bytes(b'corrupt primary\n')
    resumed_text = call('resume_queue', [s1, '-ini', ini1])
    assert 'GPU Edwards resume:' in resumed_text and '.bak.ckpt' in resumed_text
    assert not queue.read_text().strip() and finished.exists()
    save = next(restart.glob('ed_*.save')); resumed_rows = records(save); assert len(resumed_rows) == 4
    reference, ref_rows, _ = worker('uninterrupted', (1 << 347)-1, 100000)
    assert save.read_bytes() == reference.read_bytes() and resumed_rows == ref_rows
    assert not list(restart.glob('*.ckpt'))
    stage2('resumed_stage2', save, 200000)
    # Stage2 queue receipts must prevent replay on the second run.
    q2 = out/'stage2_worktodo.txt'; f2 = out/'stage2_finished.txt'; result = out/'queue_results.jsonl'
    q2.write_text(f'ECMSTAGE2=1,2,347,-1,"{fixtures[1]}",13230,0,1\n')
    ini2 = out/'queue_stage2.ini'
    ini2.write_text(f'device={args.device}\nstage2_worktodo={q2}\nstage2_finished={f2}\nstage2_results_file={result}\nstage2_log_file={out/"queue_stage2.log"}\nstage2_b2=13230\nstage2_tune_runtime=0\n')
    call('stage2_queue', [s2, '--ini', ini2, '--d', 210, '--factor-only'])
    before = result.read_bytes(); call('stage2_queue_empty', [s2, '--ini', ini2, '--d', 210, '--factor-only'])
    assert result.read_bytes() == before and len(before.splitlines()) == 1 and f2.exists()
    report['complete'] = True; persist(); print('All GPU Edwards pipeline gates passed', flush=True)


if __name__ == '__main__': main()
