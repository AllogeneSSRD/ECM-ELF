"""Independent Edwards endpoint, libecm backend and result JSON regression gates."""
import argparse
import ast
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import secrets
import time
import zlib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('oracle', ROOT/'tools/test/test_edwards_gpu_pipeline.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def verify_json(r):
    assert r['worktype'] == 'ECM' and r['program']['name'] == 'ECM-ELF'
    assert 'security-code' not in r and r['fft-length'] == 0
    target = str(r['exponent']) if 'exponent' in r else 'N='+str(r['N'])
    factors = ','.join(r.get('factors', []))
    known = ','.join(r.get('known-factors', []))
    canonical = ';'.join([target, 'ECM', factors, known, str(r['b1']), str(r.get('b2', '')),
        'E'+str(r['Edwards']['sigma']), '0', 'invariant:0', r['program']['name'],
        r['program']['version'], '', '', r['os']['os'], r['os']['architecture'], r['timestamp']])
    assert r['checksum'] == {'version': 1, 'checksum': f'{zlib.crc32(canonical.encode()):08X}'}
    n = ((1 << r['exponent'])-1)//math.prod(map(int, r.get('known-factors', []))) if 'exponent' in r else int(r['N'])
    if r['status'] == 'F':
        assert r['stage'] in (1, 2)
        assert all(1 < int(f) < n and n % int(f) == 0 for f in r['factors'])
    else:
        assert r['status'] == 'NF' and 'stage' not in r and 'factors' not in r
    return n


def offline_autoprimenet(rows):
    """Execute only the upstream parser prefix; submission code is excluded."""
    source = ROOT/'.refactor/AutoPrimeNet/autoprimenet.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_result')
    cut = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'no_report' for t in n.targets))
    fn.body = fn.body[:cut]+[ast.Return(value=ast.Tuple(elts=[ast.Name('ar', ast.Load()), ast.Name('result_type', ast.Load())], ctx=ast.Load()))]
    constants = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'PRIMENET_AR'}
    scope = dict(json=json, JSONDecodeError=json.JSONDecodeError, SCRIPT={},
                 SEC=SimpleNamespace(Internals='Internals'), Assignment=lambda: SimpleNamespace(uid=0, n=0),
                 PRIMENET_AR=SimpleNamespace(**{k: k for k in constants}), exponent_to_str=lambda a: f'M{a.n}')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), scope)
    for row in rows:
        parsed, kind = scope['parse_result'](SimpleNamespace(set=lambda *a: None),
            SimpleNamespace(user_id='', computer_id=''), None, '', 0, 'offline.jsonl', json.dumps(row))
        assert parsed['b1'] == row['b1'] and kind == ('ECM_FACTOR' if row['status'] == 'F' else 'ECM_NOFACTOR')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    s1, s2, out = args.stage1.resolve(), args.stage2.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('use a fresh evidence directory')
    sha = lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
    report = dict(identity={'stage1': sha(s1), 'stage2': sha(s2), 'test': sha(__file__)}, checks=[], complete=False)
    def persist():
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def call(name, cmd, data=None, good=True):
        start = time.perf_counter()
        r = subprocess.run(list(map(str, cmd)), input=data, cwd=out, capture_output=True, timeout=600)
        (out/(name+'.log')).write_bytes(r.stdout+r.stderr)
        assert (r.returncode == 0) == good, (name, r.returncode, (r.stdout+r.stderr)[-2000:])
        report['checks'].append(dict(name=name, exit=r.returncode, wall_seconds=time.perf_counter()-start))
        persist()
        print('PASS', name, flush=True)
        return r.stdout.decode(errors='replace')
    def stage1(name, n, sigma=17, b1=1000, count=4, threads=1, gpu=False, append=None, impl='scalar'):
        save = append or out/(name+'.save')
        call(name, [s1, '--curve', 'ed', '--target', 'gpu' if gpu else 'cpu', '--impl', 'auto' if gpu else impl,
            '--stage1-threads', threads, '-d', 1, '-sigma', sigma, '-gpucurves', count,
            '--tmp-dir', out/(name+'_scratch'), '-savea' if append else '-save', save, b1, b1], str(n).encode())
        rows = oracle.records(save)
        assert len(rows) >= count
        return save, rows
    def stage2(name, save, backend, b2=40000, extra=(), good=True):
        results = out/(name+'.jsonl')
        call(name, [s2, '--save', save, '--b2', b2, '--backend', backend, '--results', results, *extra], good=good)
        rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
        for r in rows:
            verify_json(r)
        offline_autoprimenet(rows)
        return rows
    fixtures = []
    for bits in (127, 347, 2203):
        n = (1 << bits)-1
        save, rows = stage1('scalar_m'+str(bits), n, threads=1)
        threaded, other = stage1('threads_m'+str(bits), n, threads=3)
        assert rows == other
        fixtures.append(save)
        if args.gpu:
            gpu, gpu_rows = stage1('gpu_m'+str(bits), n, gpu=True)
            assert rows == gpu_rows
        for backend in ('ntt', 'plain'):
            result = stage2(backend+'_m'+str(bits), save, backend, extra=['--exponent', bits])
            assert len(result) == len(rows) and all(r['status'] == 'NF' for r in result)
    save, first = stage1('append_first', (1 << 127)-1, count=2)
    _, combined = stage1('append_second', (1 << 127)-1, sigma=19, count=2, append=save)
    assert len(combined) == 4 and combined[:2] == first
    # Explicit SIMD must either pass or reject unavailable hardware clearly.
    simd_save = out/'simd_probe.save'
    probe = subprocess.run(list(map(str, [s1, '--curve', 'ed', '--target', 'cpu', '--impl', 'simd', '-sigma', 17,
        '-gpucurves', 4, '--tmp-dir', out/'simd_scratch', '-save', simd_save, 1000, 1000])),
        input=str((1 << 127)-1).encode(), cwd=out, capture_output=True)
    (out/'simd_probe.log').write_bytes(probe.stdout+probe.stderr)
    report['simd_available'] = probe.returncode == 0
    if report['simd_available']:
        assert oracle.records(simd_save) == oracle.records(fixtures[0])
        simd_factor, simd_rows = stage1('simd_factor', 1009*((1 << 127)-1), impl='simd')
        assert len(simd_rows) == 4 and all(r[-1] == 'FACTOR' for r in simd_rows)
    else:
        assert b'AVX512' in probe.stdout+probe.stderr or b'IFMA' in probe.stdout+probe.stderr
    persist()
    save, rows = stage1('stage1_factor', 1009*((1 << 127)-1), b1=1000)
    assert any(row[-1] == 'FACTOR' for row in rows)
    for backend in ('ntt', 'plain'):
        found = stage2('saved_factor_'+backend, save, backend)
        assert len(found) == len(rows) and any(r.get('stage') == 1 for r in found)
    # Factor is beyond Stage1; compare both backends to direct prime ladders.
    n = 10007*((1 << 127)-1)
    save, rows = stage1('stage2_factor', n, b1=20, count=1)
    a24, x, z = oracle.stage1(n, 17, 20)
    assert math.gcd(z, n) == 1
    qx, qz = oracle.mont.ladder(10007, x, z, a24, n)
    # Full direct prime-ladder oracle proves a factor exists in (20,40000].
    primes = [v for v in range(21, 40001) if all(v % d for d in range(2, math.isqrt(v)+1))]
    product = 1
    for prime in primes:
        _, rz = oracle.mont.ladder(prime, x, z, a24, n)
        product = product*rz % n
    expected = math.gcd(product, n)
    assert 1 < expected < n
    for backend in ('ntt', 'plain'):
        result = stage2('stage2_factor_'+backend, save, backend)
        assert len(result) == 1 and result[0].get('stage') == 2 and int(result[0]['factors'][0]) % expected == 0
    cofactor = ((1 << 130)-1)//3
    save, _ = stage1('cofactor', cofactor, b1=20, count=1)
    result = stage2('cofactor_metadata', save, 'plain', extra=['--exponent', 130, '--known-factor', 3])
    assert verify_json(result[0]) == cofactor
    original = fixtures[0].read_text().splitlines()[0]+'\n'
    for name, bad in [('checksum', original.replace('CHECKSUM=', 'CHECKSUM=9')),
                      ('scalar', original.replace('48LCM', 'LCM')),
                      ('duplicate', original+'SIGMA=18;'),
                      ('identity', original.replace('ECM-EDWARDS-AM', 'ECM'))]:
        save = out/('bad_'+name+'.save');save.write_text(bad)
        assert not stage2('reject_'+name, save, 'ntt', good=False)
    assert not stage2('reject_metadata', fixtures[0], 'ntt', extra=['--exponent', 131], good=False)
    for index in (0, 2):
        result = stage2('automatic_'+str(index), fixtures[index], 'auto', extra=['--curves', 1])
        assert result[0]['stage2-engine']['backend'] == ('ntt' if index == 0 else 'plain')
    assert not stage2('reject_b2_overflow', fixtures[0], 'ntt', b2=10**12+1, good=False)
    assert not stage2('reject_save_alias', fixtures[0], 'ntt', extra=['--results', fixtures[0]], good=False)
    # maxmem is a libecm planner budget, not a process RSS cap. Small budgets
    # may select more blocks rather than fail; validate a bounded small case.
    low_memory = stage2('small_memory', fixtures[2], 'ntt', extra=['--memory-mib', 1, '--curves', 1])
    assert len(low_memory) == 1 and low_memory[0]['status'] == 'NF'
    assert not stage2('reject_zero_memory', fixtures[2], 'ntt', extra=['--memory-mib', 0], good=False)
    private_user=secrets.token_hex(12)+'"\\\n'
    selected = stage2('selection', fixtures[0], 'plain', extra=['--skip', 1, '--curves', 1,
                     '--user', private_user, '--computer', secrets.token_hex(12), '--aid', secrets.token_hex(16)])
    assert len(selected) == 1 and selected[0]['Edwards']['sigma'] == 18 and selected[0]['user'] == private_user
    default = out/'default';default.mkdir()
    call('default_save', [s1, '--curve', 'ed', '--target', 'cpu', '--impl', 'scalar', '-sigma', 17,
        '-gpucurves', 1, '--tmp-dir', default, 20, 20], str((1 << 127)-1).encode())
    assert len(oracle.records(next(default.glob('ed_*.save')))) == 1
    # A failed text save publication must preserve the queue and checkpoints.
    queue_dir = out/'queue_failure';queue_dir.mkdir()
    blocker = queue_dir/next(default.glob('ed_*.save')).name;blocker.mkdir()
    queue = queue_dir/'worktodo.txt';task = 'ECM2=1,2,127,-1,20,0,1,17\n';queue.write_text(task)
    ini = queue_dir/'ecm.ini';finished = queue_dir/'finished.txt'
    ini.write_text(f'curve=ed\ntarget=cpu\nimpl=scalar\nworktodo={queue}\nfinished={finished}\ntmp_dir={queue_dir}\nlog_file=\n')
    call('save_failure_queue', [s1, '-ini', ini], good=False)
    assert queue.read_text() == task and not finished.exists()
    # Pre-signaled stop and a real stop during libecm must not produce NF.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE];kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
    event = kernel.CreateEventW(ctypes.byref(sa), True, False, None)
    try:
        result = out/'stopped_stage2.jsonl'
        with (out/'stopped_stage2.log').open('wb') as log:
            proc = subprocess.Popen(list(map(str, [s2, '--save', fixtures[0], '--b2', 10**12,
                '--results', result, '--stop-handle', event])), close_fds=False, stdout=log, stderr=subprocess.STDOUT, cwd=out)
            time.sleep(0.7);kernel.SetEvent(event);assert proc.wait(timeout=120) == 130
        assert not result.exists()
        report['checks'].append(dict(name='stop_inside_stage2', exit=130))
        persist()
    finally:
        kernel.CloseHandle(event)
    event = kernel.CreateEventW(ctypes.byref(sa), True, False, None)
    try:
        restart = out/'cpu_restart';restart.mkdir()
        queue = restart/'worktodo.txt';task = 'ECM2=1,2,127,-1,1000000,0,1,17\n';queue.write_text(task)
        ini = restart/'ecm.ini';finished = restart/'finished.txt'
        ini.write_text(f'curve=ed\ntarget=cpu\nimpl=scalar\nckpt_seconds=0.001\nworktodo={queue}\nfinished={finished}\ntmp_dir={restart}\nlog_file=\n')
        with (out/'stopped_stage1.log').open('wb') as log:
            proc = subprocess.Popen([str(s1), '-ini', str(ini), '--stage1-stop-handle', str(event)],
                close_fds=False, stdout=log, stderr=subprocess.STDOUT, cwd=out)
            deadline = time.monotonic()+60
            while not list(restart.glob('*.ckpt')) and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            kernel.SetEvent(event)
            assert proc.wait(timeout=120) != 0
        assert queue.read_text() == task and list(restart.glob('*.ckpt')) and not list(restart.glob('*.save'))
        text = call('cpu_resume', [s1, '-ini', ini])
        assert 'resume STAGE1 checkpoint' in text
        assert len(oracle.records(next(restart.glob('*.save')))) == 1
        assert not list(restart.glob('*.ckpt')) and finished.exists()
    finally:
        kernel.CloseHandle(event)
    report['complete'] = True
    persist()
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
