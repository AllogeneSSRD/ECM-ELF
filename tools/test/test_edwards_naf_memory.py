"""Edwards Stage1 memory admission, worker cache reuse, and endpoint checks."""

import argparse
import ctypes
from ctypes import wintypes
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

from stage1_checkpoint_decode import batch
import test_edwards_gpu_pipeline as oracle


def write_config(folder, window, implementation, device, interval=0, limit_mb=0):
    config = folder / 'ecm.toml'
    execution = ('target = "cpu"\n'
                 f'cpu_implementation = "{implementation}"\n'
                 'cpu_threads = 2\n') if device is None else (
                 'target = "gpu"\n'
                 'gpu_backend = "cuda"\n'
                 'gpu_algorithm = "cgbn"\n'
                 f'gpu_device = {device}\n')
    config.write_text(
        'config_version = 3\n'
        '[shared]\n'
        'worktodo_file = "worktodo.txt"\n'
        'log_file = ""\n'
        '[stage1]\n'
        'param = "e"\n'
        f'{execution}'
        'sigma = 20260922\n'
        f'checkpoint_interval_seconds = {interval}\n'
        'exponent_cache_dir = "off"\n'
        'handoff.mode = "off"\n'
        'verbose = true\n'
        f'edwards_naf_window_bits = {window}\n'
        f'naf_memory_mb = {limit_mb}\n'
        f'save_dir = {json.dumps(folder.as_posix())}\n', encoding='utf-8')
    return config


def memory_env(device, budget):
    env = {key: value for key, value in os.environ.items() if not key.startswith(('ECM_', 'NTT_'))}
    if budget is not None:
        key = 'ECM_EDWARDS_CPU_MEMORY_LIMIT_MIB' if device is None else 'ECM_EDWARDS_GPU_MEMORY_MIB'
        env[key] = str(budget)
    return env


def run_queue(exe, folder, *, window=12, implementation='simd', budget=None, b1=100,
              device=None, success=True, limit_mb=0):
    folder.mkdir()
    config = write_config(folder, window, implementation, device, limit_mb=limit_mb)
    saves = [folder / f'e_m8191_{name}_{b1}.save' for name in ('first', 'second')]
    queue = folder / 'worktodo.txt'
    work = ''.join(f'ECMSTAGE1=1,2,8191,-1,"{p.as_posix()}",{b1},0,16\n' for p in saves)
    queue.write_text(work, encoding='utf-8')
    result = subprocess.run([str(exe), '--config', str(config)], cwd=folder, env=memory_env(device, budget),
                            capture_output=True, timeout=120)
    raw = result.stdout + result.stderr
    (folder / 'run.log').write_bytes(raw)
    log = raw.decode(errors='replace')
    assert (result.returncode == 0) == success, (folder, result.returncode, log[-3000:])
    if not success:
        assert queue.read_text() == work and not any(p.stat().st_size for p in saves if p.exists())
        return log, None
    assert not queue.read_text().strip()
    curves = [oracle.records(p) for p in saves]
    assert curves[0] == curves[1] and len(curves[0]) == 16
    assert [row[2] for row in curves[0]] == list(range(20260922, 20260938))
    return log, curves[0]


def checkpoint_window(path, device):
    _, rows = batch(path)
    payload = rows[0][1]
    if not payload:
        return None
    if device is None:
        # Prime95 Edwards checkpoint after the common 52-byte header.
        bits, done, entries = struct.unpack_from('<III', payload, 108)
        assert entries and entries & (entries - 1) == 0
        return (entries.bit_length() + 1, done, bits) if done else None
    assert payload.startswith((b'ECMEDB3\0', b'ECMEDB4\0'))
    window, _, _, done = struct.unpack_from('<IIII', payload, 12)
    bits, = struct.unpack_from('<Q', payload, 40)
    return (window, done, bits) if done else None


def stop_and_resume(exe, folder, device):
    folder.mkdir()
    config = write_config(folder, 12, 'scalar', device, interval=0.001)
    b1 = 10000
    save = folder / f'e_m8191_restart_{b1}.save'
    queue = folder / 'worktodo.txt'
    work = f'ECMSTAGE1=1,2,8191,-1,"{save.as_posix()}",{b1},0,1\n'
    queue.write_text(work, encoding='utf-8')
    command = [str(exe), '--config', str(config)]

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(SecurityAttributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    security = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
    event = kernel.CreateEventW(ctypes.byref(security), True, False, None)
    assert event
    with (folder / 'stopped.log').open('wb') as log:
        process = subprocess.Popen(command + ['--stage1-stop-handle', str(event)], cwd=folder,
                                   env=memory_env(device, 8 if device is None else 1), close_fds=False,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 30
            while process.poll() is None and time.monotonic() < deadline:
                path = next(folder.glob('.e_s1_*.ckpt'), None)
                try:
                    if path and checkpoint_window(path, device):
                        break
                except (PermissionError, FileNotFoundError):
                    pass
                time.sleep(0.005)
            else:
                raise AssertionError(f'{folder}: no partial checkpoint before exit or timeout')
            assert kernel.SetEvent(event)
            assert process.wait(timeout=60) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            kernel.CloseHandle(event)

    assert queue.read_text() == work and not (save.exists() and save.stat().st_size)
    window, done, bits = checkpoint_window(path, device)
    assert 2 <= window < 12 and 0 < done < bits
    if device is not None:
        saved = oracle.checkpoint(path)
        assert saved['requested_window'] == 12 and saved['window'] == window and done < saved['total']
    frozen = path.read_bytes()
    write_config(folder, 12, 'scalar', device)
    for label, budget in [('blocked', 0), ('resumed', None)]:
        result = subprocess.run(command, cwd=folder, env=memory_env(device, budget),
                                capture_output=True, timeout=120)
        raw = result.stdout + result.stderr
        (folder / f'{label}.log').write_bytes(raw)
        if budget is not None:
            assert result.returncode != 0 and path.read_bytes() == frozen and queue.read_text() == work
        else:
            assert result.returncode == 0 and not path.exists() and not queue.read_text().strip(), raw[-3000:]
            expected = b'resume STAGE1 checkpoint' if device is None else f'window={window}'.encode()
            assert expected in raw, raw[-3000:]
    checked = oracle.records(save)
    assert len(checked) == 1 and checked[0][2:4] == (20260922, b1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, help='Also check constrained execution on this CUDA device')
    parser.add_argument('--restart', action='store_true', help='Check downgraded checkpoint recovery on Windows')
    args = parser.parse_args()
    exe, out = args.stage1.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Identical endpoint inputs recur on each backend and in both queue rows.
    oracle.stage1 = lru_cache(maxsize=32)(oracle.stage1)

    log, baseline = run_queue(exe, out / 'scalar', window=6, implementation='scalar')
    assert re.findall(r'CPU Edwards NAF:.*\bcache=(\w+)', log) == ['miss', 'hit'], log
    print('PASS scalar multi-curve and cross-task cache reuse', flush=True)

    log, limited = run_queue(exe, out / 'limited', limit_mb=16)
    assert 'WARNING: CPU Edwards memory:' in log, log
    assert re.findall(r'CPU Edwards memory:.*\bbudget=(\d+) bytes', log) == ['16000000'] * 2, log
    assert re.findall(r'CPU Edwards NAF:.*\bcache=(\w+)', log) == ['miss', 'hit'], log
    assert limited == baseline
    print('PASS limited IFMA keeps every requested curve and the same endpoints', flush=True)

    log, _ = run_queue(exe, out / 'unavailable', limit_mb=1, b1=1_000_000, success=False)
    assert 'ERROR: CPU Edwards memory:' in log, log
    print('PASS insufficient memory preserves the queue and publishes no curves', flush=True)
    if args.restart:
        stop_and_resume(exe, out / 'cpu_restart', None)
        print('PASS CPU partial checkpoint keeps its lowered window across restart', flush=True)

    if args.device is not None:
        log, limited = run_queue(exe, out / 'gpu_limited', device=args.device, limit_mb=1)
        assert 'WARNING: GPU Edwards memory:' in log, log
        assert re.findall(r'GPU Edwards device memory:.*\bbudget_bytes=(\d+)', log) == ['1000000'] * 2, log
        assert limited == baseline
        print('PASS limited CUDA keeps every requested curve and the same endpoints', flush=True)

        log, _ = run_queue(exe, out / 'gpu_unavailable', device=args.device, budget=0, success=False)
        assert 'memory' in log.lower(), log
        print('PASS insufficient CUDA memory preserves the queue and publishes no curves', flush=True)
        if args.restart:
            stop_and_resume(exe, out / 'gpu_restart', args.device)
            print('PASS CUDA partial checkpoint keeps its lowered window across restart', flush=True)


if __name__ == '__main__':
    main()
