"""Real stopped workers: V1 binary migration, V2 settings and random sigma restore."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from test_edwards_gpu_pipeline import checkpoint, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--naf-baseline', type=Path, help='Old seven-tier NAF worker for cross-container V2 restore')
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); old = a.baseline.resolve(); new = a.stage1.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(baseline_sha256=sha(old), stage1_sha256=sha(new), tool_sha256=sha(__file__),
        oracle_sha256=sha(Path(__file__).with_name('test_edwards_gpu_pipeline.py')),
        device=a.device, checks=[], complete=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ECM_', 'NTT_'))}
    class Attributes(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(Attributes), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    def persist(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def call(name, cmd, input=None, good=True, extra=None):
        r = subprocess.run(list(map(str, cmd)), input=input, capture_output=True,
            env=env | (extra or {}), cwd=out, timeout=600)
        raw = r.stdout+r.stderr; log = out/(name+'.log'); log.write_bytes(raw)
        assert (r.returncode == 0) == good, (name, r.returncode, raw[-2000:])
        report['checks'].append(dict(name=name, exit=r.returncode, log_sha256=sha(log))); persist()
        print('PASS', name, flush=True); return raw.decode('utf-8', 'replace')
    cases=[
        ('binary_v1', old, 347, 4, True, 0, 0),
        ('naf_v2', new, 347, 4, True, 8, 5),
        ('naf_random', new, 521, 2, False, 6, 4)]
    if a.naf_baseline:
        previous=a.naf_baseline.resolve()
        report['naf_baseline_sha256']=sha(previous)
        cases.append(('naf_v2_old_tier',previous,677,2,True,8,5))
    for name, source, bits, count, fixed, window, doubling in cases:
        folder = out/name; folder.mkdir(); ini = folder/'ecm.ini'; queue = folder/'worktodo.txt'; finished = folder/'finished.txt'
        task = f'ECM2=1,2,{bits},-1,100000,0,{count}'+(',20260922' if fixed else '')+'\n'
        queue.write_text(task)
        base_ini = f'curve=ed\ntarget=gpu\ndevice={a.device}\nckpt_seconds=0.001\nckpt_backups=2\nworktodo={queue}\nfinished={finished}\ntmp_dir={folder}\nlog_file=\n'
        ini.write_text(base_ini+f'naf_w={window}\n')
        sa = Attributes(ctypes.sizeof(Attributes), None, True)
        event = kernel.CreateEventW(ctypes.byref(sa), True, False, None); assert event
        with (out/(name+'_stopped.log')).open('wb') as log:
            child_env = env | ({'ECM_EDWARDS_DBL_ALGO': str(doubling)} if doubling else {})
            proc = subprocess.Popen([str(source), '-ini', str(ini), '--stage1-stop-handle', str(event)],
                stdout=log, stderr=subprocess.STDOUT, close_fds=False, creationflags=subprocess.CREATE_NO_WINDOW,
                env=child_env, cwd=out)
            try:
                deadline = time.monotonic()+120
                ready=False
                while not ready and time.monotonic() < deadline:
                    assert proc.poll() is None, ('worker finished before checkpoint', name)
                    candidates=[v for v in list(folder.glob('*.gpu.ckpt'))+list(folder.glob('.e_s1_*.ckpt')) if '.bak' not in v.name]
                    for candidate in candidates:
                        try:ready=checkpoint(candidate)['done']>0
                        except (AssertionError,ValueError,PermissionError,FileNotFoundError):continue
                        if ready:break
                    time.sleep(0.01)
                assert ready and kernel.SetEvent(event)
                assert proc.wait(timeout=120) != 0
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()
                kernel.CloseHandle(event)
        assert queue.read_text() == task and not finished.exists()
        primary = next(v for v in list(folder.glob('*.gpu.ckpt'))+list(folder.glob('.e_s1_*.ckpt')) if '.bak' not in v.name); frozen = out/(name+'_checkpoint.txt'); frozen.write_bytes(primary.read_bytes())
        data = checkpoint(primary)
        if name=='naf_v2_old_tier':
            assert 'container_bits=1024' in (out/(name+'_stopped.log')).read_text(encoding='utf-8')
        assert (data['window'], data['doubling'], data['b1'], data['count']) == (window, doubling, 100000, count)
        assert 0 < data['done'] < data['total']
        if window:
            ini.write_text(base_ini+'naf_w=3\n'); before = primary.read_bytes()
            text = call(name+'_reject_window', [new, '-ini', ini], good=False)
            assert 'checkpoint algorithm differs' in text and primary.read_bytes() == before and queue.read_text() == task
        # Default configuration must adopt the stored mode, window and algorithm.
        ini.write_text(base_ini+'naf_w=0\n')
        before = primary.read_bytes()
        text = call(name+'_reject_mode', [new, '-ini', ini], good=False,
            extra={'ECM_EDWARDS_STAGE1_ALGO': 'binary' if window else 'naf'})
        assert 'checkpoint algorithm differs' in text and primary.read_bytes() == before and queue.read_text() == task
        text = call(name+'_resume', [new, '-ini', ini])
        assert f'window={window} doubling={doubling}' in text and 'GPU Edwards resume:' in text
        if name=='naf_v2_old_tier': assert 'container_bits=768' in text
        assert not queue.read_text().strip() and finished.exists() and not list(folder.glob('.e_s1_*.ckpt'))
        if primary.name.endswith('.gpu.ckpt'):assert primary.read_bytes()==frozen.read_bytes()
        rows = records(next(folder.glob('e_*.save')))
        assert [r[2] for r in rows] == [r['sigma'] for r in data['rows']]
        # Binary and NAF have identical affine endpoints; independently validate
        # all rows above, then compare with fresh explicitly selected NAF runs.
        for index, row in enumerate(rows):
            save = out/f'{name}_full_{index}.save'
            call(f'{name}_full_{index}', [new, '--curve', 'ed', '--target', 'gpu', '-d', a.device,
                '--naf-w', window or 6, '-sigma', row[2], '-gpucurves', 1, '--tmp-dir', out/f'{name}_tmp_{index}',
                '-save', save, 100000, 0], f'(2^{bits}-1)\n'.encode(),
                extra={'ECM_EDWARDS_DBL_ALGO': str(doubling or 4)})
            assert records(save) == [row]
        report.setdefault('restarts', []).append(dict(name=name, columns=data['done'], total=data['total'],
            window=window, doubling=doubling, sigmas=[r['sigma'] for r in data['rows']], checkpoint_sha256=sha(frozen)))
    report['complete'] = True; persist()


if __name__ == '__main__': main()
