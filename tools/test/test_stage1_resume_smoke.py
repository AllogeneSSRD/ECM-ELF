"""Stop real Stage1 arithmetic, then compare its resumed save to a fresh run."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from stage1_checkpoint_decode import batch, progressed
from test_ecm_domain_stage1 import points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--curve', choices=['mont', 'ed'], default='mont')
    p.add_argument('--param', type=int, default=3)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--b1', type=int, default=100000)
    a = p.parse_args()
    exe = a.exe.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  curve=a.curve, param=a.param, b1=a.b1, device=a.device)
    class Security(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('descriptor', wintypes.LPVOID), ('inherit', wintypes.BOOL)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateEventW.argtypes = [ctypes.POINTER(Security), wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateEventW.restype = wintypes.HANDLE
    kernel.SetEvent.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    def setup(name):
        work = out/name
        work.mkdir()
        queue = work/'worktodo.txt'
        task = f'ECM=1,2,347,-1,{a.b1},0,8,1000\n'
        queue.write_text(task)
        ini = work/'ecm.toml'
        ini.write_text('[shared]\nworktodo='+json.dumps(str(queue))+'\n[stage1]\n'+
                       f'curve="{a.curve}"\ntarget="gpu"\nimpl="auto"\ngpu_param={a.param}\ndevice={a.device}\n'+
                       'exp_cache="off"\nlog_file=""\nckpt_seconds=0.001\n'+
                       'tmp_dir='+json.dumps(str(work))+'\nfinished='+json.dumps(str(work/'finished.txt'))+'\n')
        return work, queue, ini, task
    work, queue, ini, task = setup('resumed')
    security = Security(ctypes.sizeof(Security), None, True)
    event = kernel.CreateEventW(ctypes.byref(security), True, False, None)
    assert event, ctypes.get_last_error()
    command = [str(exe), '--config', str(ini)]
    with (out/'stop.log').open('wb') as log:
        proc = subprocess.Popen(command+['--stage1-stop-handle', str(event)], cwd=work,
                                stdout=log, stderr=log, close_fds=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic()+90
            while time.monotonic() < deadline:
                assert proc.poll() is None, 'worker completed before arithmetic stop'
                snapshots = [v for v in work.glob('*.ckpt') if '.bak' not in v.name]
                try:
                    if snapshots and progressed(batch(snapshots[0])[1]):
                        break
                except (PermissionError, FileNotFoundError):
                    pass
                time.sleep(.01)
            else:
                raise AssertionError('no arithmetic snapshot')
            assert kernel.SetEvent(event)
            assert proc.wait(timeout=120) != 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            kernel.CloseHandle(event)
    assert queue.read_text() == task
    snapshot = next(v for v in work.glob('*.ckpt') if '.bak' not in v.name)
    shutil.copy2(snapshot, out/'stopped_checkpoint.bin')
    identity, records = batch(snapshot)
    assert [sigma for sigma, _ in records] == list(range(1000, 1008))
    report['checkpoint_identity'] = identity
    # Disable periodic writes only after the stop has produced real restore data.
    ini.write_text(ini.read_text().replace('ckpt_seconds=0.001', 'ckpt_seconds=0'))
    fresh_work, _, fresh_ini, _ = setup('fresh')
    for name, folder, settings in [('resume', work, ini), ('fresh', fresh_work, fresh_ini)]:
        result = subprocess.run([str(exe), '--config', str(settings)], cwd=folder,
                                capture_output=True, timeout=180)
        (out/(name+'.log')).write_bytes(result.stdout+result.stderr)
        assert result.returncode == 0, (name, result.returncode)
    assert not queue.read_text().strip() and not list(work.glob('*.ckpt'))
    prefix = 'e' if a.curve == 'ed' else 'm'+str(a.param)
    restored = points(next(work.glob(prefix+'_*.save')))
    fresh = points(next((out/'fresh').glob(prefix+'_*.save')))
    assert len(restored) == 8 and restored == fresh, 'restored mathematical save differs'
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, records=len(restored), curve=a.curve, param=a.param)))


if __name__ == '__main__':
    main()
