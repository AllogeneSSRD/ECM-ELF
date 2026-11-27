"""Exercise NTT Stage1 handoff and CPU Stage2 range ownership."""

import argparse
from contextlib import contextmanager
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

from test_stage1_ntt_task_batches import pausable_process


@contextmanager
def hold_curves(path, records):
    if os.name == 'nt':
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        class Overlapped(ctypes.Structure):
            _fields_ = [('internal', ctypes.c_size_t), ('internal_high', ctypes.c_size_t),
                        ('offset', wintypes.DWORD), ('offset_high', wintypes.DWORD),
                        ('event', wintypes.HANDLE)]

        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
        handle = kernel.CreateFileW(str(path), 0xC0000000, 3, None, 4, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            for record in records:
                lock = Overlapped(offset=record)
                if not kernel.LockFileEx(handle, 3, 0, 1, 0, ctypes.byref(lock)):
                    raise ctypes.WinError(ctypes.get_last_error())
            yield
        finally:
            kernel.CloseHandle(handle)
    else:
        import fcntl

        with path.open('a') as lease:
            for record in records:
                fcntl.lockf(lease, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, record)
            yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Interrupt Stage1 after the first PART and resume')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    b1, b2 = (25000, 26000) if args.resume else (20, 100)
    save = out / f'e_m4423_{b1}.save'
    config = out / 'ecm.toml'
    queue = out / 'worktodo.txt'
    config.write_text(f'''config_version = 4
[shared]
worktodo_file = "worktodo.txt"
finished_file = "finished.txt"
log_file = ""
tune_file = "{(out / 'tune.toml').as_posix()}"
[stage1]
target = "gpu"
param = "e"
gpu_algorithm = "ntt"
stage1_gpu_device = {args.gpu}
stage1_save_dir = "{out.as_posix()}"
stage1_submission_file = "stage1-submission.json"
ntt_max_batch = 2
handoff_mode = "ecm-elf"
handoff_workers = "2,3"
sigma = 20260922
edwards_naf_window_bits = 6
[stage2]
device = "cpu"
cpu_backend = "plain"
b2_round_chain = false
stage2_submission_file = "stage2-submission.json"
''', encoding='utf-8')

    def run(exe, label, *flags, expected=0):
        proc = subprocess.run([exe.resolve(), '--config', config, *flags], cwd=out,
                              capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        log = proc.stdout + proc.stderr
        (out / f'{label}.log').write_text(log, encoding='utf-8')
        assert proc.returncode == expected, (proc.returncode, log)
        return log

    queue.write_text(f'ECMSTAGE1=1,2,4423,-1,"{save.as_posix()}",{b2},0,5\n', encoding='utf-8')
    initial = ''
    if args.resume:
        log_path = out / 'stage1_interrupted.log'
        command = [args.stage1.resolve(), '--config', config]
        with pausable_process(command, out, os.environ.copy(), log_path) as (proc, stop):
            deadline = time.monotonic() + 100
            while proc.poll() is None and time.monotonic() < deadline:
                if 'stage1_handoff_part:' in log_path.read_text(encoding='utf-8', errors='replace'):
                    break
                time.sleep(0.02)
            pending = queue.read_text()
            assert proc.poll() is None and pending.count('ECMSTAGE2PART=') == 1 and 'ECMSTAGE2=' not in pending
            stop()
            proc.wait(timeout=60)
        initial = log_path.read_text(encoding='utf-8', errors='replace')
    stage1 = initial + run(args.stage1, 'stage1')
    assert stage1.count('stage1_handoff_part:') == 2, stage1
    rows = queue.read_text(encoding='utf-8').splitlines()
    part1, part2 = (line for line in rows if line.startswith('ECMSTAGE2PART='))
    final, = (line for line in rows if line.startswith('ECMSTAGE2='))

    def fields(line):
        return next(csv.reader([line.split('=', 1)[1]]))

    assert fields(part1)[-2:] == ['0', '2']
    assert fields(part2)[-2:] == ['2', '2']
    assert fields(final)[-3:-1] == ['0', '5']
    checkpoint = save.with_name(save.name + '.stage2.ckpt')
    assert checkpoint.is_file()
    assert 'completed=2' in run(args.stage2, 'part1', '--worker', '2', '--once')
    submission = out / 'stage2-submission.json'
    assert not submission.exists() or not submission.read_text().strip()

    save.write_text(save.read_text().replace('METHOD=', 'WHO=renamed-host; METHOD='), encoding='utf-8')
    part2_edited = part2.rsplit(',', 3)[0] + f',{b2 + 500},1,3'
    queue.write_text(queue.read_text().replace(part2, part2_edited), encoding='utf-8')
    with hold_curves(checkpoint.with_name(checkpoint.name + '.claims'), [3, 4]):
        log = run(args.stage2, 'final_wait', '--worker', '2', '--once')
        assert 'curves_this_run=1' in log and 'stage2_handoff_wait:' in log, log
        assert final in queue.read_text()
    overlap = run(args.stage2, 'part2', '--worker', '3', '--once')
    assert 'completed=3' in overlap and 'curves_this_run=2' in overlap, overlap
    assert 'curves_this_run=0' in run(args.stage2, 'final', '--worker', '2', '--once')
    result, = map(json.loads, submission.read_text().splitlines())
    assert result['status'] == 'NF' and result['curves'] == 5 and result['b2'] == b2, result
    bounds = {}
    for row in checkpoint.read_text().splitlines():
        columns = row.split(maxsplit=4)
        if len(columns) == 5 and all(item.isdecimal() for item in columns[:4]):
            bounds[int(columns[0])] = int(columns[3])
    assert bounds == {1: b2, 2: b2, 3: b2 + 500, 4: b2 + 500, 5: b2}, bounds
    assert not list(out.glob(save.name + '.stage2-*'))

    queue.write_text(f'[Worker #2]\n{part1}\n', encoding='utf-8')
    log = run(args.stage2, 'late_part', '--worker', '2', '--once')
    assert 'curves_this_run=0' in log and 'stage2_part_done:' in log, log
    assert 'ECMSTAGE2' not in queue.read_text()
    print('PASS partial recovery, overlap, busy records, B2 edit, metadata edit, single NF and late PART')


if __name__ == '__main__':
    main()
