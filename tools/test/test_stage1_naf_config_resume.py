"""Resume real Edwards checkpoints after changing Stage1 NAF configuration."""

import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import time

from stage1_checkpoint_decode import batch
from test_edwards_gpu_pipeline import records
from test_stage1_ntt_task_batches import pausable_process


def saved_progress(path, backend):
    try:
        _, rows = batch(path)
        if backend in ('cpu', 'cpu-simd'):
            for _, payload in rows:
                if len(payload) >= 120 and not payload.startswith(b'ECM_EDWARDS_DONE_V1'):
                    cursor, dictionary = struct.unpack_from('<II', payload, 112)
                    if cursor and dictionary in (64, 8):
                        return cursor
            return 0
        payload = rows[0][1]
        if backend == 'cgbn' and payload.startswith((b'ECMEDB3\0', b'ECMEDB4\0')):
            return struct.unpack_from('<I', payload, 24)[0]
        for tag in (b'CUDA_ED_THREAD_V1', b'OCL_ED_NAF_V1'):
            if payload.startswith(tag):
                return struct.unpack_from('<I', payload, len(tag))[0]
    except (AssertionError, FileNotFoundError, PermissionError, ValueError, struct.error):
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--previous-stage1', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=('cgbn', 'cpu', 'cpu-simd', 'thread', 'opencl'), required=True)
    parser.add_argument('--change', choices=('window', 'mode', 'from-binary', 'doubling'), default='window')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    path = args.out.resolve()
    path.mkdir(parents=True)
    config = path / 'ecm.toml'
    queue = path / 'worktodo.txt'
    save = path / 'e_resume_100000.save'
    next_save = path / 'e_next_20.save'
    starter = (args.previous_stage1 or args.stage1).resolve()
    worker = args.stage1.resolve()
    gpu = not args.backend.startswith('cpu')
    threads = args.backend in ('thread', 'opencl')
    count = 8 if args.backend == 'cpu-simd' else 2
    if args.change != 'window' and args.backend != 'cgbn':
        parser.error('changing mode or doubling alone requires CGBN')
    task = f'ECMSTAGE1=1,2,521,-1,"{save.as_posix()}",0,0,{count}'
    next_task = f'ECMSTAGE1=1,2,521,-1,"{next_save.as_posix()}",0,0,{count}'
    checkpoint = path / '.e_s1_m521_100000.ckpt'
    base = (
        'config_version=4\n[shared]\nworktodo_file="worktodo.txt"\nfinished_file="finished.txt"\nlog_file=""\n'
        '[stage1]\nhandoff_mode="off"\ncheckpoint_interval_seconds=0.001\nexponent_cache_dir="off"\n'
        f'stage1_save_dir="{path.as_posix()}"\nsigma=20260922\n'
        + (f'target="gpu"\nstage1_gpu_device={args.device}\n' +
           f'stage1_gpu_backend="{"opencl" if args.backend == "opencl" else "cuda"}"\n' +
           f'gpu_algorithm="{"thread" if args.backend == "thread" else "cgbn"}"\n' if gpu else
           f'target="cpu"\ncpu_implementation="{"simd" if args.backend == "cpu-simd" else "scalar"}"\n'
           'stage1_cpu_threads=1\n')
    )

    def settings(window, doubling):
        policy = f'edwards_naf_window_bits={window}\n'
        if doubling is not None:
            policy += f'edwards_doubling={doubling}\n'
        config.write_text(base + policy, encoding='utf-8')

    start_binary = args.change == 'from-binary'
    settings(0 if start_binary else 8, None if args.change == 'doubling' else 1)
    queue.write_text(task + '\n', encoding='utf-8')
    environment = {k: v for k, v in os.environ.items() if k not in
                   ('ECM_EDWARDS_STAGE1_ALGO', 'ECM_EDWARDS_DBL_ALGO')}
    if start_binary:
        environment['ECM_EDWARDS_STAGE1_ALGO'] = 'binary'
    if args.change == 'doubling':
        environment['ECM_EDWARDS_DBL_ALGO'] = '5'
    with pausable_process([str(starter), '--config', str(config)], path, environment,
                          path / 'stopped.log') as (process, stop):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and process.poll() is None:
            if saved_progress(checkpoint, args.backend):
                stop()
                assert process.wait(timeout=60) != 0
                break
            time.sleep(0.002)
        else:
            raise AssertionError((path / 'stopped.log').read_text(encoding='utf-8', errors='replace'))
    assert queue.read_text().strip() == task and saved_progress(checkpoint, args.backend)
    (path / 'frozen.ckpt').write_bytes(checkpoint.read_bytes())

    if args.change == 'mode':
        environment['ECM_EDWARDS_STAGE1_ALGO'] = 'binary'
        # A resumed NAF batch retains its window even when the fresh-mode
        # environment would reject an explicitly configured NAF window.
    elif args.change == 'doubling':
        environment['ECM_EDWARDS_DBL_ALGO'] = '4'
        queue.write_text(task + '\n' + next_task + '\n', encoding='utf-8')
    else:
        environment.pop('ECM_EDWARDS_STAGE1_ALGO', None)
        settings(5, 0 if threads or start_binary else 1)
        queue.write_text(task + '\n' + next_task + '\n', encoding='utf-8')
    process = subprocess.run([worker, '--config', config], cwd=path, env=environment,
                             capture_output=True, timeout=180)
    raw = process.stdout + process.stderr
    (path / 'resume.log').write_bytes(raw)
    log = raw.decode('utf-8', 'replace')
    assert process.returncode == 0, log[-4500:]
    assert 'ECMSTAGE1=' not in queue.read_text()
    assert not checkpoint.exists()
    assert [row[2] for row in records(save)] == list(range(20260922, 20260922 + count))
    if args.backend == 'cgbn':
        mode = 'window=0 doubling=0' if start_binary else 'window=8 doubling=5'
        assert re.search(r'GPU Edwards resume: columns=[1-9]\d* ' + mode, log), log
    elif threads:
        assert re.search(r'naf=8 doubling=5 .*resume=[1-9]\d*', log), log
    else:
        assert re.search(r'resume (?:STAGE1 checkpoint|from \.ckpt) @ bitnum=[1-9]', log, re.IGNORECASE), log
        assert re.search(r'Stage1 execution: .*\bnaf=8\b', log), log
    if args.change != 'mode':
        assert len(records(next_save)) == count
        fresh = r'\bnaf=8\b.*\bdoubling=4\b' if args.change == 'doubling' else r'\bnaf=5\b'
        assert re.search(r'Stage1 execution: .*' + fresh, log), log
    print(f'{args.backend}: checkpoint used saved NAF and completed after changing {args.change}.')


if __name__ == '__main__':
    main()
