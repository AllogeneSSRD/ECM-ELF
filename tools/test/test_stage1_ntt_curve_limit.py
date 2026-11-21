"""Resume real NTT task checkpoints after changing only worktodo's curve limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

from stage1_checkpoint_decode import batch
from test_stage1_ntt_task_batches import (
    EXPONENT, SIGMA, BOUND, STAGE2_BOUND, CHECKPOINT,
    assert_finished, batch_launches, stop_inside_later_batch,
    checkpoint_cursor,
)
from test_stage1_ntt_production import edwards_stage1, ntt_executions, saved_points


def crc32c(data):
    crc = 0xffffffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    return crc ^ 0xffffffff


def write_batch(path, identity, rows):
    encoded = identity.encode()
    data = b'ECMS1B2\0' + struct.pack('<IIQ', 2, len(rows), len(encoded)) + encoded
    for sigma, payload in rows:
        data += struct.pack('<QQII', sigma, len(payload), crc32c(payload), 0) + payload
    path.write_bytes(data + struct.pack('<I', crc32c(data)))
    assert batch(path) == (identity, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--seed-exe', type=Path, help='Optional older worker to create the checkpoint')
    parser.add_argument('--seed-directory', type=Path, help='Clone a paused fixture from an earlier run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    seed_exe = (args.seed_exe or exe).resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed = out / 'seed'
    if args.seed_directory:
        shutil.copytree(args.seed_directory.resolve(strict=True), seed)
    else:
        seed.mkdir()
    original_limit = 65
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ECM_', 'NTT_'))}

    def config(work, maximum=16, algorithm='ntt'):
        (work / 'ecm.toml').write_text(
            'config_version=4\n[shared]\nworktodo_file="worktodo.txt"\nfinished_file="finished.txt"\n'
            f'log_file=""\ntune_file="{work.as_posix()}/tune.toml"\n[stage1]\n'
            f'param="e"\ntarget="gpu"\nstage1_gpu_device={args.device}\n'
            f'stage1_save_dir="{work.as_posix()}"\nhandoff_mode="ecm-elf"\n'
            f'edwards_naf_window_bits=6\nexponent_cache_dir="off"\ncheckpoint_interval_seconds=0.01\n'
            f'ntt_max_batch={maximum}\nntt_pad_batch=0\ngpu_algorithm="{algorithm}"\n', encoding='utf-8')

    def todo(work, limit, sigma=SIGMA):
        (work / 'worktodo.txt').write_text(
            f'ECM=1,2,{EXPONENT},-1,{BOUND},{STAGE2_BOUND},{limit},{sigma}\n', encoding='ascii')

    config(seed)
    todo(seed, original_limit)
    command = [str(seed_exe), '--config', str(seed / 'ecm.toml')]
    if args.seed_directory:
        interrupted = batch_launches((seed / 'stop.log').read_text(encoding='utf-8'))[-1]
        cursor = checkpoint_cursor(seed / CHECKPOINT)
    else:
        interrupted, cursor = stop_inside_later_batch(seed, command, env, 120)
    completed = interrupted['offset']
    batch_end = completed + interrupted['curves']
    assert interrupted['total'] == original_limit and completed and cursor and batch_end < original_limit, interrupted
    fresh_cap = max(1, interrupted['curves'] // 2)
    report = dict(seed_exe=str(seed_exe), exe=str(exe),
                  exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  seed_identity=batch(seed / CHECKPOINT)[0],
                  checkpoint_sha256=hashlib.sha256((seed / CHECKPOINT).read_bytes()).hexdigest(),
                  interrupted=interrupted, cursor=cursor, new_batch_cap=fresh_cap, checks=[])

    for name, limit, expected, active in (
        ('unchanged_limit', original_limit, original_limit, True),
        ('below_completed', 1, batch_end, True),
        ('inside_active', completed + 1, batch_end, True),
        ('at_active_end', batch_end, batch_end, True),
        ('increase_limit', original_limit + 32, original_limit + 32, True),
        ('completed_prefix_only', 1, completed, False),
        ('reserved_without_native', 1, batch_end, True),
        ('resize_twice', 1, 0, True),
    ):
        work = out / name
        shutil.copytree(seed, work)
        config(work, fresh_cap, 'cgbn[]')  # A matching checkpoint must still select NTT.
        todo(work, limit)
        saved_batch, saved_cursor = interrupted, cursor
        if name == 'resize_twice':
            todo(work, original_limit + 32)
            saved_batch, saved_cursor = stop_inside_later_batch(
                work, [str(exe), '--config', str(work / 'ecm.toml')], env, 120)
            assert batch(work / CHECKPOINT)[0].endswith('|ntt-task-v2')
            expected = saved_batch['offset'] + saved_batch['curves']
            todo(work, limit)
        if not active or name == 'reserved_without_native':
            path = work / CHECKPOINT
            identity, [(sigma, payload)] = batch(path)
            first, preferred, width, capacity, count, native = struct.unpack_from('<QIIIIQ', payload, 10)
            assert count == completed and width == interrupted['curves']
            width, capacity = (width, capacity) if active else (0, 0)
            payload = payload[:10] + struct.pack('<QIIIIQ', first, preferred, width, capacity, count, 0) + payload[42 + native:]
            write_batch(path, identity, [(sigma, payload)])
            saved_cursor = 0
        run = subprocess.run([str(exe), '--config', str(work / 'ecm.toml')], cwd=work, env=env,
                             capture_output=True, timeout=120)
        log = run.stdout.decode('utf-8', errors='replace') + run.stderr.decode('utf-8', errors='replace')
        (work / 'resume.log').write_text(log, encoding='utf-8')
        assert run.returncode == 0, (name, run.returncode, log[-4000:])
        launches = batch_launches(log)
        if active:
            assert launches and launches[0] == {**saved_batch, 'total': limit, 'resumed': 1}, launches
            assert int(ntt_executions(log)[0]['ntt_resume']) == saved_cursor
            assert all(row['curves'] <= fresh_cap for row in launches[1:]), launches
            if limit <= saved_batch['offset'] + saved_batch['curves']:
                assert len(launches) == 1, launches
        else:
            assert not launches and not ntt_executions(log), log
        save, _ = assert_finished(work, expected)
        if name == 'increase_limit':
            points = saved_points(save, EXPONENT, BOUND, edwards=True)
            modulus = (1 << EXPONENT) - 1
            for lane in (0, completed, expected - 1):
                _, x, z = edwards_stage1(modulus, SIGMA + lane, BOUND)
                assert points[SIGMA + lane] == x * pow(z, -1, modulus) % modulus
        report['checks'].append(dict(name=name, requested=limit, published=expected, launches=launches))
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('PASS', name, flush=True)

    # Changing scheduling limits does not relax the mathematical sigma identity.
    work = out / 'changed_sigma'
    shutil.copytree(seed, work)
    config(work)
    todo(work, original_limit, SIGMA + 1)
    paths = [p for p in work.glob('*.ckpt') if '.publish.' not in p.name]
    before = {p.name: p.read_bytes() for p in paths}
    queue = (work / 'worktodo.txt').read_bytes()
    run = subprocess.run([str(exe), '--config', str(work / 'ecm.toml')], cwd=work, env=env,
                         capture_output=True, timeout=30)
    (work / 'rejected.log').write_bytes(run.stdout + run.stderr)
    assert run.returncode != 0 and not batch_launches(run.stdout.decode('utf-8', errors='replace'))
    assert {p.name: p.read_bytes() for p in paths} == before and (work / 'worktodo.txt').read_bytes() == queue
    report['checks'].append(dict(name='changed_sigma_rejected', checkpoint_unchanged=True))
    print('PASS changed_sigma_rejected', flush=True)

    report['complete'] = True
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
