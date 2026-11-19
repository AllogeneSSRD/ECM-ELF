"""Verify Edwards factor endpoints, cofactor Stage2, and Stage1 sigma policy."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from stage1_checkpoint_decode import ntt_inflight
from test_edwards_gpu_pipeline import records
from test_stage1_ntt_task_batches import checkpoint_cursor, pausable_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    stage2 = args.stage2.resolve()
    out.mkdir(parents=True)
    stage1 = out / args.stage1.name
    shutil.copy2(args.stage1, stage1)
    shutil.copy2(args.stage1.parent / 'gmp-10.dll', out)

    def run(exe, case, label, *flags):
        process = subprocess.run([exe, '--config', case / 'ecm.toml', *flags], cwd=case,
                                 capture_output=True, timeout=180)
        log = (process.stdout + process.stderr).decode('utf-8', 'replace')
        (case / f'{label}.log').write_text(log, encoding='utf-8')
        assert process.returncode == 0, log
        return log

    def case(name, task, stage1_options, stage2_options='', handoff='ecm-elf'):
        path = out / name
        path.mkdir()
        task = re.sub(r'"(e_[^"]+\.save)"', lambda match: f'"{(path / match[1]).as_posix()}"', task)
        (path / 'worktodo.txt').write_text(task + '\n', encoding='utf-8')
        (path / 'ecm.toml').write_text(
            'config_version=4\n[shared]\nworktodo_file="worktodo.txt"\nfinished_file="finished.txt"\n'
            f'log_file=""\n[stage1]\nhandoff_mode="{handoff}"\nexponent_cache_dir="off"\n'
            f'stage1_save_dir="{path.as_posix()}"\n{stage1_options}\n'
            '[stage2]\ndevice="cpu"\ncpu_backend="plain"\nb2_round_chain=false\n'
            f'stage2_results_file="stage2.jsonl"\n{stage2_options}\n', encoding='utf-8')
        return path

    m127 = (1 << 127) - 1
    for enabled in (False, True):
        target = 1009 * 10007
        path = case(f'handoff_{enabled}',
                    f'ECMSTAGE1={target},2,127,-{target},"e_input_20.save",40000,0,1',
                    'target="cpu"\ncpu_implementation="scalar"\nsigma=17',
                    'stage2_after_factored=true' if enabled else '')
        run(stage1, path, 'stage1')
        original = path / 'e_input_20.save'
        assert records(original) == [(target * m127, 1009, 17, 20, 'FACTOR')]
        assert 'QCHECKSUM=' in original.read_text()
        queue = path / 'worktodo.txt'
        if enabled:
            task = queue.read_text().strip()
            assert task.startswith('ECMSTAGE2=')
            fields = next(csv.reader([task.split('=', 1)[1]]))
            assert fields[-1] == '1009'
            projected, = records(Path(fields[-5]))
            assert projected[0] == 10007 * m127 and projected[2:] == (17, 20, 'POINT')
            run(stage2, path, 'stage2', '--once')
            assert not queue.read_text().strip()
            result, = map(json.loads, (path / 'stage2.jsonl').read_text().splitlines())
            assert result['stage'] == 2 and result['factors'] == ['10007']
        else:
            assert not queue.read_text().strip()
            assert 'validated 0 curves' in run(stage2, path, 'skip_factor', '--save', original,
                                                '--b2', '40000', '--dry-run')
            queue.write_text(f'ECMSTAGE2N="{original.as_posix()}",40000,0,1\n', encoding='utf-8')
            assert 'curves_this_run=0' in run(stage2, path, 'skip_factor_queue', '--once')
            assert not (path / 'stage2.jsonl').exists()

    target = 137 * 10007
    path = case('construction_hit',
                f'ECMSTAGE1={target},2,127,-{target},"e_construction_20.save",40000,0,1',
                'target="cpu"\ncpu_implementation="scalar"\nsigma=17')
    run(stage1, path, 'stage1')
    construction = path / 'e_construction_20.save'
    assert records(construction) == [(target * m127, 137, 17, 20, 'FACTOR')]
    assert 'QCHECKSUM=' in construction.read_text()

    path = case('identity', 'ECMSTAGE1=1,2,7,-1,"e_identity_100.save",10000,0,4',
                'target="cpu"\ncpu_implementation="scalar"\nsigma=2026')
    run(stage1, path, 'stage1')
    assert set(row[-1] for row in records(path / 'e_identity_100.save')) == {'IDENTITY'}
    assert not (path / 'worktodo.txt').read_text().strip()

    def consecutive(sigmas):
        return sigmas == list(range(sigmas[0], sigmas[0] + len(sigmas)))

    cpu = 'target="cpu"\ncpu_implementation="scalar"\nrandom_sigma=0'
    first_task = 'ECMSTAGE1=1,2,127,-1,"e_sigmas_20.save",1000,0,12'
    second_task = first_task.replace('e_sigmas_20.save', 'e_next_sigmas_20.save')
    path = case('cpu_random_start', first_task + '\n' + second_task, cpu, handoff='off')
    run(stage1, path, 'stage1')
    starts = set()
    for name in ('e_sigmas_20.save', 'e_next_sigmas_20.save'):
        sigmas = [row[2] for row in records(path / name)]
        assert len(sigmas) == 12 and consecutive(sigmas)
        starts.add(sigmas[0])
    assert len(starts) == 2
    path = case('cpu_explicit_start', first_task, cpu + '\nsigma=17')
    run(stage1, path, 'stage1')
    assert [row[2] for row in records(path / 'e_sigmas_20.save')] == list(range(17, 29))

    if args.gpu is not None:
        for algorithm in ('cgbn', 'ntt'):
            for random in (False, True):
                path = case(f'{algorithm}_{random}',
                            'ECMSTAGE1=1,2,4423,-1,"e_sigmas_20.save",1000,0,12',
                            f'target="gpu"\ngpu_algorithm="{algorithm}"\n'
                            f'stage1_gpu_device={args.gpu}\ngpu_init_threads=3\n'
                            f'random_sigma={str(random).lower()}')
                log = run(stage1, path, 'stage1')
                assert f'stage1_{algorithm}_edwards_setup: threads=3' in log, log
                sigmas = [row[2] for row in records(path / 'e_sigmas_20.save')]
                assert len(sigmas) == len(set(sigmas)) == 12
                assert consecutive(sigmas) == (not random)
                if not random:
                    assert sigmas[0] != 6

        path = case('ntt_random_start_resume', 'ECMSTAGE1=1,2,4423,-1,"e_resume_5000.save",10000,0,8',
                    f'target="gpu"\ngpu_algorithm="ntt"\nstage1_gpu_device={args.gpu}\n'
                    'random_sigma=false\nedwards_naf_window_bits=6\ncheckpoint_interval_seconds=0.01')
        checkpoint = path / '.e_s1_m4423_5000.ckpt'
        command = [str(stage1), '--config', str(path / 'ecm.toml')]
        with pausable_process(command, path, os.environ, path / 'stopped.log') as (process, stop):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                if checkpoint_cursor(checkpoint):
                    stop()
                    assert process.wait(timeout=25) != 0
                    break
                time.sleep(0.005)
            else:
                raise AssertionError((path / 'stopped.log').read_text(encoding='utf-8', errors='replace'))
        sigmas, _ = ntt_inflight(checkpoint)
        assert len(sigmas) == len(set(sigmas)) == 8 and consecutive(sigmas)
        config = path / 'ecm.toml'
        config.write_text(config.read_text().replace('random_sigma=false', 'random_sigma=true'), encoding='utf-8')
        log = run(stage1, path, 'resume')
        assert 'stage1_ntt_batch_resume:' in log
        saved = (path / 'e_resume_5000.save').read_text()
        assert [int(value) for value in re.findall(r'; SIGMA=(\d+);', saved)] == sigmas

    print('Edwards factor endpoints, cofactor handoff and sigma policy passed.')


if __name__ == '__main__':
    main()
