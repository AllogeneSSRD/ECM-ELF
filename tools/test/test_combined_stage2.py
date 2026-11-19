"""Check Stage2 device dispatch and compare CPU/CUDA factors in one executable."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from check_opencl_stage2_worker import records, run, save_line, stop_after_first_start
from test_cpu_stage2_queue import HIT, POINTS


def setup(folder, gpu, device=None, worker=None):
    folder.mkdir()
    config = (
        'config_version = 4\n[shared]\nworktodo_file = "worktodo.txt"\n'
        'finished_file = "finished.txt"\nlog_file = ""\ntune_file = ""\n'
        f'[stage2]\nstage2_save_dir = "."\nstage2_gpu_device = {gpu}\nd = 210\n'
        'stage2_gpu_backend = "cuda"\ncpu_backend = "plain"\n'
        'curve_concurrency = "serial"\ntune_runtime_enabled = false\n'
        'stage2_results_file = "results.jsonl"\n'
    )
    if device:
        config += f'device = "{device}"\n'
    if worker:
        config += f'[worker.3]\ndevice = "{worker}"\n'
    (folder / 'ecm.toml').write_text(config, encoding='utf-8')
    (folder / 'hit.save').write_text(HIT, encoding='utf-8')


def policy(folder, rule):
    with (folder / 'ecm.toml').open('a', encoding='utf-8') as config:
        config.write(f'stage2_algorithm = "{rule}"\n')


def edwards_width_fixture(bits):
    # Used for dry-run policy boundaries only. The mixed queue uses real Stage1 saves.
    n = f'{(1 << bits) - 1:x}'
    checksum = 14695981039346656037
    for byte in f'ECM-EDWARDS-AM|1|48LCM|{n}|1|26|20|POINT'.encode():
        checksum = ((checksum ^ byte) * 1099511628211) % (1 << 64)
    return ('METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM; SIGMA=26; B1=20; '
            f'N=0x{n}; X=0x1; STATUS=POINT; CHECKSUM={checksum};\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--wide-edwards-save', type=Path)
    parser.add_argument('--opencl-exe', type=Path)
    parser.add_argument('--opencl-device', type=int, default=1)
    args = parser.parse_args()
    exe, out = args.exe.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output([exe, '--gui-queue-protocol'], text=True).strip() == 'ECM_QUEUE_OWNER=1'
    assert subprocess.check_output([exe, '--version'], text=True).startswith('ECM-ELF ')

    for name, global_device in (('default', None), ('auto', 'AuTO')):
        case = out / name
        setup(case, args.gpu, global_device)
        assert 'CPU Edwards Stage2' in run(exe, case, ['--help'])

    invalid = out / 'invalid'
    setup(invalid, args.gpu, 'unknown')
    assert 'stage2_execution_device must be [auto|cpu|gpu]' in run(exe, invalid, ['--help'], expect=2)

    override = out / 'cli'
    setup(override, args.gpu, 'gPu')
    assert 'CPU Edwards Stage2' in run(exe, override, ['--target', 'cPu', '--help'])
    assert 'GPU worker:' in run(exe, override, ['--target', 'GPU', '--help'], label='gpu_help')

    mont = out / 'montgomery_on_cpu'
    setup(mont, args.gpu, 'cpu')
    (mont / 'mont.save').write_text(save_line((1 << 127) - 1), encoding='utf-8')
    task = 'ECMSTAGE2N="mont.save",300,0,1'
    queue = mont / 'worktodo.txt'
    queue.write_text(task + '\n', encoding='utf-8')
    log = run(exe, mont, ['--once'], expect=1)
    assert 'CPU Stage2 requires Edwards saves' in log and not records(mont), log
    assert queue.read_text(encoding='utf-8').strip() == task

    for device, global_device in (('cPu', 'GPU'), ('gPu', 'CPU')):
        case = out / device.lower()
        setup(case, args.gpu, global_device, device)
        task = 'ECMSTAGE2N=N/A,"hit.save",40000,0,1'
        (case / 'worktodo.txt').write_text(f'[Worker #3]\n{task}\n', encoding='utf-8')
        log = run(exe, case, ['--worker', '3', '--once', '--gui-events'])
        if device.lower() == 'cpu':
            assert 'CPU Edwards Stage2 backend=plain' in log, log
        else:
            assert 'curve_start:' in log and 'CPU Edwards Stage2' not in log, log
        submission = json.loads((case / 'results.json_3.txt').read_text(encoding='utf-8'))
        assert submission['stage'] == 2 and submission['factors'] == ['10007']
        assert records(case)[0]['Edwards']['sigma'] == 17
        assert task not in (case / 'worktodo.txt').read_text(encoding='utf-8')
        assert task in (case / 'finished.txt').read_text(encoding='utf-8')

    for bits, expected in ((4095, 'cuda'), (4096, 'gwnum'), (16384, 'cuda'), (16385, 'gwnum')):
        case = out / f'bits_{bits}'
        setup(case, args.gpu)
        (case / 'width.save').write_text(edwards_width_fixture(bits), encoding='utf-8')
        if bits >= 16384:
            policy(case, 'CUDA[], GWNUM[4096：]')
        log = run(exe, case, ['--save', 'width.save', '--b2', '1000', '--dry-run'])
        assert f'bits={bits} selected={expected}' in log, log
        if bits == 4096:
            log = run(exe, case, ['--save', 'width.save', '--b2', '1000', '--dry-run', '--backend', 'plain'],
                      label='explicit_backend')
            assert 'selected=' not in log and 'validated 1 curves' in log, log

    blocked = out / 'no_usable_rule'
    setup(blocked, args.gpu)
    policy(blocked, 'cuda[]')
    (blocked / 'wide.save').write_text(edwards_width_fixture(16385), encoding='utf-8')
    task = 'ECMSTAGE2N="wide.save",1000,0,1\n'
    queue = blocked / 'worktodo.txt'
    queue.write_text(task, encoding='utf-8')
    log = run(exe, blocked, [], expect=2)
    assert 'no usable rule for 16385-bit Edwards' in log, log
    assert queue.read_text(encoding='utf-8') == task and not records(blocked)

    bad_policy = out / 'invalid_rule'
    setup(bad_policy, args.gpu)
    policy(bad_policy, 'gwnum[4096:4096]')
    assert 'invalid stage2_algorithm' in run(exe, bad_policy, ['--help'], expect=2)

    case = out / 'worker_policy'
    setup(case, args.gpu)
    policy(case, 'cpu-plain')
    with (case / 'ecm.toml').open('a', encoding='utf-8') as config:
        config.write('[worker.3]\nstage2_algorithm = "cuda[:1024], gwnum[]"\n')
    command = ['--save', 'hit.save', '--b2', '1000', '--dry-run']
    assert 'selected=cpu-plain' in run(exe, case, command)
    assert 'selected=cuda' in run(exe, case, ['--worker', '3', *command], label='worker_3')

    if args.wide_edwards_save:
        for once in (False, True):
            case = out / ('mixed_once' if once else 'mixed')
            setup(case, args.gpu)
            shutil.copy2(args.wide_edwards_save, case / 'wide.save')
            (case / 'small.save').write_text(POINTS, encoding='utf-8')
            narrow = 'ECMSTAGE2=1,2,127,-1,"small.save",40000,0,2'
            wide = 'ECMSTAGE2N="wide.save",1000,0,1'
            (case / 'worktodo.txt').write_text(f'{narrow}\n{wide}\n', encoding='utf-8')
            if once:
                stop_after_first_start(exe, case, ['--gui-events'])
                assert len(records(case)) == 1 and wide in (case / 'worktodo.txt').read_text()
            log = run(exe, case, ['--gui-events', *(['--once'] if once else [])])
            assert 'bits=127 selected=cuda' in log, log
            if once:
                assert 'selected=gwnum' not in log and wide in (case / 'worktodo.txt').read_text(), log
                log = run(exe, case, ['--gui-events', '--once'], label='next_task')
            assert 'selected=gwnum' in log and 'CPU Edwards Stage2 backend=gwpoly' in log, log
            assert len(records(case)) == 3, (case, records(case))
            assert 'ECMSTAGE2' not in (case / 'worktodo.txt').read_text()
            assert narrow in (case / 'finished.txt').read_text() and wide in (case / 'finished.txt').read_text()

    if args.opencl_exe:
        portable = out / 'opencl_bundle'
        portable.mkdir()
        shutil.copy2(exe, portable)
        shutil.copy2(exe.parent / 'gmp-10.dll', portable)
        shutil.copy2(args.opencl_exe, portable)
        shutil.copytree(args.opencl_exe.parent / 'opencl_resources', portable / 'opencl_resources')
        case = out / 'opencl'
        setup(case, args.opencl_device)
        policy(case, 'opencl[128:], cpu-plain[]')
        task = 'ECMSTAGE2N="hit.save",40000,0,1'
        (case / 'small.save').write_text(POINTS, encoding='utf-8')
        small = 'ECMSTAGE2N="small.save",2000,0,1'
        (case / 'worktodo.txt').write_text(f'{task}\n{small}\n', encoding='utf-8')
        log = run(portable / exe.name, case, ['--gui-events'])
        assert 'selected=opencl' in log and 'selected=cpu-plain' in log, log
        assert 'tasks_this_run=2' in log and len(records(case)) == 2, log
        assert task in (case / 'finished.txt').read_text()
    print('Verified Stage2 policy boundaries, per-worker CPU/CUDA and factor 10007.')


if __name__ == '__main__':
    main()
