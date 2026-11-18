"""Check Stage2 device dispatch and compare CPU/CUDA factors in one executable."""
import argparse
import json
from pathlib import Path
import subprocess

from check_opencl_stage2_worker import records, run, save_line
from test_cpu_stage2_queue import HIT


def setup(folder, gpu, device=None, worker=None):
    folder.mkdir()
    config = (
        'config_version = 3\n[shared]\nworktodo_file = "worktodo.txt"\n'
        'finished_file = "finished.txt"\nlog_file = ""\ntune_file = ""\n'
        f'[stage2]\nsave_dir = "."\ngpu_device = {gpu}\nd = 210\n'
        'gpu_backend = "cuda"\ncpu.backend = "plain"\n'
        'cuda.curve_concurrency = "serial"\ntune.runtime_enabled = false\n'
        'results_file = "results.jsonl"\n'
    )
    if device:
        config += f'device = "{device}"\n'
    if worker:
        config += f'[worker.3]\nstage2.device = "{worker}"\n'
    (folder / 'ecm.toml').write_text(config, encoding='utf-8')
    (folder / 'hit.save').write_text(HIT, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
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
    setup(mont, args.gpu, 'auto')
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
    print('Verified default CPU, per-worker CPU/CUDA and factor 10007.')


if __name__ == '__main__':
    main()
