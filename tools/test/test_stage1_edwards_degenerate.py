"""Exercise lane isolation and factor publication with real Edwards Stage1 workers."""

import argparse
import json
from pathlib import Path
import subprocess


def run_worker(worker, path, options, target, known, sigma, count):
    path.mkdir(parents=True)
    save = path / 'e_case_2.save'
    queue = path / 'worktodo.txt'
    queue.write_text(
        f'ECMSTAGE1={target},"{save.as_posix()}",0,0,{count}{known}\n', encoding='utf-8')
    config = path / 'ecm.toml'
    config.write_text(
        'config_version=4\n[shared]\nworktodo_file="worktodo.txt"\nfinished_file="finished.txt"\n'
        'log_file=""\n[stage1]\nhandoff_mode="off"\nexponent_cache_dir="off"\n'
        f'stage1_save_dir="{path.as_posix()}"\nsigma={sigma}\n{options}\n', encoding='utf-8')
    proc = subprocess.run([worker, '--config', config], cwd=path, capture_output=True, timeout=120)
    raw = proc.stdout + proc.stderr
    (path / 'run.log').write_bytes(raw)
    log = raw.decode('utf-8', 'replace')
    assert proc.returncode == 0, log[-6500:]
    assert not queue.read_text().strip(), log
    rows = {}
    for line in save.read_text().splitlines():
        fields = dict(field.strip().split('=', 1) for field in line.split(';') if field.strip())
        # Compare canonical outcomes. Different implementations may emit
        # different projective representatives of the same cofactor endpoint.
        key = int(fields['SIGMA'])
        assert key not in rows
        rows[key] = int(fields['N'], 0), int(fields['X'], 0), fields['STATUS']
        if 'QX' in fields:
            cofactor = rows[key][0] // rows[key][1]
            assert int(fields['QX'], 0) % cofactor or int(fields['QZ'], 0) % cofactor
    submissions = path / 'results.json.txt'
    published = {}
    if submissions.exists():
        for line in submissions.read_text().splitlines():
            outcome = json.loads(line)
            if outcome['status'] == 'F':
                key = int(outcome['Edwards']['sigma'])
                assert key not in published
                published[key] = int(outcome['factors'][0])
    assert published == {key: value for key, (_, value, status) in rows.items() if status == 'FACTOR'}
    return log, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--opencl', type=Path)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--opencl-device', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', action='append',
                        choices=('scalar', 'ifma', 'cgbn', 'cuda-thread', 'opencl', 'ntt-xonly', 'ntt-naf'))
    args = parser.parse_args()
    if args.backend and 'opencl' in args.backend and not args.opencl:
        parser.error('--backend opencl requires --opencl')
    stage1 = args.stage1.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True)
    gpu = f'target="gpu"\nstage1_gpu_device={args.device}\ngpu_init_threads=3\n'
    cpu = 'target="cpu"\nstage1_cpu_threads=1\nedwards_naf_window_bits=3\n'
    backends = {
        'scalar': (stage1, cpu + 'cpu_implementation="scalar"'),
        'ifma': (stage1, cpu + 'cpu_implementation="simd"'),
        'cgbn': (stage1, gpu + 'gpu_algorithm="cgbn"'),
        'cuda-thread': (stage1, gpu + 'gpu_algorithm="thread"'),
        'ntt-xonly': (stage1, gpu + 'gpu_algorithm="ntt"\nedwards_naf_window_bits=0'),
        'ntt-naf': (stage1, gpu + 'gpu_algorithm="ntt"\nedwards_naf_window_bits=6'),
    }
    if args.opencl:
        backends['opencl'] = (args.opencl.resolve(), 'target="gpu"\nstage1_gpu_backend="opencl"\n'
                              f'stage1_gpu_device={args.opencl_device}\ngpu_algorithm="cgbn"')
    for name in args.backend or backends:
        worker, options = backends[name]
        if name.startswith('ntt'):
            # The NTT device computes modulo the M76 carrier. The curve
            # constructor sees the actual target 229*457 after division.
            n = 229 * 457
            known, remainder = divmod((1 << 76) - 1, n)
            assert remainder == 0
            target, known = '1,2,76,-1', f',"{known}"'
            factor_sigma, factor = 275, 229
            omitted, keep = (276,), (275, 277)
        else:
            n = 137 * 1009
            target, known = '1,2,18,-123911', ''
            factor_sigma, factor = 677, 137
            omitted, keep = (679, 680), (677, 678, 681)

        def run(label, first, count):
            return run_worker(worker, out / name / label, options, target, known, first, count)

        log, mixed = run('mixed', keep[0], keep[-1] - keep[0] + 1)
        assert set(mixed) == set(keep)
        assert mixed[factor_sigma] == (n, factor, 'FACTOR')
        for sigma in omitted:
            assert f'sigma={sigma} gcd=N' in log, log
        for sigma in keep:
            _, single = run(f'sigma_{sigma}', sigma, 1)
            assert single == {sigma: mixed[sigma]}
        _, empty = run('all_degenerate', omitted[0], len(omitted))
        assert not empty
        if name == 'ifma':
            # Remove the construction-factor lane to exercise the packed
            # path directly, including an inactive lane before the first valid one.
            simd_log, simd = run('packed', 679, 3)
            assert 'field           :' in simd_log, simd_log
            assert simd == {681: mixed[681]}
        print(f'{name}: isolated {omitted}, published factor {factor}, and matched singleton results',
              flush=True)


if __name__ == '__main__':
    main()
