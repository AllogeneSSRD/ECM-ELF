"""Independent endpoint gates for the five CUDA Stage1 execution paths.

Use a local build with PRAC enabled and containers covering M127 and M521.
This exercises the production CLI, task, curve workflow, runtime and save codec.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from test_cuda_prac import rows, run
from test_edwards_gpu_pipeline import records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    exe = args.exe.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  device=args.device, complete=False, cases=[])
    def passed(name, count):
        report['cases'].append(dict(name=name, curves=count))
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print('PASS', name, count, flush=True)
    for bits in (127, 521):
        for torsion in ('lcm', 'choose12'):
            settings = (f'(2^{bits}-1)', 1000, 4, 20260922, torsion)
            cpu = out/f'm{bits}_{torsion}'/'cpu'
            run(exe, cpu, *settings, 'cpu', args.device)
            expected = rows(cpu)
            assert len(expected) == 4
            for algorithm in ('ladder', 'resident', 'prac'):
                folder = cpu.parent/algorithm
                run(exe, folder, *settings, algorithm, args.device, registers=0)
                assert rows(folder) == expected, (bits, torsion, algorithm)
                passed(f'mont_{bits}_{torsion}_{algorithm}', 4)
        points = None
        for algorithm in ('binary', 'naf'):
            folder = out/f'ed_{bits}_{algorithm}'
            folder.mkdir()
            save = folder/'points.save'
            env = dict(os.environ, ECM_EDWARDS_STAGE1_ALGO=algorithm)
            env.pop('ECM_EDWARDS_DBL_ALGO', None)
            proc = subprocess.run([str(exe), '--curve', 'ed', '--target', 'gpu',
                '-d', str(args.device), '-sigma', '20260922', '-gpucurves', '4',
                '--tmp-dir', str(folder/'tmp'), '-save', str(save), '1000', '0'],
                cwd=folder, env=env, input=f'(2^{bits}-1)\n'.encode(),
                capture_output=True, timeout=120)
            (folder/'run.log').write_bytes(proc.stdout+proc.stderr)
            assert proc.returncode == 0, (algorithm, proc.stdout[-2000:], proc.stderr)
            actual = records(save)  # Independently recompute each full 48*lcm point.
            assert len(actual) == 4
            if points is not None:
                assert actual == points
            points = actual
            passed(f'ed_{bits}_{algorithm}', 4)
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
