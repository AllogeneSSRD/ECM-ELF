"""Exercise real CLI sigma policies and reject invalid batches before execution.

RNG endpoints are covered deterministically by the host contract. These cases
check that actual CPU/CUDA/OpenCL workflows apply the intended policies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from test_ecm_domain_stage1 import points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--opencl', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, modulus='2^521-1', b1=1000, curves=8,
                  device=args.device, random=[], rejected=[], u64_endpoints=[], binaries={})
    for name in ('cuda', 'opencl'):
        path = getattr(args, name).resolve()
        report['binaries'][name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def run(name, backend, target, curve, param=None, fixed=None, count=8, algorithm='ladder'):
        work = out/name
        work.mkdir()
        source = getattr(args, backend).resolve()
        exe = work/source.name
        shutil.copy2(source, exe)
        for library in source.parent.glob('*.dll'):
            shutil.copy2(library, work/library.name)
        shutil.copy2(args.runtime, work/'gmp-10.dll')
        if backend == 'opencl':
            shutil.copytree(args.resources, work/'opencl_resources')
        command = [str(exe), '--curve', curve, '--target', target,
                   '--exp-cache', 'off', '--ckpt', '0', '--tmp-dir', str(work/'tmp'),
                   '-gpucurves', str(count), '-save', str(work/'point.save')]
        if target == 'cpu':
            command += ['--impl', 'scalar', '--stage1-threads', '1']
        else:
            command += ['-d', str(args.device)]
            if curve == 'mont':
                command += ['--gpu-param', str(param)]
        if fixed is not None:
            command += ['--sigma', str(fixed)]
        command += ['1000']
        env = dict(os.environ, ECM_GPU_STAGE1_ALGO=algorithm, ECM_EDWARDS_STAGE1_ALGO='binary')
        env.pop('ECM_GPU_STAGE1_SAMPLE_SECONDS', None)
        process = subprocess.run(command, cwd=work, env=env, input='2^521-1\n',
                                 capture_output=True, encoding='utf-8', errors='replace', timeout=120)
        text = process.stdout + process.stderr
        (work/'run.log').write_text(text, encoding='utf-8')
        return work, process.returncode, text

    cases = [
        ('cpu_mont', 'cuda', 'cpu', 'mont', 0, 'independent53'),
        ('cpu_ed', 'cuda', 'cpu', 'ed', None, 'independent53'),
        ('cuda_mont0', 'cuda', 'gpu', 'mont', 0, 'contiguous53'),
        ('cuda_mont2', 'cuda', 'gpu', 'mont', 2, 'contiguous32'),
        ('cuda_mont3', 'cuda', 'gpu', 'mont', 3, 'contiguous32'),
        ('cuda_ed', 'cuda', 'gpu', 'ed', None, 'independent53'),
        ('opencl_mont3', 'opencl', 'gpu', 'mont', 3, 'contiguous32'),
    ]
    for name, backend, target, curve, param, policy in cases:
        work, code, text = run(name, backend, target, curve, param)
        assert code == 0, (name, code, text[-2500:])
        sigmas = [int(s) for s in re.findall(r'\bSIGMA=(\d+);',
                  (work/'point.save').read_text(encoding='utf-8', errors='replace'))]
        assert len(sigmas) == 8, (name, sigmas)
        if policy == 'independent53':
            assert all(5 < s < 2**53 for s in sigmas), (name, sigmas)
        else:
            assert sigmas == list(range(sigmas[0], sigmas[0] + 8)), (name, sigmas)
            if policy == 'contiguous53':
                assert 5 < sigmas[0] < 2**53, (name, sigmas)
            else:
                assert sigmas[0] >= (2 if param == 2 else 1) and sigmas[-1] < 2**32
        report['random'].append(dict(name=name, policy=policy, sigmas=sigmas))
        print('PASS random', name, flush=True)

    expected = None
    for algorithm in ('cpu', 'ladder', 'resident', 'prac'):
        work, code, text = run('u64_' + algorithm, 'cuda',
                               'cpu' if algorithm == 'cpu' else 'gpu', 'mont',
                               0, 2**64 - 4, 4,
                               algorithm='ladder' if algorithm == 'cpu' else algorithm)
        assert code == 0, (algorithm, code, text[-2500:])
        actual = points(work/'point.save')
        assert len(actual) == 4
        if expected is None:
            expected = actual
        else:
            assert actual == expected, ('last u64 batch differs', algorithm)
        report['u64_endpoints'].append(algorithm)
        print('PASS u64 endpoint', algorithm, flush=True)

    for name, backend, target, curve, param, first, count, message in [
        ('invalid_param2_min', 'cuda', 'gpu', 'mont', 2, 1, 8, '>= 2'),
        ('invalid_cuda_u32', 'cuda', 'gpu', 'mont', 3, 2**32-7, 8, '<= 2^32'),
        ('invalid_opencl_u32', 'opencl', 'gpu', 'mont', 3, 2**32-7, 8, '<= 2^32'),
        ('invalid_mont_u64', 'cuda', 'cpu', 'mont', 0, 2**64-1, 2, '2^64-1'),
        ('invalid_ed_u64', 'cuda', 'gpu', 'ed', None, 2**64-1, 2, '2^64-1'),
    ]:
        work, code, text = run(name, backend, target, curve, param, first, count)
        assert code != 0 and message in text, (name, code, text)
        assert not (work/'point.save').exists(), ('invalid input reached execution', name)
        report['rejected'].append(name)
        print('PASS rejected', name, flush=True)
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
