"""Compare real Stage1 math against a frozen binary across curve backends."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def points(path):
    records = []
    # Legacy WHO metadata may use the host code page. Math fields are ASCII;
    # retain the raw save and exclude WHO from the comparison.
    for line in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        fields = dict(re.findall(r'(\w+)\s*=\s*([^;]+)', line))
        if 'METHOD' in fields:
            records.append(tuple(fields.get(key) for key in ('METHOD', 'PARAM', 'SIGMA', 'B1', 'N', 'X', 'STATUS')))
    return sorted(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'exe', 'runtime', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bits', type=int, default=521)
    parser.add_argument('--sigma', type=int, default=1000)
    parser.add_argument('--modes', default='cpu_mont,cpu_ed,gpu_mont0,gpu_mont2,gpu_mont3,gpu_ed')
    parser.add_argument('--resources', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    modulus = f'2^{args.bits}-1'
    report = dict(complete=False, device=args.device, modulus=modulus, b1=1000,
                  curves=8, first_sigma=args.sigma, cases=[],
                  binaries={name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in [('baseline', args.baseline), ('current', args.exe)]})
    for mode in args.modes.split(','):
        actual = {}
        for name, source in [('baseline', args.baseline), ('current', args.exe)]:
            work = out/mode/name
            work.mkdir(parents=True)
            exe = work/source.name
            shutil.copy2(source, exe)
            for library in source.parent.glob('*.dll'):
                shutil.copy2(library, work/library.name)
            shutil.copy2(args.runtime, work/'gmp-10.dll')
            if args.resources:
                shutil.copytree(args.resources, work/'opencl_resources')
            target, curve = mode.split('_', 1)
            edwards = curve == 'ed'
            command = [str(exe), '--curve', 'ed' if edwards else 'mont', '--target', target,
                       '--exp-cache', 'off', '--ckpt', '0', '--sigma', str(args.sigma),
                       '-gpucurves', '8', '--tmp-dir', str(work), '-save', str(work/'point.save')]
            if target == 'gpu':
                command += ['-d', str(args.device)]
                if not edwards:
                    command += ['--gpu-param', curve[-1]]
            else:
                command += ['--impl', 'scalar', '--stage1-threads', '1']
            command += ['1000']
            start = time.perf_counter()
            result = subprocess.run(command, cwd=work, input=modulus+'\n', capture_output=True,
                                    encoding='utf8', errors='replace', timeout=180)
            elapsed = time.perf_counter()-start
            (work/'run.log').write_text(result.stdout+result.stderr, encoding='utf8')
            assert result.returncode == 0, (mode, name, result.returncode, (result.stdout+result.stderr)[-2500:])
            actual[name] = points(work/'point.save')
            assert len(actual[name]) == 8, (mode, name, len(actual[name]))
            report['cases'].append(dict(mode=mode, binary=name, seconds=elapsed,
                                        command=command, records=len(actual[name])))
        assert actual['baseline'] == actual['current'], ('mathematical save differs', mode)
        print('PASS', mode, flush=True)
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
