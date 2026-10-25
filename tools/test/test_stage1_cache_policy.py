"""Exercise normalized cache policy through the real Stage1 CLI.

N=1111 and Montgomery sigma=26 expose a construction factor after exponent
preparation, keeping the B1>=cache-minimum regression small and deterministic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'runtime', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for mode in ('cli_off', 'config_off', 'explicit_directory', 'optional_write_failure'):
        work = out / mode
        work.mkdir()
        exe = work / 'ecm_cuda.exe'
        shutil.copy2(args.exe, exe)
        shutil.copy2(args.runtime, work / 'gmp-10.dll')
        ini = work / 'ecm.toml'
        ini.write_text('[shared]\nuser=""\ncomputer=""\n' +
                       ('[stage1]\nexp_cache="off"\n' if mode == 'config_off' else ''),
                       encoding='utf-8')
        command = [str(exe), '--config', str(ini), '--curve', 'mont', '--target', 'cpu',
                   '--impl', 'scalar', '--stage1-threads', '1',
                   '-sigma', '26', '-gpucurves', '1', '--tmp-dir', str(work)]
        if mode == 'cli_off':
            command += ['--exp-cache', 'off']
        elif mode in ('explicit_directory', 'optional_write_failure'):
            cache = work / 'cache'
            if mode == 'optional_write_failure':
                cache.write_text('a file cannot hold cache entries', encoding='utf-8')
            command += ['--exp-cache', str(cache)]
        command += ['1000000', '1000000']
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=work, input=b'1111\n', capture_output=True, timeout=60)
        (work / 'run.log').write_bytes(proc.stdout + proc.stderr)
        assert proc.returncode == 0, (mode, 'inspect run.log')
        rows = list(work.rglob('ecm_B1_1000000_t*_v1.bin'))
        if mode == 'explicit_directory':
            assert rows and all(p.parent == cache for p in rows), rows
        else:
            assert not rows, (mode, rows)
        cases.append(dict(name=mode, seconds=time.perf_counter() - start,
                          cached_files=len(rows), returncode=proc.returncode))
    result = dict(complete=True, binary_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                  cases=cases, b1=1000000, sigma=26, modulus=1111, gpu_runs=0)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
