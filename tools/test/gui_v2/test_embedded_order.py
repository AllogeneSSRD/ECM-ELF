"""Exercise GUI self-hosted analysis from a relocated directory without helper EXEs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.exe.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    portable = out / '中文-portable'
    portable.mkdir()
    exe = portable / 'ecm_gui_v2.exe'
    shutil.copy2(source, exe)
    for name in ('gmp-10.dll', 'libpari.dll'):
        shutil.copy2(source.parent / name, portable / name)
    shutil.copytree(source.parent / 'pari-data', portable / 'pari-data')
    assert not (portable / 'ecm_group_order_ui.exe').exists()
    assert not (portable / 'ecm_diagnostics_host.exe').exists()
    env = os.environ.copy()
    env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    gp_only = out / 'gp-only'
    gp_only.mkdir()
    shutil.copy2(source, gp_only / exe.name)
    version = subprocess.run([gp_only / exe.name, '--version'], env=env, capture_output=True, timeout=10)
    assert version.returncode == 0 and b'ECM-ELF GUI v2' in version.stdout, version.stderr
    cases = [('small', '1009', '26', '996', '249'),
             ('user', '3436193553215096358758503', '1955774944',
              '3436193553215760556775100', '286349462767980046397925'),
             ('p192', '6277101735386680763835789423207666416083908700390324961279', '26',
              '6277101735386680763835789423165887359574653668381181441896', None)]
    results = []
    for name, prime, sigma, group, point in cases:
        request, response = out / (name + '.in.json'), out / (name + '.out.json')
        request.write_text(json.dumps(dict(factor=prime, sigma=sigma, profile=0, backend=2, gp='')), encoding='utf-8')
        run = subprocess.run([exe, '--group-order', request, response], cwd=out, env=env,
                             capture_output=True, timeout=12)
        (out / (name + '.log')).write_bytes(run.stdout + run.stderr)
        assert run.returncode == 0 and response.is_file(), (name, run.returncode, run.stderr)
        value = json.loads(response.read_text())
        assert value['status'] == 'complete' and value['native_engine'] == 'pari', value
        assert value['group_order'] == group and (point is None or value['point_order'] == point), value
        assert value['all_proven'] and value['point_minimality'], value
        results.append(name)
    (out / 'result.json').write_text(json.dumps(dict(passed=True, cases=results,
        gui_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), helper_executables=False), indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=results, helper_executables=False)))


if __name__ == '__main__':
    main()
