"""Compile and exercise the shared bounded D search and runtime protections."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    gmp = ROOT / 'third_party/gmp-zen3/dist'; exe = out / 'fixture.exe'
    command = out / 'compile.cmd'
    command.write_text('@echo off\ncall "' + str(args.vcvars) + '" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"' + str(gmp / 'include') + '" "' +
        str(ROOT / 'tools/test/stage2_d_search_fixture.cpp') + '" /Fe:"' + str(exe) +
        '" /Fo:"' + str(out / 'fixture.obj') + '" /link /LIBPATH:"' + str(gmp / 'lib') + '" gmp.lib\n', encoding='utf-8')
    compiled = subprocess.run(['cmd', '/c', str(command)], capture_output=True, text=True, errors='replace', timeout=120)
    (out / 'compile.log').write_text(compiled.stdout + compiled.stderr, encoding='utf-8')
    if compiled.returncode:
        raise RuntimeError(compiled.stdout + compiled.stderr)
    for dll in (gmp / 'bin').glob('*.dll'):
        shutil.copy2(dll, out / dll.name)
    result = subprocess.run([str(exe)], capture_output=True, text=True, errors='replace', timeout=120)
    (out / 'fixture.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    report = json.loads(result.stdout)
    sources = ['src/core/ecm_stage2_d_search.h', 'src/core/ecm_stage2_runtime_tune.h', 'src/core/ecm_stage2_runtime_cache.h',
               'src/core/ecm_stage2_tune_portable.h', 'tools/test/stage2_d_search_fixture.cpp',
               'tools/test/test_stage2_d_search.py']
    report['sources'] = {}
    for name in sources:
        src = ROOT / name; dst = out / 'sources' / name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); report['sources'][name] = hashlib.sha256(src.read_bytes()).hexdigest()
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
