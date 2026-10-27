"""Exercise the real Stage1 instance generator without a CUDA compiler."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*\n', encoding='utf-8')
    fixture = out / 'generate.cmake'
    fixture.write_text('''cmake_minimum_required(VERSION 3.20)
set(CMAKE_SOURCE_DIR "${REPO}")
set(CMAKE_BINARY_DIR "${OUTPUT}")
set(CMAKE_CUDA_COMPILER_VERSION 13.3)
if(NOT DEFINED ECM_CUDA_FULL_BUILD)
    set(ECM_CUDA_FULL_BUILD ON)
endif()
include("${REPO}/tools/build/internal/cuda_sources.cmake")
include("${REPO}/tools/build/internal/cuda_modules.cmake")
ecm_stage1_generate_modules()
file(WRITE "${OUTPUT}/normalized.txt" "${ECM_TIERS}")
''', encoding='utf-8')
    checks = []
    for name, raw, normalized in [
        ('plain', '384,1024', '384,1024'),
        ('spaces', ' 384, 1024 ', '384,1024'),
        ('duplicates', '1024,384,384', '384,1024'),
        ('all', '', ''),
    ]:
        folder = out / name
        folder.mkdir()
        command = ['cmake', '-DREPO=' + ROOT.as_posix(),
                   '-DOUTPUT=' + folder.as_posix(), '-DECM_TIERS=' + raw,
                   '-P', str(fixture)]
        result = subprocess.run(command, capture_output=True, text=True)
        (folder / 'run.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0, (name, result.stderr)
        files = sorted((folder / 'cuda_modules/stage1').glob('*.cu'))
        expected = 125 if not normalized else 8
        assert len(files) == expected, (name, len(files), expected)
        assert (folder / 'normalized.txt').read_text() == normalized, name
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        subprocess.run(command, capture_output=True, text=True, check=True)
        assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}, name
        checks.append(name)
    for name, flags, expected in [
        ('no_param2', ['-DECM_NO_PARAM2=1'], 94),
        ('dev', ['-DECM_CUDA_FULL_BUILD=OFF'], 53),
        ('edwards_only', ['-DECM_TIERS=17408'], 1),
        ('legacy_registers', ['-DECM_TIERS=384,4608,16384',
                              '-DCMAKE_CUDA_COMPILER_VERSION=11.8'], 12),
        ('prac_all', ['-DECM_CUDA_ENABLE_PRAC=ON'], 161),
        ('prac_three', ['-DECM_CUDA_ENABLE_PRAC=ON', '-DECM_TIERS=384,4608,16384'], 19),
        ('prac_dev', ['-DECM_CUDA_ENABLE_PRAC=ON', '-DECM_CUDA_FULL_BUILD=OFF'], 60),
        ('prac_edwards_only', ['-DECM_CUDA_ENABLE_PRAC=ON', '-DECM_TIERS=17408'], 1),
    ]:
        folder = out / name
        folder.mkdir()
        # Let the command select the version instead of the ordinary fixture default.
        script = fixture.read_text().replace('set(CMAKE_CUDA_COMPILER_VERSION 13.3)',
                                            'if(NOT DEFINED CMAKE_CUDA_COMPILER_VERSION)\n'
                                            'set(CMAKE_CUDA_COMPILER_VERSION 13.3)\nendif()')
        case_fixture = folder / 'generate.cmake'
        case_fixture.write_text(script, encoding='utf-8')
        command = ['cmake', '-DREPO=' + ROOT.as_posix(), '-DOUTPUT=' + folder.as_posix(),
                   *flags, '-P', str(case_fixture)]
        subprocess.run(command, capture_output=True, text=True, check=True)
        modules = json.loads((folder / 'cuda_modules/stage1/modules.json').read_text())['modules']
        assert len(modules) == expected, (name, len(modules))
        assert len({m['label'] for m in modules}) == expected
        if name == 'no_param2':
            assert all(m['param'] != 2 for m in modules)
        if name == 'dev':
            assert all(m['curve'] == 'edwards' or m['bits'] <= 1024 for m in modules)
        if name == 'legacy_registers':
            budgets = {384: 56, 4608: 128, 16384: 255}
            assert all(m['maxrregcount'] == (0 if m['curve'] == 'edwards' else budgets[m['bits']])
                       for m in modules)
        if name.startswith('prac_'):
            prac = [m for m in modules if m['family'].startswith('prac_')]
            if name == 'prac_all':
                assert len(prac) == 36 and sum(len(m['modes']) for m in prac) == 161
                assert {m['bits'] for m in prac if m['family'] == 'prac_alternate'} == {2560, 4608}
                assert {m['bits'] for m in prac if m['family'] in ('prac_single', 'prac_constants')} == {4608}
            assert all(m['maxrregcount'] == 0 for m in prac), 'legacy PRAC cap behavior changed'
            # Candidate definitions must remain outside standard/alternate closures.
            for module in prac:
                body = Path(module['source']).read_text()
                assert '#define ECM_PRAC_BITS ' + str(module['bits']) in body
                assert '#define ECM_PRAC_TPI ' + str(module['tpi']) in body
        checks.append(name)
    for name, raw, invalid in [('unknown', '999', '999'),
                               ('empty_item', '384,,1024', ''),
                               ('negative', '-384', '-384'),
                               ('separator', '384;1024', ';')]:
        folder = out / name
        folder.mkdir()
        result = subprocess.run(['cmake', '-DREPO=' + ROOT.as_posix(),
                                 '-DOUTPUT=' + folder.as_posix(), '-DECM_TIERS=' + raw,
                                 '-P', str(fixture)], capture_output=True, text=True)
        log = result.stdout + result.stderr
        (folder / 'run.log').write_text(log, encoding='utf-8')
        assert result.returncode != 0, (name, 'invalid tier accepted')
        assert 'ECM_TIERS' in log and '17408' in log and invalid in log, (name, log)
        assert not list(folder.rglob('*.cu')), name
        checks.append(name)
    powershell = out / 'normalize.ps1'
    powershell.write_text('''param([string]$Repo, [string]$Tiers='')
$ErrorActionPreference='Stop'
. (Join-Path $Repo 'tools/build/internal/cuda_build_profile.ps1')
Write-Output ('canonical='+(Get-EcmStage1Tiers $Tiers))
''', encoding='utf-8-sig')
    for raw in ['384,1024', ' 1024, 384,384 ']:
        result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(powershell), '-Repo', str(ROOT), '-Tiers', raw],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'canonical=384,1024', result.stdout
    checks.append('powershell_same_normalizer')
    # Exercise the actual standalone entry with a native path containing spaces.
    native = out / 'native path with spaces'
    result = subprocess.run(['cmake', '-DECM_STAGE1_MODULES_OUTPUT=' + str(native),
                             '-DECM_TIERS=384', '-DECM_CUDA_FULL_BUILD=ON',
                             '-DCMAKE_CUDA_COMPILER_VERSION=13.3', '-P',
                             str(ROOT / 'tools/build/internal/cuda_modules.cmake')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    catalog = json.loads((native / 'cuda_modules/stage1/modules.json').read_text())
    assert len(catalog['modules']) == 4
    assert all(Path(m['source']).is_file() for m in catalog['modules'])
    assert all('\\' not in m['source'] for m in catalog['modules'])
    checks.append('native_output_path_with_spaces')
    result = dict(complete=True, checks=checks, cuda_compiles=0, gpu_runs=0)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
