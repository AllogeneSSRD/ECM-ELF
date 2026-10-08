"""Prepare / collect administrator NCU for a real generic16384 S4 launch.

Skip the first S4 arithmetic selftest and select the next exact NW256/plain
kernel. The saved input and complete output / check coverage are bound to a
completed unprofiled reference matrix. Replay timing is diagnostic only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys, 'set_int_max_str_digits'): sys.set_int_max_str_digits(0)
HELPER = ROOT / 'tools/bench/bench_stage2_production.py'
spec = importlib.util.spec_from_file_location('frozen_stage2', HELPER)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
sha, read, fields = helper.sha, helper.read, helper.fields
KERNEL = '_Z16s4_reduce_kernelILi256ELb0EEvPKyyiyyyS1_yiiS1_yPyyS2_S2_iiyyyi'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--ncu', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe'))
    a = p.parse_args(); exe = a.exe.resolve(); out = a.output.resolve()
    ref = read(a.reference)
    if not ref['complete']: raise ValueError('unprofiled reference is incomplete')
    case = next(c for c in ref['cases'] if c['name'] == 'generic16384')
    identity = helper.freeze(exe)
    key = next((k for k, v in ref['identity'].items() if v['binary_sha256'] == identity['binary_sha256']), None)
    if key is None: raise ValueError('binary not in the reference matrix')
    expected = next(r for r in ref['runs'] if r['case'] == case['name'] and r['key'] == key)
    saved = Path(case['save'])
    if sha(saved) != case['save_sha256']: raise ValueError('reference save changed')
    for name, want in identity['sources'].items():
        if sha(exe.parent / 'sources' / name) != want: raise ValueError('frozen source changed')
    if a.collect_only:
        command = read(out / 'command.json')
        if command['identity'] != identity or command['reference_sha256'] != sha(a.reference):
            raise ValueError('capture/reference identity differs')
        for name, field in (('collector_capture.py', 'tool_sha256'), ('helper_capture.py', 'helper_sha256')):
            if sha(out / name) != command[field]:
                raise ValueError('capture tool snapshot differs: ' + name)
        if (out / 'exit.txt').read_text(encoding='utf-8-sig').strip() != '0': raise ValueError('capture failed')
        with (out / 'metrics.csv').open('wb') as stream:
            subprocess.run([str(a.ncu), '--rename-kernels', '0', '--import', str(out / 'trace.ncu-rep'),
                '--csv', '--page', 'raw'], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
        rows = list(csv.DictReader((out / 'metrics.csv').read_text(encoding='utf-8-sig').splitlines()))
        samples = [row for row in rows if row.get('ID', '').isdigit()]
        if len(samples) != 1:
            raise ValueError('capture must contain exactly one real kernel')
        sample = samples[0]
        if (sample['Grid Size'] != '(5, 1, 1)' or sample['Block Size'] != '(128, 1, 1)'
                or sample['Device'] != '1'
                or not re.search(r's4_reduce_kernel<(?:\(int\))?256, (?:\(bool\))?(?:0|false)>', sample['Kernel Name'])):
            raise ValueError('captured kernel/geometry/device differs from the actual reference launch')
        text = (out / 'engine.log').read_text(encoding='utf-8')
        result = [json.loads(s) for s in (out / 'results.jsonl').read_text().splitlines()]
        if len(result) != 1 or result[0]['bad_factors'] or result[0]['N_hex'] != expected['result']['N_hex']:
            raise ValueError('profiled arithmetic input/result differs')
        if result[0]['factors'] != expected['result']['factors'] or fields(text, 'descent_values') != expected['leaf']:
            raise ValueError('complete output fingerprint differs')
        coverage = fields(text, 's4_multiply_stats')
        for k in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases', 'gmp_checked', 'full_checks'):
            if coverage[k] != expected['coverage'][k]: raise ValueError('check coverage differs: ' + k)
        for token in ('gmp_check_bad=0', 'gmp_selftest_bad=0', 'pending=0', 's4_div_check: cases=800 bad=0'):
            if token not in text: raise ValueError('mandatory check missing')
        (out / 'collector_collect.py').write_bytes(Path(__file__).read_bytes())
        (out / 'summary.json').write_text(json.dumps(dict(command=command, result=result[0], leaf=expected['leaf'],
            coverage=coverage, report_sha256=sha(out / 'trace.ncu-rep'), metrics_sha256=sha(out / 'metrics.csv'),
            captured_kernel={k: sample[k] for k in ('ID', 'Kernel Name', 'Grid Size', 'Block Size', 'Device')},
            collector_sha256=sha(__file__), scope='One real NW256 S4 launch, after its startup selftest; replay changes execution, not a full Stage2 performance sample.'), indent=2) + '\n')
        print('Collected', out); return
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh output directory')
    (out / 'collector_capture.py').write_bytes(Path(__file__).read_bytes())
    (out / 'helper_capture.py').write_bytes(HELPER.read_bytes())
    ini = out / 'manual.ini'; ini.write_text('device=1\n')
    app = [str(exe), '--ini', str(ini), '--save', str(saved), '--device', '1', '--b2', str(case['B2']),
           '--d', str(case['D']), '--arena-mb', '6300', '--factor-only', '--log', str(out / 'engine.log'),
           '--results', str(out / 'results.jsonl')]
    if read(exe.parent / 'build_manifest.json').get('engine') == 'production': app += ['--log-level', 'debug']
    env = dict(NTT_D_MODEL='0', NTT_NO_PROGRESS='1', NTT_GIANT_SEED_PAIR='1', NTT_GIANT_BASE_CPU='0',
               NTT_GIANT_CHAIN_BLOCK='64', NTT_GIANT_CHAIN_MIN='32768', NTT_FOLD_DEVICE_MAX_MB='640', CUDA_LAUNCH_BLOCKING='0')
    wrapper = out / 'app.cmd'
    prefix = [str(a.ncu), '--rename-kernels', '0', '--devices', '1', '--target-processes', 'all',
              '--clock-control', 'none', '--cache-control', 'none', '--kernel-name-base', 'mangled',
              '--kernel-name', 'regex:' + KERNEL, '--launch-skip', '1', '--launch-count', '1']
    for section in ('LaunchStats', 'SpeedOfLight', 'Occupancy', 'SchedulerStats', 'WarpStateStats', 'InstructionStats', 'MemoryWorkloadAnalysis'):
        prefix += ['--section', section]
    prefix += ['--metrics', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
               '--export', str(out / 'trace')]
    command = prefix + ['C:/Windows/System32/cmd.exe', '/d', '/c', str(wrapper)]
    if any(any(c in s for c in '&|<>%!^\r\n"\'') for s in app + command): raise ValueError('unsafe wrapper argument')
    reset = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
    wrapper.write_text('@echo off\n' + reset + ''.join(f'set "{k}={v}"\n' for k, v in env.items()) +
        ' '.join('"' + s + '"' for s in app) + f' > "{out / "app.log"}" 2>&1\nexit /b %errorlevel%\n')
    profile = out / 'profile.cmd'
    profile.write_text('@echo off\n' + ' '.join('"' + s + '"' for s in command) +
        f' > "{out / "run.log"}" 2>&1\nexit /b %errorlevel%\n')
    quote = lambda s: "'" + str(s).replace("'", "''") + "'"
    script = "$ErrorActionPreference='Stop'\n$env:PSModulePath=$env:WINDIR+'\\System32\\WindowsPowerShell\\v1.0\\Modules'\n"
    script += 'Set-Location -LiteralPath ' + quote(ROOT) + '\n'
    script += '& C:/Windows/System32/cmd.exe /d /c ' + quote(profile) + '\n'
    script += '$taskExit=$LASTEXITCODE\nSet-Content -LiteralPath ' + quote(out / 'exit.txt') + ' -Value $taskExit\nexit $taskExit\n'
    (out / 'capture.ps1').write_text(script)
    (out / 'command.json').write_text(json.dumps(dict(identity=identity, reference_sha256=sha(a.reference),
        save_sha256=sha(saved), key=key, command=command, environment=env, tool_sha256=sha(__file__),
        helper_sha256=sha(HELPER), expected_grid=[5, 1, 1], expected_block=[128, 1, 1],
        selection='Exact generic256 mangled name; skip index0 selftest grid1, capture index1 real grid5. Candidate order verified in Systems; baseline must match dimensions in actual NCU output.'), indent=2) + '\n')
    print(out / 'capture.ps1')


if __name__ == '__main__': main()
