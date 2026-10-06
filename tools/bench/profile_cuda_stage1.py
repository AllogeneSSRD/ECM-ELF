#!/usr/bin/env python3
"""Capture short, cache-backed Stage1 profiles; profiler rates are not benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def collect(root, tool, ncu, exit_code=None):
    report = root / ('trace.ncu-rep' if tool == 'ncu' else 'trace.nsys-rep')
    result = dict(exit_code=exit_code, report_present=report.exists(), report=str(report))
    log_text = (root / 'run.log').read_text(encoding='utf-8', errors='replace')
    if 'ERR_NVGPUCTRPERM' in log_text: result['failure_reason'] = 'gpu_counters_permission'
    elif not report.exists(): result['failure_reason'] = 'no_report; inspect run.log and app.log'
    if report.exists() and tool == 'ncu':
        with (root / 'metrics.csv').open('wb') as out:
            export = subprocess.run([str(ncu), '--rename-kernels', '0', '--import', str(report), '--csv', '--page', 'raw'],
                                    stdout=out, stderr=subprocess.STDOUT, timeout=60)
        result['csv_exit_code'] = export.returncode
    (root / 'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not report.exists() or result.get('csv_exit_code', 0): raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--tool', choices=['ncu', 'nsys'], default='ncu')
    p.add_argument('--algorithm', choices=['resident', 'prac'], default='prac')
    p.add_argument('--bits', type=int, default=4423)
    p.add_argument('--b1', type=int, default=10000000)
    p.add_argument('--curves', type=int, default=768)
    p.add_argument('--tpi', type=int, choices=[0, 16, 32], default=0)
    p.add_argument('--registers', type=int, choices=[0, 168, 255], default=255)
    p.add_argument('--variant', choices=['baseline', 'compact', 'outline-add', 'single-add'], default='baseline')
    p.add_argument('--target-ms', type=float, default=100)
    p.add_argument('--window', choices=['prefix', 'middle', 'tail'])
    p.add_argument('--window-count', type=int, default=32)
    p.add_argument('--window-chunk', type=int, default=0)
    p.add_argument('--window-warmup', type=int, default=2)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--seconds', type=float, default=6)
    p.add_argument('--launch-skip', type=int, default=2)
    p.add_argument('--memory', action='store_true', help='Include NCU memory workload counters for call/local-memory studies')
    p.add_argument('--exp-cache', type=Path, required=True)
    p.add_argument('--ncu', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe'))
    p.add_argument('--nsys', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true', help='Prepare Windows wrappers; run admin.ps1 elevated separately')
    p.add_argument('--collect-only', action='store_true', help='Export an existing prepared/admin capture without running GPU work')
    a = p.parse_args(); exe = a.exe.resolve(strict=True); root = a.output.resolve()
    if a.collect_only:
        metadata = json.loads((root / 'command.json').read_text(encoding='utf-8'))
        exit_file = root / 'exit.txt'
        collect(root, metadata['profiler'], a.ncu,
                int(exit_file.read_text().strip()) if exit_file.exists() else None)
        return
    if not 10 <= a.target_ms <= 500:
        p.error('PRAC target must be in 10..500 ms')
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ECM_GPU_STAGE1_ALGO=a.algorithm, ECM_PRAC_REG_TARGET=str(a.registers),
               ECM_STAGE1_TPI=str(a.tpi), ECM_GPU_STAGE1_SAMPLE_SECONDS=str(a.seconds), ECM_PRAC_VARIANT=a.variant,
               ECM_PRAC_PLAN_CACHE=str(a.exp_cache.resolve()), ECM_PRAC_TARGET_MS=str(a.target_ms))
    env_keys = ('ECM_GPU_STAGE1_ALGO', 'ECM_PRAC_REG_TARGET', 'ECM_STAGE1_TPI',
                'ECM_GPU_STAGE1_SAMPLE_SECONDS', 'ECM_PRAC_VARIANT', 'ECM_PRAC_PLAN_CACHE', 'ECM_PRAC_TARGET_MS')
    for key in tuple(env):
        if key.startswith('ECM_PRAC_WINDOW'): env.pop(key)
    if a.window:
        if a.algorithm != 'prac' or not 1 <= a.window_count <= 32 or not 0 <= a.window_chunk <= a.window_count or not 0 <= a.window_warmup <= 32:
            p.error('Window capture requires PRAC, count 1..32, chunk 0..count, warmup 0..32')
        env.update(ECM_PRAC_WINDOW=a.window, ECM_PRAC_WINDOW_COUNT=str(a.window_count),
                   ECM_PRAC_WINDOW_CHUNK=str(a.window_chunk), ECM_PRAC_WINDOW_WARMUP=str(a.window_warmup))
        env_keys += ('ECM_PRAC_WINDOW', 'ECM_PRAC_WINDOW_COUNT', 'ECM_PRAC_WINDOW_CHUNK', 'ECM_PRAC_WINDOW_WARMUP')
    env.pop('ECM_GPU_DUMP', None)
    app = [str(exe), '-gpu', '-d', str(a.device), '--gpu-param', '0', '-sigma', '0:26',
           '-gpucurves', str(a.curves), '--ckpt', '0', '--exp-cache', str(a.exp_cache.resolve()),
           '-v', '-savea', 'completed.save', str(a.b1), '0']
    if a.tool == 'ncu':
        prefix = [str(a.ncu), '--clock-control', 'none', '--cache-control', 'none',
                  '--rename-kernels', '0', '--check-exit-code', '0',
                  '--kernel-name', 'regex:kernel_suyama_domain', '--launch-skip', str(a.launch_skip),
                  '--launch-count', '1', '--section', 'SpeedOfLight', '--section', 'Occupancy',
                  '--section', 'SchedulerStats', '--section', 'WarpStateStats',
                  '--export', str(root / 'trace')]
        if a.memory:
            prefix += ['--metrics', ','.join((
                'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
                'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
                'l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct',
                'l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct'))]
        report = root / 'trace.ncu-rep'
    else:
        prefix = [str(a.nsys), 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                  '--output', str(root / 'trace')]
        report = root / 'trace.nsys-rep'
    # Nsight Systems does not reliably forward its piped stdin to the child.
    # Give the actual application its own file redirection and log instead.
    if any(any(c in token for c in '&|<>%!^\r\n"\'') for token in app + prefix + [str(root)]):
        p.error('Windows wrapper paths/arguments cannot contain shell metacharacters or quotes')
    (root / 'n.txt').write_text(f'(2^{a.bits}-1)\n', encoding='ascii')
    app_wrapper = root / 'app.cmd'
    app_wrapper.write_text('@echo off\n'+' '.join('"'+x+'"' for x in app)+
        ' < "'+str(root/'n.txt')+'" > "'+str(root/'app.log')+'" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
    if a.tool == 'ncu': prefix += ['--target-processes', 'all']
    command = prefix + ['C:/Windows/System32/cmd.exe', '/d', '/c', str(app_wrapper)]
    (root / 'command.json').write_text(json.dumps(dict(command=command, profiler=a.tool,
        binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), environment={k:env[k] for k in env_keys},
        measurement='profile only; replay/instrumentation changes timing',
        input_scope='reset-point subproduct window' if a.window else 'ordinary production prefix'), indent=2)+'\n', encoding='utf-8')
    if a.prepare_only:
        wrapper = root / 'profile.cmd'
        settings = 'set ECM_GPU_DUMP=\nset ECM_PRAC_WINDOW=\nset ECM_PRAC_WINDOW_COUNT=\nset ECM_PRAC_WINDOW_CHUNK=\nset ECM_PRAC_WINDOW_WARMUP=\nset ECM_PRAC_WINDOW_DUMP=\n' + ''.join('set '+k+'='+env[k]+'\n' for k in env_keys)
        wrapper.write_text('@echo off\ncd /d "'+str(root)+'"\n'+settings+
            ' '.join('"'+x+'"' for x in command)+' > "'+str(root/'run.log')+'" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
        # Elevation is explicit; this prepared file never starts a privileged process itself.
        ps = "$task = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' -ArgumentList @('/d', '/c', '\""+str(wrapper)+"\"') -Verb RunAs -WindowStyle Hidden -PassThru\n"
        ps += "$task.Id | Set-Content -LiteralPath '"+str(root/'pid.txt')+"'\n$task.WaitForExit()\n$task.ExitCode | Set-Content -LiteralPath '"+str(root/'exit.txt')+"'\n"
        (root / 'admin.ps1').write_text(ps, encoding='utf-8')
        print(root/'admin.ps1', flush=True); return
    with (root / 'run.log').open('wb') as log:
        proc = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    collect(root, a.tool, a.ncu, proc.returncode)


if __name__ == '__main__': main()
