"""Prepare an administrator NCU capture of the actual giant seed kernel.

Use --collect-only after running the generated admin.ps1. Kernel replay and
profiling durations are diagnostic evidence, never Stage2 benchmark samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['original', 'paired'], required=True)
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--ncu', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe'))
    a = p.parse_args()
    exe, saved, out = a.exe.resolve(), a.save.resolve(), a.output.resolve()
    frozen_path = exe.parent / 'frozen_sources_manifest.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['binary_sha256'] != sha(exe):
        raise ValueError('Frozen binary differs')
    for name, want in frozen['sources'].items():
        if sha(exe.parent / 'sources' / name) != want:
            raise ValueError('Frozen compiled source differs: ' + name)
    if a.collect_only:
        meta = json.loads((out / 'command.json').read_text(encoding='utf-8'))
        if meta['binary_sha256'] != sha(exe) or meta['save_sha256'] != sha(saved) or meta['mode'] != a.mode:
            raise ValueError('Capture identity differs')
        if int((out / 'exit.txt').read_text(encoding='utf-8-sig').strip()) != 0:
            raise ValueError('Administrator capture failed; inspect run.log')
        with (out / 'metrics.csv').open('wb') as dest:
            subprocess.run([str(a.ncu), '--rename-kernels', '0', '--import', str(out / 'trace.ncu-rep'),
                            '--csv', '--page', 'raw'], stdout=dest, stderr=subprocess.STDOUT, check=True, timeout=60)
        result = json.loads((out / 'results.jsonl').read_text(encoding='utf-8').splitlines()[-1])
        log = (out / 'engine.log').read_text(encoding='utf-8')
        if result['bad_factors'] or result['factors'] or not all(s in log for s in ('gmp_check_bad=0', 'gmp_selftest_bad=0', 'pending=0')):
            raise ValueError('Profiled application arithmetic failed')
        (out / 'summary.json').write_text(json.dumps(dict(command=meta, result=result,
            metrics_sha256=sha(out / 'metrics.csv'), report_sha256=sha(out / 'trace.ncu-rep'),
            scope='One actual M8191/C8 seed kernel; replay changes execution and timing. Full unprofiled A/B required.'), indent=2), encoding='utf-8')
        print('Collected', out)
        return
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    ini = out / 'manual.ini'
    ini.write_text('[gpu]\ndevice=1\n', encoding='utf-8')
    app = [str(exe), '--ini', str(ini), '--save', str(saved), '--device', '1', '--b2', '122942820', '--d', '30030',
           '--arena-mb', '4096', '--factor-only', '--log', str(out / 'engine.log'), '--results', str(out / 'results.jsonl')]
    wrapper = out / 'app.cmd'
    env = dict(NTT_NO_PROGRESS='1', NTT_D_MODEL='0', NTT_POINT_MERSENNE='1', NTT_CARRY_CHECK_FUSED='0',
        NTT_FOLD_DEVICE_MAX_MB='0', NTT_GIANT_CHAIN_MIN='0', NTT_GIANT_CHAIN_BLOCK='8',
        NTT_GIANT_SEED_PAIR='1' if a.mode == 'paired' else '0', CUDA_LAUNCH_BLOCKING='0')
    kernel = 'regex:s2g_ladder_kernel' if a.mode == 'original' else 'regex:s2g_seed_pair_kernel'
    prefix = [str(a.ncu), '--target-processes', 'all', '--clock-control', 'none', '--cache-control', 'none',
        '--kernel-name', kernel, '--launch-skip', '1' if a.mode == 'original' else '0', '--launch-count', '1',
        '--section', 'SpeedOfLight', '--section', 'Occupancy', '--section', 'SchedulerStats', '--section', 'WarpStateStats',
        '--metrics', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
        '--export', str(out / 'trace')]
    command = prefix + ['C:/Windows/System32/cmd.exe', '/d', '/c', str(wrapper)]
    if any(any(c in s for c in '&|<>%!^\r\n"\'') for s in app + command + [str(out)]):
        raise ValueError('Wrapper arguments contain shell metacharacters')
    settings = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
    settings += ''.join('set "' + k + '=' + v + '"\n' for k, v in env.items())
    wrapper.write_text('@echo off\n' + settings + ' '.join('"'+s+'"' for s in app) +
        ' > "'+str(out / 'app.log')+'" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
    profile = out / 'profile.cmd'
    profile.write_text('@echo off\n' + ' '.join('"'+s+'"' for s in command) +
        ' > "'+str(out / 'run.log')+'" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
    ps = "$stage2Profile = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' -ArgumentList @('/d','/c','\""+str(profile)+"\"') -Verb RunAs -WindowStyle Hidden -PassThru\n"
    ps += "$stage2Profile.Id | Set-Content -LiteralPath '"+str(out / 'pid.txt')+"'\n$stage2Profile.WaitForExit()\n"
    ps += "$stage2Profile.ExitCode | Set-Content -LiteralPath '"+str(out / 'exit.txt')+"'\nexit $stage2Profile.ExitCode\n"
    (out / 'admin.ps1').write_text(ps, encoding='utf-8')
    (out / 'command.json').write_text(json.dumps(dict(command=command, environment=env, mode=a.mode,
        binary_sha256=sha(exe), save_sha256=sha(saved), frozen_manifest_sha256=sha(frozen_path), tool_sha256=sha(__file__),
        selection='Skip baby ladder and capture original seed grid17; paired kernel grid8 is unique. Verified by Systems timeline.'), indent=2), encoding='utf-8')
    print(out / 'admin.ps1')


if __name__ == '__main__':
    main()
