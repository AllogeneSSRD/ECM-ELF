"""Administrator Nsight fork/join capture; preserve concurrency with range replay."""
import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from benchmark_ntt_tensor_concurrency import fields
from profile_stage2_graph import NSYS, NCU, safe, sha, systems_summary

ROOT = Path(__file__).resolve().parents[2]


def union(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    left, right = intervals[0]
    total = 0
    for start, end in intervals[1:]:
        if start > right:
            total += right-left
            left, right = start, end
        else:
            right = max(right, end)
    return total+right-left


def collect(out):
    meta = json.loads((out / 'command.json').read_text())
    if sha(meta['app'][0]) != meta['binary_sha256']:
        raise ValueError('Binary changed')
    if int((out / 'exit.txt').read_text(encoding='utf-8-sig')):
        raise ValueError('Administrator profile failed; inspect run.log')
    text = (out / 'app.log').read_text(encoding='utf-8', errors='replace')
    samples = fields(text, 'tc_concurrency_bench')
    if not samples or (meta['kind'] == 'systems' and len(samples) != 1) or any(int(s['bad']) or int(s['scene']) != meta['scene'] for s in samples):
        raise ValueError('Profiled output check failed')
    if 'live_bytes=0 fuse_live_bytes=0' not in text:
        raise ValueError('Allocation leak')
    if meta['kind'] == 'systems':
        with (out / 'export.log').open('wb') as f:
            subprocess.run([str(NSYS), 'export', '--type=sqlite', '--force-overwrite=true', '--output', str(out / 'trace.sqlite'), str(out / 'trace.nsys-rep')], stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
        summary = systems_summary(out / 'trace.sqlite', meta['device'])
        c = sqlite3.connect(out / 'trace.sqlite')
        rows = c.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=?', (meta['device'],)).fetchall()
        tensor = [(a, b) for a, b, name in rows if 'tc_goldilocks_' in name]
        cuda = [(a, b) for a, b, name in rows if 'tc_goldilocks_' not in name]
        tu, cu, both = union(tensor), union(cuda), union(tensor+cuda)
        summary['concurrency'] = dict(tensor_union_seconds=tu/1e9, cuda_union_seconds=cu/1e9,
            combined_union_seconds=both/1e9, tensor_cuda_overlap_seconds=(tu+cu-both)/1e9,
            overlap_fraction_of_tensor=(tu+cu-both)/tu if tu else 0,
            note='GPU kernel interval overlap, not proof of same-SM or same-cycle instruction co-issue.')
        c.close()
    else:
        with (out / 'metrics.pending.csv').open('wb') as f:
            subprocess.run([str(NCU), '--rename-kernels', '0', '--import', str(out / 'trace.ncu-rep'), '--csv', '--page', 'raw'], stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
        (out / 'metrics.pending.csv').replace(out / 'metrics.csv')
        with (out / 'metrics.csv').open(encoding='utf-8-sig') as f:
            metrics = list(csv.DictReader(f))
        if len(metrics) < 2:
            raise ValueError('No range counters')
        device = fields(text, 'tc_concurrency_device')[0]
        name = re.search(r'name=(.+) sm=', text)[1]
        for row in metrics[1:]:
            if row.get('device__attribute_display_name') != name or int(row.get('device__attribute_multiprocessor_count', -1)) != int(device['sms']):
                raise ValueError('NCU captured a different physical device')
        summary = dict(units=metrics[0], ranges=metrics[1:], note='Range replay preserves the fork/join schedule; profiling times excluded from benchmark.')
    summary.update(meta=meta, sample=samples[-1], application_outputs_checked=len(samples), arithmetic_verified=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Collected', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path)
    p.add_argument('--shape', type=Path)
    p.add_argument('--scene', type=int, default=6)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--execution', choices=['native', 'graph'], default='graph')
    p.add_argument('--kind', choices=['systems', 'compute'], default='systems')
    p.add_argument('--replay', choices=['range', 'app-range'], default='app-range')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-admin', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if a.collect_only:
        collect(out)
        return
    if not a.exe or not a.shape or not 0 <= a.scene < 13:
        p.error('exe, shape, scene 0..12 required')
    out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    manifest = json.loads((exe.parent / 'manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe) != manifest['sha256'].lower():
        raise ValueError('Build identity differs')
    for name, digest in manifest['sources'].items():
        if sha(ROOT / name) != digest.lower():
            raise ValueError('Compiled source changed')
    shape = json.loads(a.shape.read_text())
    app = [str(exe), str(a.device), '--bench', str(shape['k']), str(shape['batch']), '12', str(shape.get('threads', 256)),
           '1', str(shape['iterations']), str(int(a.execution == 'graph')), str(a.scene)]
    wrapper = out / 'app.cmd'
    settings = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\nset "CUDA_LAUNCH_BLOCKING=0"\n'
    redirect = ' >> "' if a.kind == 'compute' else ' > "'
    wrapper.write_text('@echo off\ncd /d "' + str(ROOT) + '"\n' + settings + safe(app) + redirect + str(out / 'app.log') + '" 2>&1\nexit /b %errorlevel%\n')
    if a.kind == 'systems':
        prefix = [str(NSYS), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node',
                  '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true', '--output', str(out / 'trace')]
    else:
        # The target alone selects its GPU. A runtime ordinal filter produced
        # empty ranges for this single-context probe; verify the device by name.
        prefix = [str(NCU), '--rename-kernels', '0', '--target-processes', 'all', '--clock-control', 'none', '--cache-control', 'none',
                  '--replay-mode', a.replay,
                  '--section', 'SpeedOfLight', '--section', 'ComputeWorkloadAnalysis', '--section', 'MemoryWorkloadAnalysis', '--export', str(out / 'trace')]
    command = prefix + ['C:/Windows/System32/cmd.exe', '/d', '/c', str(wrapper)]
    profile = out / 'profile.cmd'
    profile.write_text('@echo off\n' + safe(command) + ' > "' + str(out / 'run.log') + '" 2>&1\nexit /b %errorlevel%\n')
    ps = "$tensorProfile = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' -ArgumentList @('/d','/c','\"" + str(profile) + "\"') -Verb RunAs -WindowStyle Hidden -PassThru\n"
    ps += "$tensorProfile.Id | Set-Content -LiteralPath '" + str(out / 'pid.txt') + "'\n$tensorProfile.WaitForExit()\n"
    ps += "$tensorProfile.ExitCode | Set-Content -LiteralPath '" + str(out / 'exit.txt') + "'\nexit $tensorProfile.ExitCode\n"
    (out / 'admin.ps1').write_text(ps)
    (out / 'command.json').write_text(json.dumps(dict(app=app, shape=shape, device=a.device, scene=a.scene, kind=a.kind,
        execution=a.execution, replay=a.replay, append_application_passes=a.kind == 'compute', binary_sha256=sha(exe), manifest=manifest, command=command, collector_sha256=sha(__file__)), indent=2))
    if a.run_admin:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(out / 'admin.ps1')], check=True, timeout=600)
        collect(out)
    else:
        print(out / 'admin.ps1')


if __name__ == '__main__':
    main()
