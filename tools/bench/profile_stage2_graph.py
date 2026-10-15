"""Administrator Nsight captures of an exact completed fixed-D Graph A/B case.

Profiling runs are diagnostic only. Preserve and compare the application's
arithmetic/count invariants with the unprofiled reference before interpreting it.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from benchmark_stage2_short_ntt import parse, fields

ROOT = Path(__file__).resolve().parents[2]
NSYS = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')
NCU = Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(args):
    if any(any(c in x for c in '&|<>%!^\r\n"\'') for x in args):
        raise ValueError('Shell metacharacters in wrapper arguments')
    return ' '.join('"' + x + '"' for x in args)


def verify(meta):
    exe = Path(meta['app'][0])
    frozen = json.loads((exe.parent / 'frozen_sources_manifest.json').read_text(encoding='utf-8'))
    if sha(exe) != frozen['binary_sha256'] or sha(exe) != meta['binary_sha256']:
        raise ValueError('Frozen binary changed')
    for name, digest in frozen['sources'].items():
        if sha(exe.parent / 'sources' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for path, digest in meta['inputs'].items():
        if sha(path) != digest:
            raise ValueError('Input changed: ' + path)


def systems_summary(path, device):
    with sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        c.text_factory = lambda b: b.decode('utf-8', 'replace')
        tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
        def rows(sql, args=()):
            return [dict(r) for r in c.execute(sql, args)]
        kernels = rows('select k.*,s.value name,m.value mangled from CUPTI_ACTIVITY_KIND_KERNEL k '
                       'join StringIds s on s.id=k.demangledName join StringIds m on m.id=k.mangledName '
                       'where k.deviceId=? order by k.start', (device,))
        if not kernels:
            raise ValueError('No target device kernels')
        owners = {(k['globalPid'], k['contextId']) for k in kernels}
        if len(owners) != 1:
            raise ValueError('Expected one application GPU context')
        # Invocation indices count every instance of this mangled kernel, matching
        # NCU --kernel-id, regardless of length/batch or Graph/native launch.
        groups, invocation = {}, {}
        for k in kernels:
            mangled = k['mangled']
            invocation[mangled] = invocation.get(mangled, 0) + 1
            graph_node = bool(k.get('graphNodeId'))
            key = (mangled, k['gridX'], k['gridY'], k['blockX'], k['dynamicSharedMemory'], k.get('launchType'), graph_node)
            if key not in groups:
                tile = 'tile_kernel<' in k['name']
                groups[key] = dict(name=k['name'], mangled=mangled, grid=[k['gridX'], k['gridY'], k['gridZ']],
                    block=[k['blockX'], k['blockY'], k['blockZ']], shared_bytes=k['dynamicSharedMemory'],
                    registers=k['registersPerThread'], local_bytes=k['localMemoryPerThread'],
                    launch_type=k.get('launchType'), graph_node=graph_node, count=0, seconds=0,
                    length=k['gridX'] * (k['dynamicSharedMemory']//8) if tile else None,
                    batch=k['gridY'] if tile else None, first_invocation=invocation[mangled])
            group = groups[key]
            group['count'] += 1
            group['seconds'] += (k['end']-k['start']) / 1e9
        intervals = []
        copies = []
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            if table not in tables:
                continue
            intervals += [(r[0], r[1]) for r in c.execute('select start,end from ' + table + ' where deviceId=?', (device,))]
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            copies = rows('select e.label kind,count(*) count,sum(m.bytes) bytes,sum(m.end-m.start)/1e9 seconds '
                          'from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                          'where m.deviceId=? group by m.copyKind', (device,))
        apis = {}
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
            if table in tables:
                apis[table] = rows('select s.value name,count(*) count,sum(r.end-r.start)/1e9 seconds from ' + table +
                    ' r join StringIds s on s.id=r.nameId group by r.nameId order by seconds desc')
        intervals.sort()
        left, right = intervals[0]
        first, union, gaps = left, 0, []
        for start, end in intervals[1:]:
            if start > right:
                union += right-left
                gaps.append((start-right)/1e9)
                left, right = start, end
            else:
                right = max(right, end)
        union += right-left
        span = right-first
        return dict(owners=[list(v) for v in owners], window=dict(span=span/1e9, event_union=union/1e9,
                    no_own_gpu_event=(span-union)/1e9, no_own_gpu_event_percent=100*(span-union)/span),
                    kernels=sorted(groups.values(), key=lambda x: -x['seconds']), apis=apis, copies=copies,
                    largest_gap_seconds=sorted(gaps, reverse=True)[:20],
                    scope='Own process GPU kernel/copy/memset union including selftests and tail. '
                          'Gaps are not proof of whole-card idle; API waits overlap GPU execution. '
                          'Profiler timings are excluded from performance A/B.')


def collect(out):
    meta = json.loads((out / 'command.json').read_text(encoding='utf-8'))
    verify(meta)
    if int((out / 'exit.txt').read_text(encoding='utf-8-sig').strip()) != 0:
        raise ValueError('Administrator capture failed; inspect run.log')
    actual = parse(out / 'engine.log', out / 'results.jsonl')
    reference = meta['reference']
    for key in ('correctness', 'histogram', 'convolutions', 'chunks'):
        # JSON normalization converts integer histogram keys to strings.
        if json.dumps(actual[key], sort_keys=True) != json.dumps(reference[key], sort_keys=True):
            if key != 'histogram' or {str(k): v for k, v in actual[key].items()} != reference[key]:
                raise ValueError('Profile arithmetic/workload changed: ' + key)
    graph = fields((out / 'engine.log').read_text(encoding='utf-8'), 'ntt_graph_stats')
    if graph['requested'] != meta['environment']['NTT_SHORT_GRAPH'] or int(graph['failures']):
        raise ValueError('Graph request/capture differs')
    if 'NTT_S4_PACK_ZERO' in meta['environment']:
        pack = fields((out / 'engine.log').read_text(encoding='utf-8'), 's4_pack_zero_stats')
        enabled = int(meta['environment']['NTT_S4_PACK_ZERO'])
        if int(pack['requested']) != enabled or bool(int(pack['words'])) != bool(enabled):
            raise ValueError('Pack/zero profile request differs')
    if meta['kind'] == 'systems':
        exported = out / 'trace.sqlite'
        if not exported.exists():
            with (out / 'export.log').open('wb') as f:
                subprocess.run([str(NSYS), 'export', '--type=sqlite', '--force-overwrite=true', '--output', str(exported),
                                str(out / 'trace.nsys-rep')], stdout=f, stderr=subprocess.STDOUT, check=True, timeout=120)
        summary = systems_summary(exported, meta['device'])
        if meta.get('host_trace'):
            with sqlite3.connect(exported) as database:
                exists = database.execute("select count(*) from sqlite_master where type='table' and name='NVTX_EVENTS'").fetchone()[0]
                count = database.execute('select count(*) from NVTX_EVENTS where eventType in (59,70) and end is not null').fetchone()[0] if exists else 0
            if not count:
                raise ValueError('NTT_HOST_TRACE was requested but no CPU NVTX ranges were captured')
            summary['host_trace_ranges'] = count
        summary['report_sha256'] = sha(out / 'trace.nsys-rep')
    else:
        # Failed exports must not overwrite a previously collected metric table.
        exported_metrics = out / 'metrics.pending.csv'
        with exported_metrics.open('wb') as f:
            subprocess.run([str(NCU), '--rename-kernels', '0', '--import', str(out / 'trace.ncu-rep'), '--csv', '--page', 'raw'],
                           stdout=f, stderr=subprocess.STDOUT, check=True, timeout=60)
        exported_metrics.replace(out / 'metrics.csv')
        with (out / 'metrics.csv').open(encoding='utf-8-sig') as f:
            metrics = list(csv.DictReader(f))
        if len(metrics) != 2:
            raise ValueError('Expected units row and exactly one profiled kernel')
        units, values = metrics
        shape = meta['kernel']
        for field, wanted in [('Grid Size', shape['grid']), ('Block Size', shape['block'])]:
            if [int(v) for v in re.findall(r'\d+', values[field])] != wanted:
                raise ValueError('NCU geometry differs from Systems selection: ' + field)
        dynamic = float(values['launch__shared_mem_per_block_dynamic'].replace(',', ''))
        dynamic *= {'byte': 1, 'Kbyte': 1000, 'Mbyte': 1000000}[units['launch__shared_mem_per_block_dynamic'].split('/')[0]]
        if round(dynamic) != shape['shared_bytes'] or int(values['launch__registers_per_thread']) != shape['registers']:
            raise ValueError('NCU resources differ from Systems selection')
        if values['Kernel Name'] != shape['name']:
            raise ValueError('NCU kernel differs from Systems selection')
        selected_metrics = {k: dict(value=values[k], unit=units[k]) for k in values if k.startswith((
            'gpu__time_', 'gpc__cycles_elapsed.avg.per_second', 'sm__throughput.', 'gpu__dram_throughput.',
            'sm__warps_active.', 'smsp__warps_eligible.avg.per_cycle', 'smsp__issue_active.',
            'smsp__average_warps_issue_stalled_', 'launch__registers_', 'launch__shared_mem_',
            'launch__occupancy_', 'derived__local_spilling_'))}
        summary = dict(metrics_sha256=sha(out / 'metrics.csv'), report_sha256=sha(out / 'trace.ncu-rep'),
                       metrics=selected_metrics, geometry_verified=True,
                       selected=meta['kernel'], scope='Kernel replay diagnostic; exclude profiling time from performance A/B.')
    summary.update(application=actual, graph=graph, command_sha256=sha(out / 'command.json'), arithmetic_verified=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Validated capture:', out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, help='Completed A/B *.command.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['systems', 'compute'], default='systems')
    p.add_argument('--kernel', type=Path, help='JSON shape selected from Systems summary, including first_invocation')
    p.add_argument('--all-cuda-apis', action='store_true', help='Systems only; include normally filtered CUDA calls')
    p.add_argument('--host-trace', action='store_true', help='Systems only; enable instrumented Stage2 CPU NVTX ranges')
    p.add_argument('--run-admin', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if a.collect_only:
        collect(out)
        return
    if not a.reference:
        p.error('--reference required for new capture')
    if (a.all_cuda_apis or a.host_trace) and a.kind != 'systems':
        p.error('--all-cuda-apis/--host-trace requires --kind systems')
    out.mkdir(parents=True, exist_ok=False)
    ref = a.reference.resolve()
    record = json.loads(ref.read_text(encoding='utf-8'))
    if record['returncode']:
        raise ValueError('Reference application failed')
    cmd = record['command'][:]
    def arg(name):
        return cmd[cmd.index(name)+1]
    source_log, source_result = Path(arg('--log')), Path(arg('--results'))
    reference = parse(source_log, source_result)
    inputs = {arg('--ini'): sha(arg('--ini')), arg('--save'): sha(arg('--save'))}
    cmd[cmd.index('--log')+1] = str(out / 'engine.log')
    cmd[cmd.index('--results')+1] = str(out / 'results.jsonl')
    env = dict(record['environment'], CUDA_LAUNCH_BLOCKING='0')
    if a.host_trace:
        env['NTT_HOST_TRACE'] = '1'
    if any(not re.fullmatch(r'(NTT_[A-Z0-9_]+|CUDA_LAUNCH_BLOCKING)', k) or
           not re.fullmatch(r'[A-Za-z0-9_,.\-]+', v) for k, v in env.items()):
        raise ValueError('Unsupported environment value in administrator wrapper')
    wrapper = out / 'app.cmd'
    settings = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
    settings += ''.join('set "' + k + '=' + v + '"\n' for k, v in env.items())
    wrapper.write_text('@echo off\ncd /d "' + str(ROOT) + '"\n' + settings + safe(cmd) +
                       ' > "' + str(out / 'app.log') + '" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
    kernel = None
    if a.kind == 'systems':
        prefix = [str(NSYS), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                  '--cuda-graph-trace=node', '--force-overwrite=true', '--output', str(out / 'trace')]
        if a.all_cuda_apis:
            prefix += ['--cuda-trace-all-apis=true']
    else:
        if not a.kernel:
            p.error('--kernel required for compute')
        kernel = json.loads(a.kernel.read_text(encoding='utf-8'))
        prefix = [str(NCU), '--rename-kernels', '0', '--devices', arg('--device'), '--target-processes', 'all',
                  '--clock-control', 'none', '--cache-control', 'none', '--graph-profiling', 'node',
                  '--kernel-name-base', 'mangled', '--kernel-id', '::' + kernel['mangled'] + ':' + str(kernel['first_invocation']),
                  '--launch-count', '1']
        for section in ('LaunchStats', 'SpeedOfLight', 'Occupancy', 'SchedulerStats', 'WarpStateStats', 'InstructionStats', 'MemoryWorkloadAnalysis'):
            prefix += ['--section', section]
        prefix += ['--export', str(out / 'trace')]
    profile = out / 'profile.cmd'
    command = prefix + ['C:/Windows/System32/cmd.exe', '/d', '/c', str(wrapper)]
    profile.write_text('@echo off\n' + safe(command) + ' > "' + str(out / 'run.log') +
                       '" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
    ps = "$stage2Profile = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' -ArgumentList @('/d','/c','\"" + str(profile) + "\"') -Verb RunAs -WindowStyle Hidden -PassThru\n"
    ps += "$stage2Profile.Id | Set-Content -LiteralPath '" + str(out / 'pid.txt') + "'\n$stage2Profile.WaitForExit()\n"
    ps += "$stage2Profile.ExitCode | Set-Content -LiteralPath '" + str(out / 'exit.txt') + "'\nexit $stage2Profile.ExitCode\n"
    (out / 'admin.ps1').write_text(ps, encoding='utf-8')
    meta = dict(kind=a.kind, device=int(arg('--device')), app=cmd, environment=env, reference=reference,
                all_cuda_apis=a.all_cuda_apis, host_trace=a.host_trace,
                reference_file=str(ref), reference_sha256=sha(ref), inputs=inputs, binary_sha256=sha(cmd[0]),
                command=command, kernel=kernel, collector_sha256=sha(__file__))
    verify(meta)
    (out / 'command.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if a.run_admin:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(out / 'admin.ps1')],
                       check=True, timeout=600)
        collect(out)
    else:
        print(out / 'admin.ps1')


if __name__ == '__main__':
    main()
