"""Freeze exact integer probes, screen policies, then independently remeasure.

GPU settings are read only. A complete wide product and canonical Montgomery
product are separate workloads; high/low feedback prevents truncated products.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from bench_stage2_point_matrix import environment
from bench_stage2_version import check_gpu_processes, load, write, stamp
from gpu_telemetry import TelemetryRecorder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape(bits):
    prime = 2**64 - 2**32 + 1
    bpw = next(b for b in range(31, 15, -1) if ((bits+b-1)//b)*(2**b-1)**2 < prime)
    digits = (bits+bpw-1)//bpw
    return 1 << (2*digits).bit_length(), bpw


def freeze(build, dest, repo):
    manifest = load(build/'manifest.json')
    assert sha(build/'cgbn_ntt_multiply_probe.exe') == manifest['sha256'].lower()
    assert manifest['ntt_abi_schema'] == 1 and manifest['production_runtime']
    assert manifest['production_module_group'] == 'STAGE2_NTT' and manifest['gl_fixed_mode'] == 3
    assert manifest['default_stream'] == 'legacy'
    dest.mkdir(parents=True)
    (dest/'.gitignore').write_text('*\n')
    for group in ('compiled_objects', 'production_objects'):
        for path, digest in manifest[group].items():
            assert sha(path) == digest.lower(), path
    for name, digest in manifest['sources'].items():
        source = build/'sources'/name if (build/'sources').is_dir() else repo/name
        assert sha(source) == digest.lower(), name
        target = dest/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ('manifest.json', 'cgbn_ntt_multiply_probe.exe', 'gmp-10.dll'):
        shutil.copy2(build/name, dest/name)
    assert sha(dest/'gmp-10.dll') == manifest['gmp_dll_sha256'].lower()
    defines = dict(item.split('=', 1) for item in manifest['local_defines'])
    return dict(bits=int(defines['ECM_CROSS_BITS']), tpi=int(defines['ECM_CROSS_TPI']),
                folder=str(dest), exe_sha256=sha(dest/'cgbn_ntt_multiply_probe.exe'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', choices=('smoke', 'check', 'screen', 'final', 'sanitizer'), required=True)
    p.add_argument('--selection', type=Path, help='Completed independent screening summary for final measurements')
    p.add_argument('--process-policy', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--counts', default='1,192,1536')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--sanitizer', type=Path)
    p.add_argument('--sanitizer-tool', choices=('memcheck', 'synccheck'), default='memcheck')
    a = p.parse_args()
    if a.suite == 'final' and (not a.selection or a.rounds < 3):
        p.error('final requires --selection and at least three rounds')
    if a.suite == 'sanitizer' and (not a.sanitizer or not a.sanitizer.is_file()):
        p.error('sanitizer suite requires --sanitizer PATH')
    repo = Path(__file__).resolve().parents[2]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    allowed = load(a.process_policy)['allowed_processes']
    report = dict(complete=False, started_utc=stamp(), suite=a.suite, device=a.device,
                  warmups=0, allowed_processes=allowed, builds=[], runs=[], failures=[],
                  timing='GPU-resident dependent chain; pack/NTT/carry/unpack/REDC included; oracle/setup/graph build/copies excluded')
    for index, build in enumerate(a.build):
        report['builds'].append(freeze(build.resolve(), out/'bin'/str(index), repo))
    assert len({(b['bits'], b['tpi']) for b in report['builds']}) == len(report['builds'])
    shutil.copy2(__file__, out/'collector.py')
    write(out/'summary.json', report)

    def run(build, count, iters, backend, op, tile=12, samples=3, kind=0, fault=0, round_index=None):
        folder = out/f'call{len(report["runs"]):04d}'
        folder.mkdir()
        cmd = [str(Path(build['folder'])/'cgbn_ntt_multiply_probe.exe'),
               *map(str, (a.device, count, iters, backend, op, tile, samples, fault, kind))]
        env = environment()
        if a.suite == 'sanitizer':
            cmd = [str(a.sanitizer), '--tool', a.sanitizer_tool, '--error-exitcode', '87', *cmd]
            env['NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE'] = 'named-pipes'
        write(folder/'command.json', cmd)
        mon = TelemetryRecorder(a.device, folder/'telemetry.jsonl', interval=.05)
        try:
            check_gpu_processes(mon.monitor, out, folder/'before.json', allowed=allowed)
            result = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, text=True, timeout=300)
            console = result.stdout + result.stderr
            (folder/'console.log').write_text(console, encoding='utf-8')
            write(folder/'exit.json', dict(returncode=result.returncode))
            check_gpu_processes(mon.monitor, out, folder/'after.json', allowed=allowed)
        finally:
            mon.close()
        row = dict(bits=build['bits'], tpi=build['tpi'], exe_sha256=build['exe_sha256'],
                   count=count, iterations=iters, backend=backend, op=op, tile=tile, kind=kind,
                   folder=str(folder), round=round_index)
        if fault:
            assert result.returncode == 3 and 'crossover_bad:' in console, folder
            row['fault_rejected'] = True
        else:
            assert result.returncode == 0, folder
            if a.suite == 'sanitizer':
                assert 'ERROR SUMMARY: 0 errors' in console, folder
            lines = [line for line in console.splitlines() if line.startswith('crossover:')]
            assert len(lines) == 1, folder
            fields = dict(re.findall(r'(\w+)=([^\s]+)', lines[0]))
            for key, expected in dict(bits=build['bits'], tpi=build['tpi'], count=count,
                                      iterations=iters, backend=backend, op=op, bad=0, guards=1).items():
                assert int(fields[key]) == expected, (folder, key)
            times = [float(t) for t in fields.pop('samples_ms').split(',')]
            assert len(times) == samples and all(t > 0 for t in times)
            if backend:
                assert (int(fields['length']), int(fields['bpw'])) == shape(build['bits'])
            row.update(fields=fields, samples_ms=times, median_ms=statistics.median(times),
                       batch_ms=statistics.median(times)/iters, input_hash=fields['input_hash'])
        report['runs'].append(row)
        write(out/'summary.json', report)
        print(a.suite, row['bits'], row['tpi'], count, backend, op, tile,
              row.get('batch_ms', 'fault-rejected'), flush=True)
        return row

    try:
        builds = sorted(report['builds'], key=lambda b: (b['bits'], b['tpi']))
        if a.suite in ('smoke', 'check', 'sanitizer'):
            for build in builds:
                cases = [(33, 3, 1)] if a.suite != 'check' else [(1, 3, 0), (33, 1, 1), (67, 3, 1)]
                for count, iters, kind in cases:
                    for backend in (0, 1, 2):
                        for op in (0, 1):
                            run(build, count, iters, backend, op, samples=1, kind=kind)
                if a.suite == 'check':
                    for backend in (0, 1, 2):
                        run(build, 33, 3, backend, 0, samples=1, kind=1, fault=1)
                    for op in (0, 1):
                        run(build, 67, 3, 2, op, tile=6, samples=1, kind=1)
        elif a.suite == 'screen':
            counts = [int(c) for c in a.counts.split(',')]
            for build in builds:
                for count in counts:
                    iters = 32 if count == 1 else 2 if count >= 1000 else 8
                    for op in (0, 1):
                        run(build, count, iters, 0, op)
            for build in builds:
                length, _ = shape(build['bits'])
                tiles = sorted({min(t, length.bit_length()-1) for t in (6, 9, 12)})
                for count in counts:
                    iters = 32 if count == 1 else 2 if count >= 1000 else 8
                    for op in (0, 1):
                        # Raw NTT is TPI-independent. Modular reduction is not.
                        if op == 0 and build != next(b for b in builds if b['bits'] == build['bits']):
                            continue
                        for tile in tiles:
                            run(build, count, iters, 2, op, tile=tile)
        else:
            selection = load(a.selection)
            assert selection['complete'] and not selection['failures'] and selection['suite'] == 'screen'
            available = {(b['bits'], b['tpi']): b for b in builds}
            cells = sorted({(r['bits'], r['count'], r['op']) for r in selection['runs']})
            for bits, count, op in cells:
                choices = []
                group = [r for r in selection['runs'] if (r['bits'], r['count'], r['op']) == (bits, count, op)]
                assert len({r['input_hash'] for r in group}) == 1
                for backend in (0, 2):
                    best = min((r for r in group if r['backend'] == backend), key=lambda r: r['batch_ms'])
                    build = available[(bits, best['tpi'])]
                    assert build['exe_sha256'] == best['exe_sha256']
                    choices.append((build, best['iterations'], backend, best['tile']))
                build, iters, _, tile = choices[1]
                choices.append((build, iters, 1, tile))
                for repeat in range(a.rounds):
                    for build, iters, backend, tile in (choices if repeat % 2 == 0 else list(reversed(choices))):
                        run(build, count, iters, backend, op, tile, samples=5, round_index=repeat)
        report.update(complete=True, finished_utc=stamp())
    except Exception as error:
        report['failures'].append(str(error))
        raise
    finally:
        write(out/'summary.json', report)


if __name__ == '__main__':
    main()
