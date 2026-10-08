"""Profile actual hot Stage2 NTT launches with administrator Nsight Compute.

Prepare eight serial captures from a completed production Systems trace and
unprofiled matrix. An isolated host-range build must have byte-identical GPU
SASS/resources. Match the first kernel inside its explicit profiling range;
collection rejects a different grid/block/shared-memory shape.
Replay metrics are diagnostic, not uninstrumented Stage2 performance results.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'tools/bench/bench_stage2_production.py'
spec = importlib.util.spec_from_file_location('stage2_production', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, fields = helper.read, helper.fields
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def select_shapes(trace, source_root, sources):
    runtime = Path(source_root) / 'src/cuda/stage2/ntt_runtime.cuh'
    outer = Path(source_root) / 'src/cuda/stage2/ntt_coop_outer.cuh'
    for path in (runtime, outer):
        if sha(path) != sources[path.relative_to(source_root).as_posix()]:
            raise ValueError('frozen launcher changed')
    body = runtime.read_text()
    for token in ('const unsigned int blocks = (unsigned int)(n >> c.t)',
                  'const int smem = (int)(1ull << c.t) * (int)sizeof(unsigned long long)',
                  'dim3 gr(blocks, (unsigned int)nbatch)'):
        if token not in body:
            raise ValueError('tile geometry derivation requires the verified launcher')
    if 'dim3 grid((unsigned int)(n/((1ull<<m)*v)),(unsigned int)nbatch)' not in outer.read_text():
        raise ValueError('outer geometry derivation requires the verified launcher')
    query = """select k.*,s.value name,m.value mangled from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id=k.demangledName join StringIds m on m.id=k.mangledName
        where k.deviceId=1 order by k.start,k.correlationId"""
    counts, selected, owners = {}, {}, set()
    with sqlite3.connect('file:' + trace.as_posix() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for global_index, raw in enumerate(conn.execute(query)):
            r = dict(raw)
            owners.add((r['globalPid'], r['contextId']))
            if r['streamId'] != 7 or r['launchType'] != 1:
                raise ValueError('selection requires the traced single-stream direct launch sequence')
            if 'tile_kernel<' not in r['name'] and 'outer_coop_kernel<' not in r['name']:
                continue
            name = re.sub(r'\((?:bool|int)\)\s*', '', r['name'])
            if 'tile_kernel<' in name:
                m = re.search(r'tile_kernel<(false|true|[01]),\s*(false|true|[01])>', name)
                if not m or m[2] not in ('true', '1'):
                    raise ValueError('unexpected tile specialization')
                tile = r['dynamicSharedMemory'] // 8
                if tile <= 0 or tile & (tile-1) or r['dynamicSharedMemory'] % 8:
                    raise ValueError('invalid tile shared memory')
                n, kind, inverse = r['gridX'] * tile, 'tile', m[1] in ('true', '1')
            else:
                m = re.search(r'outer_coop_kernel<(\d+),\s*(false|true|[01])>', name)
                if not m or not 5 <= int(m[1]) <= 8:
                    raise ValueError('unexpected cooperative specialization')
                radix = int(m[1])
                n = r['gridX'] * (1 << radix) * (16 if radix == 8 else 32)
                kind, inverse = 'outer_m' + m[1], m[2] in ('true', '1')
            direction = 'inverse' if inverse else 'forward'
            key = None
            if n == 1 << 27 and r['gridY'] == 1 and kind in ('tile', 'outer_m7', 'outer_m8'):
                key = 'n27_' + kind + '_' + direction
            if n == 1 << 11 and r['gridY'] == 990 and kind == 'tile':
                key = 'n11_b990_tile_' + direction
            if key and key not in selected:
                selected[key] = dict(name=r['name'], mangled=r['mangled'],
                    matched_launches_before=counts.get(r['mangled'], 0),
                    launch_skip_before_match=global_index, ntt_words=n, nbatch=r['gridY'],
                    grid=[r['gridX'], r['gridY'], r['gridZ']],
                    block=[r['blockX'], r['blockY'], r['blockZ']],
                    dynamic_shared_bytes=r['dynamicSharedMemory'],
                    static_shared_bytes=r['staticSharedMemory'], registers=r['registersPerThread'],
                    first_start_ns=r['start'], correlation_id=r['correlationId'])
            counts[r['mangled']] = counts.get(r['mangled'], 0) + 1
    if len(selected) != 8 or len(owners) != 1:
        raise ValueError('requires all eight shapes in one own GPU1 CUDA context')
    return selected


def safe_args(args):
    if any(any(c in s for c in '&|<>%!^\r\n"\'') for s in args):
        raise ValueError('unsafe Windows wrapper argument')
    return ' '.join('"' + s + '"' for s in args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--systems', type=Path, required=True, help='Completed same-binary Systems directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--collect-only', action='store_true')
    p.add_argument('--range-project', type=Path, required=True,
                   help='Isolated host-range project; requires full GPU instruction/resource equivalence with only anonymous TU IDs normalized')
    p.add_argument('--ncu', type=Path, default=Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe'))
    a = p.parse_args()
    exe, out, systems = a.exe.resolve(), a.output.resolve(), a.systems.resolve()
    range_binding = None
    if a.range_project:
        project = a.range_project.resolve()
        generated = read(project/'instrumentation.json')
        build = read(exe.parent/'build_manifest.json')
        if {n:h.lower() for n,h in build['source_hashes'].items()} != generated['generated_sources']:
            raise ValueError('instrumented build differs from the exact generated closure')
        for name, want in generated['generated_sources'].items():
            if sha(project/name) != want:
                raise ValueError('generated source changed: ' + name)
        frozen = exe.parent/'frozen_sources_manifest.json'
        if not frozen.exists():
            for name in generated['generated_sources']:
                dest = exe.parent/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((project/name).read_bytes())
            frozen.write_text(json.dumps(dict(binary_sha256=sha(exe),sources=generated['generated_sources']),indent=2)+'\n')
        base_identity = helper.freeze(Path(generated['base_exe']))
        proof = read(exe.parent/'gpu_equivalence.json')
        if (not proof['complete'] or not proof['normalized_raw_identical']
                or not proof['normalized_resources_identical'] or proof['kernels'] != 172
                or proof['base_binary_sha256'] != base_identity['binary_sha256']
                or proof['instrumented_binary_sha256'] != sha(exe)):
            raise ValueError('host-range build lacks exact GPU equivalence')
        for info in proof['files'].values():
            if sha(info['path']) != info['sha256']:
                raise ValueError('raw GPU SASS/resource evidence changed')
        if proof['tool_sha256'] != sha(ROOT/'tools/bench/verify_stage2_ntt_range.py'):
            raise ValueError('GPU equivalence verifier changed')
        if generated['base_sources'] != base_identity['sources']:
            raise ValueError('instrumentation base differs')
        range_binding = dict(project_sha256=sha(project/'instrumentation.json'),
            gpu_equivalence_sha256=sha(exe.parent/'gpu_equivalence.json'),base_identity=base_identity)
    identity, reference, manifest = helper.freeze(exe), read(a.reference), read(systems/'manifest.json')
    traced_identity = range_binding['base_identity'] if range_binding else identity
    if not reference['complete'] or traced_identity['binary_sha256'] != manifest['sha256']:
        raise ValueError('requires a completed reference and the exact traced binary')
    key = next((k for k,v in reference['identity'].items() if v == traced_identity), None)
    if key is None:
        raise ValueError('binary is absent from the reference matrix')
    case = next(c for c in reference['cases'] if c['name'] == 'm4423_large')
    expected = next(r for r in reference['runs'] if r['case'] == case['name'] and r['key'] == key and r['category'] == 'timing')
    if sha(case['save']) != case['save_sha256'] or manifest['save_sha256'] != case['save_sha256']:
        raise ValueError('saved curve changed')
    if manifest['sources'] != traced_identity['sources'] or manifest['frozen_manifest_sha256'] != traced_identity['snapshot_sha256']:
        raise ValueError('Systems source identity differs')
    for name, want in identity['sources'].items():
        if sha(exe.parent/'sources'/name) != want:
            raise ValueError('frozen dependency changed: ' + name)
    for flag, val in (('--device','1'), ('--d',str(case['D'])), ('--b2',str(case['B2'])), ('--arena-mb','6300')):
        if manifest['command'][manifest['command'].index(flag)+1] != val:
            raise ValueError('Systems/reference configuration differs')
    trace = systems/'trace.sqlite'
    if read(systems/'audit.json')['trace_sha256'] != sha(trace):
        raise ValueError('Systems trace changed')
    if fields((systems/'engine.log').read_text(), 'descent_values') != expected['leaf']:
        raise ValueError('Systems complete output differs')
    shapes = select_shapes(trace, manifest['source_root'], traced_identity['sources'])
    binding = dict(identity=identity, reference_sha256=sha(a.reference),
        systems_manifest_sha256=sha(systems/'manifest.json'), trace_sha256=sha(trace),
        save_sha256=sha(case['save']), helper_sha256=sha(HELPER), tool_sha256=sha(__file__),
        shapes=shapes,range_instrumentation=range_binding)
    if a.collect_only:
        plan = read(out/'plan.json')
        if plan['binding'] != binding or sha(out/'collector_capture.py') != binding['tool_sha256'] or sha(out/'helper_capture.py') != binding['helper_sha256']:
            raise ValueError('capture tools, inputs or selection changed')
        if (out/'exit.txt').read_text(encoding='utf-8-sig').strip() != '0':
            raise ValueError('serial capture did not complete')
        results = {}
        for target, shape in shapes.items():
            dest = out/target
            if (dest/'exit.txt').read_text(encoding='utf-8-sig').strip() != '0':
                raise ValueError('capture failed: ' + target)
            for name, want in plan['files'].items():
                if sha(out/name) != want:
                    raise ValueError('prepared capture file changed: ' + name)
            with (dest/'metrics.csv').open('wb') as stream:
                subprocess.run([str(a.ncu), '--rename-kernels','0','--import',str(dest/'trace.ncu-rep'),
                    '--csv','--page','raw'],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
            rows = list(csv.DictReader((dest/'metrics.csv').read_text(encoding='utf-8-sig').splitlines()))
            samples = [r for r in rows if r.get('ID','').isdigit()]
            if len(samples) != 1:
                raise ValueError('exactly one kernel required: ' + target)
            sample, units = samples[0], rows[0]
            vector = lambda x: '(' + ', '.join(str(v) for v in x) + ')'
            norm = lambda x: re.sub(r'\((?:bool|int)\)\s*', '', x).replace(' ', '').replace('false','0').replace('true','1')
            for column, want in [('Grid Size',vector(shape['grid'])), ('Block Size',vector(shape['block'])), ('Device','1')]:
                if sample[column] != want:
                    raise ValueError('wrong actual geometry/device: ' + target + '/' + column)
            if norm(sample['Kernel Name']) != norm(shape['name']):
                raise ValueError('wrong kernel specialization: ' + target)
            numeric = lambda k: float(sample[k].replace(',',''))
            for suffix, key in (('dynamic','dynamic_shared_bytes'), ('static','static_shared_bytes')):
                column = 'launch__shared_mem_per_block_' + suffix
                scale = {'byte/block':1, 'Kbyte/block':1000}.get(units[column])
                if scale is None or abs(numeric(column)*scale-shape[key]) > 0.01:
                    raise ValueError('wrong actual shared memory: ' + target + '/' + suffix)
            if numeric('launch__registers_per_thread') != shape['registers']:
                raise ValueError('register identity differs: ' + target)
            text = (dest/'engine.log').read_text()
            if range_binding:
                both = text + (dest/'app.log').read_text()
                if len(re.findall(r'^s2_ncu_range: target='+re.escape(target)+r' .* begin=1$',both,re.M)) != 1 or len(re.findall(r'^s2_ncu_range_end: end=1$',both,re.M)) != 1:
                    raise ValueError('exact profiling range did not execute once: ' + target)
            records = [json.loads(s) for s in (dest/'results.jsonl').read_text().splitlines()]
            if len(records) != 1:
                raise ValueError('one complete curve required')
            result = records[0]
            for k in ('N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors'):
                if result[k] != expected['result'][k]:
                    raise ValueError('arithmetic input/output differs: ' + k)
            if fields(text,'descent_values') != expected['leaf']:
                raise ValueError('complete leaf fingerprint differs')
            coverage = fields(text,'s4_multiply_stats')
            for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'):
                if coverage[k] != expected['coverage'][k]:
                    raise ValueError('mandatory arithmetic coverage differs: ' + k)
            for token in ('mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0',
                          'gmp_check_bad=0','gmp_selftest_bad=0','pending=0'):
                if token not in text:
                    raise ValueError('missing mandatory check: ' + token)
            metrics = {k:dict(value=v,unit=units.get(k,'')) for k,v in sample.items()
                if k.startswith(('gpu__time_duration','dram__bytes','dram__throughput','sm__throughput',
                    'sm__warps_active','smsp__issue_active','smsp__average_warps_issue_stalled_',
                    'smsp__inst_executed','gpc__cycles_elapsed','launch__','l1tex__t_sectors_pipe_lsu_mem_local_op_'))}
            artifacts = {name:sha(dest/name) for name in ('metrics.csv','trace.ncu-rep','engine.log','results.jsonl','app.log','run.log','exit.txt')}
            results[target] = dict(shape=shape,kernel={k:sample[k] for k in ('Kernel Name','Grid Size','Block Size','Device')},
                metrics=metrics,artifacts=artifacts,leaf=expected['leaf'],coverage=coverage,result=result)
        (out/'summary.json').write_text(json.dumps(dict(complete=True,binding=binding,results=results,
            scope='Eight actual GPU1 launches with full curve/check identity. Replay metrics only; no whole-curve speedup or production capacity certification.'),indent=2)+'\n')
        print(json.dumps(dict(complete=True,kernels=len(results))));return
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('use an empty output directory')
    (out/'collector_capture.py').write_bytes(Path(__file__).read_bytes())
    (out/'helper_capture.py').write_bytes(HELPER.read_bytes())
    environment = dict(manifest['env'],CUDA_LAUNCH_BLOCKING='0')
    capture_lines = ["$ErrorActionPreference='Stop'", "$env:PSModulePath=$env:WINDIR+'\\System32\\WindowsPowerShell\\v1.0\\Modules'", "Set-Location -LiteralPath '" + str(ROOT) + "'"]
    commands = {}
    order = ['n27_tile_forward','n27_tile_inverse','n27_outer_m7_forward',
             'n27_outer_m8_forward','n27_outer_m8_inverse','n27_outer_m7_inverse',
             'n11_b990_tile_forward','n11_b990_tile_inverse']
    for target in order:
        shape = shapes[target]
        dest = out/target;dest.mkdir()
        ini = dest/'manual.ini';ini.write_text('device=1\n')
        app = [str(exe),'--ini',str(ini),'--save',case['save'],'--device','1',
            '--b2',str(case['B2']),'--d',str(case['D']),'--arena-mb','6300',
            '--factor-only','--log-level','debug','--log',str(dest/'engine.log'),'--results',str(dest/'results.jsonl')]
        reset = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
        wrapper = dest/'app.cmd'
        range_env = f'set "S2_NCU_TARGET={target}"\n' if range_binding else 'set "S2_NCU_TARGET="\n'
        wrapper.write_text('@echo off\n'+reset+range_env+''.join(f'set "{k}={v}"\n' for k,v in environment.items())+
            safe_args(app)+f' > "{dest / "app.log"}" 2>&1\nexit /b %errorlevel%\n')
        command = [str(a.ncu),'--rename-kernels','0','--devices','1','--target-processes','all',
            '--clock-control','none','--cache-control','none','--kernel-name-base','mangled']
        if range_binding:
            command += ['--profile-from-start','0','--kernel-name',shape['mangled']]
        else:
            command += ['--kernel-id','::'+shape['mangled']+':'+str(shape['matched_launches_before']+1)]
        command += ['--launch-skip','0','--launch-count','1']
        for section in ('LaunchStats','SpeedOfLight','Occupancy','SchedulerStats','WarpStateStats','InstructionStats','MemoryWorkloadAnalysis'):
            command += ['--section',section]
        command += ['--metrics','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
            '--export',str(dest/'trace'),'C:/Windows/System32/cmd.exe','/d','/c',str(wrapper)]
        profile = dest/'profile.cmd'
        profile.write_text('@echo off\n'+safe_args(command)+f' > "{dest / "run.log"}" 2>&1\nexit /b %errorlevel%\n')
        quote = lambda s: "'" + str(s).replace("'", "''") + "'"
        capture_lines += [f"if((Get-FileHash -LiteralPath {quote(exe)} -Algorithm SHA256).Hash.ToLower() -ne '{identity['binary_sha256']}'){{throw 'binary changed'}}",
            'Set-Content -LiteralPath '+quote(out/'step.txt')+' -Value '+quote(target),
            '& C:/Windows/System32/cmd.exe /d /c '+quote(profile),
            '$taskExit=$LASTEXITCODE', 'Set-Content -LiteralPath '+quote(dest/'exit.txt')+' -Value $taskExit',
            'if($taskExit -ne 0){Set-Content -LiteralPath '+quote(out/'exit.txt')+' -Value $taskExit;exit $taskExit}']
        commands[target] = dict(app=app,ncu=command,selection=shape)
    capture_lines += ['Set-Content -LiteralPath '+quote(out/'step.txt')+" -Value 'all captures terminal'",
        'Set-Content -LiteralPath '+quote(out/'exit.txt')+' -Value 0','exit 0']
    (out/'capture_all.ps1').write_text('\n'.join(capture_lines)+'\n')
    files = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'plan.json').write_text(json.dumps(dict(binding=binding,environment=environment,commands=commands,files=files,
        scope='Serial administrator capture of an explicit host profiling range in a GPU-identical isolated build; verify actual geometry and complete curve output/checks against the original production reference.'),indent=2)+'\n')
    print(out/'capture_all.ps1')


if __name__ == '__main__':
    main()
