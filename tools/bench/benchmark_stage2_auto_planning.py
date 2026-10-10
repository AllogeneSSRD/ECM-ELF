"""Interleave old/new complete Auto B2 invocations with identical inputs.

Measures full process, worker planning and engine total separately. Preserves
every warmup and formal run. No GPU configuration changes; hardware telemetry
is read only. The caller must keep compilation and other device work stopped.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-exe','exe','profile','stage1-profile','save','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--stage1-batch',type=int,default=8)
    p.add_argument('--stage1-exponent',choices=('lcm','choose12'),default='lcm')
    p.add_argument('--lower-b2',type=int,default=12500000000)
    p.add_argument('--upper-b2',type=int,default=23700000000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--powershell',type=Path,required=True)
    a = p.parse_args()
    assert a.device >= 0 and 3 <= a.repeats <= 20 and 0 < a.lower_b2 <= a.upper_b2
    out = a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    identities = {}

    def freeze(path):
        path = path.resolve();h = sha(path);identities[str(path)] = h
        copy = out/'inputs'/h/path.name;copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,copy);assert sha(copy) == h
        return copy

    for path in (a.profile,a.stage1_profile,a.save,Path(__file__)):
        freeze(path)
    binaries = dict(baseline=a.baseline_exe.resolve(),current=a.exe.resolve())
    for exe in binaries.values():
        freeze(exe);freeze(exe.parent/'gmp-10.dll');freeze(exe.parent/'build_manifest.json')
        manifest = json.loads((exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
        assert manifest['sha256'].lower() == sha(exe)
        if exe == binaries['current']:
            for relative,expected in manifest['source_hashes'].items():
                assert sha(ROOT/relative) == expected.lower(), relative
                freeze(ROOT/relative)
    assert sha(binaries['baseline'].parent/'gmp-10.dll') == sha(binaries['current'].parent/'gmp-10.dll')
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8-sig'))
    assert profile['summary']['complete'] == 1 and profile['profile']['format'] == 4
    policy = profile['policy'];environment = dict(os.environ)
    for key,value in policy['environment'].items():
        assert type(value) is int and value >= 0
        environment['NTT_'+key.upper()] = str(value)
    ini = out/'ecm.ini';ini.write_text('verbose=false\nstage2_debug_log=false\nstage2_b2=0\n')
    common = ['--ini',str(ini),'--save',str(a.save.resolve()),'--device',str(a.device),
        '--batch-mb',str(policy['batch_mb']),'--arena-mb',str(policy['arena_mb']),
        '--owner-budget-mb',str(policy['fold_mb']),'--log-level','quiet',
        '--tune-profile',str(a.profile.resolve()),'--auto-b2',
        '--stage1-tune-profile',str(a.stage1_profile.resolve()),'--stage1-batch',str(a.stage1_batch),
        '--stage1-exponent',a.stage1_exponent,'--curves','1']
    cases = dict(unrestricted=[],bounded=['--auto-min-b2',str(a.lower_b2),'--auto-max-b2',str(a.upper_b2)])
    rows = [];choices = {};mandatory = checked = 0
    report = dict(complete=False,repeats=a.repeats,warmups=1,source_identities=identities,
                  device=a.device,runs=rows,hardware_changes=0)
    def publish():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    publish();monitor = None;telemetry = None;proc = None
    try:
        smi = shutil.which('nvidia-smi');assert smi
        telemetry = (out/'telemetry.csv').open('w')
        monitor = subprocess.Popen([smi,'--query-gpu=timestamp,index,uuid,utilization.gpu,clocks.current.sm,temperature.gpu,power.draw,memory.used',
            '--format=csv','-lms','1000'],stdout=telemetry,stderr=subprocess.STDOUT)
        for repeat in range(a.repeats+1):
            order = [(case,variant) for case in cases for variant in binaries]
            if repeat % 2:order.reverse()
            for case,variant in order:
                stem = f'{case}_{variant}_{repeat}'
                receipt = out/(stem+'.jsonl');log = out/(stem+'.log')
                command = [str(binaries[variant])]+common+cases[case]+['--results',str(receipt),'--log',str(log)]
                start = time.perf_counter()
                with (out/(stem+'.console.log')).open('w') as console:
                    proc = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=console,stderr=subprocess.STDOUT)
                    print(f'{stem} parent_pid={proc.pid}',flush=True)
                    if repeat == 0:
                        # Module collection is confined to discarded warmups.
                        module_script = out/(stem+'.modules.ps1')
                        module_script.write_text(
                            '$ErrorActionPreference="Stop"\n'+
                            f'$parentPid={proc.pid}\n'+
                            '$child=$null\nfor($i=0;$i -lt 20;$i++){\n'+
                            ' $child=Get-CimInstance Win32_Process -Filter "ParentProcessId=$parentPid" | Where-Object Name -eq "ecm_cuda_stage2.exe" | Select-Object -First 1\n'+
                            ' if($child){break};Start-Sleep -Milliseconds 100\n}\n'+
                            'if(-not $child){throw "worker module sample unavailable"}\n'+
                            '$rows=@()\nforeach($taskPid in @($parentPid,$child.ProcessId)){\n'+
                            ' $mods=@((Get-Process -Id $taskPid).Modules | Where-Object { $_.ModuleName -in @("ecm_cuda_stage2.exe","gmp-10.dll") } | ForEach-Object {\n'+
                            ' [pscustomobject]@{path=$_.FileName;sha256=(Get-FileHash -LiteralPath $_.FileName -Algorithm SHA256).Hash.ToLowerInvariant()} })\n'+
                            ' $rows+=[pscustomobject]@{pid=$taskPid;modules=$mods}\n}\n$rows | ConvertTo-Json -Depth 5\n')
                        capture = subprocess.run([str(a.powershell),'-NoProfile','-File',str(module_script)],
                            capture_output=True,text=True,errors='replace',timeout=15)
                        (out/(stem+'.modules.json')).write_text(capture.stdout)
                        (out/(stem+'.modules.stderr')).write_text(capture.stderr)
                        assert capture.returncode == 0, capture.stderr
                        modules = json.loads(capture.stdout)
                        assert len(modules) == 2 and all(len(row['modules']) == 2 for row in modules)
                        expected = {str(binaries[variant]).lower():sha(binaries[variant]),
                            str(binaries[variant].parent/'gmp-10.dll').lower():sha(binaries[variant].parent/'gmp-10.dll')}
                        assert all(expected[m['path'].lower()] == m['sha256'] for r in modules for m in r['modules'])
                    code = proc.wait(timeout=600)
                process = time.perf_counter()-start
                assert code == 0, (stem,code)
                value = json.loads(receipt.read_text(encoding='utf-8-sig'))
                assert value['status'] == 'stage2_completed' and value['hits'] == value['bad_factors'] == 0
                assert value['requested_B2'] == value['requested_D'] == value['requested_carrier_exponent'] == 0
                choice = value['auto_plan']
                assert choice['T1_source'] == 'measured_stage1_profile' and choice['memory_rejected'] == choice['route_rejected'] == 0
                stable = {k:v for k,v in choice.items() if k != 'free_bytes'}
                if case not in choices:choices[case] = stable
                assert choices[case] == stable, ('Auto decision/model changed',stem)
                text = log.read_text(encoding='utf-8-sig')
                wall = next(line for line in text.splitlines() if line.startswith('stage2_full_wall:'))
                assert 'clean=1' in wall
                engine = float(re.search(r'\btotal=([0-9.]+)',wall)[1])
                assert process >= value['seconds'] >= engine
                assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b',text)
                assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b',text)
                tests = [line for line in text.splitlines() if line.startswith('s4_multiply_stats:')]
                assert len(tests) == 1
                fields = dict(re.findall(r'(\w+)=([^\s]+)',tests[0]))
                assert fields['gmp_selftest_bad'] == fields['gmp_check_bad'] == '0'
                mandatory += int(fields['gmp_selftest_cases']);checked += int(fields['gmp_checked'])
                assert int(fields['gmp_selftest_cases']) > 0 and int(fields['gmp_checked']) > 0
                row = dict(case=case,variant=variant,repeat=repeat,engine_seconds=engine,
                    process_seconds=process,worker_seconds=value['seconds'],planning_seconds=value['auto_planning_seconds'])
                rows.append(row);publish();print(json.dumps(row),flush=True)
        summary = {}
        for case in cases:
            summary[case] = {}
            for variant in binaries:
                formal = [r for r in rows if r['case']==case and r['variant']==variant and r['repeat']>0]
                assert len(formal) == a.repeats
                summary[case][variant] = {key: statistics.median(r[key] for r in formal)
                    for key in ('engine_seconds','process_seconds','worker_seconds','planning_seconds')}
        assert all(sha(Path(path)) == h for path,h in identities.items())
        report.update(summary=summary,choices=choices,curves=len(rows),
                      mandatory_cases=mandatory,gmp_checked=checked,arithmetic_bad=0,actual_modules_match=True)
        publish()
    except BaseException as error:
        report['failure'] = repr(error);publish();raise
    finally:
        # Do not leave a curve running if warmup module collection fails.
        if proc is not None and proc.poll() is None:
            proc.wait(timeout=600)
        if monitor is not None:monitor.terminate();monitor.wait(timeout=10)
        if telemetry is not None:telemetry.close()
    with (out/'telemetry.csv').open() as stream:
        samples = list(csv.reader(stream))[1:]
    device_samples = [r for r in samples if len(r)==8 and r[1].strip()==str(a.device)]
    expected_uuid = profile['device']['uuid_hex'].replace('-','').lower()
    assert device_samples and all(r[2].strip().removeprefix('GPU-').replace('-','').lower()==expected_uuid for r in device_samples)
    loaded = [r for r in device_samples if float(r[3].split()[0])>50]
    assert loaded
    report['telemetry'] = dict(device_samples=len(device_samples),loaded_samples=len(loaded),
        sm_mhz={key:fn(float(r[4].split()[0]) for r in loaded) for key,fn in
                (('min',min),('median',statistics.median),('max',max))},
        power_w={key:fn(float(r[6].split()[0]) for r in loaded) for key,fn in
                 (('min',min),('median',statistics.median),('max',max))})
    report['complete'] = True
    publish()
    print(json.dumps(dict(complete=True,curves=len(rows),summary=summary)))


if __name__ == '__main__':
    main()
