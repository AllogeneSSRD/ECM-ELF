"""Fixed-shape CUDA G-tree layout A/B with mandatory arithmetic checks.

Uses one binary and copied Stage1 points; no tune rows or production queue edits.
The override is deliberately excluded from automatic selection and calibration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from gpu_telemetry import TelemetryRecorder

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(20000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(text, prefix):
    found = re.findall(r'^' + re.escape(prefix) + r': (.*)$', text, re.M)
    if not found:
        raise AssertionError('missing ' + prefix)
    return dict(re.findall(r'(\w+)=([^\s]+)', found[-1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--save', type=Path)
    p.add_argument('--record', type=int, default=1)
    p.add_argument('--d', type=int, default=1141140)
    p.add_argument('--b2', type=int, default=1300000000000)
    p.add_argument('--carrier', type=int, default=5987)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--order', default='0,1,1,0,0,1')
    p.add_argument('--fixtures', action='store_true')
    a = p.parse_args()
    if not a.fixtures and (not a.save or a.record < 1):
        p.error('benchmark requires --save and record >= 1')
    if a.device < 0 or a.carrier < 0 or a.carrier > 16384:
        p.error('device >= 0 and carrier in 0..16384')
    order = [int(x) for x in a.order.split(',')]
    if not order or any(x not in (0,1) for x in order):
        p.error('order must contain 0/1')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__,out/'collector_snapshot.py')
    identities = {str(a.exe.resolve()):sha(a.exe)}
    if a.fixtures:
        from test_edwards_stage2_relocation import save
        save(out/'fixture.save',(1<<127)-1,26,20)
        point = (out/'fixture.save').read_text()
        a.d=2310; a.b2=1000000; a.carrier=0
    else:
        identities[str(a.save.resolve())]=sha(a.save)
        points=[s for s in a.save.read_text(encoding='utf-8-sig').splitlines() if s.startswith('METHOD=')]
        if a.record>len(points):
            p.error('record exceeds available save points')
        point=points[a.record-1]+'\n'
    base={k:v for k,v in os.environ.items() if not k.startswith('NTT_') and k!='CUDA_VISIBLE_DEVICES'}
    report=dict(complete=False, identities=identities, independent_ecm_coverage=False,
        D=a.d,B2=a.b2,carrier=a.carrier,device=a.device,runs=[])
    report['collector_sha256']=sha(out/'collector_snapshot.py')
    report.update(N_hex=re.search(r'\bN=0x([\da-fA-F]+);',point)[1],
        B1=int(re.search(r'\bB1=(\d+);',point)[1]),sigma=int(re.search(r'\bSIGMA=(\d+);',point)[1]))
    report['target_bits']=int(report['N_hex'],16).bit_length()
    modes=[('checked',1,0),('fault',1,1)] if a.fixtures else [(f'{i:02d}_layout{layout}',layout,0) for i,layout in enumerate(order)]
    for label,layout,fault in modes:
        directory=out/label; directory.mkdir()
        (directory/'point.save').write_text(point,encoding='utf-8')
        (directory/'ecm.toml').write_text('[stage2]\nstage2_auto_carrier=false\nstage2_auto_b2=false\n'
            'stage2_tune_runtime=false\nstage2_short_calibration=false\n',encoding='utf-8')
        sentinel=directory/'unused_tune.toml'; sentinel.write_text('must never be parsed or rewritten\n',encoding='utf-8')
        original=sha(sentinel)
        env=base|{'NTT_GROOT_BALANCED':str(layout)}
        if a.fixtures:
            env.update(NTT_GROOT_DEVICE_TEST='1',NTT_GROOT_DEVICE_CHECK='1',NTT_GROOT_DEVICE_TEST_BAD=str(fault))
        command=[str(a.exe.resolve()),'--config',str(directory/'ecm.toml'),
            '--save',str(directory/'point.save'),'--curves','1','--device',str(a.device),
            '--b2',str(a.b2),'--d',str(a.d),'--carrier-exponent',str(a.carrier),
            '--auto-carrier','0','--tune-runtime','0','--short-calibration','0',
            '--short-ntt-batch-tune','0','--tune-profile',str(sentinel),'--factor-only',
            '--arena-mb','0','--owner-budget-mb','960','--batch-mb','64','--ntt-counts','1',
            '--results',str(directory/'internal.jsonl'),'--submission',str(directory/'results.txt'),
            '--log',str(directory/'engine.log'),'--debug-log-file',str(directory/'debug.log'),'--log-level','batches']
        (directory/'command.json').write_text(json.dumps(dict(command=command,environment={k:v for k,v in env.items() if k.startswith('NTT_')}),indent=2),encoding='utf-8')
        monitor=None
        if not a.fixtures and shutil.which('nvidia-smi'):
            monitor=TelemetryRecorder(a.device,directory/'telemetry.jsonl')
        start=time.perf_counter()
        try:
            with (directory/'console.log').open('wb') as log:
                result=subprocess.run(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            elapsed=time.perf_counter()-start
        finally:
            if monitor is not None:
                monitor.close()
        text=(directory/'console.log').read_text(encoding='utf-8',errors='replace')
        if (directory/'engine.log').exists():
            text+='\n'+(directory/'engine.log').read_text(encoding='utf-8',errors='replace')
        if (directory/'debug.log').exists():
            text+='\n'+(directory/'debug.log').read_text(encoding='utf-8',errors='replace')
        assert sha(sentinel)==original,'experiment touched tune profile'
        row=dict(label=label,layout=layout,wall_seconds=elapsed,exit=result.returncode,fault=bool(fault))
        if monitor is not None:
            row.update(telemetry_identity=monitor.identity,telemetry_samples=monitor.samples)
        if fault:
            assert result.returncode and 'resident GMP node mismatch' in text and not (directory/'internal.jsonl').exists()
        else:
            assert result.returncode==0,(label,text[-3000:])
            raw=json.loads((directory/'internal.jsonl').read_text(encoding='utf-8-sig'))
            assert raw['status']=='NF' and not raw['stage2-engine'].get('tune_plan',{}).get('selected',False)
            assert raw['source-build']==identities[str(a.exe.resolve())]
            row.update(engine=fields(text,'stage2_full_wall'),split=fields(text,'real_batched_split'),
                layout_info=fields(text,'groot_layout'),checks=fields(text,'s4_multiply_stats'),
                root=fields(text,'real_batched_rootfold'))
            assert int(row['layout_info']['balanced'])==layout
            # Diagnostic fixture arithmetic is intentionally not eligible as a
            # clean performance/tune observation, even when all checks pass.
            assert row['engine']['clean']==('0' if a.fixtures else '1')
            assert int(row['checks']['gmp_checked'])>0 and int(row['checks']['gmp_selftest_cases'])>0
            assert row['checks']['gmp_check_bad']==row['checks']['gmp_selftest_bad']=='0'
            if a.fixtures:
                fixture=fields(text,'gdevice_fixture');assert fixture['cases']=='48' and fixture['bad']=='0'
        report['runs'].append(row)
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(label=label,wall=elapsed,exit=result.returncode,split=row.get('split'))),flush=True)
    if a.fixtures:
        # Reject incompatible uses before any calibration/profile access. The
        # invalid TOML sentinel also proves fixed-layout runs do not read rows
        # whose shape identity still describes the legacy layout.
        report['guards']=[]
        for name,changes,extra,value in [
            ('bad_value',{},[],'2'),('automatic_D',{'--d':'0'},[],'1'),
            ('sampling',{'--tune-runtime':'1'},[],'1'),
            ('auto_carrier',{'--auto-carrier':'1'},[],'1'),
            ('calibration',{'--short-calibration':'1'},[],'1'),
            ('short_batch',{'--short-ntt-batch-tune':'1'},[],'1'),
            ('auto_B2',{'--b2':'0'},['--auto-b2'],'1'),('tune',{},['--tune','ecm'],'1')]:
            guard=out/('guard_'+name);guard.mkdir()
            for item in ('point.save','ecm.toml','unused_tune.toml'):
                shutil.copy2(directory/item,guard/item)
            guard_command=[arg.replace(str(directory),str(guard)) for arg in command]+extra
            for flag,new_value in changes.items():
                guard_command[guard_command.index(flag)+1]=new_value
            if name=='tune':
                # A formal tune cannot accept save/curve CLI arguments. Use its
                # valid grammar and a fixed TOML D to reach the layout guard.
                with (guard/'ecm.toml').open('a',encoding='utf-8') as config:
                    config.write('stage2_d=2310\n')
                guard_command=[str(a.exe.resolve()),'--config',str(guard/'ecm.toml'),
                    '--device',str(a.device),'--tune','ecm']
            rejected=subprocess.run(guard_command,cwd=guard,env=base|{'NTT_GROOT_BALANCED':value},
                capture_output=True,timeout=30)
            text=(rejected.stdout+rejected.stderr).decode('utf-8',errors='replace')
            (guard/'console.log').write_text(text,encoding='utf-8')
            (guard/'command.json').write_text(json.dumps(guard_command,indent=2),encoding='utf-8')
            assert rejected.returncode and not (guard/'internal.jsonl').exists(),name
            assert ('NTT_GROOT_BALANCED must be 0 or 1' if name=='bad_value' else 'G-root layout experiments require') in text,name
            assert sha(guard/'unused_tune.toml')==original,name
            report['guards'].append(name)
    if not a.fixtures:
        digests={(r['root']['digest_words'],r['root']['digest_sum'],r['root']['digest_xor']) for r in report['runs']}
        assert len(digests)==1,'G roots changed with layout'
        report['root_digest_identical']=True
    assert all(sha(Path(path))==digest for path,digest in identities.items()),'original input changed'
    report['complete']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
