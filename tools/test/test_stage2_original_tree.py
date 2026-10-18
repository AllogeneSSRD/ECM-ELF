"""Exercise balanced G roots with independent GMP nodes and fault sensitivity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from test_edwards_stage2_relocation import save,fields
from test_stage2_request_program import verify_topology,verify_s4_memory,native_records

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh evidence directory')
    case=save(out/'ed.save',(1<<127)-1,26,20);ini=out/'test.ini';ini.write_text('user=\ncomputer=\n')
    rows=[]
    for fault in (0,1):
        folder=out/('fault' if fault else 'checked');folder.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        env.update(NTT_GROOT_DEVICE_TEST='1',NTT_GROOT_DEVICE_CHECK='1',NTT_GROOT_DEVICE_TEST_BAD=str(fault),
                   NTT_TARGET_LEAF_HASH='1',NTT_REQUEST_AUDIT='1')
        cmd=list(map(str,[a.exe.resolve(),'--ini',ini,'--save',case['save'],'--b2',1000000,'--d',2310,
            '--device',a.device,'--relocation',0,'--factor-only','--short-calibration',0,'--tune-runtime',0,
            '--results',folder/'internal.jsonl','--submission',folder/'results.json.txt',
            '--log',folder/'engine.log','--debug-log-file',folder/'debug.log','--log-level','debug']))
        (folder/'command.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('NTT_')}),indent=2))
        run=subprocess.run(cmd,cwd=folder,env=env,capture_output=True,timeout=300)
        (folder/'driver.log').write_bytes(run.stdout+run.stderr)
        text=(run.stdout+run.stderr).decode('utf-8',errors='replace')+(folder/'debug.log').read_text()
        if fault:
            assert run.returncode!=0 and 'resident GMP node mismatch' in text
            assert not (folder/'internal.jsonl').exists()
        else:
            assert run.returncode==0
            fixture=fields(text,'gdevice_fixture');assert fixture['cases']=='48' and fixture['bad']=='0'
            result=json.loads((folder/'internal.jsonl').read_text());assert not result['factors'] and not result['relocation']
        rows.append(dict(fault=fault,exit=run.returncode,files={str(f.relative_to(folder)):sha(f) for f in folder.iterdir() if f.is_file()}))
        print('PASS','fault rejected' if fault else '48 independent GMP products and nodes',flush=True)
    # Both runtime policies use a 258-leaf tail. The wide arithmetic target
    # is a valid 8255-bit carrier of M127, independent of external datasets.
    for label,carrier,full in [('request_order',0,2),('request_order_wide',127*65,4)]:
        folder=out/label;folder.mkdir();b2=30030*(full*2880+258-2)
        common=list(map(str,[a.exe.resolve(),'--ini',ini,'--save',case['save'],'--b2',b2,'--d',30030,'--device',a.device,
            '--relocation',0,'--carrier-exponent',carrier,'--short-calibration',0,'--tune-runtime',0,'--arena-mb',6300,'--owner-budget-mb',640,'--batch-mb',256]))
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};env.update(NTT_REQUEST_AUDIT='1')
        cmd=common+list(map(str,['--factor-only','--results',folder/'internal.jsonl','--submission',folder/'results.json.txt',
            '--log',folder/'engine.log','--debug-log-file',folder/'debug.log','--log-level','debug']))
        run=subprocess.run(cmd,cwd=folder,env=env,capture_output=True,timeout=300)
        (folder/'driver.log').write_bytes(run.stdout+run.stderr);assert run.returncode==0
        query=subprocess.run(common+['--plan-only'],cwd=folder,env=env,capture_output=True,timeout=60)
        (folder/'plan.log').write_bytes(query.stdout+query.stderr);assert query.returncode==0
        records=[json.loads(s) for s in query.stdout.decode().splitlines() if s.startswith('{')]
        plans=[r for r in records if r.get('type')=='stage2_plan'];assert len(plans)==1
        plan=plans[0];request=verify_topology(plan);memory=verify_s4_memory(plan);assert memory
        text=(folder/'debug.log').read_text();assert int(fields(text,'groot_layout')['balanced'])==bool(carrier);signatures=native_records(text,'stage2_request_audit')
        for phase in request['phases']:
            for key in ('multiplier','addend'):assert int(signatures[phase['phase']][key])==phase[key]
        for key in ('multiplier','addend'):
            assert int(signatures['all'][key])==request[key]
            assert int(fields(text,'stage2_s4_program_audit')[key])==memory[key]
        rows.append(dict(request_order=True,carrier=carrier,files={str(f.relative_to(folder)):sha(f) for f in folder.iterdir() if f.is_file()}))
        print('PASS',label,'complete request/route order and 258-leaf tail',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(complete=True,engine_sha256=sha(a.exe),source_sha256=sha(__file__),runs=rows),indent=2)+'\n')
if __name__=='__main__':main()
