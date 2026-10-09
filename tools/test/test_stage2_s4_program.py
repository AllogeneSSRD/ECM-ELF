"""GPU diagnostic gates for conditional S4 request routes and owned checkpoints.

Uses current device settings. No power writes; these are not performance samples.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import freeze,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--large-save',type=Path,help='Optional M8011 cofactor save for D1381380/B2=2.6e12')
    p.add_argument('--large-reference',type=Path,help='Completed previous same-input full-leaf gate for large save')
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();exe=a.exe.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('use a fresh output directory')
    if bool(a.large_save)!=bool(a.large_reference):raise ValueError('large save and reference must be provided together')
    out.mkdir(parents=True,exist_ok=True)
    fixtures=json.loads(a.fixtures.read_text(encoding='utf-8'))
    if not fixtures['complete']:raise ValueError('complete independent input manifest required')
    inputs={c['exponent']:c for c in fixtures['cases']}
    helper=ROOT/'tools/bench/bench_stage2_carrier.py';verifier=Path(__file__).with_name('test_stage2_request_program.py')
    result=dict(complete=False,identity=freeze(exe),tool_sha256=sha(__file__),helper_sha256=sha(helper),
        verifier_sha256=sha(verifier),fixture_sha256=sha(a.fixtures),rows=[],curves_executed=0)
    def call(command,log):
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        proc=subprocess.run(command,env=env,capture_output=True,timeout=1200)
        log.write_bytes(proc.stdout+proc.stderr)
        if proc.returncode:raise ValueError('gate failed: '+str(log))
    try:
        cases=[('m37',37,inputs[37]['D'],inputs[37]['B2'],False,True),
            ('P63_tail1',37,254,254*(3*63-1),True,False),
            ('P64_tail1',37,256,256*(3*64-1),False,False),
            ('P65_tail1',37,262,262*(3*65-1),True,False),
            ('generic8193',0,210,13230,True,False)]
        matrices=[]
        for name,e,d,b2,raw,oracle in cases:
            source=inputs[16384 if e==0 else e]
            if sha(source['save'])!=source['save_sha256']:raise ValueError('save changed')
            command=[sys.executable,str(helper),'--exe',str(exe),'--save',source['save'],
                '--carrier-exponent',str(e),'--comparison','phase-output','--b2',str(b2),'--d',str(d),
                '--device',str(a.device),'--mode','check','--projection-only','--memory-ledger',
                '--request-audit','--require-resident','--output',str(out/name)]
            if raw:command+=['--trim-phase-raw']
            if oracle:command+=['--fixtures',str(a.fixtures.resolve())]
            call(command,out/(name+'_driver.log'))
            matrix=out/name/'measurements.json';m=json.loads(matrix.read_text(encoding='utf-8'))
            if not m['complete'] or m['identity']!=result['identity']:raise ValueError('invalid native matrix')
            matrices.append(matrix);result['curves_executed']+=len(m['runs'])
            result['rows'].append(dict(name=name,command=command,matrix=str(matrix),matrix_sha256=sha(matrix)))
            print(name,'OK',flush=True)
        if a.large_save:
            command=[sys.executable,str(helper),'--exe',str(exe),'--save',str(a.large_save.resolve()),
                '--carrier-exponent','8011','--comparison','products',
                '--b2','2600000000000','--d','1381380','--device',str(a.device),'--fold-mb','1024',
                '--baby-mb','640','--mode','check','--projection-only','--memory-ledger','--request-audit',
                '--require-resident','--trim-phase-raw','--output',str(out/'m8011')]
            call(command,out/'m8011_driver.log')
            matrix=out/'m8011/measurements.json';m=json.loads(matrix.read_text(encoding='utf-8'))
            old=json.loads(a.large_reference.read_text(encoding='utf-8'))
            if not m['complete'] or not old['complete'] or m['identity']!=result['identity']:raise ValueError('invalid large reference')
            if m['input']!=old['input']:raise ValueError('large reference target/save/bounds differ')
            for r in old['runs']:
                if sha(r['debug_log'])!=r['debug_sha256'] or sha(r['log'])!=r['log_sha256']:raise ValueError('old large raw evidence changed')
                if r['leaf']!=m['runs'][0]['leaf'] or sorted(r['result']['factors'])!=sorted(m['runs'][0]['result']['factors']):
                    raise ValueError('large complete target leaves/factors differ from old binary')
            matrices.append(matrix);result['curves_executed']+=len(m['runs'])
            result['rows'].append(dict(name='m8011',command=command,matrix=str(matrix),matrix_sha256=sha(matrix),
                reference=str(a.large_reference.resolve()),reference_sha256=sha(a.large_reference),old_complete_leaf_equal=True))
            print('m8011 OK',flush=True)
        command=[sys.executable,str(verifier),'--exe',str(exe),'--matrix',*[str(m) for m in matrices],
            '--output',str(out/'verification')]
        call(command,out/'verification_driver.log')
        verification=json.loads((out/'verification/checks.json').read_text(encoding='utf-8'))
        if not verification['complete'] or len(verification['rows'])!=result['curves_executed']:
            raise ValueError('incomplete route/owned-checkpoint verification')
        for key,path in [('tool_sha256',__file__),('helper_sha256',helper),('verifier_sha256',verifier),('fixture_sha256',a.fixtures)]:
            if sha(path)!=result[key]:raise ValueError('source/input changed during gate')
        result.update(complete=True,verification_sha256=sha(out/'verification/checks.json'))
    except Exception as exc:
        result['error']=str(exc);raise
    finally:
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
