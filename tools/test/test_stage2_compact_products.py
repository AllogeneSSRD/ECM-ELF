"""Same-binary GPU gates for compact block-product buffers, with independent
GMP fixtures, complete leaf/factor equivalence and owned allocation ledgers.
Uses the selected GPU's current settings; never changes power or clocks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import freeze,sha,fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sanitizer',type=Path,help='Optional compute-sanitizer.exe for the P65 gate')
    args=parser.parse_args()
    out,exe,fixture=args.output.resolve(),args.exe.resolve(),args.fixtures.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    prepared=json.loads(fixture.read_text(encoding='utf-8'))
    if not prepared['complete']:raise ValueError('incomplete independent inputs')
    inputs={c['exponent']:c for c in prepared['cases']}
    helper=ROOT/'tools/bench/bench_stage2_carrier.py'
    identity,tool_sha,helper_sha,fixture_sha=freeze(exe),sha(__file__),sha(helper),sha(fixture)
    rows=[]
    result=dict(complete=False,identity=identity,tool_sha256=tool_sha,helper_sha256=helper_sha,
                fixture_sha256=fixture_sha,device=args.device,rows=rows)
    try:
        cases=[(f'm{e}',e,inputs[e]['D'],inputs[e]['B2'],False,True) for e in (37,67,29,253,16384)]
        cases+=[('generic8193',0,210,13230,False,False),
                ('P63',37,254,254*(2*63+17-2),False,False),
                ('P64',37,256,256*(2*64+17-2),False,False),
                ('P65',37,262,262*(2*65+17-2),False,False),
                ('G1_P65',37,262,262*20,False,False),
                ('frontier_fallback',37,210,13230,True,True)]
        for name,e,d,b2,fallback,oracle in cases:
            source=inputs[16384 if e==0 else e]
            if sha(source['save'])!=source['save_sha256']:raise ValueError('save changed')
            command=[sys.executable,str(helper),'--exe',str(exe),'--save',source['save'],
                     '--carrier-exponent',str(e),'--comparison','products','--b2',str(b2),'--d',str(d),
                     '--device',str(args.device),'--mode','check','--projection-only','--memory-ledger',
                     '--trim-phase-raw','--output',str(out/name)]
            if oracle:command+=['--fixtures',str(fixture)]
            if fallback:command+=['--frontier-alloc-fail']
            proc=subprocess.run(command,capture_output=True,timeout=600)
            (out/(name+'_driver.log')).write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:raise ValueError(name+' failed; raw evidence retained')
            m=json.loads((out/name/'measurements.json').read_text(encoding='utf-8'))
            if not m['complete'] or len(m['runs'])!=2 or m['identity']!=identity:raise ValueError('invalid gate matrix')
            rows.append(dict(name=name,command=command,matrix=str(out/name/'measurements.json'),
                             matrix_sha256=sha(out/name/'measurements.json'),
                             memory_reduction_bytes=m['product_memory_reduction_bytes'],
                             descent_reduction_bytes=m['product_descent_reduction_bytes']))
            print(name,'OK',flush=True)
        # Inject only a host-side wrong fixture answer. The device is untouched;
        # an expected arithmetic failure proves the GMP gate actually rejects it.
        env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        env.update(NTT_D_MODEL='0',NTT_S3_PRODUCTS_TEST='1',NTT_S3_PRODUCTS_TEST_BAD='1')
        command=[str(exe),'--ini',str(out/'P65/manual.ini'),'--save',inputs[37]['save'],
                 '--carrier-exponent','37','--device',str(args.device),'--d','262','--b2','37990',
                 '--curves','1','--factor-only']
        bad=subprocess.run(command,env=env,capture_output=True,timeout=60)
        (out/'oracle_rejection.log').write_bytes(bad.stdout+bad.stderr)
        if not bad.returncode or b'product fixture GMP mismatch' not in bad.stderr:
            raise ValueError('GMP product fixture accepted wrong host answer')
        result['oracle_rejection']=dict(command=command,environment={k:v for k,v in env.items() if k.startswith('NTT_')},
                                         returncode=bad.returncode,log_sha256=sha(out/'oracle_rejection.log'))
        if args.sanitizer:
            sanitizer=args.sanitizer.resolve()
            env.pop('NTT_S3_PRODUCTS_TEST_BAD')
            env['NTT_S3_COMPACT_PRODUCTS']='1'
            # Windows TCP injection listens under the target executable's name
            # and can prompt for firewall access on every new build path.
            if os.name=='nt':env['NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE']='named-pipes'
            debug=out/'sanitizer.debug.log'
            safe=[str(sanitizer),'--tool','memcheck','--error-exitcode','86']+command+['--debug-log-file',str(debug)]
            run=subprocess.run(safe,env=env,capture_output=True,timeout=600)
            log=out/'sanitizer.log';log.write_bytes(run.stdout+run.stderr)
            if run.returncode or b'ERROR SUMMARY: 0 errors' not in log.read_bytes():
                raise ValueError('compute-sanitizer failed; see retained log')
            check=fields(debug.read_text(encoding='utf-8'),'s3_product_fixture')
            if check['bad']!='0' or check['cases']!='45':raise ValueError('sanitizer product fixture incomplete')
            result['sanitizer']=dict(command=safe,binary_sha256=sha(sanitizer),log_sha256=sha(log),
                                    debug_sha256=sha(debug),fixture=check,returncode=run.returncode,
                                    connection_override=env.get('NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE'))
            print('compute-sanitizer OK',flush=True)
        if sha(__file__)!=tool_sha or sha(helper)!=helper_sha or sha(fixture)!=fixture_sha or sha(exe)!=identity['binary_sha256']:
            raise ValueError('tool/input/build changed during check')
        result.update(complete=True,curves_executed=2*len(rows)+(1 if args.sanitizer else 0),
                      expected_rejected_invocations=1)
    except Exception as exc:
        result['error']=str(exc);raise
    finally:
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
