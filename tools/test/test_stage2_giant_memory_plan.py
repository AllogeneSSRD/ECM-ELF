"""Native plan-only giant policy checks. Initializes the selected CUDA device
for planning, executes no curve arithmetic and changes no GPU power settings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import freeze, sha
from test_stage2_workspace_plan import verify_giant_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--save',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--carrier-exponent',type=int,default=8011)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out,exe,save = args.output.resolve(),args.exe.resolve(),args.save.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    identity,save_sha,tool_sha = freeze(exe),sha(save),sha(__file__)
    verifier = Path(__file__).with_name('test_stage2_workspace_plan.py')
    verifier_sha = sha(verifier)
    ini = out/'manual.ini'
    ini.write_text(f'device={args.device}\n',encoding='utf-8')
    cases = [
        ('legacy_points',1381380,2600000000000,{}),
        ('legacy_products',1381380,2600000000000,{'NTT_S3_COMPACT_PRODUCTS':'0'}),
        ('bounded_points',1381380,2600000000000,{'NTT_GIANT_CHUNK_FLOOR':'1'}),
        ('ladder_tail',1381380,1381380*(253440+32000-2),{}),
        ('forced_ladder',1381380,2600000000000,{'NTT_GIANT_LADDER':'1'}),
        ('legacy_seed',1381380,2600000000000,{'NTT_GIANT_SEED_DEVICE':'0'}),
        ('resident_budget_fallback',1381380,2600000000000,{'NTT_DEVICE_GLEAF_MAX_MB':'1'}),
        ('short_chain',2310,100000000,{'NTT_GIANT_CHAIN_MIN':'0','NTT_GIANT_CHAIN_SMALL_BLOCK':'4'}),
        ('small_prime_no_reuse',2310,100000000,{'NTT_SMALL_PRIME_REUSE':'0'}),
        ('diagnostic_rejected',1381380,2600000000000,{'NTT_GIANT_CHAIN_CHECK':'1'}),
        ('huge_B2',1381380,9000000000000000000,{}),
    ]
    rows = []
    result = dict(complete=False,identity=identity,save=str(save),save_sha256=save_sha,
                  tool_sha256=tool_sha,verifier_sha256=verifier_sha,curves_executed=0,rows=rows)
    try:
        for name,d,b2,overrides in cases:
            env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
            env.update(NTT_D_MODEL='0',NTT_GIANT_POINT_BUDGET_KB='262144',NTT_GIANT_CHUNK_FLOOR='0',
                       NTT_GIANT_CHAIN_BLOCK='64',NTT_GIANT_CHAIN_MIN='32768',NTT_SMALL_PRIME_REUSE='1')
            env.update(overrides)
            command = [str(exe),'--ini',str(ini),'--save',str(save),'--carrier-exponent',str(args.carrier_exponent),
                       '--device',str(args.device),'--b2',str(b2),'--d',str(d),'--arena-mb','6300',
                       '--owner-budget-mb','1024','--batch-mb','256','--plan-only']
            proc = subprocess.run(command,env=env,capture_output=True,timeout=60)
            log = out/(name+'.log');log.write_bytes(proc.stdout+proc.stderr)
            if name in ('legacy_seed','small_prime_no_reuse','short_chain'):
                key = {'legacy_seed':'NTT_GIANT_SEED_DEVICE','small_prime_no_reuse':'NTT_SMALL_PRIME_REUSE',
                       'short_chain':'NTT_GIANT_CHAIN_SMALL_BLOCK'}[name]
                expected = '0' if name=='short_chain' else '1'
                if not proc.returncode or ('production Stage2 requires '+key+'='+expected) not in log.read_text(encoding='utf-8'):
                    raise ValueError('production algorithm guard did not reject '+key)
                rows.append(dict(name=name,command=command,engine_rejected=True,returncode=proc.returncode,
                    environment=overrides,log_sha256=sha(log)))
                print(name,'production guard OK',flush=True)
                continue
            if proc.returncode:
                raise ValueError(name+' plan failed; raw output retained')
            plans = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines() if line.startswith('{')]
            if len(plans)!=1 or plans[0]['curves_executed']:
                raise ValueError('one plan-only result is required')
            plan = plans[0];verify_giant_memory(plan)
            m = plan['giant_memory']
            if name=='diagnostic_rejected':
                if m['valid'] or m['reason']!='diagnostic_workspace_not_modeled':
                    raise ValueError('diagnostic allocator footprint accepted')
            elif not m['valid']:
                raise ValueError('ordinary giant policy unsupported')
            if name=='ladder_tail' and not (m['chunks'][0]['route']=='chain' and m['chunks'][1]['route']=='ladder' and m['final_point_capacity']==32000):
                raise ValueError('chain -> ladder retained-capacity transition missing')
            if name=='resident_budget_fallback' and any(c['resident'] for c in m['chunks']):
                raise ValueError('resident budget incorrectly accepted')
            if sha(exe)!=identity['binary_sha256'] or sha(save)!=save_sha or sha(__file__)!=tool_sha:
                raise ValueError('source/input/binary changed during query')
            rows.append(dict(name=name,command=command,environment={k:v for k,v in env.items() if k.startswith('NTT_')},
                             plan=plan,log_sha256=sha(log)))
            print(name,'OK',flush=True)
        if sha(exe)!=identity['binary_sha256'] or sha(save)!=save_sha or sha(__file__)!=tool_sha or sha(verifier)!=verifier_sha:
            raise ValueError('source/input/binary changed during query')
        result['complete']=True
    except Exception as exc:
        result['error']=str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
