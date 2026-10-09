"""Validate predictive request topology against completed native arithmetic runs.

Plans initialize CUDA but execute no curve. The independent dense oracle checks
ordered requests, then compares all five phase counters and optional native
ordering signatures. No-eviction NTT retention is not a process peak/admission.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from bench_stage2_production import fields, freeze, sha

PHASES = ('ftree','gtrees','fold','descent','inverse')


def dense_degrees(p):
    pad = 1 << (p-1).bit_length()
    d = [0]*pad+[1]*p+[0]*(pad-p)
    for i in range(pad-1,0,-1):
        d[i] = d[2*i]+d[2*i+1]
    return d,pad


@lru_cache(maxsize=32)
def dense_tree(p, phase):
    d,pad = dense_degrees(p)
    result,base = [],pad//2
    while base:
        groups = {}
        for i in range(base,2*base):
            a,b = d[2*i:2*i+2]
            if a and b:
                key = tuple(sorted((a+1,b+1)))
                groups[key] = groups.get(key,0)+1
        result.extend((phase,a,b,nb,0,a+b-1) for (a,b),nb in sorted(groups.items()))
        base //= 2
    return tuple(result)


def dense_program(p, points):
    result = list(dense_tree(p,0))
    size = 1
    while size < p+1:
        nxt = min(2*size,p+1)
        result.extend(((4,nxt,size,1,0,nxt),(4,size,nxt,1,0,nxt)))
        size = nxt
    h = 0
    for begin in range(0,points,p):
        g = min(p,points-begin)
        result.extend(dense_tree(g,1))
        if not h:
            h = g+1
        else:
            product = g+h
            result.append((2,g+1,h,1,0,product))
            if product > p:
                k = product-p
                result.extend(((2,k,k,1,0,k),(2,k,p+1,1,0,p)))
                h = p
            else:
                h = product
    result.append((3,p,p,1,0,p))
    d,pad = dense_degrees(p)
    base = 1
    while base < pad:
        groups = {}
        for i in range(base,2*base):
            a,b = d[2*i:2*i+2]
            if a and b:
                for key in ((a,b),(b,a)):
                    groups[key] = groups.get(key,0)+1
        result.extend((3,a+b,b+1,nb,b,a) for (a,b),nb in sorted(groups.items()))
        base *= 2
    return result


def native_records(text, prefix):
    records = [fields(line,prefix) for line in text.splitlines() if line.startswith(prefix+':')]
    if len(records) != len({r['phase'] for r in records}):
        raise ValueError('duplicate phase records')
    return {r['phase']:r for r in records}


def verify_topology(plan):
    request = plan['request_program']
    if not request['valid'] or not request['supported'] or request['admission_model'] or request['process_peak_complete']:
        raise ValueError('wrong conditional topology contract')
    if plan['G']>128:
        raise ValueError('dense gate limited to 128 G roots; compressed huge-repeat fixture is CPU-only')
    actual = [tuple(r[k] for k in ('phase','ma','mb','pairs','first','count'))
              for block in request['blocks'] for _ in range(block['repeat']) for r in block['requests']]
    expected = dense_program(plan['P'],plan['I'])
    if actual!=expected:
        raise ValueError('predictive request order differs from independent dense topology')
    for i,phase in enumerate(request['phases']):
        if phase['phase']!=PHASES[i] or phase['groups']!=sum(r[0]==i for r in expected) or phase['pairs']!=sum(r[3] for r in expected if r[0]==i):
            raise ValueError('phase aggregate differs from independent topology')
    return request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--matrix',type=Path,nargs='+',required=True)
    p.add_argument('--plan-matrix',type=Path,help='Optional completed same-build tree planner matrix')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    exe,out = a.exe.resolve(),a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    identity,tool_sha = freeze(exe),sha(__file__)
    for name,want in identity['sources'].items():
        if sha(ROOT/name)!=want:
            raise ValueError('current compiled source differs: '+name)
    rows = []
    data = dict(complete=False,identity=identity,tool_sha256=tool_sha,rows=rows,plan_rows=[])
    try:
        if a.plan_matrix:
            plans = json.loads(a.plan_matrix.read_text(encoding='utf-8'))
            if not plans['complete'] or plans['identity']!=identity:
                raise ValueError('matching complete plan matrix required')
            for row in plans['rows']:
                raw = a.plan_matrix.parent/(row['name']+'.log')
                if sha(raw)!=row['log_sha256']:
                    raise ValueError('native plan raw evidence changed')
                responses = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
                if responses!=[row['plan']]:
                    raise ValueError('stored plan differs from raw native response')
                verify_topology(row['plan'])
                data['plan_rows'].append(dict(name=row['name'],matrix_sha256=sha(a.plan_matrix),log_sha256=row['log_sha256']))
        for m,path in enumerate(a.matrix):
            source = json.loads(path.read_text(encoding='utf-8'))
            matrix_sha = sha(path)
            if not source['complete'] or source['mode']!='check' or source['identity']!=identity:
                raise ValueError('completed matching-build arithmetic checks required')
            if sha(source['input']['save'])!=source['input']['save_sha256'] or sha(path.parent/'collector.py')!=source['tool_sha256']:
                raise ValueError('runtime input or collector changed')
            for i,run in enumerate(source['runs']):
                if run['environment'].get('NTT_REQUEST_AUDIT')!='1' or run['category']!='check':
                    raise ValueError('native request audit required; no timing rows')
                if sha(run['debug_log'])!=run['debug_sha256'] or sha(run['log'])!=run['log_sha256']:
                    raise ValueError('raw runtime evidence changed')
                results = run['command'][run['command'].index('--results')+1]
                if sha(results)!=run['result_sha256']:
                    raise ValueError('published arithmetic result changed')
                text = Path(run['log']).read_text(encoding='utf-8')+'\n'+Path(run['debug_log']).read_text(encoding='utf-8')
                if fields(text,'target_descent_values')!=run['leaf'] or int(run['coverage']['gmp_selftest_bad']) or int(run['coverage']['gmp_check_bad']):
                    raise ValueError('native arithmetic evidence differs or failed')
                for tag in ('real_batched_folddevice','scaled_root_device','scaled_frontier_device'):
                    if fields(text,tag)['enabled']!='1':
                        raise ValueError('resident topology required: '+tag)
                if int(run['workspace']['legacy_mallocs']) or int(fields(text,'real_batched_breakdown')['arena_overflow']):
                    raise ValueError('per-call allocation fallback is outside this contract')
                original = run['command']
                command = [str(exe),'--plan-only']
                for option in ('--ini','--save','--b2','--d','--device','--carrier-exponent','--arena-mb','--owner-budget-mb','--batch-mb'):
                    if option in original:
                        command.extend(original[original.index(option):original.index(option)+2])
                env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
                env.update(run['environment'])
                log = out/f'{m}_{i}_plan.log'
                proc = subprocess.run(command,env=env,capture_output=True,timeout=60)
                log.write_bytes(proc.stdout+proc.stderr)
                if proc.returncode:
                    raise ValueError('native plan query failed')
                plans = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines() if line.startswith('{')]
                if len(plans)!=1:
                    raise ValueError('expected one native plan')
                plan = plans[0]
                request = verify_topology(plan)
                memories = native_records(text,'s4_phase_memory')
                signatures = native_records(text,'stage2_request_audit')
                for phase in request['phases']:
                    name = phase['phase']
                    for key in ('groups','pairs','chunks','request_big_peak_bytes','request_output_peak_bytes'):
                        if int(memories[name][key])!=phase[key]:
                            raise ValueError(f'{run["name"]}: {name}.{key} differs from execution')
                    for key in ('multiplier','addend'):
                        if int(signatures[name][key])!=phase[key]:
                            raise ValueError(f'{run["name"]}: {name} ordered request signature differs')
                for key in ('multiplier','addend'):
                    if int(signatures['all'][key])!=request[key]:
                        raise ValueError('cross-phase request order differs')
                cache = request['cache_shapes']
                table,base = (sum(c[key] for c in cache) for key in ('table_bytes','base_bytes'))
                big = request['shared_big_peak_bytes'] if plan['tree_workspace']['pool'] else request['keyed_big_retained_bytes']
                if table!=request['table_retained_bytes'] or base!=request['base_retained_bytes'] or big+table+base+request['digit_retained_bytes']!=request['ntt_retained_bytes']:
                    raise ValueError('no-eviction NTT payload components do not balance')
                peak = int(run['workspace']['full_peak_bytes'])
                if peak>request['ntt_retained_bytes']:
                    raise ValueError('actual NTT retained peak exceeds conditional no-eviction model')
                if sha(path)!=matrix_sha or sha(exe)!=identity['binary_sha256'] or sha(__file__)!=tool_sha:
                    raise ValueError('evidence, build or checker changed')
                rows.append(dict(matrix=str(path.resolve()),matrix_sha256=matrix_sha,run=run['name'],command=command,
                                 plan=plan,log_sha256=sha(log),native_phases=memories,native_signatures=signatures,
                                 native_ntt_full_peak_bytes=peak))
                print(path.parent.name,run['name'],'OK',flush=True)
        data['complete'] = True
    except Exception as exc:
        data['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
