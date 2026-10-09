"""Check native padded-tree geometry and two/three-buffer workspace policy.

Initializes CUDA only to query a plan; executes no curve arithmetic. Retains
commands, inputs, frozen build identity and every native JSON response.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from bench_stage2_production import fields, freeze, sha
from calibrate_stage2_d import phi
from test_stage2_request_program import verify_s4_memory


def verify_giant_memory(plan):
    """Independent component formula; do not add its peak to the NTT peak."""
    m = plan['giant_memory']
    if m['version'] != 2 or m['process_peak_complete'] or m['admission_model']:
        raise ValueError('wrong giant component scope')
    if not m['valid']:
        if m['reason'] not in ('diagnostic_workspace_not_modeled','small_prime_interval_not_modeled'):
            raise ValueError('unexpected unsupported giant model')
        return
    v = m['policy']
    p, n, w = plan['P'], plan['I'], (plan['bits']+63)//64
    q, cap, segfix, base = m['chunk_points'], v['initial_points'], 0, False
    if not q or q % p or m['point_chunks'] != (n+q-1)//q:
        raise ValueError('wrong giant chunk geometry')
    def workspace(values=0,products=0):
        return 8*(5*w+cap*(1+2*w)+(values+products)*w+((segfix+1)*w if segfix else 0)+(2*w if base else 0))
    if workspace() != m['initial_bytes']:
        raise ValueError('wrong initial giant workspace')
    peak, count, points = workspace(), 0, 0
    for c in m['chunks']:
        x = c['points']
        chain = not v['force_ladder'] and x >= v['chain_min']
        block = v['short_block'] if v['short_block'] and x < v['short_max'] and v['short_block'] < v['chain_block'] else v['chain_block']
        chains = (x+block-1)//block
        seeds = 2*chains+1 if chain else x
        cap = max(cap,seeds)
        base |= chain and v['seed_device'] and v['seed_pair']
        ns = (x+v['segment']-1)//v['segment']
        coordinate = 16*x*w
        resident = v['resident_requested'] and v['resident_eligible'] and v['exact_segments'] and coordinate <= v['resident_limit_bytes']
        segment = 8*ns*w if chain or resident else 0
        if segment and v['exact_segments']:
            segfix = max(segfix,v['segment'])
        ng = (ns+v['group']-1)//v['group'] if resident else 0
        groups = 8*(ng+v['group']+1)*w if resident else 0
        legacy = 8*(4*chains+2)*w if chain and not v['seed_device'] else 0
        prepare = workspace()+(coordinate if chain else 0)+segment+legacy
        tree = workspace()+((coordinate if chain else 0)+segment+groups if resident else 0)
        expected = dict(route='chain' if chain else 'ladder',block=block if chain else 0,seeds=seeds,
                        resident=resident,segments=ns,groups=ng,point_capacity=cap,workspace_bytes=workspace(),
                        coordinate_bytes=coordinate,segment_bytes=segment,group_bytes=groups,
                        legacy_seed_bytes=legacy,prepare_bytes=prepare,tree_bytes=tree)
        if any(c[k] != value for k,value in expected.items()):
            raise ValueError('giant component differs from independent formula')
        peak = max(peak,prepare,tree)
        count += c['repeat']
        points += x*c['repeat']
    products = (p+v['accumulation_block']-1)//v['accumulation_block'] if v['compact_products'] else p
    if m['value_bytes']!=8*p*w or m['product_bytes']!=8*products*w:
        raise ValueError('wrong separate value/product payload')
    if count != m['point_chunks'] or points != n or len(m['chunks'])>2 or workspace()!=m['after_giant_bytes'] or workspace(p,products)!=m['accumulation_bytes'] or cap!=m['final_point_capacity'] or max(peak,workspace(p,products))!=m['peak_bytes']:
        raise ValueError('wrong retained giant capacity or compressed chunk sequence')


def verify_ntt_memory(plan):
    """Check scope, successful-prefix accounting and no-eviction equivalence."""
    memory, request = plan['ntt_memory'], plan['request_program']
    if memory['version'] != 1 or memory['process_peak_complete'] or memory['admission_model'] or memory['cold_trim_modeled'] or memory['fallback_modeled']:
        raise ValueError('wrong NTT allocator model scope')
    if memory['cap_bytes'] != plan['arena_cap_bytes'] or memory['pool'] != plan['tree_workspace']['pool']:
        raise ValueError('allocator policy differs from native plan')
    payload = memory['final_payload']
    if payload['total_bytes'] != sum(payload[k] for k in ('big_bytes','digit_bytes','table_bytes','base_bytes')):
        raise ValueError('final allocator components do not balance')
    if memory['peak_bytes'] < payload['total_bytes'] or (memory['cap_bytes'] and memory['peak_bytes'] > memory['cap_bytes']):
        raise ValueError('invalid owned arena prefix peak')
    if not request['valid']:
        if memory['valid'] or memory['finished']:
            raise ValueError('unsupported topology accepted by allocator model')
        return
    if not memory['valid'] or not memory['supported']:
        raise ValueError('supported allocator program rejected')
    counters = memory['counters']
    chunks = sum(p['chunks'] for p in request['phases'])
    if memory['finished']:
        if memory['stopped_at'] is not None or memory['reason'] != 'ok' or counters['calls'] != chunks or len(memory['checkpoints']) != len(request['blocks']):
            raise ValueError('completed allocator sequence differs from request program')
        if not counters['cap_evictions'] and not memory['carry_check']:
            if memory['peak_bytes'] != request['ntt_retained_bytes'] or payload['total_bytes'] != request['ntt_retained_bytes']:
                raise ValueError('no-eviction state differs from independent retention calculation')
            if counters['fuse_builds'] != len(request['cache_shapes']) or counters['fuse_hits']+counters['fuse_builds'] != counters['calls']:
                raise ValueError('no-eviction fuse reuse differs from request count')
    else:
        stop = memory['stopped_at']
        if memory['reason'] not in ('fuse_cap_refusal','big_cap_refusal','digits_cap_refusal') or not stop or not 0 < counters['calls'] <= chunks:
            raise ValueError('wrong successful-prefix cap refusal')
        block = request['blocks'][stop['block']]
        r = block['requests'][stop['request_index']]
        if not 0 <= stop['repeat_index'] < block['repeat'] or any(stop[k] != r[k] for k in ('ma','mb','pairs')) or not stop['N'] or not stop['slices']:
            raise ValueError('cap refusal location differs from request program')
    for point in memory['checkpoints']:
        if point['peak_bytes'] < point['payload']['total_bytes'] or point['peak_bytes'] > memory['peak_bytes']:
            raise ValueError('invalid allocator checkpoint peak')


def dense_groups(p):
    pad = 1 << (p-1).bit_length()
    degree = [0]*pad+[1]*p+[0]*(pad-p)
    for i in range(pad-1,0,-1):
        degree[i] = degree[2*i]+degree[2*i+1]
    result,base = [],pad//2
    while base:
        groups = {}
        for i in range(base):
            a,b = degree[2*base+2*i:2*base+2*i+2]
            if a and b:
                key = tuple(sorted((a+1,b+1)))
                groups[key] = groups.get(key,0)+1
        result.extend((a,b,count) for (a,b),count in sorted(groups.items()))
        base //= 2
    return result


def payload(groups,lengths,bits,buffers,physical,budget,chunk_max):
    big = output = pairs = chunks = 0
    digits,keyed = {},{}
    for a,b,nb in groups:
        n,slots = lengths[b],2*b-1
        c,test = 1,nb
        while test:
            if 8*test*((buffers if physical else 3)*n+slots+(2 if physical else 0)) <= budget:
                c = test
                break
            test //= 2
        if chunk_max:
            c = min(c,chunk_max)
        pairs += nb
        chunks += (nb+c-1)//c
        big = max(big,8*buffers*n*c)
        output = max(output,8*((bits+63)//64)*(a+b-1)*c)
        for offset in range(0,nb,c):
            count = min(c,nb-offset)
            digits[n,count] = max(digits.get((n,count),0),8*(slots+2)*count)
            keyed[n,count] = 24*n*count
    return dict(groups=len(groups),pairs=pairs,chunks=chunks,shared_big_peak_bytes=big,
                digit_retained_bytes=sum(digits.values()),keyed_big_retained_bytes=sum(keyed.values()),
                output_peak_bytes=output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--carrier-exponent', type=int, default=0)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--d',type=int,nargs='+',default=[810810,1021020,1381380,1531530,524288])
    p.add_argument('--runtime-check',type=Path,help='Validate F-tree capacity predictions against a completed same-build arithmetic matrix')
    p.add_argument('--batch-mb',type=int,default=256)
    a = p.parse_args()
    if any(d<6 or d%2 for d in a.d) or a.batch_mb<=0:
        p.error('D must be even and at least 6; batch budget must be positive')
    out, exe, save = a.output.resolve(), a.exe.resolve(), a.save.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    identity, save_sha, tool_sha = freeze(exe), sha(save), sha(__file__)
    ini = out/'manual.ini'
    ini.write_text(f'device={a.device}\n', encoding='utf-8')
    rows = []
    data = dict(complete=False, identity=identity, save=str(save), save_sha256=save_sha,
                tool_sha256=tool_sha, curves_executed=0, rows=rows)
    try:
        groups_by_d = {d:dense_groups(phi(d)//2) for d in a.d}
        coefficients = {2}|{b for groups in groups_by_d.values() for _,b,_ in groups}
        # Independently query each transform via a native fold at P=h.
        # phi(4h)/2=h for power-of-two h; D6 supplies h=1.
        lengths = {}
        anchor_rows = data['anchors'] = []
        for b in sorted(coefficients):
            h = b-1
            d = 6 if h==1 else 4*h
            env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
            env['NTT_D_MODEL'] = '0'
            command = [str(exe),'--ini',str(ini),'--save',str(save),
                '--carrier-exponent',str(a.carrier_exponent),'--device',str(a.device),
                '--b2','2600000000000','--d',str(d),'--plan-only']
            proc = subprocess.run(command,env=env,capture_output=True,timeout=60)
            log = out/f'anchor_{b}.log'
            log.write_bytes(proc.stdout+proc.stderr)
            if proc.returncode:
                raise ValueError('native anchor query failed')
            plans = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines() if line.startswith('{')]
            if len(plans)!=1 or plans[0]['P']!=h:
                raise ValueError('wrong anchor geometry')
            lengths[b] = plans[0]['fold_length']
            if b==2:
                lengths[1] = plans[0]['tree_length']
            anchor_rows.append(dict(coefficients=b,command=command,plan=plans[0],log_sha256=sha(log)))
        for d in a.d:
            for pool,reuse,physical in ((pool,reuse,physical) for pool,reuse in ((1,0),(1,1),(0,0),(0,1)) for physical in (0,1)):
                name = f'd{d}_pool{pool}_reuse{reuse}_physical{physical}'
                env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
                env.update(NTT_D_MODEL='0', NTT_ARENA_WORKSPACE_POOL=str(pool),
                           NTT_WORKSPACE_REUSE_BQ=str(reuse),NTT_S4_WORKSPACE_BUDGET=str(physical),NTT_S4_BATCH_MB=str(a.batch_mb))
                command = [str(exe), '--ini', str(ini), '--save', str(save),
                    '--carrier-exponent', str(a.carrier_exponent), '--device', str(a.device),
                    '--b2', '2600000000000', '--d', str(d), '--curves', '1',
                    '--arena-mb', '6300', '--owner-budget-mb', '1024','--batch-mb',str(a.batch_mb), '--plan-only']
                proc = subprocess.run(command, env=env, capture_output=True, timeout=60)
                (out/(name+'.log')).write_bytes(proc.stdout+proc.stderr)
                if proc.returncode:
                    raise ValueError(name+' failed; raw output retained')
                candidates = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines()
                              if line.startswith('{')]
                if len(candidates) != 1:
                    raise ValueError('expected one native JSON plan')
                plan = candidates[0]
                verify_ntt_memory(plan)
                verify_giant_memory(plan)
                verify_s4_memory(plan)
                buffers = 2 if pool and reuse else 3
                degree = phi(d)//2
                tree_coeffs = max((b for _,b,_ in groups_by_d[d]),default=1)
                nf, nt = plan['fold_length'], plan['tree_length']
                expected_arena = 8*(buffers*nf+2*degree+1+
                                    2*(buffers*nt+2*tree_coeffs-1))
                expected = dict(type='stage2_plan', curves_executed=0, D=d, P=degree,
                                workspace_buffers=buffers, carrier_exponent=a.carrier_exponent,
                                geometry_version=3,tree_operand_coefficients=tree_coeffs,tree_length=lengths[tree_coeffs],
                                fold_big_bytes=8*buffers*nf, arena_estimate_bytes=expected_arena)
                for key, value in expected.items():
                    if plan[key] != value:
                        raise ValueError(f'{name}: {key}={plan[key]} expected {value}')
                if plan['arena_estimate_fits'] != (expected_arena <= (6300 << 20)):
                    raise ValueError('arena fit predicate differs from reported estimate')
                tree = plan['tree_workspace']
                if not tree['supported'] or tree['pool']!=bool(pool) or tree['batch_bytes']!=a.batch_mb*(1<<20) or plan['arena_estimate_kind']!='legacy_additive':
                    raise ValueError('wrong model scope/arena estimate kind')
                want = payload(groups_by_d[d],lengths,plan['bits'],buffers,bool(physical),tree['batch_bytes'],tree['chunk_max'])
                for key,value in want.items():
                    if tree[key]!=value:
                        raise ValueError(f'{name}: tree.{key}={tree[key]} expected {value}')
                cache = {v['N']:v for v in tree['cache_shapes']}
                if set(cache)!={lengths[b] for _,b,_ in groups_by_d[d]}:
                    raise ValueError('cache shapes differ from exact tree requests')
                table = sum(v['table_bytes'] for v in cache.values())
                base = sum(v['base_bytes'] for v in cache.values())
                retained = (want['shared_big_peak_bytes'] if pool else want['keyed_big_retained_bytes'])+want['digit_retained_bytes']+table+base
                if tree['table_retained_bytes']!=table or tree['base_retained_bytes']!=base or tree['ntt_retained_bytes']!=retained:
                    raise ValueError('NTT retention differs from cache/physical payload ledger')
                if sha(exe) != identity['binary_sha256'] or sha(save) != save_sha or sha(__file__) != tool_sha:
                    raise ValueError('build, input or collector changed')
                rows.append(dict(name=name, command=command,
                    environment={k: v for k, v in env.items() if k.startswith('NTT_')},
                    plan=plan, log_sha256=sha(out/(name+'.log'))))
                print(name, 'OK', flush=True)
        if a.runtime_check:
            check = json.loads(a.runtime_check.read_text(encoding='utf-8'))
            if not check['complete'] or check['identity']['binary_sha256']!=identity['binary_sha256'] or check['input']['save_sha256']!=save_sha:
                raise ValueError('runtime matrix has wrong build/input or is incomplete')
            verified = []
            for run in check['runs']:
                policy = run['environment']
                pool = int(policy.get('NTT_ARENA_WORKSPACE_POOL','1'))
                reuse = int(policy['NTT_WORKSPACE_REUSE_BQ'])
                physical = int(policy.get('NTT_S4_WORKSPACE_BUDGET','0'))
                matches = [r for r in rows if r['plan']['D']==int(run['shape']['D']) and
                    int(r['environment']['NTT_ARENA_WORKSPACE_POOL'])==pool and
                    int(r['environment']['NTT_WORKSPACE_REUSE_BQ'])==reuse and
                    int(r['environment']['NTT_S4_WORKSPACE_BUDGET'])==physical]
                command = run['command']
                batch = int(command[command.index('--batch-mb')+1])
                if len(matches)!=1 or batch!=a.batch_mb:
                    raise ValueError('runtime policy has no matching plan')
                debug = Path(run['debug_log'])
                if sha(debug)!=run['debug_sha256']:
                    raise ValueError('runtime raw evidence changed')
                record = fields(next(line for line in debug.read_text(encoding='utf-8').splitlines()
                    if line.startswith('s4_phase_memory: phase=ftree ')),'s4_phase_memory')
                tree = matches[0]['plan']['tree_workspace']
                for actual,predicted in [('groups','groups'),('pairs','pairs'),('chunks','chunks'),
                    ('request_big_peak_bytes','shared_big_peak_bytes'),('request_output_peak_bytes','output_peak_bytes')]:
                    if int(record[actual])!=tree[predicted]:
                        raise ValueError(f'actual F-tree {actual} differs from plan')
                if int(record['live_big_peak_bytes'])!=tree['shared_big_peak_bytes'] or int(run['workspace']['legacy_mallocs']):
                    raise ValueError('runtime tree pool has fallback or different physical peak')
                if int(record['ntt_payload_peak_bytes'])!=tree['ntt_retained_bytes']:
                    raise ValueError('actual F-tree NTT retention differs from complete component plan')
                verified.append(dict(name=run['name'],record=record))
            data['runtime'] = dict(path=str(a.runtime_check),sha256=sha(a.runtime_check),verified=verified)
        data['complete'] = True
    except Exception as exc:
        data['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
