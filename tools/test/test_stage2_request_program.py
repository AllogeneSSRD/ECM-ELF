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
from stage2_memory_ledger import parse as parse_memory_ledger

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
    if request['version']>=2:
        blocks=request['blocks']
        for i,block in enumerate(blocks):
            leaves=(plan['I']%plan['P'] if i==len(blocks)-2 and plan['I']%plan['P'] else plan['P']) if 0<i<len(blocks)-1 else 0
            if block['tree_leaves']!=leaves:raise ValueError('tree lease leaves differ from dense topology')
            for r in block['requests']:
                source='host' if r['phase'] in (0,4) else 'tree_raw' if r['phase']==1 else 'fold_owner' if r['phase']==2 or r['first']==0 else 'frontier_owner'
                if r['input']!=source:raise ValueError('resident/host input route differs')
    for i,phase in enumerate(request['phases']):
        if phase['phase']!=PHASES[i] or phase['groups']!=sum(r[0]==i for r in expected) or phase['pairs']!=sum(r[3] for r in expected if r[0]==i):
            raise ValueError('phase aggregate differs from independent topology')
    return request


def verify_s4_memory(plan):
    """Independent integer event simulator; S4 component only, never admission."""
    m=plan['s4_memory']
    if m['version']!=1 or m['process_peak_complete'] or m['admission_model'] or m['fallback_modeled']:
        raise ValueError('wrong S4 component scope')
    if not m['valid']:
        if m['reason'] not in ('unsupported_request_program','unsupported_resident_policy'):raise ValueError('unexpected S4 rejection')
        return None
    verify_topology(plan)
    v=m['policy'];w=(plan['bits']+63)//64;desc={s['operand']:s for s in m['shapes']}
    fields=('raw_a','raw_b','output','pack_a','pack_b','modulus','shape_constant','canonical','selftest_digits','selftest_output','tree_metadata')
    live={name:0 for name in fields};peak=0;allocations=frees=requests=chunks=trees=0;keys=set();sig=[1,0]
    def record():
        nonlocal peak
        peak=max(peak,sum(live.values()))
    def allocate(name,n):
        nonlocal allocations
        if not n:return
        live[name]+=n;allocations+=1;record()
    def release(name,n=None):
        nonlocal frees
        n=live[name] if n is None else n
        if n:live[name]-=n;frees+=1;record()
    def reserve(name,n):
        if n>live[name]:release(name);allocate(name,n)
    def snapshot():return {name+'_bytes':n for name,n in live.items()}|{'total_bytes':sum(live.values())}
    def words(values):
        for value in values:
            sig[0]=(sig[0]*0x9e3779b185ebca87)&((1<<64)-1)
            sig[1]=(sig[1]*0x9e3779b185ebca87+value)&((1<<64)-1)
    def boundary(kind,n=0):words((0x5334424f554e4400,kind,n))
    def trim():
        if v['trim_raw']:release('raw_a');release('raw_b')
        if v['trim_output']:release('output')
    allocate('modulus',8*w)
    blocks=plan['request_program']['blocks']
    for bi,block in enumerate(blocks):
        if bi==1:
            if snapshot()!=m['after_inverse']:raise ValueError('S4 inverse checkpoint differs')
            boundary(1);trim()
        if bi+1==len(blocks):
            if snapshot()!=m['after_giant']:raise ValueError('S4 giant checkpoint differs')
            boundary(4);trim()
        if block['repeat']>128:raise ValueError('dense S4 gate requires bounded G repeats')
        for _ in range(block['repeat']):
            leaves=block['tree_leaves'];tree=bool(leaves)
            if tree:
                trees+=1;boundary(2,leaves);reserve('raw_a',16*leaves*w)
                reserve('raw_b',8*w*(leaves+(leaves+1)//2) if v['compact_raw'] and leaves>1 else 0 if v['compact_raw'] else 16*leaves*w)
                pad=1<<(leaves-1).bit_length()
                if pad>1:allocate('tree_metadata',24*(pad//2))
            for r in block['requests']:
                if tree and r['phase']!=1:release('tree_metadata');boundary(3,leaves);tree=False
                operand=max(r['ma'],r['mb']);q=desc[operand]
                if q['slots']!=2*operand-1 or q['slot_words']!=(q['slot_bits']+q['bpw']-1)//q['bpw']:
                    raise ValueError('invalid native S4 descriptor')
                c=1
                test=r['pairs']
                while test:
                    words_per=(v['buffers'] if v['physical_chunks'] else 3)*q['N']+q['slots']+(2 if v['physical_chunks'] else 0)
                    if 8*test*words_per<=v['batch_bytes']:c=test;break
                    test//=2
                if v['chunk_max']:c=min(c,v['chunk_max'])
                key=(q['slot_bits'],q['slot_words'],q['bpw'])
                if key not in keys:
                    keys.add(key);allocate('shape_constant',8*w)
                    allocate('selftest_digits',96*q['slot_words']*8);allocate('selftest_output',96*w*8)
                    release('selftest_digits');release('selftest_output')
                reserve('output',max(1,(c if v['chunk_output'] else r['pairs'])*(r['count'] if v['output_window'] else q['slots'])*w)*8)
                if r['input']=='host':reserve('raw_a',8*c*operand*w);reserve('raw_b',8*c*operand*w)
                if not live['canonical']:allocate('canonical',8)
                requests+=1;chunks+=(r['pairs']+c-1)//c
                words((0x5334524551554553,r['phase'],r['ma'],r['mb'],r['pairs'],r['first'],r['count'],q['N'],q['slots'],c,
                    ('host','tree_raw','fold_owner','frontier_owner').index(r['input'])))
            if tree:release('tree_metadata');boundary(3,leaves)
    if snapshot()!=m['final_payload'] or len(keys)!=m['shape_count']:raise ValueError('S4 final state differs')
    for _ in keys:release('shape_constant',8*w)
    for field in ('modulus','canonical','output','raw_a','raw_b','pack_a','pack_b'):release(field)
    if snapshot()!=m['released_payload'] or peak!=m['peak_bytes']:raise ValueError('S4 peak/destruction differs')
    if m['counters']!=dict(requests=requests,chunks=chunks,trees=trees,allocations=allocations,frees=frees):
        raise ValueError('S4 counters differ')
    if sig!=[m['multiplier'],m['addend']]:raise ValueError('S4 ordering signature differs')
    return m


def verify_s4_ledger(exe,run,prediction,identity):
    """Actual allocation sites at three boundaries; no component peak claim."""
    source=exe.parent/'sources/src/cuda/ecm_cuda_stage2.cu'
    if sha(source)!=identity['sources']['src/cuda/ecm_cuda_stage2.cu']:
        raise ValueError('compiled ledger source differs')
    text=source.read_text(encoding='utf-8')
    needles=dict(raw_a='CK(cudaMalloc(&d_rawA,',raw_b='CK(cudaMalloc(&d_rawB,',
        output='CK(cudaMalloc(&C.d_out,',pack_a='CK(cudaMalloc(&C.d_packA,',pack_b='CK(cudaMalloc(&C.d_packB,',
        modulus='CK(cudaMalloc(&R.dn,',shape_constant='CK(cudaMalloc(&S->dy,',canonical='CK(cudaMalloc(&R.dbad,',
        selftest_digits='CK(cudaMalloc(&dd, dig.size()',selftest_output='CK(cudaMalloc(&dout, (size_t)(cases * R.w)',
        tree_metadata='CK(cudaMalloc(&meta.device,')
    sites={}
    for field,needle in needles.items():
        if text.count(needle)!=1:raise ValueError('allocation anchor not unique: '+field)
        sites[field+'_bytes']='ecm_cuda_stage2.cu:'+str(text[:text.index(needle)].count('\n')+1)
    ledger=parse_memory_ledger(Path(run['debug_log']).read_text(encoding='utf-8'))
    result=[]
    for snapshot,key in [('after_inverse','after_inverse'),('after_giant_loop','after_giant'),('after_descent','final_payload')]:
        live={s['site']:int(s['bytes']) for s in ledger['sites'] if s['scope']=='live' and s['snapshot']==snapshot}
        actual={field:live.get(site,0) for field,site in sites.items()};actual['total_bytes']=sum(actual.values())
        if actual!=prediction[key]:raise ValueError('native S4 owned checkpoint differs: '+snapshot)
        result.append(dict(snapshot=snapshot,payload=actual))
    return result


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
                s4=verify_s4_memory(plan)
                if not s4:raise ValueError('resident run lacks S4 model')
                audit=fields(text,'stage2_s4_program_audit')
                if any(int(audit[k])!=s4[k] for k in ('multiplier','addend')):
                    raise ValueError('native S4 route/boundary order differs')
                s4_ledgers=verify_s4_ledger(exe,run,s4,identity) if source.get('memory_ledger') else []
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
                                 native_ntt_full_peak_bytes=peak,native_s4_signature=audit,s4_ledger_checkpoints=s4_ledgers))
                print(path.parent.name,run['name'],'OK',flush=True)
        data['complete'] = True
    except Exception as exc:
        data['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
