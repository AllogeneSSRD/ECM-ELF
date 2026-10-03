"""Gate resident G roots with independent GMP nodes and complete real Stage2 outputs."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', type=Path, default=repo/'build_cuda_cmake/stage2_tree_gpu.exe')
p.add_argument('--device', type=int, default=1)
p.add_argument('--output', type=Path)
args = p.parse_args()
exe = args.exe.resolve()
out = (args.output or repo/'build_cuda_cmake'/('_gdevice_gate_'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()): raise RuntimeError('Use a fresh output directory')
base = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_ASYNC='1', NTT_S4_PACK_DIRECT='1', NTT_S4_FLAT_DIRECT='1',
            NTT_S4_GROOT_ONLY='1', NTT_ARENA_WORKSPACE_POOL='1', NTT_FUSE_COMPACT_SCRATCH='1',
            NTT_S4_CHUNK_OUTPUT='1', NTT_S4_CHUNK_MAX='2', NTT_S4_SAMPLE='96',
            NTT_S4_CHECK_EVERY='8', NTT_S4_ORACLE_ASYNC='1', NTT_S4_ORACLE_PACK='1',
            NTT_S4_OLDTAIL='0', NTT_GROOT_LEAF_STAGING='1', NTT_GROOT_COMPACT_RAW='1', NTT_GROOT_LEAF_CHUNK='0', NTT_SCALED_DESCENT='1', NTT_NO_PROGRESS='1', NTT_NAME_MAX='1')
rows = []

def run(name, n, b1=20, b2=1000, d=210, **settings):
    command = [str(exe), '--real', '--n-hex', format(n, 'x'), '--sigma', '26',
               '--b1', str(b1), '--b2', str(b2), '--d', str(d), '--device', str(args.device)]
    proc = subprocess.run(command, env=base | settings, cwd=out, capture_output=True, text=True, timeout=240)
    text = proc.stdout + proc.stderr
    log = out/(name+'.log'); log.write_text(text, encoding='utf-8')
    assert proc.returncode == 0, (name, proc.returncode, str(log))
    assert not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]|slot_canonical_bad=[1-9]', text), (name, str(log))
    def stat(prefix):
        line = re.search(r'(?m)^'+prefix+r': (.*)', text)
        assert line, (name, prefix)
        return dict(re.findall(r'(\w+)=(\S+)', line[1]))
    device = {k: int(v) for k, v in stat('real_batched_gdevice').items()}
    memory = {k: int(v) for k, v in stat('real_batched_gmemory').items()}
    assert len(memory) == 11
    assert memory['pinned_words'] + memory['pageable_words'] == device['leaf_words']
    assert memory['compact_raw'] == int(settings.get('NTT_GROOT_COMPACT_RAW','1'))
    assert memory['leaf_staging'] == int(settings.get('NTT_GROOT_LEAF_STAGING','1'))
    root = stat('real_batched_groot')
    leaf = stat('descent_values')
    stage = re.search(r'stage2:.*hits=(\d+) bad_factors=0 factors=(\S*) hit_primes=(\S*)', text)
    assert stage, (name, str(log))
    eligible = settings.get('NTT_GROOT_DEVICE') == '1' and not any(
        settings.get(k) == v for k, v in (('NTT_S4_HOSTPACK','1'), ('NTT_S4_PACK_DIRECT','0'),
                                        ('NTT_S4_FINAL_READBACK','1'), ('NTT_S4_GROOT_ONLY','0'), ('NTT_S4_OFF','1')))
    if eligible:
        assert device['trees'] > 0 and device['fallbacks'] == 0 and device['raw_peak_bytes'] > 0
        assert memory['rawA_peak_bytes'] + memory['rawB_peak_bytes'] == device['raw_peak_bytes']
        assert memory['rawA_peak_bytes'] >= memory['rawB_peak_bytes']
        if memory['leaf_staging'] and settings.get('NTT_S4_ASYNC') != '0':
            assert memory['pinned_trees'] > 0 and memory['pinned_words'] > 0
            if settings.get('NTT_GROOT_LEAF_CHUNK') == '3': assert memory['pinned_slices'] > memory['pinned_trees']
        elif not memory['leaf_staging']: assert memory['pinned_trees'] == 0 and memory['legacy_trees'] > 0
        else: assert memory['leaf_fallbacks'] > 0 and memory['pinned_trees'] == 0
        assert device['pairs'] > 0 and device['metadata_words'] == 3*device['pairs']
        if settings.get('NTT_S4_CARRY_TRACE') == '1': assert device['trace_words'] > 0
        else: assert device['resident_words'] > 0 and device['trace_words'] == 0
    elif settings.get('NTT_GROOT_DEVICE') == '1':
        assert device['trees'] == 0 and device['fallbacks'] > 0 and device['resident_words'] == 0
    if settings.get('NTT_GROOT_DEVICE_TEST') == '1':
        fixture = stat('gdevice_fixture'); w = (n.bit_length()+63)//64
        assert int(fixture['cases']) == 48 and int(fixture['words']) == 3420*w and fixture['bad'] == '0'
        assert int(fixture['checked_nodes']) == (9036 if eligible else 0)
        assert int(fixture['checked_words']) == (41272*w if eligible else 0)
        assert re.search(r'stage2_full_wall:.*clean=0', text)
    if settings.get('NTT_S4_OFF') != '1':
        window = stat('real_batched_outputwindow'); final = stat('real_batched_finalreadback')
        assert int(window['d2h_words']) + device['resident_words'] == int(final['copied_words']) + int(final['avoided_words'])
        trace = stat('real_batched_carrytrace')
    else: trace = dict(words='0', signature='unavailable_s4_off')
    row = dict(case=name, log=str(log), n_hex=format(n,'x'), settings=settings, device=device, memory=memory,
               root_words=root['root_words'], root_hash=root['root_hash'], leaves=leaf['leaves'], leaf_words=leaf['words'],
               leaf_hash=leaf['hash'], hits=stage[1], factors=stage[2], hit_primes=stage[3],
               trace_words=trace['words'], trace_hash=trace['signature'])
    rows.append(row); print('PASS', name, flush=True)
    return row

for bits, n in ((4,15),(6,35),(64,(1<<64)-59),(129,(1<<128)+1),(521,(1<<521)-1),(5261,(1<<5261)-1)):
    for window in ('0','1'):
        run(f'fixture_{bits}_{window}', n, b1=2 if n<64 else 20, NTT_GROOT_DEVICE='1',
            NTT_GROOT_DEVICE_TEST='1', NTT_GROOT_DEVICE_CHECK='1', NTT_S4_OUTPUT_WINDOW=window)
for bits, n in ((129,(1<<128)+1),(5261,(1<<5261)-1)):
    for mode in ('blocking','refused','trace'):
        run(f'fixture_{bits}_{mode}', n, NTT_GROOT_DEVICE='1', NTT_GROOT_DEVICE_TEST='1', NTT_GROOT_DEVICE_CHECK='1',
            NTT_S4_OUTPUT_WINDOW='1', NTT_S4_ASYNC='0' if mode=='blocking' else '1',
            NTT_ARENA_CAP_KB='1' if mode=='refused' else '', NTT_S4_CARRY_TRACE='1' if mode=='trace' else '0')
for mode, setting in (('host',dict(NTT_S4_HOSTPACK='1')), ('copied',dict(NTT_S4_PACK_DIRECT='0')),
                      ('readback',dict(NTT_S4_FINAL_READBACK='1')), ('full_tree',dict(NTT_S4_GROOT_ONLY='0')),
                      ('cpu',dict(NTT_S4_OFF='1',NTT_SCALED_DESCENT='0'))):
    run('fallback_'+mode, (1<<128)+1, NTT_GROOT_DEVICE='1', NTT_GROOT_DEVICE_TEST='0' if mode=='full_tree' else '1',
        NTT_GROOT_DEVICE_CHECK='1', NTT_S4_OUTPUT_WINDOW='1', **setting)

for name, n, b1, b2, d in (
    ('short',(1<<64)-59,20,1000,210), ('single_tail',(1<<128)+1,1000,400000,2310),
    ('frozen',(1<<128)+1,1000,1000000,210), ('sharp',(1<<128)+1,1000,114000,210),
    ('multi',(1<<521)-1,1000,1000000,2310), ('long',(1<<5261)-1,20,114000,210)):
    for window in ('0','1'):
        a = run(f'{name}_{window}_host',n,b1,b2,d,NTT_GROOT_DEVICE='0',NTT_S4_OUTPUT_WINDOW=window,NTT_S4_CARRY_TRACE='1')
        b = run(f'{name}_{window}_device',n,b1,b2,d,NTT_GROOT_DEVICE='1',NTT_GROOT_DEVICE_CHECK='1',
                NTT_S4_OUTPUT_WINDOW=window,NTT_S4_CARRY_TRACE='1')
        for k in ('root_words','root_hash','leaves','leaf_words','leaf_hash','hits','factors','hit_primes','trace_words','trace_hash'):
            assert a[k] == b[k], (name,window,k,a[k],b[k])

# Independently toggle both memory changes. Chunk=3 forces alternating pinned slots and
# reuse while uploads are outstanding; n=1 also exercises the no-multiply endpoint fence.
for bits, n in ((129,(1<<128)+1),(5261,(1<<5261)-1)):
    variants = []
    for compact in ('0','1'):
        for staging in ('0','1'):
            variants.append(run(f'memory_fixture_{bits}_{compact}_{staging}',n,NTT_GROOT_DEVICE='1',
                NTT_GROOT_DEVICE_TEST='1',NTT_GROOT_DEVICE_CHECK='1',NTT_S4_OUTPUT_WINDOW='1',
                NTT_GROOT_COMPACT_RAW=compact,NTT_GROOT_LEAF_STAGING=staging,NTT_GROOT_LEAF_CHUNK='3'))
    for r in variants[1:]:
        for k in ('root_words','root_hash','leaf_words','leaf_hash','hits','factors','hit_primes'):
            assert r[k] == variants[0][k], (bits,k)
    assert variants[2]['device']['raw_peak_bytes'] < variants[0]['device']['raw_peak_bytes']

for bits, n, b1, b2 in ((129,(1<<128)+1,1000,1000000),(5261,(1<<5261)-1,20,114000)):
    variants = []
    for compact in ('0','1'):
        variants.append(run(f'memory_real_{bits}_{compact}',n,b1,b2,210,NTT_GROOT_DEVICE='1',
            NTT_GROOT_DEVICE_CHECK='1',NTT_S4_OUTPUT_WINDOW='1',NTT_S4_CARRY_TRACE='1',
            NTT_GROOT_COMPACT_RAW=compact,NTT_GROOT_LEAF_STAGING=compact,NTT_GROOT_LEAF_CHUNK='3'))
    for k in ('root_words','root_hash','leaf_words','leaf_hash','hits','factors','hit_primes','trace_words','trace_hash'):
        assert variants[0][k] == variants[1][k], (bits,k)

for name, settings, message in (
    ('frontier_fault',dict(NTT_GROOT_DEVICE_CHECK='1',NTT_GROOT_DEVICE_TEST_BAD='1'),'resident GMP node mismatch'),
    ('carry_fault',dict(NTT_GROOT_DEVICE_TEST='1',NTT_S4_CARRY_TEST_BAD='1'),'deferred carry check')):
    env = base | dict(NTT_GROOT_DEVICE='1',NTT_S4_OUTPUT_WINDOW='1') | settings
    command = [str(exe),'--real','--n-hex','ffffffffffffffc5','--sigma','26','--b1','20','--b2','1000','--d','210','--device',str(args.device)]
    proc = subprocess.run(command,env=env,cwd=out,capture_output=True,text=True,timeout=240)
    text = proc.stdout+proc.stderr; log=out/(name+'.log');log.write_text(text,encoding='utf-8')
    assert proc.returncode != 0 and message in text and ('gdevice_fault:' in text if name=='frontier_fault' else 'first resident interior' in text), (name,str(log))
    rows.append(dict(case=name,log=str(log),fault_caught=True));print('PASS',name,flush=True)

summary = dict(passed=len(rows),failed=0,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runs=rows,
               scope='Real S4 resident gather/scatter, independent GMP every small node, 48 root fixtures per fixture run; '
                     'complete roots/leaves/returned coefficient fingerprints match host tree on actual Stage2, plus independent raw capacity/leaf staging toggles, forced staging reuse and fatal frontier/carry faults. No speed claim.')
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed',flush=True)
