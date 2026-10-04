"""Gate device seeds and exact returned-Z products against independent arithmetic."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess

repo = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', type=int, default=1)
a = p.parse_args()
exe = a.exe.resolve()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    raise RuntimeError('Use a fresh output directory')
spec = importlib.util.spec_from_file_location('mont_ref', repo/'tools/stat/suyama_mont_ref.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
base = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_OLDTAIL='0', NTT_S4_PACK_DIRECT='1', NTT_S4_FLAT_DIRECT='1',
            NTT_GROOT_DEVICE='1', NTT_SCALED_DESCENT='1', NTT_S4_OUTPUT_WINDOW='1',
            NTT_S4_CHUNK_OUTPUT='1', NTT_GFINV_BATCH='1', NTT_FOLD_FLAT='1',
            NTT_NO_PROGRESS='1', NTT_NAME_MAX='1', NTT_GIANT_CHAIN_MIN='0',
            NTT_GFINV_SEG_EXACT='1', NTT_GFINV_SEG_CHECK='1',
            NTT_S4_CARRY_TRACE='1', NTT_GROOT_DEVICE_CHECK='1')
rows = []


def invoke(name, n, b1=20, b2=1000, d=210, **settings):
    cmd = [str(exe), '--real', '--n-hex', format(n, 'x'), '--sigma', '26',
           '--b1', str(b1), '--b2', str(b2), '--d', str(d), '--device', str(a.device)]
    proc = subprocess.run(cmd, env=base | settings, capture_output=True, text=True,
                          timeout=300, cwd=out)
    text = proc.stdout+proc.stderr
    log = out/(name+'.log')
    log.write_text(text, encoding='utf-8')
    return proc.returncode, text, log


def run(name, n, b1=20, b2=1000, d=210, **settings):
    rc, text, log = invoke(name, n, b1, b2, d, **settings)
    assert rc == 0 and not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]', text), (name, rc, str(log))

    def stat(prefix):
        match = re.search(r'(?m)^'+prefix+r': (.*)', text)
        assert match, (name, prefix)
        return dict(re.findall(r'(\w+)=(\S+)', match[1]))

    seed = stat('real_giant_seed')
    proj = stat('real_batched_projective')
    chain = stat('real_giant_chain')
    leaf = stat('descent_values')
    root = stat('real_batched_groot')
    trace = stat('real_batched_carrytrace')
    stage = re.search(r'stage2:.*hits=(\d+) bad_factors=0 factors=(\S*) hit_primes=(\S*)', text)
    assert stage
    assert seed['enabled'] == settings.get('NTT_GIANT_SEED_DEVICE', '0')
    assert seed['exact_segments'] == settings.get('NTT_GFINV_SEG_EXACT', '1')
    if int(chain['chunks']):
        assert int(seed['segments']) > 0
        if settings.get('NTT_GFINV_SEG_CHECK', '1') == '1':
            assert seed['segments'] == seed['segment_checks']
        if seed['enabled'] == '1':
            assert seed['chunks'] == chain['chunks'] and seed['points'] == chain['seed_points']
            w = (n.bit_length()+63)//64
            assert int(seed['avoided_d2h_bytes']) == int(seed['avoided_h2d_bytes']) == 16*w*int(seed['points'])
            assert int(seed['avoided_cpu_modmuls']) == int(seed['avoided_montmuls']) == 2*int(seed['points'])
            if settings.get('NTT_GIANT_SEED_CHECK') == '1':
                assert int(seed['checked_words']) == 2*w*int(seed['points'])
    else:
        assert seed['chunks'] == seed['points'] == '0'
    if settings.get('NTT_GFINV_SEG_TEST') == '1':
        f = stat('segment_product_fixture')
        assert f['cases'] == '91' and f['bad'] == '0' and int(f['checks']) > 0
        assert int(f['legacy_different']) > 0
    assert re.search(r'stage2_full_wall:.*clean=0', text)
    row = dict(case=name, seed=seed, projective=proj, root_hash=root['root_hash'],
               root_words=root['root_words'], leaf_hash=leaf['hash'], leaves=leaf['leaves'],
               words=leaf['words'], trace_words=trace['words'], trace_hash=trace['signature'],
               hits=stage[1], factors=stage[2], hit_primes=stage[3], log=str(log))
    rows.append(row)
    print('PASS', name, flush=True)
    return row


def fnv(values, w):
    h = 1469598103934665603
    for v in values:
        for j in range(w):
            h = ((h ^ ((v >> (64*j)) & ((1 << 64)-1))) * 1099511628211) & ((1 << 64)-1)
    return str(h)


def independent_values(n, b1, b2, d, extra):
    # Ordinary affine Python oracle; does not use GPU segment products or the tree algorithm.
    q = oracle.stage1(26, b1, n, torsion=extra)['x']
    _, a24, _, _ = oracle.suyama_curve(26, n)
    def affine(k):
        x, z = oracle.ladder(k, q, 1, a24, n)
        assert math.gcd(z, n) == 1, ('unit-only oracle case has nonunit Z', k)
        return x*pow(z, -1, n) % n
    baby = [affine(j) for j in range(1, d//2+1) if math.gcd(j, d) == 1]
    values = [1]*len(baby)
    for i in range(1, b2//d+3):
        u = affine(i*d)
        values = [v*(x-u) % n for v, x in zip(values, baby)]
    return values


for bits, n in ((64, (1 << 64)-59), (129, (1 << 128)+1),
                (4423, (1 << 4423)-1), (5261, (1 << 5261)-1)):
    run('fixture_'+str(bits), n, NTT_GFINV_SEG_TEST='1', NTT_GIANT_SEED_DEVICE='1',
        NTT_GIANT_SEED_CHECK='1')

for name, n, b1, b2, d, extra, independent in (
    ('unit64', (1 << 64)-59, 20, 40000, 210, 1, True),
    ('unit127', (1 << 127)-1, 20, 13230, 210, 12, True),  # 65-point tail: one-point chain/segment
    ('frozen', (1 << 128)+1, 1000, 1000000, 210, 1, True),
    ('tail', (1 << 128)+1, 1000, 400000, 2310, 1, False),
    ('composite', 15, 2, 1000, 210, 1, False),
    ('mixed', 35, 2, 1000, 210, 1, False),
    ('choose12', (1 << 4423)-1, 20, 114000, 210, 12, True),
    ('wide', (1 << 5261)-1, 20, 114000, 210, 1, True)):
    pair = [run(name+'_'+mode, n, b1, b2, d, NTT_GIANT_SEED_DEVICE=mode,
                NTT_GIANT_SEED_CHECK='1' if mode == '1' else '0', NTT_STAGE1_EXTRA=str(extra))
            for mode in ('0', '1')]
    for key in ('root_hash', 'root_words', 'leaf_hash', 'leaves', 'words', 'trace_words',
                'trace_hash', 'hits', 'factors', 'hit_primes', 'projective'):
        assert pair[0][key] == pair[1][key], (name, key)
    if name == 'frozen':
        assert pair[1]['factors'] == '59649589127497217' and pair[1]['hit_primes'] == '114713'
        assert pair[1]['leaf_hash'] == '7706779146789021619'
    if independent:
        assert int(pair[1]['projective']['affine_fallback_points']) == 0
        vals = independent_values(n, b1, b2, d, extra)
        expected = fnv(vals, (n.bit_length()+63)//64)
        assert pair[1]['leaf_hash'] == expected, (name, 'independent monic leaf hash', pair[1]['leaf_hash'], expected)
        rows.append(dict(case=name+'_independent_monic', leaf_hash=expected, leaf_count=len(vals)))
        print('PASS', name+'_independent_monic', flush=True)
        if name == 'unit127':
            legacy = run(name+'_legacy_scale', n, b1, b2, d, NTT_GIANT_SEED_DEVICE='1',
                         NTT_GIANT_SEED_CHECK='1', NTT_GFINV_SEG_EXACT='0', NTT_GFINV_SEG_CHECK='0',
                         NTT_STAGE1_EXTRA=str(extra))
            w = (n.bit_length()+63)//64
            points = b2//d+2
            scale = pow(pow(2, 64*w, n), points-(points+15)//16, n)
            assert legacy['leaf_hash'] == fnv([v*scale % n for v in vals], w)
            assert legacy['leaf_hash'] != expected
            assert legacy['factors'] == pair[1]['factors']
            rows.append(dict(case=name+'_legacy_unit_relation', exponent=points-(points+15)//16))
            print('PASS', name+'_legacy_unit_relation', flush=True)

# Device option on a forced ladder chunk must be a harmless fallback.
pair = [run('ladder_fallback_'+mode, (1 << 127)-1, NTT_GIANT_SEED_DEVICE=mode,
            NTT_GIANT_CHAIN_MIN='999999999') for mode in ('0', '1')]
assert pair[0]['leaf_hash'] == pair[1]['leaf_hash'] and pair[1]['seed']['chunks'] == '0'

for name, settings, reason in (
    ('fixture_fault', dict(NTT_GFINV_SEG_TEST='1', NTT_GFINV_SEG_TEST_BAD='1'), 'segment fixture GMP mismatch'),
    ('legacy_rejected', dict(NTT_GFINV_SEG_EXACT='0'), 'segment product GMP mismatch')):
    rc, text, log = invoke(name, (1 << 64)-59, **settings)
    assert rc != 0 and reason in text, (name, rc, str(log))
    rows.append(dict(case=name, fault_caught=True, log=str(log)))
    print('PASS', name, flush=True)

(out/'summary.json').write_text(json.dumps(dict(passed=len(rows), failed=0, runs=rows,
    binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()), indent=2), encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed', flush=True)
