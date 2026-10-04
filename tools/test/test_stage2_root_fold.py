"""Same-binary root-to-fold correctness, lifetime and fallback gates."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--exe', required=True)
p.add_argument('--output', required=True)
p.add_argument('--device', type=int, default=1)
a = p.parse_args()
exe = Path(a.exe).resolve()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
env.update({k: '1' for k in ['NTT_FOLD_DEVICE', 'NTT_FOLD_FLAT', 'NTT_GROOT_DEVICE',
    'NTT_DEVICE_GLEAF', 'NTT_GFINV_BATCH', 'NTT_GFINV_SEG_EXACT', 'NTT_GIANT_SEED_DEVICE',
    'NTT_SMALL_PRIME_REUSE', 'NTT_SCALED_DESCENT', 'NTT_S4_OUTPUT_WINDOW', 'NTT_S4_CHUNK_OUTPUT',
    'NTT_S4_MERSENNE', 'NTT_NO_PROGRESS']})
env.update(NTT_NAME_MAX='1', NTT_S4_SAMPLE='96', NTT_S4_CHECK_EVERY='8')
checks = []

def run(name, n, flags, b2=12000, d=210, fail=False):
    e = env.copy()
    e.update(flags)
    args = [str(exe), '--real', '--n', str(n), '--sigma', '26', '--b1', '1000',
            '--b2', str(b2), '--d', str(d), '--device', str(a.device)]
    r = subprocess.run(args, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding='utf-8', errors='replace')
    (out / (name + '.log')).write_text(r.stdout, encoding='utf-8')
    if fail:
        assert r.returncode != 0 and 'root-to-fold word mismatch' in r.stdout, (name, r.stdout[-2000:])
        checks.append({'name': name, 'poison_rejected': True, 'exit': r.returncode})
        return
    assert r.returncode == 0, (name, r.stdout[-2000:])
    for field in ['gmp_selftest_bad', 'gmp_check_bad', 'bad_factors']:
        assert all(int(x) == 0 for x in re.findall(field + r'=(\d+)', r.stdout)), name
    shape = re.search(r'real_batched_shape: P=(\d+) giant_points=(\d+) num_poly_g=(\d+)', r.stdout)
    leaf = re.search(r'descent_values:.*hash=(\d+)', r.stdout)[1]
    tail = re.search(r'stage2: algorithm=tree_gpu_batched .*?hits=(\d+) bad_factors=0 factors=([^ ]*) hit_primes=([^ ]*)', r.stdout).groups()
    line = re.search(r'real_batched_rootfold: (.*)', r.stdout)[1]
    root = dict(re.findall(r'(\w+)=([^ ]+)', line))
    fd = dict(re.findall(r'(\w+)=([^ ]+)', re.search(r'real_batched_folddevice: (.*)', r.stdout)[1]))
    complete = re.search(r'real_batched_groot: .*root_hash_complete=(\d+)', r.stdout)[1]
    row = {'name': name, 'exit': 0, 'leaf': leaf, 'tail': tail, 'root': root,
           'fold': fd, 'fnv_complete': complete, 'shape': list(map(int, shape.groups()))}
    checks.append(row)
    return row

def pair(name, n, flags=None, b2=12000, d=210, direct=True):
    flags = flags or {}
    control = run(name + '_host', n, dict(flags, NTT_GROOT_TO_FOLD='0'), b2, d)
    candidate = run(name + '_device', n, dict(flags, NTT_GROOT_TO_FOLD='1',
                    NTT_GROOT_TO_FOLD_CHECK='1', NTT_FOLD_DEVICE_CHECK='1') if direct else
                    dict(flags, NTT_GROOT_TO_FOLD='1'), b2, d)
    assert candidate['leaf'] == control['leaf'] and candidate['tail'] == control['tail'], name
    for field in ['digest_words', 'digest_sum', 'digest_xor', 'digest_kind']:
        assert candidate['root'][field] == control['root'][field], (name, field)
    assert candidate['root']['requested'] == '1', name
    if direct:
        W = (n.bit_length() + 63) // 64
        _, I, G = candidate['shape']
        words = (I + G) * W
        assert int(candidate['root']['trees']) == G and int(candidate['root']['words']) == words, name
        assert int(candidate['root']['digest_words']) == words, name
        assert int(candidate['root']['checked_words']) == 2 * words, name
        assert int(candidate['root']['avoided_h2d_bytes']) == 8 * words, name
        assert int(candidate['root']['avoided_d2h_bytes']) == 8 * words, name
        assert candidate['fnv_complete'] == '0' and control['fnv_complete'] == '1', name
    else:
        assert candidate['root']['trees'] == '0' and candidate['fnv_complete'] == '1', name
    return candidate

for bits, n in [(65, 2**65-59), (127, 2**127-1), (129, 2**128+1), (8192, 2**8192-1)]:
    pair('bits_' + str(bits), n)
pair('single_leaf_roots', 2**128+1, b2=1100, d=6)
factor = pair('frozen_factor', 2**128+1, b2=1000000)
assert factor['tail'][1] == '59649589127497217'
for name, flags in [('window_full', {'NTT_S4_OUTPUT_WINDOW': '0'}),
                    ('blocking', {'NTT_S4_ASYNC': '0'}),
                    ('host_leaf', {'NTT_DEVICE_GLEAF': '0'})]:
    pair(name, 2**128+1, flags)
for name, flags, b2 in [('fold_off', {'NTT_FOLD_DEVICE': '0'}, 12000),
                      ('budget', {'NTT_FOLD_DEVICE_MAX_MB': '0'}, 12000),
                      ('host_groot', {'NTT_GROOT_DEVICE': '0'}, 12000),
                      ('full_tree', {'NTT_S4_GROOT_ONLY': '0'}, 12000),
                      ('copy_pack', {'NTT_S4_PACK_DIRECT': '0'}, 12000),
                      ('trace', {'NTT_S4_CARRY_TRACE': '1'}, 12000),
                      ('whole_readback', {'NTT_S4_FINAL_READBACK': '1'}, 12000),
                      ('single_G', {}, 2000)]:
    pair(name, 2**128+1, flags, b2=b2, direct=False)
run('poison', 2**128+1, {'NTT_GROOT_TO_FOLD': '1', 'NTT_GROOT_TO_FOLD_CHECK': '1',
    'NTT_GROOT_TO_FOLD_TEST_BAD': '1'}, fail=True)
summary = {'exe': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
           'device': a.device, 'checks': checks, 'passed': len(checks), 'failed': 0}
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'passed': len(checks), 'failed': 0, 'sha256': summary['sha256']}))
