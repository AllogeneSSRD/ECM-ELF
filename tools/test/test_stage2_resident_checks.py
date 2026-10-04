"""Check combined oracle/carry scheduling on the production resident pipeline."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--exe', required=True)
p.add_argument('--output', required=True)
p.add_argument('--device', type=int, default=1)
a = p.parse_args()
exe, out = Path(a.exe).resolve(), Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
env.update({k: '1' for k in ['NTT_FOLD_DEVICE', 'NTT_FOLD_FLAT', 'NTT_GROOT_DEVICE',
    'NTT_GROOT_TO_FOLD', 'NTT_DEVICE_GLEAF', 'NTT_GFINV_BATCH', 'NTT_GFINV_SEG_EXACT',
    'NTT_GIANT_SEED_DEVICE', 'NTT_SMALL_PRIME_REUSE', 'NTT_SCALED_DESCENT',
    'NTT_S4_OUTPUT_WINDOW', 'NTT_S4_CHUNK_OUTPUT', 'NTT_S4_MERSENNE', 'NTT_NO_PROGRESS']})
env.update(NTT_NAME_MAX='1', NTT_S4_SAMPLE='96', NTT_S4_CHECK_EVERY='8', NTT_S4_CHUNK_MAX='2')
checks = []

def run(name, n, oracle, carry, flags=None, fail=None, b2=12000):
    e = dict(env, NTT_S4_ORACLE_ASYNC=str(oracle), NTT_S4_CARRY_BATCH=str(carry))
    e.update(flags or {})
    r = subprocess.run([str(exe), '--real', '--n', str(n), '--sigma', '26', '--b1', '1000',
                        '--b2', str(b2), '--d', '210', '--device', str(a.device)], env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding='utf-8', errors='replace')
    (out / (name + '.log')).write_text(r.stdout, encoding='utf-8')
    if fail:
        assert r.returncode != 0 and fail in r.stdout, (name, r.stdout[-2500:])
        checks.append(dict(name=name, exit=r.returncode, fault_rejected=fail))
        return
    assert r.returncode == 0, (name, r.stdout[-2500:])
    for field in ['gmp_selftest_bad', 'gmp_check_bad', 'bad_factors', 'slot_canonical_bad']:
        assert all(int(x) == 0 for x in re.findall(field + r'=(\d+)', r.stdout)), name
    def fields(prefix):
        return dict(re.findall(r'(\w+)=([^ ]+)', re.search('^' + prefix + ': (.*)', r.stdout, re.M)[1]))
    o, c, root, fd = [fields(x) for x in ['s4_oracle_stats', 'real_batched_carrydefer',
                                         'real_batched_rootfold', 'real_batched_folddevice']]
    assert int(o['async']) == oracle and o['selected'] == o['compared'], name
    assert int(o['pending']) == int(o['fallbacks']) == 0, name
    assert int(o['queued']) == (int(o['selected']) if oracle else 0), name
    assert int(c['batch_enabled']) == carry and c['chunks_deferred'] == c['checked_chunks'], name
    assert int(c['chunks_deferred']) > 0, name
    assert int(c['finishes']) < int(c['checked_chunks']) if carry else c['finishes'] == c['checked_chunks'], name
    if int(fd['enabled']):
        assert int(root['trees']) > 0 and root['words'] == root['digest_words'], name
        assert int(root['avoided_h2d_bytes']) == int(root['avoided_d2h_bytes']) == 8*int(root['words']), name
        assert 'root_hash_complete=0' in r.stdout, name
    else:
        assert int(root['trees']) == int(root['digest_words']) == 0 and 'root_hash_complete=1' in r.stdout, name
    row = dict(name=name, exit=0, oracle=o, carry=c, root=root, fold=fd,
               leaf=re.search(r'descent_values:.*hash=(\d+)', r.stdout)[1],
               tail=re.search(r'stage2: algorithm=tree_gpu_batched .*?hits=(\d+) bad_factors=0 factors=([^ ]*) hit_primes=([^ ]*)', r.stdout).groups())
    checks.append(row)
    return row

def compare(control, candidate):
    assert candidate['leaf'] == control['leaf'] and candidate['tail'] == control['tail'], candidate['name']
    for field in ['samples', 'selected', 'signature']:
        assert candidate['oracle'][field] == control['oracle'][field], (candidate['name'], field)
    for field in ['digest_words', 'digest_sum', 'digest_xor', 'words', 'trees']:
        assert candidate['root'][field] == control['root'][field], (candidate['name'], field)
    assert candidate['carry']['checked_chunks'] == control['carry']['checked_chunks'], candidate['name']

for bits in [65, 127, 129, 8192]:
    n = 2**128+1 if bits == 129 else 2**bits-1
    control = run(f'n{bits}_00', n, 0, 0)
    for oracle, carry in [(1, 0), (0, 1), (1, 1)]:
        compare(control, run(f'n{bits}_{oracle}{carry}', n, oracle, carry))
for name, flags in [('ring1', {'NTT_S4_ORACLE_RING': '1'}),
                    ('ring8_window0', {'NTT_S4_ORACLE_RING': '8', 'NTT_S4_OUTPUT_WINDOW': '0'}),
                    ('blocking_output', {'NTT_S4_ASYNC': '0'}),
                    ('host_fold_fallback', {'NTT_FOLD_DEVICE_MAX_MB': '0'})]:
    compare(run(name+'_00', 2**128+1, 0, 0, flags), run(name+'_11', 2**128+1, 1, 1, flags))
run('poison_oracle', 2**127-1, 1, 1,
    {'NTT_S4_SAMPLE': '1', 'NTT_S4_CHECK_EVERY': '1', 'NTT_S4_ORACLE_TEST_BAD': '1'},
    fail='device reduction disagrees with GMP')
run('poison_carry', 2**127-1, 1, 1, {'NTT_S4_CARRY_TEST_BAD': '1'}, fail='deferred carry check')
summary = dict(exe=str(exe), sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               device=a.device, passed=len(checks), failed=0, checks=checks)
(out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(dict(passed=len(checks), failed=0)))
