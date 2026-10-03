"""Gate contiguous host fold with independent GMP remainder and full Stage2 outputs."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', type=Path, default=repo/'build_cuda_cmake/stage2_tree_gpu.exe')
p.add_argument('--device', type=int, default=1)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
exe = args.exe.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    raise RuntimeError('Use a fresh output directory')
env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
env.update(NTT_S4_OLDTAIL='0', NTT_S4_PACK_DIRECT='1', NTT_S4_FLAT_DIRECT='1',
           NTT_S4_GROOT_ONLY='1', NTT_GROOT_DEVICE='1', NTT_GROOT_LEAF_STAGING='1',
           NTT_GROOT_COMPACT_RAW='1', NTT_SCALED_DESCENT='1', NTT_S4_OUTPUT_WINDOW='1',
           NTT_S4_CHUNK_OUTPUT='1', NTT_S4_CARRY_BATCH='0', NTT_GFINV_BATCH='1',
           NTT_S4_SAMPLE='96', NTT_S4_CHECK_EVERY='8', NTT_S4_ORACLE_ASYNC='1',
           NTT_NO_PROGRESS='1', NTT_NAME_MAX='1')
rows = []


def run(name, n, b1=20, b2=1000, d=210, **settings):
    command = [str(exe), '--real', '--n-hex', format(n, 'x'), '--sigma', '26',
               '--b1', str(b1), '--b2', str(b2), '--d', str(d), '--device', str(args.device)]
    proc = subprocess.run(command, env=env | settings, capture_output=True, text=True,
                          timeout=240, cwd=out)
    text = proc.stdout + proc.stderr
    log = out/(name+'.log')
    log.write_text(text, encoding='utf-8')
    assert proc.returncode == 0, (name, proc.returncode, str(log))
    assert not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]|slot_canonical_bad=[1-9]', text), (name, str(log))

    def stat(prefix):
        m = re.search(r'(?m)^'+prefix+r': (.*)', text)
        assert m, (name, prefix, str(log))
        return dict(re.findall(r'(\w+)=(\S+)', m[1]))

    fold = stat('real_batched_foldflat')
    enabled = '1' if settings.get('NTT_FOLD_FLAT') == '1' and settings.get('NTT_S4_OFF') != '1' else '0'
    assert fold['enabled'] == enabled
    shape = stat('real_batched_shape')
    if enabled == '1':
        assert fold['folds'] == shape['loops']
        assert int(fold['muls']) >= int(fold['folds'])
        if int(shape['loops']):
            assert int(fold['sub_coeffs']) > 0 and int(fold['peak_bytes']) > 0
            assert stat('scaled_descent')['root_inverse_reused'] == '1'
    if settings.get('NTT_FOLD_FLAT_TEST') == '1':
        assert stat('fold_flat_fixture')['cases'] == '30'
        assert stat('fold_flat_fixture')['bad'] == '0'
        assert re.search(r'stage2_full_wall:.*clean=0', text)
    root, leaf, proj = (stat(x) for x in ('real_batched_groot', 'descent_values', 'real_batched_projective'))
    trace = stat('real_batched_carrytrace')
    stage = re.search(r'stage2:.*hits=(\d+) bad_factors=0 factors=(\S*) hit_primes=(\S*)', text)
    assert stage, (name, str(log))
    row = dict(case=name, log=str(log), fold=fold, projective=proj, root_hash=root['root_hash'],
               root_words=root['root_words'], leaf_hash=leaf['hash'], leaves=leaf['leaves'],
               words=leaf['words'], trace_words=trace['words'], trace_hash=trace['signature'],
               hits=stage[1], factors=stage[2], hit_primes=stage[3])
    rows.append(row)
    print('PASS', name, flush=True)
    return row


for label, n in (('composite', 15), ('129', (1 << 128)+1), ('4423', (1 << 4423)-1)):
    run('fixture_'+label, n, b1=2 if n == 15 else 20,
        NTT_FOLD_FLAT='1', NTT_FOLD_FLAT_TEST='1')

for name, n, b1, b2, d, chain in (
    ('composite', 15, 2, 1000, 210, '0'),
    ('mixed', 35, 2, 1000, 210, '999999999'),
    ('frozen', (1 << 128)+1, 1000, 1000000, 210, '999999999'),
    ('tail', (1 << 128)+1, 1000, 400000, 2310, '0'),
    ('choose12', (1 << 4423)-1, 20, 114000, 210, '0'),
    ('long', (1 << 5261)-1, 20, 114000, 210, '0')):
    pair = [run(name+'_'+flat, n, b1, b2, d, NTT_FOLD_FLAT=flat,
                NTT_GIANT_CHAIN_MIN=chain, NTT_S4_CARRY_TRACE='1',
                NTT_GROOT_DEVICE_CHECK='1', NTT_STAGE1_EXTRA='12' if name == 'choose12' else '1')
            for flat in ('0', '1')]
    for k in ('root_hash', 'root_words', 'leaf_hash', 'leaves', 'words',
              'trace_words', 'trace_hash', 'hits', 'factors', 'hit_primes', 'projective'):
        assert pair[0][k] == pair[1][k], (name, k, pair[0][k], pair[1][k])
    if name == 'frozen':
        assert pair[1]['factors'] == '59649589127497217'
        assert pair[1]['hit_primes'] == '114713'
        assert pair[1]['leaf_hash'] == '7706779146789021619'
    if name in ('composite', 'mixed'):
        assert int(pair[1]['projective']['affine_fallback_points']) > 0

pair = [run('s4_off_'+flat, (1 << 128)+1, 1000, 114000, 210, NTT_FOLD_FLAT=flat,
            NTT_S4_OFF='1', NTT_SCALED_DESCENT='0', NTT_GROOT_DEVICE='0') for flat in ('0', '1')]
for k in ('root_hash', 'root_words', 'leaf_hash', 'leaves', 'words', 'hits', 'factors', 'hit_primes'):
    assert pair[0][k] == pair[1][k], ('s4_off', k)

command = [str(exe), '--real', '--n-hex', 'ffffffffffffffc5', '--sigma', '26',
           '--b1', '20', '--b2', '1000', '--d', '210', '--device', str(args.device)]
proc = subprocess.run(command, env=env | dict(NTT_FOLD_FLAT='1', NTT_FOLD_FLAT_TEST='1',
                                             NTT_FOLD_FLAT_TEST_BAD='1'),
                      capture_output=True, text=True, timeout=240, cwd=out)
text = proc.stdout + proc.stderr
(out/'fixture_fault.log').write_text(text, encoding='utf-8')
assert proc.returncode != 0 and 'flat fold GMP mismatch' in text
rows.append(dict(case='fixture_fault', fault_caught=True))
summary = dict(passed=len(rows), failed=0, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), runs=rows)
(out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed', flush=True)
