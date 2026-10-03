"""Exercise the actual GPU scaled descent and compare full nodes/leaves with GMP."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exe', type=Path, default=repo / 'build_cuda_cmake/stage2_tree_gpu.exe')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
exe = args.exe.resolve()
out = (args.output or repo / 'build_cuda_cmake' / ('_scaled_gate_' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()): raise RuntimeError('Use a fresh output directory')
base = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_ASYNC='1', NTT_S4_PACK_DIRECT='1', NTT_S4_FLAT_DIRECT='1',
            NTT_S4_GROOT_ONLY='1', NTT_ARENA_WORKSPACE_POOL='1', NTT_FUSE_COMPACT_SCRATCH='1',
            NTT_S4_CHUNK_OUTPUT='1', NTT_S4_CHUNK_MAX='2', NTT_S4_SAMPLE='96',
            NTT_S4_CHECK_EVERY='8', NTT_S4_ORACLE_ASYNC='1', NTT_S4_ORACLE_PACK='1',
            NTT_S4_OLDTAIL='0', NTT_NO_PROGRESS='1', NTT_NAME_MAX='1')
rows = []

def run(name, n, bound=20, b2=1000, d=210, **settings):
    env = base | settings
    command = [str(exe), '--real', '--n-hex', format(n, 'x'), '--sigma', '26',
               '--b1', str(bound), '--b2', str(b2), '--d', str(d), '--device', str(args.device)]
    p = subprocess.run(command, env=env, cwd=out, capture_output=True, text=True, timeout=180)
    text = p.stdout + p.stderr
    log = out / (name + '.log'); log.write_text(text, encoding='utf-8')
    assert p.returncode == 0, (name, p.returncode, str(log))
    assert not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]|slot_canonical_bad=[1-9]', text), (name, str(log))
    leaves = re.search(r'descent_values: leaves=(\d+) words=(\d+) hash=(\d+)', text)
    stage = re.search(r'stage2:.*hits=(\d+) bad_factors=0 factors=(\S*) hit_primes=(\S*)', text)
    assert leaves and stage, (name, 'missing final output', str(log))
    row = dict(case=name, log=str(log), n_hex=format(n, 'x'), settings=settings,
               leaves=int(leaves[1]), leaf_words=int(leaves[2]), leaf_hash=leaves[3],
               hits=stage[1], factors=stage[2], hit_primes=stage[3])
    if settings.get('NTT_SCALED_TEST') == '1':
        fixture = re.search(r'scaled_fixture: cases=(\d+) states=(\d+) words=(\d+) leaves=(\d+) reused=(\d+) root_divisions=(\d+) bad=0', text)
        w = (n.bit_length() + 63) // 64
        assert fixture and tuple(map(int, fixture.groups())) == (150, 3180, 6330*w, 1305, 120, 30), (name, 'fixture counts', str(log))
        assert re.search(r'stage2_full_wall:.*clean=0', text)
        row['fixture_states'] = 3180; row['fixture_leaf_values'] = 1305
    if settings.get('NTT_SCALED_DESCENT') == '1':
        stat = re.search(r'scaled_descent: enabled=1 (.*)', text)
        assert stat, (name, 'candidate not taken')
        row['scaled'] = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', stat[1])}
        if settings.get('NTT_SCALED_CHECK') == '1':
            assert row['scaled']['states'] == row['scaled']['checked_states']
            assert row['scaled']['words'] == row['scaled']['checked_words']
            assert row['scaled']['leaves'] == row['leaves']
    if settings.get('NTT_S4_DESCENT_CHECK') == '1':
        assert re.search(r'descent_check:.*mismatching_coefficients=0', text)
    rows.append(row)
    print('PASS', name, flush=True)
    return row

# Per-node GMP long division and per-leaf Horner, all padding patterns and H degrees.
for n in (15, 35):
    run(f'fixture_n{n}', n, bound=2, NTT_SCALED_TEST='1', NTT_SCALED_DESCENT='1',
        NTT_SCALED_CHECK='1', NTT_S4_OUTPUT_WINDOW='1')
for bits, n in ((64, (1 << 64)-59), (129, (1 << 128)+1), (521, (1 << 521)-1)):
    for mode in ('window_off', 'pinned', 'blocking', 'host', 'refused'):
        run(f'fixture_{bits}_{mode}', n, NTT_SCALED_TEST='1', NTT_SCALED_DESCENT='1',
            NTT_SCALED_CHECK='1', NTT_S4_OUTPUT_WINDOW='0' if mode == 'window_off' else '1',
            NTT_S4_ASYNC='0' if mode == 'blocking' else '1',
            NTT_S4_HOSTPACK='1' if mode == 'host' else '0',
            NTT_ARENA_CAP_KB='1' if mode == 'refused' else '')
# Real Stage2 H, including projective G scaling, zero/short H, and cached root inverse.
for name, n, bound, b2, d in (
    ('short', (1 << 64)-59, 20, 1000, 210),
    ('root_degree', (1 << 64)-59, 20, 4620, 210),  # imax = P: initial H has degree P
    ('frozen', (1 << 128)+1, 1000, 1000000, 210),
    ('sharp', (1 << 128)+1, 1000, 114000, 210),
    ('nonpower', (1 << 128)+1, 1000, 400000, 2310),
    ('multi', (1 << 521)-1, 1000, 1000000, 2310)):
    for window in ('0', '1'):
        control = run(f'{name}_{window}_division', n, bound, b2, d, NTT_SCALED_DESCENT='0', NTT_S4_OUTPUT_WINDOW=window)
        candidate = run(f'{name}_{window}_scaled', n, bound, b2, d, NTT_SCALED_DESCENT='1',
                        NTT_SCALED_CHECK='1', NTT_S4_DESCENT_CHECK='1', NTT_S4_OUTPUT_WINDOW=window)
        for key in ('leaves', 'leaf_words', 'leaf_hash', 'hits', 'factors', 'hit_primes'):
            assert control[key] == candidate[key], (name, window, key, control[key], candidate[key])
        if name == 'root_degree': assert candidate['scaled']['root_divisions'] == 1
        if name in ('frozen', 'multi'): assert candidate['scaled']['root_inverse_reused'] == 1
summary = dict(passed=len(rows), failed=0, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               fixture_cases=sum(150 for row in rows if 'fixture_states' in row),
               fixture_states=sum(row.get('fixture_states', 0) for row in rows),
               fixture_leaf_values=sum(row.get('fixture_leaf_values', 0) for row in rows), runs=rows,
               scope='Actual S4 scaled GPU multiply; independent GMP every node and Horner every fixture leaf. '
                     'Real complete leaf fingerprints, factors and hit primes match division; slow descent checked too. '
                     'Includes full/windowed outputs, two-slice tail, pinned/blocking/host/fallback paths. No speed claim.')
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed', flush=True)
