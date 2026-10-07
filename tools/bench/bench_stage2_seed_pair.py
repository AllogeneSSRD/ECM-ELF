"""Same-binary ABBA for rebased adjacent giant seeds, using verified saves.

Full affine/seed/segment comparisons run outside performance measurements.
Results retain every run, including slow repeats; no cost profile is exported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from bench_stage2_budget_scaling import fields
from calibrate_stage2_d import parse
from ecm_cost_model import features
from bench_stage2_short_policy import chunks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bits', type=int, nargs='+', default=[2203, 4423, 8191])
    p.add_argument('--d', type=int, default=30030)
    p.add_argument('--points', type=int, nargs='+')
    p.add_argument('--b2', type=int, nargs='+', help='Additional exact B2 values; omit --points to test only these')
    p.add_argument('--blocks', type=int, nargs='+', default=[8, 64])
    p.add_argument('--chunk-tail', type=int, nargs='*', default=[])
    p.add_argument('--owner-mb', type=int, default=0)
    p.add_argument('--arena-mb', type=int, default=4096)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--resume-gates', action='store_true', help='Verify a finished timing matrix and continue interrupted gates')
    a = p.parse_args()
    if a.points is None:
        a.points = [] if a.b2 else [4096, 8192, 24977]
    if (a.repeats < 1 or a.d < 6 or a.d % 2 or a.d > 200000000 or a.owner_mb < 0 or a.arena_mb <= 0 or
            len(set(a.bits)) != len(a.bits) or len(set(a.blocks)) != len(a.blocks) or
            any(not 4 <= b <= 1 << 20 for b in a.blocks) or
            any(n < 3 or n > 5000000 or a.d * (n - 2) <= 1000 for n in a.points) or
            any(not 2 <= n < 8192 for n in a.chunk_tail) or
            any(v <= 1000 or v // a.d + 2 > 5000000 for v in (a.b2 or []))):
        p.error('Invalid range, budget or duplicate geometry')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not a.resume_gates:
        raise ValueError('Use a fresh output directory')
    exe = a.exe.resolve()
    root = Path(__file__).resolve().parents[2]
    build_path = exe.parent / 'build_manifest.json'
    build = read(build_path)
    if build['sha256'].lower() != sha(exe) or build['gl_fixed_mode'] != 3 or build['outer_unroll_u'] != 0:
        raise ValueError('Need matching PTX3/outer0 build receipt')
    # Freeze the exact compiled dependencies before any experiments or formatting edits.
    frozen_path = exe.parent / 'frozen_sources_manifest.json'
    if not frozen_path.exists():
        for name, want in build['source_hashes'].items():
            source = root / name
            if sha(source) != want.lower():
                raise ValueError('Compiled source changed: ' + name)
            dest = exe.parent / 'sources' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
        frozen_path.write_text(json.dumps(dict(binary_sha256=sha(exe),
            sources={n: h.lower() for n, h in build['source_hashes'].items()}), indent=2), encoding='utf-8')
    frozen = read(frozen_path)
    study = read(a.study)
    if not study['complete'] or any(b not in study['controls']['bits'] for b in a.bits):
        raise ValueError('Need independently verified Stage1 widths')
    saves = {b: Path(study['saves'][str(b)]['path']) for b in a.bits}
    names = ('bench_stage2_seed_pair.py', 'bench_stage2_short_policy.py',
             'bench_stage2_budget_scaling.py', 'calibrate_stage2_d.py', 'ecm_cost_model.py')
    identity = dict(binary_sha256=sha(exe), build_manifest_sha256=sha(build_path),
        snapshot_sha256=sha(frozen_path), study_sha256=sha(a.study),
        tools={n: sha(Path(__file__).with_name(n)) for n in names})
    (out / ('collector_' + identity['tools']['bench_stage2_seed_pair.py'][:12] + '.py')).write_bytes(Path(__file__).read_bytes())

    def verify():
        if (sha(exe) != identity['binary_sha256'] or frozen['binary_sha256'] != sha(exe) or
                sha(build_path) != identity['build_manifest_sha256'] or
                sha(frozen_path) != identity['snapshot_sha256'] or sha(a.study) != identity['study_sha256']):
            raise ValueError('Binary, build, snapshot or study changed')
        if {n: h.lower() for n, h in build['source_hashes'].items()} != frozen['sources']:
            raise ValueError('Compiled dependency closure differs from snapshot')
        for b, saved in saves.items():
            if sha(saved) != study['saves'][str(b)]['sha256']:
                raise ValueError('Verified save changed')
        for n, h in frozen['sources'].items():
            if sha(exe.parent / 'sources' / n) != h:
                raise ValueError('Frozen source changed: ' + n)
        for n, h in identity['tools'].items():
            if sha(Path(__file__).with_name(n)) != h:
                raise ValueError('Collector changed: ' + n)

    verify()
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_D_MODEL='0', NTT_STAGE1_Q_DUMP='1',
        NTT_CARRY_CHECK_FUSED='0', NTT_POINT_MERSENNE='1', NTT_GIANT_CHAIN_MIN='0',
        NTT_FOLD_DEVICE_MAX_MB=str(a.owner_mb), CUDA_LAUNCH_BLOCKING='0')
    device = json.loads(subprocess.run([str(exe), '--cost-device-info', '--device', '1'],
        env=env, capture_output=True, check=True, timeout=60).stdout)
    if device['uuid_hex'] != '8a67b1f8ef1c3177a822813a7ac2224d':
        raise ValueError('Wrong GPU')
    cases = []
    for b in a.bits:
        f = features(a.d, a.d * 4094, b, 0)
        size = chunks(dict(f, I=5000000))[0]
        bounds = {a.d * (n - 2) for n in a.points + [size + t for t in a.chunk_tail]}
        bounds.update(a.b2 or [])
        for b2 in sorted(bounds):
            for block in a.blocks:
                cases.append(dict(bits=b, points=b2 // a.d + 2, block=block, B2=b2))
    if not cases:
        p.error('Select at least one point count, B2 or chunk tail')
    data = dict(schema=1, kind='giant_seed_pair_abba', identity=identity, device=device,
        controls=dict(D=a.d, arena_mb=a.arena_mb, owner_mb=a.owner_mb, sequence=['original', 'paired', 'paired', 'original'],
                      repeats=a.repeats, base_nonunit_fallback=True),
        cases=cases, warmups=[], runs=[], gates=[])
    path = out / 'measurements.json'
    gate_suffix = ''
    if a.resume_gates:
        previous = read(path)
        if previous.get('complete') or previous['cases'] != cases or previous['controls'] != data['controls']:
            raise ValueError('Only an identical unfinished timing matrix can resume gates')
        for key in ('binary_sha256', 'build_manifest_sha256', 'snapshot_sha256', 'study_sha256'):
            if previous['identity'][key] != identity[key]:
                raise ValueError('Resumed input identity changed: ' + key)
        old_tools = previous['identity']['tools']
        for name, h in old_tools.items():
            if name != 'bench_stage2_seed_pair.py' and identity['tools'][name] != h:
                raise ValueError('Resumed dependency changed: ' + name)
        initial = out / 'collector_initial.py'
        if sha(initial) != old_tools['bench_stage2_seed_pair.py']:
            raise ValueError('Retain the exact original collector before correcting it')
        for case in cases:
            timed = [r for r in previous['runs'] if r['case'] == case]
            if len(timed) != 4 * a.repeats or sorted(r['mode'] for r in timed) != ['original'] * (2*a.repeats) + ['paired'] * (2*a.repeats):
                raise ValueError('Timing matrix incomplete; refusing to hide missing trials')
        for row in previous['runs'] + previous['warmups'] + previous['gates']:
            if sha(row['log']) != row['log_sha256'] or sha(out / (row['name'] + '.jsonl')) != row['result_sha256']:
                raise ValueError('Original raw evidence changed')
            if row['category'] != 'gates' and parse(Path(row['log']).read_text(encoding='utf-8')) != row['phases']:
                raise ValueError('Original timing differs from raw log')
        data = previous
        data.setdefault('continuations', []).append(dict(identity=identity,
            reason='Gate fixtures have clean=0 and must not use the clean timing parser; all original timings retained.'))
        gate_suffix = '_r' + str(len(data['continuations']))

    def persist():
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    ini = out / 'manual.ini'
    ini.write_text('[gpu]\ndevice=1\n', encoding='utf-8')
    persist()
    reference, factor_sets = {}, {}

    def run(case, mode, index, category='runs'):
        verify()
        b, b2, block = case['bits'], case['B2'], case['block']
        gate = category == 'gates'
        name = f'm{b}_i{case["points"]}_c{block}_{mode}_{index}_{category}' + (gate_suffix if gate else '')
        log, result = out / (name + '.log'), out / (name + '.jsonl')
        cmd = [str(exe), '--ini', str(ini), '--save', str(saves[b]), '--device', '1', '--b2', str(b2),
               '--d', str(a.d), '--arena-mb', str(a.arena_mb), '--factor-only', '--log', str(log), '--results', str(result)]
        variant = dict(NTT_GIANT_SEED_PAIR='1' if mode == 'paired' else '0', NTT_GIANT_CHAIN_BLOCK=str(block),
            NTT_GIANT_CHAIN_CHECK='1' if gate else '0', NTT_GIANT_SEED_CHECK='1' if gate else '0',
            NTT_GFINV_SEG_CHECK='1' if gate else '0')
        start = time.perf_counter()
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env | variant)
        try:
            stdout, stderr = child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            child.communicate(timeout=30)
            raise RuntimeError('Owned curve timed out')
        elapsed = time.perf_counter() - start
        (out / (name + '_driver.log')).write_bytes(stdout + stderr)
        if child.returncode:
            raise RuntimeError('Curve failed: ' + name)
        text, r = log.read_text(encoding='utf-8'), read(result)
        n = (1 << b) - 1
        # Seed/segment fixtures intentionally mark full-wall timing unclean.
        clean_token = 'clean=0' if gate else 'clean=1'
        if (not all(s in text for s in ('gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0', clean_token)) or
                r['bad_factors'] or r['B1'] != 1000 or r['B2'] != b2 or r['sigma'] != 26 or
                int(r['N_hex'], 16) != n or any(not 1 < int(q) < n or n % int(q) for q in r['factors'])):
            raise ValueError('Arithmetic or input contract failed: ' + name)
        f = features(a.d, b2, b, 0)
        sizes = chunks(f)
        expected_ladders = sum((k + block - 1) // block for k in sizes)
        seed, pair = fields(text, 'real_giant_seed'), fields(text, 'real_giant_seed_pair')
        if int(seed['chunks']) != len(sizes) or int(seed['points']) != 2 * expected_ladders + len(sizes):
            raise ValueError('Actual seed schedule differs')
        if mode == 'paired':
            if (int(pair['base_builds']) != 1 or int(pair['base_nonunits']) or
                    int(pair['chunks']) != len(sizes) or int(pair['paired_ladders']) != expected_ladders or
                    int(pair['scalar_h2d_avoided']) != int(seed['points']) * 8 or
                    int(pair['base_d2h_bytes']) != 8 * ((b + 63) // 64)):
                raise ValueError('Paired path or cache not used as requested')
        elif any(int(pair[k]) for k in ('base_builds', 'chunks', 'paired_ladders', 'scalar_h2d_avoided', 'base_d2h_bytes')):
            raise ValueError('Disabled paired path performed work')
        affine = [dict(re.findall(r'(\w+)=([^\s]+)', s)) for s in re.findall(r'giant_chain_check: (.*)', text)]
        if gate:
            if (len(affine) != len(sizes) or any((int(row['points']), int(row['per_block']), int(row['mismatches'])) !=
                    (k, block, 0) for row, k in zip(affine, sizes)) or
                    int(seed['checked_words']) != 2 * ((b + 63) // 64) * int(seed['points']) or
                    seed['segments'] != seed['segment_checks']):
                raise ValueError('Independent point/seed/segment gate failed')
        leaf = fields(text, 'descent_values')['hash']
        key = (b, b2, block, mode)
        if key in reference and reference[key] != (leaf, r['factors']):
            raise ValueError('Same-policy output changed')
        reference[key] = (leaf, r['factors'])
        common = (b, b2, block)
        if common in factor_sets and factor_sets[common] != r['factors']:
            raise ValueError('Proper factor set changed')
        factor_sets[common] = r['factors']
        verify()
        record = dict(name=name, case=case, mode=mode, category=category, command=cmd, process_seconds=elapsed,
            phases=None if gate else parse(text), seed=seed, pair=pair, affine=affine, leaf_hash=leaf, factors=r['factors'],
            log=str(log), log_sha256=sha(log), result_sha256=sha(result),
            ntt=fields(text, 'ntt_workspace_stats'), multiply=fields(text, 's4_multiply_stats'))
        data[category].append(record)
        persist()
        if gate:
            print(name, 'seed/segment/affine checks passed', flush=True)
        else:
            print(name, 'full', record['phases']['full'], 'giant', record['phases']['giant'], flush=True)

    if a.resume_gates:
        for row in data['runs'] + data['warmups']:
            case = row['case']
            reference[(case['bits'], case['B2'], case['block'], row['mode'])] = (row['leaf_hash'], row['factors'])
            factor_sets[(case['bits'], case['B2'], case['block'])] = row['factors']
    else:
        for mode in ('original', 'paired'):
            run(cases[0], mode, 0, 'warmups')
        for case in cases:
            for rep in range(a.repeats):
                sequence = data['controls']['sequence'] if rep % 2 == 0 else ['paired', 'original', 'original', 'paired']
                for i, mode in enumerate(sequence):
                    run(case, mode, rep * 4 + i)
    for case in cases:
        if not any(r['case'] == case for r in data['gates']):
            run(case, 'paired', 0, 'gates')
    comparisons = []
    for case in cases:
        rows = [r for r in data['runs'] if r['case'] == case]
        stats = {m: {k: dict(mean=statistics.mean(r['phases'][k] for r in rows if r['mode'] == m),
                            median=statistics.median(r['phases'][k] for r in rows if r['mode'] == m),
                            minimum=min(r['phases'][k] for r in rows if r['mode'] == m),
                            maximum=max(r['phases'][k] for r in rows if r['mode'] == m))
                     for k in ('full', 'giant')} for m in ('original', 'paired')}
        gains = {k: 100 * (1 - stats['paired'][k]['mean'] / stats['original'][k]['mean']) for k in ('full', 'giant')}
        comparisons.append(dict(case=case, stats=stats, mean_reduction_percent=gains))
    data.update(comparisons=comparisons, complete=True)
    persist()
    for comparison in comparisons:
        print('comparison', comparison['case'], comparison['mean_reduction_percent'], flush=True)


if __name__ == '__main__':
    main()
