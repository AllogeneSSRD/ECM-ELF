"""16384-bit native Stage2 acceptance using CPU/GMP-ECM Stage1 saves.

Preparation checks Stage1 and every baby/giant coordinate independently. Unit
cases also compute the complete monic evaluation values in ordinary arithmetic.
Synthetic X=2 is used only for the full-limb Mersenne primitive case.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)  # Bounded native inputs can have 4933 decimal digits.

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from bench_stage2_budget_scaling import fields


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(values, words):
    result = 1469598103934665603
    for value in values:
        for word in range(words):
            result = ((result ^ ((value >> (64 * word)) & ((1 << 64) - 1))) * 1099511628211) & ((1 << 64) - 1)
    return str(result)


def prepare(out, gmp):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use a fresh preparation directory')
    spec = importlib.util.spec_from_file_location('wide_ref', ROOT / 'tools/stat/suyama_mont_ref.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    identity = dict(tool_sha256=sha(__file__), reference_sha256=sha(spec.origin), gmp_sha256=sha(gmp))
    fixtures = []
    for name, exponent, offset, multiplier, b1 in (
        ('generic8193', 8193, 3, 1, 20), ('m16381', 16381, 1, 1, 20),
        ('generic16384', 16384, 15, 1, 20), ('factor16384', 16374, 3, 1019, 20),
        ('base_nonunit16384', 16372, 5, 2621, 20),
    ):
        n = multiplier * ((1 << exponent) - offset)
        if n.bit_length() != (8193 if name == 'generic8193' else 16381 if name == 'm16381' else 16384):
            raise ValueError('Fixture bit width is wrong: ' + name)
        expr = f'{multiplier}*(2^{exponent}-{offset})'
        point = ref.stage1(26, b1, n)
        if point['gcd'] != 1:
            raise ValueError('Stage1 fixture is not normalizable: ' + name)
        saved = out / (name + '.save')
        gmp_save = out / (name + '_gmp.save')
        cmd = [str(gmp.resolve()), '-param', '0', '-sigma', '26', '-c', '1', '-save', str(gmp_save), str(b1), str(b1)]
        proc = subprocess.run(cmd, input=(expr + '\n').encode(), capture_output=True, timeout=120)
        (out / (name + '_gmp.log')).write_bytes(proc.stdout + proc.stderr)
        match = re.search(r'\bX=(0x[0-9a-fA-F]+)', gmp_save.read_text())
        if proc.returncode or not match or int(match[1], 16) != point['x'] or sha(gmp) != identity['gmp_sha256']:
            raise ValueError('Independent GMP-ECM Stage1 mismatch: ' + name)
        saved.write_text(f'METHOD=ECM; PARAM=0; SIGMA=26; B1={b1}; N={expr}; X=0x{point["x"]:x}; '
                         f'CHECKSUM={b1*26*n*point["x"]%4294967291};\n', encoding='utf-8')
        baby, giant, bad = [], [], []
        for j in range(1, 106):
            if math.gcd(j, 210) == 1:
                x, z = ref.ladder(j, point['x'], 1, point['a24'], n)
                if math.gcd(z, n) != 1:
                    bad.append(dict(kind='baby', index=j, gcd_hex=format(math.gcd(z, n), 'x')))
                else:
                    baby.append(x * pow(z, -1, n) % n)
        for i in range(1, 67):
            x, z = ref.ladder(i * 210, point['x'], 1, point['a24'], n)
            if math.gcd(z, n) != 1:
                bad.append(dict(kind='giant', index=i, gcd_hex=format(math.gcd(z, n), 'x')))
            else:
                giant.append(x * pow(z, -1, n) % n)
        unit = not bad
        if unit != (multiplier == 1):
            raise ValueError('Unit/nonunit fixture changed: ' + name)
        base_gcd = math.gcd(ref.ladder(210, point['x'], 1, point['a24'], n)[1], n)
        if multiplier == 1019 and (base_gcd != 1 or not any(b['gcd_hex'] == format(1019, 'x') for b in bad)):
            raise ValueError('Independent giant factor fixture changed')
        if multiplier == 2621 and base_gcd != 2621:
            raise ValueError('Independent base factor fixture changed')
        expected = {}
        if unit:
            for count in (2, 65, 66):
                values = [1] * len(baby)
                for x in giant[:count]:
                    values = [v * (b - x) % n for v, b in zip(values, baby)]
                expected[str(count)] = fnv(values, (n.bit_length() + 63) // 64)
        fixtures.append(dict(name=name, N_hex=format(n, 'x'), bits=n.bit_length(), B1=b1,
            save=str(saved.resolve()), save_sha256=sha(saved), gmp_save_sha256=sha(gmp_save), gmp_command=cmd,
            stage1_cpu_gmp_equal=True, base_gcd_hex=format(base_gcd, 'x'), nonunits=bad,
            unit=unit, expected_leaf_hash=expected, expected_factor=multiplier if multiplier > 1 else 0))
        (out / 'fixtures.json').write_text(json.dumps(dict(identity=identity, fixtures=fixtures, complete=False), indent=2), encoding='utf-8')
        print(name, 'Stage1 CPU/GMP equal; unit=' + str(unit), flush=True)
    data = dict(identity=identity, fixtures=fixtures, complete=True)
    (out / 'fixtures.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gmp-ecm', type=Path, default=Path('D:/code/GIMPS/gmp-ecm/ecm-7.0.5-znver3/ecm.exe'))
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.prepare_only:
        prepare(out, args.gmp_ecm)
        return
    if not args.fixtures or not args.exe:
        parser.error('Runtime acceptance requires --fixtures and --exe')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use a fresh runtime output directory')
    fixture_sha = sha(args.fixtures)
    prepared = json.loads(args.fixtures.read_text(encoding='utf-8'))
    if not prepared['complete'] or prepared['identity']['reference_sha256'] != sha(ROOT / 'tools/stat/suyama_mont_ref.py'):
        raise ValueError('Preparation identity mismatch')
    exe = args.exe.resolve()
    manifest_path = exe.parent / 'frozen_sources_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    identity = dict(binary_sha256=sha(exe), manifest_sha256=sha(manifest_path), fixture_sha256=fixture_sha,
                    tool_sha256=sha(__file__))
    def verify():
        if (sha(exe) != identity['binary_sha256'] or manifest['binary_sha256'] != sha(exe) or
                sha(manifest_path) != identity['manifest_sha256'] or sha(args.fixtures) != fixture_sha or
                sha(__file__) != identity['tool_sha256']):
            raise ValueError('Runtime identity changed')
        for name, want in manifest['sources'].items():
            if sha(exe.parent / 'sources' / name) != want or sha(ROOT / name) != want:
                raise ValueError('Compiled source mismatch: ' + name)
    clean = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    clean.update(NTT_NO_PROGRESS='1', NTT_D_MODEL='0', NTT_GIANT_CHAIN_MIN='0',
        NTT_GIANT_CHAIN_BLOCK='64', NTT_GFINV_SEG_CHECK='1', NTT_GIANT_SEED_CHECK='1',
        NTT_FOLD_DEVICE_MAX_MB='64', CUDA_LAUNCH_BLOCKING='0')
    ini = out / 'manual.ini'
    ini.write_text('[gpu]\ndevice=1\n', encoding='utf-8')
    data = dict(identity=identity, runs=[], protocols=[], rejections=[], complete=False)
    def persist():
        (out / 'summary.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    variants = dict(
        host=dict(NTT_POINT_MERSENNE='0', NTT_S4_MERSENNE='0', NTT_BABY_DEVICE='0',
                  NTT_DEVICE_GLEAF='0', NTT_FOLD_DEVICE='0'),
        default={}, gpu_pair=dict(NTT_GIANT_SEED_PAIR='1'),
        cpu_pair=dict(NTT_GIANT_SEED_PAIR='1', NTT_GIANT_BASE_CPU='1'),
    )
    def run(case, variant, count=65, extra=None):
        verify()
        if sha(case['save']) != case['save_sha256']:
            raise ValueError('Save identity changed')
        name = case['name'] + '_' + variant + '_i' + str(count)
        log, result = out / (name + '.log'), out / (name + '.jsonl')
        env = clean | variants.get(variant, {}) | (extra or {})
        env['NTT_GIANT_CHAIN_CHECK'] = '1' if case['unit'] else '0'
        b2 = 210 * (count - 2) if count > 2 else case['B1'] + 1
        cmd = [str(exe), '--ini', str(ini), '--save', case['save'], '--b2', str(b2), '--d', '210',
               '--device', '1', '--arena-mb', '512', '--factor-only', '--log', str(log), '--results', str(result)]
        proc = subprocess.run(cmd, env=env, capture_output=True, timeout=600)
        (out / (name + '_driver.log')).write_bytes(proc.stdout + proc.stderr)
        if proc.returncode:
            raise ValueError('Actual wide curve failed: ' + name)
        record = json.loads(result.read_text(encoding='utf-8').splitlines()[-1])
        n = int(case['N_hex'], 16)
        text = log.read_text(encoding='utf-8')
        if (record['N_hex'].lower() != case['N_hex'] or record['B1'] != case['B1'] or record['B2'] != b2 or
                record['sigma'] != 26 or record['bad_factors'] or
                any(not 1 < int(f) < n or n % int(f) for f in record['factors'])):
            raise ValueError('Wide input/proper factor mismatch: ' + name)
        for token in ('stage1_skipped=1', 'mont_selftest: cases=2048 mismatches=0',
                      'gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0', 's4_div_check: cases=800 bad=0'):
            if token not in text:
                raise ValueError('Required arithmetic check absent: ' + token)
        leaf = fields(text, 'descent_values')
        if case['unit'] and leaf['hash'] != case['expected_leaf_hash'][str(count)]:
            raise ValueError('Independent full monic evaluation mismatch: ' + name)
        if case['expected_factor'] and not any(int(f) % case['expected_factor'] == 0 for f in record['factors']):
            raise ValueError('Known factor missing: ' + name)
        seed, pair, base = (fields(text, key) for key in ('real_giant_seed', 'real_giant_seed_pair', 'real_giant_base'))
        if int(seed['checked_words']) != 2 * ((case['bits'] + 63) // 64) * int(seed['points']) or seed['segments'] != seed['segment_checks']:
            raise ValueError('Wide seed/segment coverage absent')
        if variant.endswith('pair'):
            cpu = variant == 'cpu_pair'
            if (pair['base_builds'] != '1' or int(base['cpu_builds']) != int(cpu) or
                    int(base['gpu_builds']) != int(not cpu) or
                    int(base['checked_words']) != (2*((case['bits']+63)//64) if cpu else 0)):
                raise ValueError('Wide base/cache path absent')
            if case['base_gcd_hex'] != '1' and (pair['base_nonunits'] != '1' or pair['chunks'] != '0'):
                raise ValueError('Wide nonunit base did not fall back')
        if case['unit']:
            affine = [dict(re.findall(r'(\w+)=(\S+)', s)) for s in re.findall(r'giant_chain_check: (.*)', text)]
            if not affine or sum(int(s['points']) for s in affine) != count or any(int(s['mismatches']) for s in affine):
                raise ValueError('Wide full affine comparison absent')
        if variant.startswith('legacy_descent'):
            if ('s5_descent:' not in text or 's5_dev_done:' not in text or
                    fields(text, 'descent_check')['mismatching_coefficients'] != '0' or
                    fields(text, 'descent_check_leaves')['differing_leaves'] != '0'):
                raise ValueError('Wide legacy device descent/full leaf comparison absent')
        if variant == 'mersenne_fixtures' and ('s4_mersenne_check: cases=912' not in text or
                'xadd6_selftest: cases=1280' not in text):
            raise ValueError('Extended Mersenne/xADD fixtures absent')
        if variant == 'full_mersenne_primitive' and 'point_mersenne_mode: requested=1 enabled=1 bits=16384 nw=256' not in text:
            raise ValueError('Full-limb Mersenne primitive path absent')
        data['runs'].append(dict(name=name, variant=variant, count=count, command=cmd,
            environment={k:v for k,v in env.items() if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'},
            result=record, leaf=leaf, seed=seed, pair=pair, base=base, log=str(log), log_sha256=sha(log),
            result_sha256=sha(result)))
        verify()
        persist()
        print(name, 'passed', flush=True)
    persist()
    # Exercise the two legacy boundary routes first so an invariant failure
    # is found before the main four-path fixture matrix. No case is omitted.
    full = next(c for c in prepared['fixtures'] if c['name'] == 'generic16384')
    for no_linear in (1, 0):
        run(full, 'legacy_descent' + ('_no_linear' if no_linear else ''), extra=dict(
            NTT_SCALED_DESCENT='0', NTT_S5_ON='1', NTT_S5_NO_LINEAR=str(no_linear),
            NTT_S4_DESCENT_CHECK='1'))
    for case in prepared['fixtures']:
        for variant in variants:
            run(case, variant)
        if case['base_gcd_hex'] != '1':
            pair_runs = [r for r in data['runs'] if r['name'].startswith(case['name']) and r['variant'].endswith('pair')]
            if set(pair_runs[0]['result']['factors']) != set(pair_runs[1]['result']['factors']):
                raise ValueError('Wide CPU/GPU nonunit fallback raw factors differ')
    for count in (2, 66):
        run(full, 'cpu_pair', count)
    # Additional kernel routes and division/fold primitives are independently
    # checked in the native binary; these diagnostic curves are not timings.
    run(full, 'montgomery_tail', extra=dict(NTT_S4_OLDTAIL='1', NTT_S4_MERSENNE='0'))
    mers = next(c for c in prepared['fixtures'] if c['name'] == 'm16381')
    run(mers, 'mersenne_fixtures', extra=dict(NTT_S4_MERSENNE_TEST='1', NTT_XADD6_TEST='1'))
    # A synthetic saved coordinate reaches full-limb Mersenne arithmetic. It
    # is explicitly excluded from the valid Stage1/monic-oracle fixture set.
    n = (1 << 16384) - 1
    saved = out / 'synthetic_m16384.save'
    saved.write_text(f'METHOD=ECM; SIGMA=26; B1=20; N=2^16384-1; X=2; CHECKSUM={20*26*n*2%4294967291};\n')
    synthetic = dict(name='synthetic_m16384', N_hex=format(n,'x'), bits=16384, B1=20,
        save=str(saved), save_sha256=sha(saved), unit=False, expected_factor=0)
    run(synthetic, 'full_mersenne_primitive', extra=dict(NTT_XADD6_TEST='1'))
    # Verify geometry without running curves, then a real queue with exponent
    # 16384 and [,B2][,skip][,count]. No user queue files are touched.
    command = [str(exe), '--ini', str(ini), '--save', full['save'], '--b2', '13230', '--d', '210',
               '--device', '1', '--arena-mb', '512', '--plan-only']
    verify()
    proc = subprocess.run(command, env=clean, capture_output=True, timeout=120)
    (out / 'plan_only.log').write_bytes(proc.stdout + proc.stderr)
    plans = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines() if line.startswith('{')]
    if proc.returncode or len(plans) != 1 or plans[0]['bits'] != 16384 or plans[0]['words'] != 256 or plans[0]['curves_executed'] != 0:
        raise ValueError('Wide actual NTT geometry query failed')
    data['protocols'].append(dict(name='wide_plan_only', command=command, plan=plans[0]))
    queue_dir = out / 'queue'
    queue_dir.mkdir()
    save = queue_dir / 'three.save'
    save.write_bytes(Path(full['save']).read_bytes() * 3)
    queue = queue_dir / 'worktodo.txt'
    line = 'ECMSTAGE2=1,2,16384,-15,"three.save",13230,1,1\n'
    queue.write_text(line, encoding='utf-8')
    queue_ini = queue_dir / 'ecm.ini'
    queue_ini.write_text('[queue]\nworktodo=worktodo.txt\nfinished=finished.txt\ntmp_dir=.\n'
                         '[gpu]\ndevice=1\n[stage2]\nstage2_d=210\nstage2_arena_mb=512\n', encoding='utf-8')
    result, log = queue_dir / 'result.jsonl', queue_dir / 'engine.log'
    command = [str(exe), '--ini', str(queue_ini), '--once', '--factor-only', '--results', str(result), '--log', str(log)]
    proc = subprocess.run(command, env=clean, capture_output=True, timeout=600)
    (queue_dir / 'driver.log').write_bytes(proc.stdout + proc.stderr)
    records = [json.loads(s) for s in result.read_text(encoding='utf-8').splitlines()] if result.exists() else []
    text = log.read_text(encoding='utf-8') if log.exists() else ''
    if (proc.returncode or len(records) != 1 or records[0]['record'] != 2 or records[0]['B2'] != 13230 or
            records[0]['N_hex'] != full['N_hex'] or records[0]['bad_factors'] or
            fields(text, 'descent_values')['hash'] != full['expected_leaf_hash']['65'] or
            line.strip() in queue.read_text() or line.strip() not in (queue_dir / 'finished.txt').read_text()):
        raise ValueError('Wide queue B2/skip/count/finished acceptance failed')
    data['protocols'].append(dict(name='wide_queue', command=command, result=records[0], log_sha256=sha(log)))
    for name, invalid_queue in (('oversize_save', False), ('oversize_queue', True)):
        bad_save = out / (name + '.save')
        n = (1 << 16385) - 1
        bad_save.write_text(f'METHOD=ECM; SIGMA=26; B1=20; N=2^16385-1; X=2; CHECKSUM={20*26*n*2%4294967291};\n')
        rejected = out / (name + '.jsonl')
        command = [str(exe), '--ini', str(ini), '--device', '1', '--dry-run', '--results', str(rejected)]
        if invalid_queue:
            bad_queue = out / (name + '.txt')
            bad_queue.write_text(f'ECMSTAGE2=1,2,16385,-1,"{bad_save}",13230,0,1\n', encoding='utf-8')
            before = sha(bad_queue)
            command += ['--worktodo', str(bad_queue)]
        else:
            command += ['--save', str(bad_save), '--b2', '13230']
        proc = subprocess.run(command, env=clean, capture_output=True, timeout=120)
        error = (proc.stdout + proc.stderr).decode('utf-8', errors='replace')
        (out / (name + '.log')).write_text(error, encoding='utf-8')
        expected = 'worktodo exponent exceeds' if invalid_queue else 'at most 16384 bits'
        if not proc.returncode or expected not in error or rejected.exists() or (invalid_queue and sha(bad_queue) != before):
            raise ValueError('Oversize boundary was not rejected safely')
        data['rejections'].append(dict(name=name, command=command, exit=proc.returncode, expected_error=expected))
    verify()
    data.update(complete=True, passed=len(data['runs']), failed=0,
        scope='Valid CPU/GMP Stage1 saves at 8193/16381/16384 bits, full monic oracles on unit cases, '
              'wide nonunit base/giant proper factors, tail seeds, actual reduction/descent routes; '
              'synthetic full Mersenne input is arithmetic coverage only, not a production save.')
    persist()


if __name__ == '__main__':
    main()
