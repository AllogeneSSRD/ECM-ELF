"""Owner layout arithmetic and same-binary saved Stage2 gates / timings.

Reuse masks: 0 original, 1 q/qb, 2 G/reverse, 3 both. Independent bigint
layout checks and CPU/GMP-prepared saves; no synthetic Stage1 performance input.
Each invocation requires a fresh output directory and retains every raw sample.
"""
import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
HELPER = ROOT / 'tools/bench/bench_stage2_production.py'
spec = importlib.util.spec_from_file_location('stage2_frozen', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sha, fields, read = helper.sha, helper.fields, helper.read


def layout(p, w, mode):
    cell = (p + 1) * w
    return dict(source_words=(4 if mode & 2 else 5) * cell,
                result_words=((3 if mode & 1 else 4) * p + 2) * w,
                bytes=8 * (((7 * p + 7) if mode == 3 else
                            (8 * p + 7) if mode == 2 else
                            (8 * p + 8) if mode == 1 else (9 * p + 8)) * w) + 48,
                f=0, inv=cell, h=2 * cell, g=3 * cell,
                reverse=3 * cell if mode & 2 else 4 * cell,
                t=0, q=(2 * p + 1) * w,
                qb=(2 * p + 1) * w if mode & 1 else (3 * p + 2) * w)


def cpu_gate(out):
    src = out / 'layout.cpp'
    src.write_text('#include "ecm_stage2_geometry.h"\n#include <iostream>\n'
        'int main(){ecm_stage2::Word p,w;unsigned m;while(std::cin>>p>>w>>m){'
        'ecm_stage2::FoldOwnerLayout l;bool ok=ecm_stage2::fold_owner_layout(p,w,m,l);'
        'std::cout<<ok<<" "<<ecm_stage2::owner_bytes(p,w,m);'
        'if(ok)std::cout<<" "<<l.source_words<<" "<<l.result_words<<" "<<l.bytes'
        '<<" "<<l.f<<" "<<l.inv<<" "<<l.h<<" "<<l.g<<" "<<l.reverse'
        '<<" "<<l.t<<" "<<l.q<<" "<<l.qb;std::cout<<"\\n";} }\n')
    vc = sorted(glob.glob('C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'))
    if not vc:
        raise ValueError('vcvars64.bat not found')
    batch = out / 'compile.cmd'
    batch.write_text(f'@echo off\ncall "{vc[0]}" >nul\n'
        f'cl /nologo /std:c++17 /EHsc /O2 /I"{ROOT / "src/core"}" "{src}" '
        f'/Fe:"{out / "layout.exe"}" /Fo:"{out / "layout.obj"}"\n')
    proc = subprocess.run(['cmd.exe', '/c', str(batch)], capture_output=True)
    (out / 'compile.log').write_bytes(proc.stdout + proc.stderr)
    if proc.returncode:
        raise ValueError('layout compilation failed')
    maximum = (1 << 64) - 1
    cases = [(p, w, m) for p in (1, 2, 3, 8, 17, 24, 2880, 126720, 138240)
             for w in (1, 2, 70, 128, 129, 256) for m in range(4)]
    rng = random.Random(20261008)
    cases += [(rng.randrange(1, 1 << 40), rng.randrange(1, 257), rng.randrange(4)) for _ in range(400)]
    for w in (1, 70, 256):
        for m in range(4):
            # Exact first rejected positive-width payload, independently derived.
            slope = (9, 8, 8, 7)[m]; intercept = (8, 8, 7, 7)[m]
            boundary = (((maximum - 48) // 8) // w - intercept) // slope
            cases += [(p, w, m) for p in (boundary - 1, boundary, boundary + 1, maximum)]
    cases += [(1, 1, 4), (24, 256, (1 << 32) - 1)]
    stdin = ''.join(f'{p} {w} {m}\n' for p, w, m in cases).encode()
    proc = subprocess.run([str(out / 'layout.exe')], input=stdin, capture_output=True, check=True)
    (out / 'cases.txt').write_bytes(stdin); (out / 'results.txt').write_bytes(proc.stdout)
    rows = proc.stdout.decode().splitlines()
    if len(rows) != len(cases):
        raise ValueError('CPU layout result count')
    for (p, w, m), line in zip(cases, rows):
        actual = list(map(int, line.split())); want = layout(p, w, m)
        valid = m <= 3 and want['bytes'] <= maximum
        if bool(actual[0]) != valid or actual[1] != (want['bytes'] if valid else maximum):
            raise ValueError(f'layout overflow / payload mismatch: {(p, w, m)}')
        if valid:
            if actual[2:] != list(want.values()):
                raise ValueError(f'layout offsets mismatch: {(p, w, m)}')
            for key in ('f', 'inv', 'h', 'g', 'reverse'):
                if want[key] + (p + 1) * w > want['source_words']:
                    raise ValueError('source bounds')
            for key, count in (('t', 2 * p + 1), ('q', p + 1), ('qb', p)):
                if want[key] + count * w > want['result_words']:
                    raise ValueError('result bounds')
    report = dict(complete=True, passed=len(cases), failed=0, formal_performance_samples=0,
                  header_sha256=sha(ROOT / 'src/core/ecm_stage2_geometry.h'),
                  tool_sha256=sha(__file__), probe_sha256=sha(src), binary_sha256=sha(out / 'layout.exe'))
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def native(a, out):
    exe = a.exe.resolve(); ident = helper.freeze(exe); tool = sha(__file__); hs = sha(HELPER)
    build = read(exe.parent / 'build_manifest.json')
    if build.get('engine') != 'development' or build['gl_fixed_mode'] != 3 or build['outer_unroll_u'] != 0:
        raise ValueError('requires frozen development PTX3 / outer0 binary')
    prepared = read(a.fixtures)
    if not prepared['complete'] or prepared['identity']['reference_sha256'] != sha(ROOT / 'tools/stat/suyama_mont_ref.py'):
        raise ValueError('fixture identity changed')
    fixture_sha = sha(a.fixtures)
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0', NTT_GIANT_SEED_PAIR='1', NTT_GIANT_BASE_CPU='0',
               NTT_GIANT_CHAIN_BLOCK='64', NTT_GIANT_CHAIN_MIN='32768',
               NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    ini = out / 'manual.ini'; ini.write_text(f'device={a.device}\n')
    report = dict(identity=ident, tool_sha256=tool, helper_sha256=hs, fixture_sha256=fixture_sha,
                  mode=a.mode, complete=False, runs=[], rejections=[], plans=[], formal_performance_samples=0)
    if a.resume_gates:
        if a.mode != 'gate': raise ValueError('timing cannot resume / select samples')
        previous = read(out / 'measurements_initial.json')
        if previous['complete'] or previous['identity'] != ident or previous['fixture_sha256'] != fixture_sha:
            raise ValueError('gate continuation identity differs')
        if sha(out / 'collector_initial.py') != previous['tool_sha256']:
            raise ValueError('initial collector changed')
        for row in previous['runs']:
            if sha(row['log']) != row['log_sha256'] or sha(Path(row['log']).with_suffix('.jsonl')) != row['result_sha256']:
                raise ValueError('previous verified arithmetic changed')
        report['runs'] = previous['runs']
        report['continuation'] = dict(initial_measurements_sha256=sha(out / 'measurements_initial.json'),
            initial_tool_sha256=previous['tool_sha256'], reason='No owner exists for I<=P; backend rejection precedes layout. Preserve successful rows and rejected short-tail raw output.')

    def persist():
        (out / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')

    def verify():
        if sha(exe) != ident['binary_sha256'] or sha(__file__) != tool or sha(HELPER) != hs or sha(a.fixtures) != fixture_sha:
            raise ValueError('experiment identity changed')
        if sha(exe.parent / 'build_manifest.json') != ident['build_sha256']:
            raise ValueError('build receipt changed')
        for name, digest in ident['sources'].items():
            if sha(exe.parent / 'sources' / name) != digest:
                raise ValueError('frozen source changed: ' + name)

    def call(name, args, extra, code=0):
        verify(); cmd = [str(exe), '--ini', str(ini), *args]
        proc = subprocess.run(cmd, env=env | extra, capture_output=True, timeout=900)
        driver = out / (name + '_driver.log'); driver.write_bytes(proc.stdout + proc.stderr)
        verify()
        if proc.returncode != code:
            raise ValueError(f'{name}: exit {proc.returncode}, expected {code}')
        return dict(name=name, command=cmd, environment={k: v for k, v in (env | extra).items()
                    if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'}, exit=proc.returncode,
                    driver_log_sha256=sha(driver)), driver.read_text(encoding='utf-8', errors='replace')

    def run(case, mode, name, count=65, budget=64, extra=None, category='gate', d=210, b2=None):
        previous = next((r for r in report['runs'] if r['name'] == name), None)
        if previous: return previous
        if (out / (name + '_driver.log')).exists(): name += '_continuation'
        if sha(case['save']) != case['save_sha256']:
            raise ValueError('save identity changed')
        b2 = b2 if b2 is not None else d * (count - 2) if count > 2 else case['B1'] + 1
        log = out / (name + '.log'); result = out / (name + '.jsonl')
        use = dict(NTT_FOLD_OWNER_REUSE=str(mode), NTT_FOLD_DEVICE_MAX_MB=str(budget))
        if category == 'gate':
            use.update(NTT_FOLD_DEVICE_CHECK='1', NTT_GROOT_TO_FOLD_CHECK='1',
                       NTT_GFINV_SEG_CHECK='1', NTT_GIANT_SEED_CHECK='1',
                       NTT_FOLD_DEVICE_TEST='1')
        use.update(extra or {})
        row, _ = call(name, ['--save', case['save'], '--b2', str(b2), '--d', str(d), '--device', str(a.device),
            '--arena-mb', '6300', '--factor-only', '--log-level', 'debug', '--log', str(log), '--results', str(result)], use)
        records = [json.loads(s) for s in result.read_text().splitlines()]
        if len(records) != 1: raise ValueError('result count')
        r = records[0]; text = log.read_text(encoding='utf-8'); n = int(case['N_hex'], 16)
        if r['N_hex'].lower() != case['N_hex'] or r['B1'] != case['B1'] or r['B2'] != b2 or r['sigma'] != 26 or r['bad_factors']:
            raise ValueError('saved arithmetic input mismatch')
        if any(not 1 < int(f) < n or n % int(f) for f in r['factors']): raise ValueError('invalid factor')
        if case.get('expected_factor') and not any(int(f) % case['expected_factor'] == 0 for f in r['factors']):
            raise ValueError('known factor missing')
        for token in ('stage1_skipped=1', 'mont_selftest: cases=2048 mismatches=0',
                      's4_div_check: cases=800 bad=0', 'gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0'):
            if token not in text: raise ValueError('mandatory check missing: ' + token)
        leaf = fields(text, 'descent_values')
        if case['unit'] and str(count) in case.get('expected_leaf_hash', {}) and leaf['hash'] != case['expected_leaf_hash'][str(count)]:
            raise ValueError('independent complete monic fingerprint mismatch')
        if use.get('NTT_FOLD_DEVICE_TEST') == '1' and fields(text, 'fold_device_fixture')['cases'] != '30':
            raise ValueError('fold fixture coverage')
        fd = fields(text, 'real_batched_folddevice'); shape = fields(text, 'real_shape')
        p = int(shape['P'].split('=')[-1]); want = layout(p, (case['bits'] + 63) // 64, mode)['bytes']
        if fd['requested'] == '0':
            if int(fd['layout_bytes']) or int(fd['peak_bytes']): raise ValueError('unused owner allocated')
        elif int(fd['reuse']) != mode or int(fd['layout_bytes']) != (0 if fd['fallback'] == 'backend' else want):
            raise ValueError('runtime layout accounting')
        if fd['enabled'] == '1':
            if int(fd['peak_bytes']) != want or int(fd['saved_bytes']) != layout(p, (case['bits'] + 63) // 64, 0)['bytes'] - want:
                raise ValueError('allocation payload accounting')
        elif int(fd['peak_bytes']) != 0: raise ValueError('fallback allocated owner')
        row.update(case=case['name'], reuse=mode, category=category, budget=budget, result=r, leaf=leaf,
                   owner=fd, rootfold=fields(text, 'real_batched_rootfold'), coverage=fields(text, 's4_multiply_stats'),
                   wall=fields(text, 'stage2_full_wall'), phases=fields(text, 'real_batched_split'),
                   log=str(log), log_sha256=sha(log), result_sha256=sha(result))
        report['runs'].append(row); persist(); print(name, row['wall']['total'], 's', flush=True)
        return row

    persist()
    full = next(c for c in prepared['fixtures'] if c['name'] == 'generic16384')
    if a.mode == 'gate':
        for case in prepared['fixtures']:
            group = [run(case, m, f'{case["name"]}_reuse{m}') for m in range(4)]
            for key in ('hash', 'words'):
                if len({r['leaf'][key] for r in group}) != 1: raise ValueError('cross-layout leaf mismatch')
            for key in ('digest_words', 'digest_sum', 'digest_xor'):
                if len({r['rootfold'][key] for r in group}) != 1: raise ValueError('cross-layout root digest mismatch')
        for count in (2, 66): run(full, 3, 'tail_' + str(count), count=count)
        for name, extra in [('budget', {'NTT_FOLD_DEVICE_MAX_MB': '0'}),
                            ('allocation', {'NTT_FOLD_DEVICE_ALLOC_FAIL': '1'}),
                            ('copied_pack', {'NTT_S4_PACK_DIRECT': '0'}),
                            ('sync_oracle', {'NTT_S4_ORACLE_ASYNC': '0'}),
                            ('ring_one', {'NTT_S4_ORACLE_RING': '1'})]:
            run(full, 3, name, extra={'NTT_FOLD_DEVICE_TEST': '0'} | extra)
        for m in range(4):
            row, text = call('plan_' + str(m), ['--save', full['save'], '--b2', '13230', '--d', '210', '--device', str(a.device), '--plan-only'],
                             {'NTT_FOLD_OWNER_REUSE': str(m)})
            plan = next(json.loads(s) for s in text.splitlines() if s.startswith('{'))
            if plan['owner_reuse'] != m or plan['owner_bytes'] != layout(24, 256, m)['bytes'] or plan['curves_executed'] != 0:
                raise ValueError('planner/runtime mismatch')
            row['plan'] = plan; report['plans'].append(row); persist()
        for key in ('NTT_FOLD_DEVICE_TEST_BAD', 'NTT_GROOT_TO_FOLD_TEST_BAD', 'NTT_S4_ORACLE_TEST_BAD'):
            log = out / (key + '.log'); result = out / (key + '.jsonl')
            row, _ = call(key, ['--save', full['save'], '--b2', '13230', '--d', '210', '--log', str(log), '--results', str(result)],
                          {'NTT_FOLD_OWNER_REUSE': '3', 'NTT_FOLD_DEVICE_CHECK': '1', 'NTT_GROOT_TO_FOLD_CHECK': '1', key: '1'}, code=2)
            if result.exists() or 'FATAL' not in log.read_text(encoding='utf-8'): raise ValueError('fault accepted')
            row['log_sha256'] = sha(log); report['rejections'].append(row); persist()
        for m in ('4', '-1', '', 'abc', '03'):
            row, text = call('invalid_' + (m or 'empty'), ['--save', full['save'], '--b2', '13230'], {'NTT_FOLD_OWNER_REUSE': m}, code=2)
            if 'NTT_FOLD_OWNER_REUSE must be 0..3' not in text: raise ValueError('invalid layout accepted')
            report['rejections'].append(row); persist()
    else:
        if not a.save: raise ValueError('--save required for timing')
        # This saved input is also independently prepared in the fixed-D study.
        previous = read(ROOT / 'build_cuda_cmake/_stage2_production_20261007/whole_ab/measurements.json')
        if not previous['complete'] or sha(a.save) != previous['cases'][0]['save_sha256']:
            raise ValueError('timing requires the independently validated M4423 anchor save')
        prior = previous['runs'][0]['result']
        case = dict(name='m4423', save=str(a.save.resolve()), save_sha256=sha(a.save),
                    N_hex=prior['N_hex'].lower(), B1=1000, bits=4423, unit=True)
        sequence = [0, 3, 3, 0, 3, 0, 0, 3]
        report['sequence'] = sequence; report['budgets'] = [640, 512]; persist()
        for budget in report['budgets']:
            for category, modes in [('warmup', [0, 3]), ('timing', sequence)]:
                for i, m in enumerate(modes):
                    run(case, m, f'b{budget}_{category}_{i}_reuse{m}', count=1456028, budget=budget,
                        category=category, d=1381380, b2=2011326186870)
            group = [r for r in report['runs'] if r['budget'] == budget]
            if len({r['leaf']['hash'] for r in group}) != 1 or len({tuple(sorted(r['result']['factors'])) for r in group}) != 1:
                raise ValueError('cross-layout result mismatch')
            for key in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases', 'gmp_checked', 'full_checks'):
                if len({r['coverage'][key] for r in group}) != 1: raise ValueError('check coverage changed: ' + key)
        report['summary'] = {}
        for budget in report['budgets']:
            means = {str(m): statistics.mean(float(r['wall']['total']) for r in report['runs']
                     if r['budget'] == budget and r['reuse'] == m and r['category'] == 'timing') for m in (0, 3)}
            report['summary'][str(budget)] = dict(full_mean_seconds=means, reduction_percent=100 * (1 - means['3'] / means['0']))
        report['formal_performance_samples'] = 16
    verify(); report.update(complete=True, passed=len(report['runs']) + len(report['plans']) + len(report['rejections']), failed=0)
    persist(); print(json.dumps(dict(passed=report['passed'], summary=report.get('summary'))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('layout', 'gate', 'timing'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exe', type=Path)
    p.add_argument('--fixtures', type=Path)
    p.add_argument('--save', type=Path)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--resume-gates', action='store_true', help='Continue verified gate rows with preserved initial collector / measurements; never timings')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not a.resume_gates: raise ValueError('use a fresh output directory')
    if a.mode == 'layout': cpu_gate(out)
    else:
        if not a.exe or not a.fixtures: p.error('--exe and --fixtures required')
        native(a, out)


if __name__ == '__main__':
    main()
