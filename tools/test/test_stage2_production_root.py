"""Check every production resident-root word against the original packed path.

Bind inputs and complete leaf/factor outputs to finished unprofiled production
matrices. The packing is independent; the large-root arithmetic uses the same
NTT/S4 implementation. Small-node independent GMP gates live in the native tool.
These extra diagnostic invocations are never performance samples.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from bench_stage2_production import fields, freeze, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh output directory')
    identity = freeze(exe)
    if read(exe.parent / 'build_manifest.json').get('engine') != 'production':
        raise ValueError('requires the independent production engine')
    tool = Path(__file__); helper = ROOT / 'tools/bench/bench_stage2_production.py'
    report = dict(complete=False, identity=identity, tool_sha256=sha(tool),
                  helper_sha256=sha(helper), references={}, runs=[],
                  formal_performance_samples=0)
    cases = []
    for source in args.reference:
        source = source.resolve(); matrix = read(source)
        if not matrix['complete'] or matrix['mode'] not in ('timing', 'timing-wide'):
            raise ValueError('requires complete unprofiled timing references')
        key = next((k for k, v in matrix['identity'].items() if v == identity), None)
        if key is None: raise ValueError('production binary is not in the reference')
        report['references'][str(source)] = sha(source)
        for case in matrix['cases']:
            expected = next(r for r in matrix['runs'] if r['case'] == case['name']
                            and r['key'] == key and r['category'] == 'timing')
            if expected['root']['enabled'] != '1' or expected['root']['checked_words'] != '0':
                raise ValueError('reference must use the normal resident root')
            if any(c[0]['name'] == case['name'] for c in cases):
                raise ValueError('duplicate reference case')
            cases.append((case, expected))
    (out / 'collector.py').write_bytes(tool.read_bytes())
    (out / 'helper.py').write_bytes(helper.read_bytes())

    def verify():
        if freeze(exe) != identity or sha(tool) != report['tool_sha256'] or sha(helper) != report['helper_sha256']:
            raise ValueError('compiled or collector identity changed')
        for name, want in identity['sources'].items():
            if sha(exe.parent / 'sources' / name) != want: raise ValueError('source changed: ' + name)
        for source, want in report['references'].items():
            if sha(source) != want: raise ValueError('reference changed')
        for case, _ in cases:
            if sha(case['save']) != case['save_sha256']: raise ValueError('save changed')

    def persist():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')

    persist()
    for case, expected in cases:
        verify(); name = case['name']
        log = out / (name + '.log'); result = out / (name + '.jsonl')
        # Copy the exact measured command and environment, changing only output
        # paths and enabling the complete root shadow comparison.
        command = list(expected['command'])
        for flag, path in (('--log', log), ('--results', result)):
            command[command.index(flag) + 1] = str(path)
        env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
        env.update(expected['environment'], NTT_SCALED_ROOT_CHECK='1')
        proc = subprocess.run(command, env=env, capture_output=True, timeout=600)
        driver = out / (name + '_driver.log'); driver.write_bytes(proc.stdout + proc.stderr)
        if proc.returncode: raise ValueError(f'{name}: exit {proc.returncode}')
        records = [json.loads(s) for s in result.read_text().splitlines()]
        if len(records) != 1: raise ValueError('result record count differs')
        actual = records[0]; text = log.read_text(encoding='utf-8')
        for k in ('N_hex', 'B1', 'B2', 'sigma', 'record', 'bad_factors'):
            if actual[k] != expected['result'][k]: raise ValueError('input/result differs: ' + k)
        if actual['factors'] != expected['result']['factors'] or actual['bad_factors']:
            raise ValueError('factor result differs')
        if fields(text, 'descent_values') != expected['leaf']:
            raise ValueError('complete leaf fingerprint differs')
        for token in ('mont_selftest: cases=2048 mismatches=0', 's4_div_check: cases=800 bad=0',
                      'gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0'):
            if token not in text: raise ValueError('mandatory check missing: ' + token)
        root = fields(text, 'scaled_root_device'); coverage = fields(text, 's4_multiply_stats')
        p = int(root['coefficients']); w = (int(actual['N_hex'], 16).bit_length() + 63) // 64
        if root['enabled'] != '1' or int(root['checked_words']) != p * w:
            raise ValueError('complete root shadow comparison missing')
        if int(root['check_d2h_bytes']) != 8 * p * w + int(root['avoided_h_readback_bytes']):
            raise ValueError('diagnostic input readback differs')
        for k, extra in (('launches', 1), ('poly_muls', 1), ('coeffs_reduced', p), ('gmp_selftest_cases', 0)):
            if int(coverage[k]) != int(expected['coverage'][k]) + extra:
                raise ValueError('diagnostic arithmetic count differs: ' + k)
        ledger = fields(text, 'real_batched_wall')
        elapsed = float(re.search(r'real_batched_wall:.*\(elapsed=([0-9.]+)\)', text)[1])
        if abs(float(ledger['sum']) - elapsed) > .03: raise ValueError('wall ledger not closed')
        if abs(float(ledger['sum']) - float(fields(text, 'stage2_full_wall')['main'])) > .003:
            raise ValueError('precise main ledger not closed')
        report['runs'].append(dict(name=name, command=command,
            environment={k: v for k, v in env.items() if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'},
            result=actual, root=root, coverage=coverage, ledger=ledger, leaf=expected['leaf'],
            driver_sha256=sha(driver), log_sha256=sha(log), result_sha256=sha(result)))
        persist(); print(name, 'checked_words=' + root['checked_words'], flush=True)
    verify(); report.update(complete=True, passed=len(cases), failed=0); persist()


if __name__ == '__main__': main()
