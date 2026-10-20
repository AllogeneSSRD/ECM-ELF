"""Compare short CUDA runs with a frozen pre-P0 build; never use production queues.

This is a behavioral regression gate, not a timing or architecture qualification.
Stage1 compares canonical save fields; Stage2 compares target-ring leaf fingerprints,
arithmetic result fields and successful mandatory checks. Existing independent point
and arithmetic probes remain necessary for algorithm changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='frozen old executable directory')
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = a.baseline.resolve()
    exes = dict(old1=baseline/'ecm_cuda.exe', old2=baseline/'ecm_cuda_stage2.exe',
                new1=a.stage1.resolve(), new2=a.stage2.resolve())
    identity = {k: sha(v) for k, v in exes.items()}
    config = out/'fixture.toml'
    config.write_text('[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = false\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NTT_', 'ECM_EDWARDS_'))}
    results = dict(complete=False, identity=identity, device=a.device, stage1=[], stage2=[], performance_samples=0)

    def persist():
        (out/'summary.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')

    def call(name, exe, args, input_text=None, extra_env=None):
        folder = out/name
        folder.mkdir()
        command = [str(exe), '--config', str(config), *map(str, args)]
        (folder/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        proc = subprocess.run(command, cwd=folder, env=env | (extra_env or {}),
                              input=input_text, text=True, encoding='utf-8', errors='replace',
                              capture_output=True, timeout=240)
        text = proc.stdout+proc.stderr
        (folder/'driver.log').write_text(text, encoding='utf-8')
        assert proc.returncode == 0, (name, proc.returncode, text[-3000:])
        assert {k: sha(v) for k, v in exes.items()} == identity, 'executable changed during regression'
        return folder, text

    def canonical_save(path):
        lines = []
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if 'SIGMA=' not in line:
                continue
            fields = dict(part.strip().split('=', 1) for part in line.split(';') if '=' in part)
            fields.pop('WHO', None)
            fields.pop('TIME', None)
            lines.append(fields)
        assert lines, f'no saved curves: {path}'
        return lines

    saves = {}
    # PARAM2/3 retain their existing exact-tier lookup. Use a 384-bit-container
    # input for those families when the P0 build carries only the 384 tier.
    stage1_cases = [('mont0', 'mont', 0, {}, '(2^127-1)'),
                    ('mont2', 'mont', 2, {}, '(2^256-2^32-977)'),
                    ('mont3', 'mont', 3, {}, '(2^256-2^32-977)'),
                    ('ed_naf', 'ed', None, {}, '(2^127-1)'),
                    ('ed_binary', 'ed', None, {'ECM_EDWARDS_STAGE1_ALGO': 'binary'}, '(2^127-1)')]
    for name, curve, param, extra, expression in stage1_cases:
        pair = []
        for which in ('old', 'new'):
            directory = out/f'{name}_{which}'
            save = directory/'point.save'
            args = ['--curve', curve, '--target', 'gpu', '-d', a.device, '-gpucurves', 4,
                    '-sigma', 26, '--tmp-dir', directory, '--exp-cache', 'off', '--ckpt', 0,
                    '-save', save]
            if param is not None:
                args += ['--gpu-param', param]
            args += [20, 0]
            call(f'{name}_{which}', exes[which+'1'], args, expression+'\n', extra)
            pair.append(canonical_save(save))
            if which == 'new':
                saves[name] = save
        assert pair[0] == pair[1], f'Stage1 save mismatch: {name}'
        results['stage1'].append(dict(case=name, curves=len(pair[0]), canonical_saves_equal=True))
        persist()

    def result_fields(path):
        records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        assert records, f'no internal result: {path}'
        volatile = {'program', 'source-build', 'timestamp', 'checksum', 'result_checksum',
                    'elapsed', 'time', 'seconds'}
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items()
                        if k not in volatile and not k.endswith('_seconds')}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value
        return [stable(row) for row in records]

    for name, saved, carrier, relocation, budget in [
            ('mont_normal', 'mont0', 0, 0, 640),
            ('ed_normal', 'ed_naf', 0, 0, 640),
            ('ed_carrier', 'ed_naf', 254, 0, 640),
            ('ed_relocation', 'ed_naf', 0, 1, 640),
            ('ed_host_fallback', 'ed_naf', 0, 0, 0)]:
        pair = []
        for which in ('old', 'new'):
            directory = out/f'{name}_{which}'
            args = ['--save', saves[saved], '--b2', 40000, '--d', 210, '--curves', 1,
                    '--device', a.device, '--carrier-exponent', carrier, '--relocation', relocation,
                    '--short-calibration', 0, '--tune-runtime', 0, '--owner-budget-mb', budget,
                    '--factor-only', '--results', directory/'internal.jsonl',
                    '--submission', directory/'results.json.txt', '--log', directory/'engine.log',
                    '--debug-log-file', directory/'debug.log', '--log-level', 'debug']
            folder, text = call(f'{name}_{which}', exes[which+'2'], args,
                                extra_env={'NTT_TARGET_LEAF_HASH': '1'})
            text += (folder/'debug.log').read_text(encoding='utf-8', errors='replace')
            leaves = re.findall(r'target_descent_values: ([^\r\n]+)', text)
            assert leaves, f'no target leaf fingerprint: {name}/{which}'
            assert not re.search(r'(?:mismatches|gmp_check_bad|gmp_selftest_bad)=[1-9]', text), name
            pair.append(dict(leaves=leaves[-1], result=result_fields(folder/'internal.jsonl')))
        assert pair[0] == pair[1], (name, pair)
        results['stage2'].append(dict(case=name, target_leaves_equal=True, result_fields_equal=True))
        persist()
    results['complete'] = True
    persist()
    print(json.dumps(dict(complete=True, stage1_pairs=len(results['stage1']),
                          stage2_pairs=len(results['stage2']), performance_samples=0)))


if __name__ == '__main__':
    main()
