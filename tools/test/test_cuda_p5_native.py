"""Full production-width CUDA qualification on an explicitly selected device.

Builders must first validate the probes against the actual worker objects. This
collector preserves every invocation and rejects incomplete coverage. Functional
timings are not performance samples. Use a fresh evidence directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('preflight', 'tier-gate', 'baseline', 'stage1', 'stage2', 'edwards-probe',
                'arithmetic-probe', 'fixtures', 'small-save', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path): setattr(a, key, value.resolve())
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    tiers = list(map(int, re.findall(r'^ECM_CUDA_TIER\((\d+),',
        (ROOT/'src/ecm/backend/cuda/dispatch/tiers.def').read_text(), re.M)))
    assert len(tiers) == 32 and len(set(tiers)) == 32
    assert read(a.preflight/'identity_preflight/result.json')['checks'] == 40
    assert read(a.preflight/'identity_preflight/result.json')['complete']
    tier_gate = read(a.tier_gate)
    assert tier_gate['complete'] and tier_gate['profiles'] == 16
    files = [a.stage1, a.stage2, a.edwards_probe, a.arithmetic_probe,
             a.baseline/'ecm_cuda.exe', a.baseline/'ecm_cuda_stage2.exe',
             a.fixtures, a.small_save, ROOT/'src/ecm/backend/cuda/dispatch/tiers.def']
    files += sorted({path.parent/'gmp-10.dll' for path in files[:6]})
    files += [a.stage1.parent/'stage1_local_manifest.json', a.stage2.parent/'build_manifest.json',
              a.edwards_probe.parent/'manifest.json', a.arithmetic_probe.parent/'manifest.json',
              ROOT/'tools/test/edwards_gpu_stage1_probe.cpp', ROOT/'tools/test/cuda_stage2_arithmetic_probe.cpp',
              ROOT/'config/ecm_options.json', ROOT/'config/ecm.toml.example',
              a.preflight/'identity_preflight/result.json', a.tier_gate, Path(__file__)]
    identities = {str(path): sha(path) for path in files}
    result = dict(complete=False, device=a.device, tiers=tiers, inputs=identities,
                  collector_sha256=sha(__file__), gates=[], performance_samples=0)
    env = {k:v for k,v in os.environ.items() if not k.startswith(
        ('NTT_', 'ECM_PRAC_', 'ECM_GPU_STAGE1_', 'ECM_EDWARDS_', 'CUDA_FORCE_', 'CUDA_DISABLE_'))}

    def persist():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def invoke(name, command, timeout=7200):
        folder = out/name
        folder.mkdir()
        command = list(map(str, command))
        (folder/'invocation.json').write_text(json.dumps(dict(command=command,
            cwd=str(ROOT)), indent=2)+'\n')
        with (folder/'stdout.log').open('wb') as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        text = (folder/'stdout.log').read_text(encoding='utf-8', errors='replace')
        assert proc.returncode == 0, (name, proc.returncode, text[-2500:])
        return folder, text

    def gate(name, detail):
        assert {str(path):sha(path) for path in files} == identities, 'input changed'
        outputs = {str(path.relative_to(out)):sha(path) for path in sorted((out/name).rglob('*')) if path.is_file()}
        result['gates'].append(dict(name=name, **detail, outputs=outputs))
        persist()
        print('PASS', name, json.dumps(detail), flush=True)

    def regression(name, script, flags, summary, expected):
        folder, _ = invoke(name, [sys.executable, ROOT/'tools/test'/script,
                                 *flags, '--output', out/name/'evidence', '--device', a.device])
        report = read(folder/'evidence'/summary)
        assert report['complete'], name
        for key, count in expected.items():
            assert (len(report[key]) if isinstance(report[key], list) else report[key]) == count, (name, key)
        gate(name, dict(summary=str(folder/'evidence'/summary), expected=expected))

    persist()
    try:
        regression('handoff', 'test_cuda_p0_runtime.py',
            ['--baseline', a.baseline, '--stage1', a.stage1, '--stage2', a.stage2],
            'summary.json', dict(stage1=5, stage2=5))
        regression('registry', 'test_cuda_p1_registry.py',
            ['--stage1', a.stage1, '--tiers', ','.join(map(str, tiers))],
            'result.json', dict(checks=260))
        regression('montgomery', 'test_cuda_p1_curves.py',
            ['--baseline', a.baseline/'ecm_cuda.exe', '--stage1', a.stage1,
             '--tiers', ','.join(map(str, tiers))], 'result.json', dict(cases=93))
        folder, text = invoke('edwards', [a.edwards_probe, a.device, out/'edwards/saves'])
        rows = re.findall(r'^PASS (\S+) device=(\d+) curves=(\d+).*cpu_match=1 tier=(\d+) tpi=(\d+)$', text, re.M)
        assert len(rows) == 131 and sum(int(r[2]) for r in rows) == 537
        for width in tiers:
            for suffix in ('', '_w6_a4', '_w6_a5'):
                row = [r for r in rows if r[0] == f'n{width-1}_b20'+suffix]
                assert len(row) == 1 and int(row[0][3]) == width
                assert int(row[0][1]) == a.device and int(row[0][2]) == 1
        gate('edwards', dict(cases=len(rows), curves=537, boundary_modes=96,
            independent_gmp=True, stop_resume='17/113', adaptive_restart=True))
        folder, text = invoke('arithmetic', [a.arithmetic_probe, a.device, '--all-words'])
        rows = [tuple(map(int, row)) for row in re.findall(
            r'^PASS bits=(\d+) words=(\d+) capacity=(\d+) mersenne=(\d+)$', text, re.M)]
        capacities = (4, 8, 16, 32, 64, 128, 256)
        expected = {(max(2, (w-1)*64+t), w, next(c for c in capacities if w <= c), m)
                    for w in range(1, 257) for t in (1, 63, 64) for m in (0, 1)}
        assert len(rows) == 1536 and set(rows) == expected
        tail = json.loads(text.strip().splitlines()[-1])
        assert tail == dict(complete=True, moduli=1536, comparisons=152064, bad=0)
        gate('arithmetic', tail | dict(words=[1, 256], capacities=capacities, independent_gmp=True))
        regression('stage2_runtime', 'test_cuda_p3_runtime.py',
            ['--baseline', a.baseline/'ecm_cuda_stage2.exe', '--exe', a.stage2,
             '--fixtures', a.fixtures, '--small-save', a.small_save],
            'summary.json', dict(pairs=14, plans=4, rejections=20))
        assert all(sha(out/path) == digest for g in result['gates'] for path,digest in g['outputs'].items())
        result['complete'] = True
        persist()
        print('PASS full native width qualification', flush=True)
    except BaseException as error:
        result['failure'] = repr(error)
        persist()
        raise


if __name__ == '__main__':
    main()
