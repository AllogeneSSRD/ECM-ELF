"""Edwards B1=2 small-prime GCDs, baby-proof reuse and ladder fallback on GPU1."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from test_edwards_gpu_large_bounds import save_rows
from test_edwards_gpu_pipeline import ROOT, stage1, mont, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); s1 = a.stage1.resolve(); s2 = a.stage2.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh evidence directory')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(identity=dict(stage1=sha(s1), stage2=sha(s2), test=sha(__file__),
                  reference=sha(ROOT/'tools/test/test_edwards_gpu_pipeline.py'),
                  parser=sha(ROOT/'tools/test/test_edwards_gpu_large_bounds.py')),
                  device=a.device, checks=[], complete=False, formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    ini = out/'ecm.ini'; ini.write_text(f'device={a.device}\n')
    base = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    base.update(NTT_NO_PROGRESS='1', NTT_SMALL_PRIME_CHECK='1', NTT_S4_DESCENT_CHECK='1')
    def persist(): (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    def call(name, command, input=None, extra=None, good=True):
        r = subprocess.run([str(x) for x in command], input=input, env=base | (extra or {}), cwd=out,
                           capture_output=True, timeout=300)
        log = out/(name+'.log'); log.write_bytes(r.stdout+r.stderr)
        assert (r.returncode == 0) == good, (name, r.returncode, (r.stdout+r.stderr)[-2000:])
        report['checks'].append(dict(name=name, exit=r.returncode, log_sha256=sha(log)))
        persist(); print('PASS', name, flush=True)
        return (r.stdout+r.stderr).decode('utf-8', 'replace')
    persist()
    for label, prime, sigma, ell in [('unit', 1, 17, None), ('p131_q3', 131, 31, 3),
                                    ('p67_q5', 67, 19, 5), ('p97_q7', 97, 26, 7)]:
        n = prime*((1 << 127)-1); save = out/(label+'.save')
        a24, rx, rz = stage1(n, sigma, 2)
        assert math.gcd(a24*(a24-1), n) == 1 and math.gcd(rz, n) == 1
        qx = rx*pow(rz, -1, n) % n
        assert math.gcd(qx, n) == 1
        oracle = {q: math.gcd(mont.ladder(q, qx, 1, a24, n)[1], n) for q in (3, 5, 7)}
        if ell:
            assert oracle[ell] == prime and mont.ladder(ell, qx, 1, a24, n)[0] % prime != 0
        call(label+'_stage1', [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device, '-sigma', sigma,
             '-gpucurves', 1, '-save', save, '--tmp-dir', out/(label+'_tmp'), 2, 2], str(n).encode())
        got = save_rows(save)
        assert len(got) == 1 and got[0]['status'] == 'POINT' and got[0]['x'] == qx
        name = label+'_reuse1'; result = out/(name+'.jsonl'); debug = out/(name+'_debug.log')
        call(name, [s2, '--ini', ini, '--save', save, '--b2', 1000, '--d', 210, '--device', a.device,
             '--short-calibration', 0, '--tune-runtime', 0, '--factor-only', '--results', result,
             '--log', out/(name+'_engine.log'), '--debug-log-file', debug], extra={'NTT_SMALL_PRIME_REUSE': '1'})
        row = json.loads(result.read_text().splitlines()[-1]); text = debug.read_text(encoding='utf-8')
        assert row['curve'] == 'ed' and row['param'] is None and not row['bad_factors']
        assert all(1 < int(f) < n and n % int(f) == 0 for f in row['factors'])
        assert sorted(row['factors']) == ([str(prime)] if ell else [])
        if ell is None:
            expected = values(n, qx, a24, 1000)
            assert re.findall(r'descent_values:.*?hash=(\d+)', text)[-1] == expected
        line = re.findall(r'^small_prime_reuse: (.*)$', text, re.M)[-1]
        fields = dict(re.findall(r'(\w+)=(\S+)', line))
        assert fields['primes'] == '26' and fields['bad'] == '0' and fields['requested'] == '1'
        assert int(fields['reused']) == int(fields['checked']) == 23 and int(fields['fallback']) == 3
        report['checks'][-1].update(direct_prime_gcd={str(q): str(g) for q, g in oracle.items()},
            small_prime_proof=fields, stage2_point_source='gpu_save')
        persist()
        if ell is None:
            rejected = call('reject_disabled_production_cache', [s2, '--ini', ini, '--save', save,
                '--b2', 1000, '--d', 210, '--device', a.device, '--results', out/'rejected.jsonl'],
                extra={'NTT_SMALL_PRIME_REUSE': '0'}, good=False)
            assert 'requires NTT_SMALL_PRIME_REUSE=1' in rejected and not (out/'rejected.jsonl').exists()
    report['complete'] = True; persist(); print('All Edwards small-prime gates passed', flush=True)


if __name__ == '__main__': main()
