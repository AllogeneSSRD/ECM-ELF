"""Serial Goldilocks inverse-scale GMP gates and same-binary convolution A/B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--sizes', type=int, nargs='*', default=[16, 24, 25, 26, 27])
    p.add_argument('--gate-only', action='store_true')
    a = p.parse_args()
    if any(k < 16 or k > 27 for k in a.sizes):
        raise ValueError('k must be 16..27')
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        raise ValueError('Use a fresh output directory')
    exe = a.exe.resolve()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))

    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest() == manifest['sha256'].lower()
        for name, want in manifest['sources'].items():
            assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == want.lower(), name

    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env['NTT_GL_SHORT_REDUCE'] = '1'
    data = dict(manifest=manifest, device=a.device, checks={}, runs=[],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                timing='8 ABBA+BAAB runs, warm+3 event samples/run, full convolution; all N outputs checked outside events')

    def save():
        (a.output/'measurements.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def run(name, args, extra=None, code=0):
        verify()
        r = subprocess.run([str(exe), str(a.device), *args], env=env | (extra or {}),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        text = r.stdout.decode('utf-8', errors='replace')
        (a.output/(name+'.log')).write_text(text, encoding='utf-8')
        verify()
        assert r.returncode == code, (name, r.returncode, text[-1500:])
        return text

    for short in (0, 1):
        text = run(f'primitive_{short}', ['--scale-check'], {'NTT_GL_SHORT_REDUCE': str(short)})
        m = re.search(r'ntt_scale_check: words=(\d+) bad=0', text)
        assert m and int(m[1]) > 262144
        data['checks'][f'primitive_short{short}'] = int(m[1])
        save()
    for shift in (0, 1):
        extra = {'NTT_GL_SHIFT_SCALE': str(shift)}
        text = run(f'coop_{shift}', [], extra)
        assert 'ntt_fuse_coop_check: cases=96 words=27131904 bad=0' in text
        assert 'ntt_fuse_coop_switch_check: calls=4 words=3145728 bad=0' in text
        assert f'ntt_coop_probe: device={a.device} bad=0' in text
        data['checks'][f'coop_shift{shift}'] = True
        text = run(f'fault_{shift}', [], extra | {'NTT_FUSE_COOP_BAD': '1'}, 3)
        assert re.search(r'ntt_fuse_coop_check: .*bad=[1-9]', text)
        data['checks'][f'fault_shift{shift}'] = True
        text = run(f'legacy_{shift}', ['--legacy'], extra)
        assert not re.search(r'bad=[1-9]', text)
        assert 'ntt_fuse_warp_check:' in text and 'ntt_fuse_capacity_check:' in text
        data['checks'][f'legacy_shift{shift}'] = True
        save()
    if not a.gate_only:
        for k in a.sizes:
            text = run(f'bench_k{k}', ['--bench-scale', str(k)])
            rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                    for line in re.findall(r'ntt_scale_bench: (.*)', text)]
            assert [int(r['shift']) for r in rows] == [0, 1, 1, 0, 1, 0, 0, 1]
            assert [int(r['run']) for r in rows] == list(range(1, 9))
            assert all(r['bad'] == '0' and int(r['k']) == k and int(r['N']) == 1 << k for r in rows)
            means = {str(mode): statistics.mean(float(r['seconds']) for r in rows if int(r['shift']) == mode)
                     for mode in (0, 1)}
            item = dict(k=k, means=means, gain_percent=100*(means['0']-means['1'])/means['0'], raw=rows)
            data['runs'].append(item)
            save()
            print(json.dumps({name: value for name, value in item.items() if name != 'raw'}), flush=True)
    verify()
    data['passed'] = len(data['checks'])
    data['failed'] = 0
    save()
    print(json.dumps(dict(passed=data['passed'], failed=0)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
