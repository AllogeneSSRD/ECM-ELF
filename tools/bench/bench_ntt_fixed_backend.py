"""Serial immutable Goldilocks backend gates and cross-binary NTT controls."""
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
    p.add_argument('--short', type=Path, required=True)
    p.add_argument('--ptx', type=Path, required=True)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--reference-short', type=Path,
                   help='Frozen pre-PTX runtime short probe with --bench-scale and raw sources/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--sizes', type=int, nargs='+', default=[16, 24, 25, 26, 27])
    a = p.parse_args()
    assert all(16 <= k <= 27 for k in a.sizes)
    a.output.mkdir(parents=True, exist_ok=True)
    assert not any(a.output.iterdir()), 'Use a fresh output directory'
    repo = Path(__file__).resolve().parents[2]
    exes = {'short': a.short.resolve(), 'ptx': a.ptx.resolve()}
    if a.runtime:
        exes['runtime'] = a.runtime.resolve()
    if a.reference_short:
        exes['reference_short'] = a.reference_short.resolve()
    manifests = {key: json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
                 for key, exe in exes.items()}
    assert manifests['short']['gl_fixed_mode'] == 1
    assert manifests['ptx']['gl_fixed_mode'] == 3
    assert manifests['short']['sources'] == manifests['ptx']['sources']
    if a.runtime:
        assert manifests['runtime'].get('gl_fixed_mode', -1) == -1
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env['NTT_GL_SHIFT_SCALE'] = '0'
    data = dict(manifests=manifests, device=a.device, checks={}, comparisons=[],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                timing='cross-binary ABBA processes, 8 runs/process, warm+3 events/run; all N outputs checked outside events')

    def verify(key):
        exe = exes[key]
        m = manifests[key]
        assert hashlib.sha256(exe.read_bytes()).hexdigest() == m['sha256'].lower()
        for name, want in m['sources'].items():
            base = exe.parent/'sources' if key in ('runtime','reference_short') else repo
            assert hashlib.sha256((base/name).read_bytes()).hexdigest() == want.lower(), (key, name)

    def save():
        (a.output/'measurements.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def run(key, name, args, extra=None, code=0):
        verify(key)
        control = {'NTT_GL_SHORT_REDUCE': '1', 'NTT_GL_PTX_REDUCE': '1'} if key == 'runtime' else {}
        if key == 'reference_short':
            control = {'NTT_GL_SHORT_REDUCE': '1', 'NTT_GL_PTX_REDUCE': '0'}
        r = subprocess.run([str(exes[key]), str(a.device), *args], env=env | control | (extra or {}),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        text = r.stdout.decode('utf-8', errors='replace')
        (a.output/(name+'.log')).write_text(text, encoding='utf-8')
        verify(key)
        assert r.returncode == code, (name, r.returncode, text[-2000:])
        return text

    for key, mode in [('short', 1), ('ptx', 3)]:
        ack = f'ntt_gl_reduce_mode: device={a.device} short=1 ptx={mode >> 1} fixed={mode}'
        text = run(key, key+'_primitive', ['--gl-selftest'])
        assert ack in text and '200000 device cases' in text
        data['checks'][key+'_primitive'] = True
        text = run(key, key+'_scale', ['--scale-check'])
        assert ack in text and re.search(r'ntt_scale_check: words=263357 bad=0', text)
        data['checks'][key+'_scale'] = True
        text = run(key, key+'_coop', [])
        assert ack in text
        assert 'ntt_fuse_coop_check: cases=96 words=27131904 bad=0' in text
        assert 'ntt_fuse_coop_switch_check: calls=4 words=3145728 bad=0' in text
        assert not re.search(r'local=[1-9]', text)
        data['checks'][key+'_coop'] = True
        text = run(key, key+'_fault', [], {'NTT_FUSE_COOP_BAD': '1'}, 3)
        assert re.search(r'ntt_fuse_coop_check: .*bad=[1-9]', text)
        data['checks'][key+'_fault'] = True
        text = run(key, key+'_legacy', ['--legacy'])
        assert ack in text and not re.search(r'bad=[1-9]', text)
        assert 'ntt_fuse_warp_check:' in text and 'ntt_fuse_capacity_check:' in text
        assert not re.search(r'(?:fwd|inv)_local=[1-9]', text)
        data['checks'][key+'_legacy'] = True
        # Reject contradictory controls before any arithmetic selftest kernel.
        for label, control in [('short', {'NTT_GL_SHORT_REDUCE': '0'}),
                               ('ptx', {'NTT_GL_PTX_REDUCE': str(1-(mode >> 1))})]:
            text = run(key, key+'_conflict_'+label, ['--gl-selftest'], control, 2)
            assert f'ntt_gl_backend_conflict: compiled={mode}' in text
            assert '200000 device cases' not in text
            data['checks'][key+'_conflict_'+label] = True
        # Explicit matching controls also acknowledge the actual fixed backend.
        text = run(key, key+'_explicit', ['--gl-selftest'],
                   {'NTT_GL_SHORT_REDUCE': '1', 'NTT_GL_PTX_REDUCE': str(mode >> 1)})
        assert ack in text and '200000 device cases' in text
        data['checks'][key+'_explicit'] = True
        save()

    pairs = [('short', 'ptx')]
    if a.runtime:
        pairs.append(('runtime', 'ptx'))
    if a.reference_short:
        pairs.append(('reference_short', 'ptx'))
    for base, candidate in pairs:
        for k in a.sizes:
            rows = []
            for seq, key in enumerate([base, candidate, candidate, base], 1):
                text = run(key, f'{base}_vs_{candidate}_k{k}_{seq}_{key}',
                           ['--bench-scale' if key == 'reference_short' else '--bench-ptx' if key == 'runtime' else '--bench-fixed', str(k)])
                prefix = 'ntt_scale_bench' if key == 'reference_short' else 'ntt_ptx_bench' if key == 'runtime' else 'ntt_fixed_bench'
                parsed = [dict(re.findall(r'(\w+)=(\S+)', line))
                          for line in re.findall(prefix+r': (.*)', text)]
                assert [int(r['run']) for r in parsed] == list(range(1, 9))
                assert all(r['bad'] == '0' and int(r['k']) == k and int(r['N']) == 1 << k for r in parsed)
                selected = [r for r in parsed if (key != 'runtime' or r['ptx'] == '1') and
                            (key != 'reference_short' or r['shift'] == '0')]
                assert all(int(r['backend']) == (1 if key == 'short' else 3) for r in selected if key in ('short','ptx'))
                rows.extend(dict(r, backend_name=key, process=seq) for r in selected)
            assert len({(r['passes_fwd'], r['selected_M'], r['selected_coop']) for r in rows}) == 1
            means = {key: statistics.mean(float(r['seconds']) for r in rows if r['backend_name'] == key)
                     for key in (base, candidate)}
            result = dict(base=base, candidate=candidate, k=k, means=means,
                          gain_percent=100*(1-means[candidate]/means[base]), raw=rows)
            data['comparisons'].append(result)
            save()
            print(json.dumps({key: value for key, value in result.items() if key != 'raw'}), flush=True)
    for key in exes:
        verify(key)
    data.update(passed=len(data['checks']), failed=0)
    save()
    print(json.dumps(dict(passed=data['passed'], failed=0)), flush=True)


if __name__ == '__main__':
    main()
