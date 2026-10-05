"""Serial full Stage2 ABBA+BAAB against a frozen production executable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--b2', type=int, default=2011326186870)
    p.add_argument('--d', type=int, default=1381380)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert not any(a.output.iterdir()), 'Use a fresh output directory'
    exes = {'old': a.baseline.resolve(), 'new': a.candidate.resolve()}
    manifests = {}
    for key, exe in exes.items():
        name = 'frozen_manifest.json' if key == 'old' else 'build_manifest.json'
        m = json.loads((exe.parent/name).read_text(encoding='utf-8-sig'))
        if isinstance(m['sources'], list):
            # Build signatures contain toolkit/options in addition to file hashes.
            m['sources'] = dict(s.split('=', 1) for s in m['sources'] if re.match(r'^(src|tools)/.*=[0-9A-Fa-f]{64}$', s))
        assert len(m['sources']) >= 17
        manifests[key] = m
    assert manifests['new']['gl_fixed_mode'] == 3
    save = a.save.resolve()
    save_sha = digest(save)
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_ARENA_CAP_KB='6451200', NTT_D_MODEL='0',
               NTT_GL_SHORT_REDUCE='1', NTT_GL_SHIFT_SCALE='0')
    data = dict(manifests=manifests, save=str(save), save_sha256=save_sha,
                device=a.device, b2=a.b2, d=a.d, runs=[],
                scope='serial old/new/new/old/new/old/old/new; same save/D/default checks; fixed PTX3 candidate',
                script_sha256=digest(Path(__file__)))

    def verify():
        assert digest(save) == save_sha
        for key, exe in exes.items():
            m = manifests[key]
            assert digest(exe) == m['sha256'].lower()
            for name, want in m['sources'].items():
                assert digest(exe.parent/'sources'/name) == want.lower(), (key, name)
                if key == 'new':
                    assert digest(Path(__file__).resolve().parents[2]/name) == want.lower(), name

    signature = None
    for index, key in enumerate(('old', 'new', 'new', 'old', 'new', 'old', 'old', 'new'), 1):
        verify()
        name = f'{index}_{key}'
        log = a.output/(name+'_engine.log')
        results = a.output/(name+'.jsonl')
        cmd = [str(exes[key]), '--save', str(save), '--b2', str(a.b2), '--d', str(a.d),
               '--device', str(a.device), '--log', str(log), '--results', str(results)]
        with (a.output/(name+'_driver.log')).open('wb') as output:
            r = subprocess.run(cmd, env=env | {'NTT_GL_PTX_REDUCE': '1' if key == 'new' else '0'},
                               stdout=output, stderr=subprocess.STDOUT, timeout=300)
        verify()
        assert r.returncode == 0, name
        text = log.read_text(encoding='utf-8', errors='replace')
        for token in ('stage1_skipped=1', 'baby_device: requested=1 enabled=1', 'gmp_check_bad=0',
                      'gmp_selftest_bad=0', 'clean=1', 'pending=0', 'point_arithmetic: xadd6=1'):
            assert token in text, (key, token)
        if key == 'new':
            assert f'ntt_gl_reduce_mode: device={a.device} short=1 ptx=1 fixed=3' in text
        row = json.loads(results.read_text(encoding='utf-8').splitlines()[-1])
        assert row['bad_factors'] == 0
        leaf = re.search(r'descent_values: (.*)', text)[1]
        oracle = re.search(r's4_oracle_stats:.*selected=(\d+) queued=(\d+) compared=(\d+) samples=(\d+) pending=(\d+).*signature=(\S+)', text)
        assert oracle and oracle[1] == oracle[2] == oracle[3] and oracle[5] == '0'
        mul = re.search(r's4_multiply_stats: (.*)', text)[1]
        counts = dict(re.findall(r'(\w+)=(\S+)', mul))
        coverage = {k: counts[k] for k in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases',
                                          'gmp_selftest_bad', 'gmp_checked', 'gmp_check_bad', 'full_checks')}
        current = (leaf, oracle[1], oracle[4], oracle[6], row['factors'], coverage)
        if signature is None:
            signature = current
        else:
            assert current == signature, name
        wall = re.search(r'stage2_full_wall:.*?init=([\d.]+) main=([\d.]+) total=([\d.]+)', text)
        data['runs'].append(dict(name=name, backend=key, init=float(wall[1]), main=float(wall[2]),
                                 full=float(wall[3]), leaf=leaf, oracle_signature=oracle[6],
                                 factors=row['factors'], coverage=coverage))
        (a.output/'measurements.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(name, 'full', wall[3], flush=True)
    data['means'] = {key: statistics.mean(r['full'] for r in data['runs'] if r['backend'] == key)
                     for key in exes}
    data['gain_percent'] = 100*(1-data['means']['new']/data['means']['old'])
    data['order_groups'] = []
    for rows in (data['runs'][:4], data['runs'][4:]):
        means = {key: statistics.mean(r['full'] for r in rows if r['backend'] == key) for key in exes}
        data['order_groups'].append(dict(means=means, gain_percent=100*(1-means['new']/means['old'])))
    data.update(passed=8, failed=0)
    verify()
    (a.output/'measurements.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(dict(means=data['means'], gain_percent=data['gain_percent'])), flush=True)


if __name__ == '__main__':
    main()
