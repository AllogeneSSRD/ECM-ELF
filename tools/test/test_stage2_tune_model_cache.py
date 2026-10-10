"""Compare every Auto B2 candidate against a frozen pre-cache native fixture.

Both fixtures use the same caller code and compiler; the baseline headers come
from the recorded old source snapshot. This is CPU equivalence/performance,
not CUDA curve performance or memory admission qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tomllib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--profile', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.iterations <= 10000 or not 3 <= a.repeats <= 100:
        p.error('invalid benchmark repeats')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [a.fixture.resolve(), a.baseline.resolve(), Path(__file__).resolve()]+[x.resolve() for x in a.profile]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(path): digest(path) for path in paths}
    report = dict(complete=False, identities=hashes, comparison='all candidates and ordering; exact numeric equality',
        scope='CPU candidate construction; excludes file loading, device query and joint memory planner',
        cases=[], benchmark=[], iterations=a.iterations, repeats=a.repeats)

    def publish():
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    def call(exe, mode, profile, t1, lo=0, hi=0, ratio=1, iterations=None):
        command = [str(exe.resolve()), mode, str(profile.resolve()), str(t1), str(lo), str(hi), str(ratio)]
        if iterations is not None:
            command.append(str(iterations))
        proc = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=120)
        return proc.returncode, proc.stdout, proc.stderr

    publish()
    try:
        for index, profile in enumerate(a.profile):
            data = tomllib.loads(profile.read_text(encoding='utf-8-sig'))
            bounds = [row['b2'] for row in data['ecm'].values()]
            low, high = min(bounds), max(bounds)
            # Anchors, predictions, bounded grid, exact-only and no-fit profiles
            # are supplied by the caller. T1/ratio test interior and both ends.
            cases = [(t1, 0, 0, ratio) for t1 in (.001, 3, 30, 210) for ratio in (1, 2)]
            cases += [(3, low, low, 1), (3, high, high, 1),
                      (3, low+(high-low)//3, low+2*(high-low)//3, 1),
                      (3, low-1, 0, 1), (3, 0, high+1, 1), (0, 0, 0, 1)]
            for case, (t1, lo, hi, ratio) in enumerate(cases):
                left = call(a.baseline, '--auto-grid', profile, t1, lo, hi, ratio)
                right = call(a.fixture, '--auto-grid', profile, t1, lo, hi, ratio)
                stem = f'profile_{index}_case_{case}'
                (out/(stem+'_baseline.log')).write_text(left[1]+left[2], encoding='utf-8')
                (out/(stem+'_current.log')).write_text(right[1]+right[2], encoding='utf-8')
                assert left[0] == right[0], (stem, left, right)
                count = 0
                if not left[0]:
                    old, new = json.loads(left[1]), json.loads(right[1])
                    assert old == new, 'candidate cost/order changed: '+stem
                    count = len(new)
                else:
                    assert left[2] == right[2], 'rejection reason changed: '+stem
                report['cases'].append(dict(profile=index, case=case, t1=t1, lo=lo, hi=hi,
                    ratio=ratio, accepted=left[0] == 0, candidates=count))
            timing = dict(profile=index, baseline_seconds=[], current_seconds=[], candidates=None)
            # One CPU warmup then interleaved formal runs, with reversed order.
            for repeat in range(a.repeats+1):
                order = [('baseline', a.baseline), ('current', a.fixture)]
                if repeat % 2:
                    order.reverse()
                checksums = []
                for name, exe in order:
                    status, stdout, stderr = call(exe, '--auto-time', profile, 30, iterations=a.iterations)
                    (out/f'profile_{index}_{name}_repeat{repeat}.log').write_text(stdout+stderr, encoding='utf-8')
                    assert not status, stderr
                    row = json.loads(stdout)
                    assert row['iterations'] == a.iterations and row['seconds'] > 0
                    checksums.append(row['checksum'])
                    if timing['candidates'] is None:
                        timing['candidates'] = row['candidates']
                    assert row['candidates'] == timing['candidates']
                    if repeat:
                        timing[name+'_seconds'].append(row['seconds'])
                assert checksums[0] == checksums[1]
            for name in ('baseline', 'current'):
                timing[name+'_median_seconds'] = statistics.median(timing[name+'_seconds'])
            timing['current_over_baseline'] = timing['current_median_seconds']/timing['baseline_median_seconds']
            report['benchmark'].append(timing)
            publish()
        assert all(digest(Path(path)) == value for path, value in hashes.items()), 'source input changed'
        report['complete'] = True
        publish()
    except Exception as exc:
        report['error'] = str(exc)
        publish()
        raise
    print(json.dumps(dict(passed=True, cases=len(report['cases']),
        compared_candidates=sum(c['candidates'] for c in report['cases']),
        CPU_ratios=[x['current_over_baseline'] for x in report['benchmark']])))


if __name__ == '__main__':
    main()
