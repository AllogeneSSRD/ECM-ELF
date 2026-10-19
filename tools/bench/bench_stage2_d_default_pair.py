"""Interleave two full default Stage2 matrices through the existing collector.

Each arm remains a normal bench_stage2_version study with all raw calls retained.
The paired comparison supplements, never replaces, historical reference data.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import bench_stage2_version as collector

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(study, version, exe, reference, repeats):
    return SimpleNamespace(output=study, version=version, exe=exe, reference=reference, repeats=repeats,
                           allow_shared_device=False, resume=True, max_runs=1, timeout=3600)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='Complete historical three-repeat matrix')
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--candidate-version', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    old = collector.load(a.baseline/'plan.json'); out = a.output.resolve()
    if not old['complete'] or old['failures'] or old['repeats'] != 3 or len(old['cases']) != 13:
        raise ValueError('Need the complete historical 52-cell, n=3 reference')
    reference = Path(old['reference'])
    versions = dict(baseline=old['program_version'], candidate=a.candidate_version)
    exes = dict(baseline=Path(old['exe']), candidate=a.candidate.resolve())
    recorded_build = collector.load(Path(old['runs'][0]['prefix']).with_suffix('.jsonl'))['source-build']
    if recorded_build != sha(exes['baseline']):
        raise ValueError('Historical executable differs from its recorded source-build fingerprint')
    studies = {k:out/k for k in versions}
    params = {k:arguments(studies[k], versions[k], exes[k], reference, a.repeats) for k in versions}
    if not a.resume:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, out/'controller_snapshot.py')
        manifest = collector.load(a.candidate.parent/'build_manifest.json')
        if manifest['sha256'].lower() != sha(a.candidate):
            raise ValueError('Candidate manifest differs from executable')
        for name, digest in manifest['source_hashes'].items():
            src = ROOT/name
            if sha(src) != digest.lower():
                raise ValueError('Compiled candidate source changed: '+name)
            dst = out/'candidate_sources'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        shutil.copy2(a.candidate.parent/'build_manifest.json', out/'candidate_build_manifest.json')
        plans = {k:collector.prepare(params[k], studies[k]) for k in versions}
        # Use exactly the existing collector's condition order, reversing only
        # the version order at adjacent conditions and subsequent repetitions.
        conditions = []
        for r in old['prime95_rows']:
            if r['curve_type'] != 'Edwards' or not r['benchmark_eligible']:
                continue
            case = next(c for c in old['cases'] if c['N_hex'] == r['N_hex'] and c['sigma'] == int(r['s']) and c['b1'] == r['b1'])
            conditions.append((case['id'], r['b2']))
        schedule = []; index = 0
        for repeat in range(1, a.repeats+1):
            for case, bound in conditions:
                for mode in (('gpu','gpu-carrier') if repeat%2 else ('gpu-carrier','gpu')):
                    arms = ('baseline','candidate') if (index+repeat)%2 else ('candidate','baseline')
                    for arm in arms:
                        schedule.append(dict(arm=arm, id=f'{case}_b{bound}_{mode}_r{repeat}'))
                    index += 1
        report = dict(complete=False, started_utc=collector.stamp(), versions=versions, repeats=a.repeats,
            historical_study=str(a.baseline.resolve()), historical_plan_sha256=sha(a.baseline/'plan.json'),
            exe_sha256={k:sha(v) for k,v in exes.items()},
            order='Adjacent versions per exact cell; reverse versions across cells/repeats; no warmup',
            between_invocation_idle_seconds=.2, schedule=schedule, failures=[])
        collector.write(out/'pair.json', report)
    else:
        report = collector.load(out/'pair.json')
        if (out/'controller_snapshot.py').read_bytes() != Path(__file__).read_bytes():
            raise ValueError('Controller changed; retain the study and start a new one')
        if report['complete'] or report['failures'] or report['versions'] != versions or report['repeats'] != a.repeats:
            raise ValueError('Pair complete/failed or identity changed')
    try:
        for step in report['schedule']:
            arm = step['arm']; current = collector.load(studies[arm]/'plan.json')
            if any(r['id'] == step['id'] for r in current['runs']):
                continue
            count = len(current['runs'])
            print('PAIR', count+1, '/', 52*a.repeats, arm, step['id'], flush=True)
            collector.run(params[arm])
            latest = collector.load(studies[arm]/'plan.json')
            if len(latest['runs']) != count+1 or latest['runs'][-1]['id'] != step['id']:
                raise ValueError('Collector condition order differs from the declared pair schedule')
            time.sleep(report['between_invocation_idle_seconds'])
        # max_runs=1 returns before setting the completed bit at the final call.
        for arm in versions:
            if not collector.load(studies[arm]/'plan.json')['complete']:
                collector.run(params[arm])
        for arm in versions:
            plan = collector.load(studies[arm]/'plan.json')
            if not plan['complete'] or len(plan['runs']) != 52*a.repeats:
                raise ValueError('Incomplete paired arm')
            if sha(plan['exe']) != report['exe_sha256'][arm]:
                raise ValueError('Frozen executable changed')
        report.update(complete=True, finished_utc=collector.stamp());collector.write(out/'pair.json', report)
        print('Complete:', 104*a.repeats, 'retained invocations', flush=True)
    except Exception as e:
        report['failures'].append(dict(utc=collector.stamp(), error=str(e)));collector.write(out/'pair.json', report);raise


if __name__ == '__main__':
    main()
