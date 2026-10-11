"""Continue native D searches cut short by the first tuned-carrier study budget.

Reuses valid summaries; never changes the source study/profile. Each incomplete
local window must finish before publishing the supplemented profile for holdout.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from bench_stage2_n_scaling import Monitor, read, sha, stamp, write


def windows(text):
    return [dict(re.findall(r'(\w+)=(\S+)', line)) for line in text.splitlines()
            if line.startswith('ecm_tune_window_done:')]


def finished(text, budget):
    w = windows(text)
    elapsed = re.findall(r'ecm_tune_complete: .*budget_elapsed=([\d.]+)', text)
    # A rejected warmup can close a direction without a formal observation.
    # Revisit an exhausted budget even if the local iterator reports complete.
    return len(w) == 2 and all(r['complete'] == '1' for r in w) and bool(elapsed) and float(elapsed[-1]) < budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--budget-seconds', type=int, default=600)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if min(a.budget_seconds, a.timeout) <= 0:
        p.error('budgets must be positive')
    study, out = a.study.resolve(), a.output.resolve()
    data = read(study/'measurements.json')
    if not data['complete'] or data['failures']:
        raise ValueError('wait for the original study to finish successfully')
    if sha(study/'measured.toml') != data['profile_sha256'] or sha(data['executable']) != data['executable_sha256']:
        raise ValueError('source profile/binary changed')
    if out.exists() and any(out.iterdir()):
        raise ValueError('use fresh output; never overwrite source evidence')
    out.mkdir(parents=True, exist_ok=True)
    mon = Monitor(data['device']['index'])
    if mon.identity != data['device']:
        mon.close()
        raise ValueError('GPU identity changed')
    profile = out/'stage2_ecm_tune.toml'
    shutil.copy2(study/'measured.toml', profile)
    shutil.copy2(__file__, out/'collector.py')
    cases = {c['id']: c for c in data['cases']}
    jobs = []
    for t in data['training']:
        log = study/'training'/(t['key']+'.log')
        if sha(log) != t['log_sha256']:
            raise ValueError('source tune log changed')
        if not finished(log.read_text(encoding='utf-8'), data['tune_budget_seconds']):
            jobs.append(t)
    report = dict(complete=False, created_utc=stamp(), source_manifest_sha256=sha(study/'measurements.json'),
                  source_profile_sha256=data['profile_sha256'], binary_sha256=data['executable_sha256'],
                  device=data['device'], collector_sha256=sha(__file__), jobs=len(jobs), receipts=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_') and k != 'CUDA_VISIBLE_DEVICES'}
    env.update(NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    try:
        for t in jobs:
            c = cases[t['case']]
            if sha(c['save']) != c['save_sha256']:
                raise ValueError('source save changed')
            for attempt in range(1, 4):
                prefix = out/f'{t["key"]}_attempt{attempt}'
                cmd = [data['executable'], '--ini', str(study/'bench.ini'), '--device', str(data['device']['index']),
                       '--arena-mb', str(data['memory']['arena_mb']), '--batch-mb', str(data['memory']['batch_mb']),
                       '--owner-budget-mb', str(data['memory']['fold_mb']), '--tune', 'ecm',
                       '--tune-file', str(profile), '--tune-save', c['save'],
                       '--tune-carrier-exponent', str(c['exponent']), '--tune-level', '10',
                       '--tune-b2', str(t['B2']), '--tune-repeats', '1',
                       '--tune-budget-seconds', str(a.budget_seconds*attempt)]
                write(prefix.with_suffix('.command.json'), cmd)
                before, start = sha(profile), time.perf_counter()
                print('COMPLETE WINDOW', t['key'], 'attempt', attempt, flush=True)
                with prefix.with_suffix('.log').open('wb') as f:
                    r = subprocess.run(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout)
                text = prefix.with_suffix('.log').read_text(encoding='utf-8')
                w = windows(text)
                report['receipts'].append(dict(case=c['id'], B2=t['B2'], attempt=attempt, command=cmd,
                    returncode=r.returncode, seconds=time.perf_counter()-start, before_sha256=before,
                    after_sha256=sha(profile), windows=w, log=str(prefix.with_suffix('.log')),
                    log_sha256=sha(prefix.with_suffix('.log'))))
                write(out/'report.json', report)
                if r.returncode or 'ecm_tune_complete:' not in text:
                    raise ValueError('native continuation failed')
                print(text.splitlines()[-1], flush=True)
                if finished(text, a.budget_seconds*attempt):
                    break
            else:
                raise ValueError('window still incomplete after three budgets')
        report.update(complete=True, completed_utc=stamp(), output_profile_sha256=sha(profile))
        write(out/'report.json', report)
        print('COMPLETE:', len(jobs), 'supplemented cells;', profile, flush=True)
    finally:
        mon.close()


if __name__ == '__main__':
    main()
