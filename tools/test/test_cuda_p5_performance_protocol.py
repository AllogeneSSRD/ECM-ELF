"""CPU negative gates for paired CUDA qualification and premature GPU sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from qualify_cuda_refactor import passes_metrics, ratio_interval, validate_edwards_geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    checks = []
    for model, window, doubling in [('ed', 6, 4), ('ed-5', 6, 5), ('ed-binary', 0, 0)]:
        validate_edwards_geometry(model, window, doubling)
        checks.append(model+' exact geometry accepted')
    for model, window, doubling in [('ed-binary', 0, 4), ('ed-binary', 6, 0),
                                    ('ed', 6, 0), ('ed-5', 6, 4)]:
        try: validate_edwards_geometry(model, window, doubling)
        except AssertionError: checks.append(model+' wrong geometry rejected')
        else: raise AssertionError('incorrect Edwards mode accepted')
    fast = ratio_interval([.99]*21)
    slow = ratio_interval([1.06]*21)
    uncertain = ratio_interval([.99]*12+[1.08]*9)
    assert fast['pairs'] == 21 and fast['bootstrap_resamples'] == 10000
    assert passes_metrics(dict(wall_seconds=fast, internal_seconds=fast))
    checks.append('stable paired pass')
    for name, wall, internal in [('wall regression', slow, fast),
                                 ('internal regression', fast, slow),
                                 ('uncertainty', fast, uncertain)]:
        assert not passes_metrics(dict(wall_seconds=wall, internal_seconds=internal))
        checks.append(name)
    assert uncertain['median'] < 1.05 < uncertain['high95']
    assert ratio_interval([.99]*12+[1.08]*9) == uncertain
    checks.append('deterministic confidence interval')
    outliers = ratio_interval([1.0]*10+[1.2]*11)
    assert outliers['median'] == 1.2 and outliers['pairs'] == 21
    checks.append('slow samples retained')
    boundary = ratio_interval([1.05]*21)
    assert passes_metrics(dict(wall_seconds=boundary, internal_seconds=boundary))
    checks.append('fixed inclusive five percent boundary')
    try: passes_metrics(dict(wall_seconds=fast))
    except AssertionError: checks.append('missing internal boundary rejected')
    else: raise AssertionError('incomplete timing metrics accepted')
    gate = out/'incomplete_gate.json'; gate.write_text('{"complete":false}\n')
    base = dict(acceptance_ready=True, threshold=1.05, device=1, prerequisites=[str(gate)])
    for name, patch, pairs, message in [
        ('not_ready', dict(acceptance_ready=False), 21, 'not bound to the final acceptance gates'),
        ('incomplete_gate', {}, 21, 'correctness/build gate incomplete'),
        ('too_few_pairs', {}, 20, 'at least 21 pairs'),
        ('empty_gates', dict(prerequisites=[]), 21, 'correctness/build gate incomplete'),
        ('changed_threshold', dict(threshold=1.10), 21, 'AssertionError')]:
        plan = out/(name+'.json'); plan.write_text(json.dumps(base | patch))
        target = out/(name+'_study')
        proc = subprocess.run([sys.executable, str(ROOT/'tools/bench/qualify_cuda_refactor.py'),
            '--plan', str(plan), '--output', str(target), '--pairs', str(pairs)], capture_output=True)
        text = (proc.stdout+proc.stderr).decode('utf-8', 'replace')
        (out/(name+'.log')).write_text(text)
        assert proc.returncode and message in text and not target.exists(), (name, text)
        checks.append(name+' rejected before GPU initialization')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(complete=True, checks=checks, gpu_runs=0,
        inputs={str(path):sha(path) for path in [Path(__file__), ROOT/'tools/bench/qualify_cuda_refactor.py']})
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS performance protocol:', len(checks), 'CPU gates')


if __name__ == '__main__': main()
