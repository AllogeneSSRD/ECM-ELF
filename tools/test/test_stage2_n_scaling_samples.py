"""Different-n comparisons retain frozen-input checks and explicitly record sample counts."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from analyze_stage2_n_scaling import compare_analyses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(a.baseline_analysis.read_text(encoding='utf-8'))
    current = copy.deepcopy(previous)
    for row in current['summary']:
        row['samples'] = 1
        row['std_seconds'] = None
    compare_analyses(current, previous, a.baseline_analysis, a.output / 'same_times_fixture')
    report = json.loads((a.output / 'same_times_fixture_comparison.json').read_text())
    assert all(row['current_samples'] == 1 and row['previous_samples'] in (1,3) for row in report['rows'])
    assert all(row['speedup'] == 1 and row['current_std_seconds'] is None for row in report['rows'])
    overlap = copy.deepcopy(current)
    grouped = next(r for r in overlap['summary'] if len(list(filter(None, r['factors'].split(';')))) >= 2)
    grouped['factors'] += ';' + str(math.lcm(*[int(v) for v in grouped['factors'].split(';')[:2]]))
    compare_analyses(overlap, previous, a.baseline_analysis, a.output / 'overlap_fixture')
    coverage = json.loads((a.output / 'overlap_fixture_comparison.json').read_text())
    assert all(r['factor_coverage_equal'] for r in coverage['rows'])
    assert sum(not r['factors_equal'] for r in coverage['rows']) == 1
    for field in ('N_hex','B1','sigma','save_sha256','Q_sha256'):
        altered = copy.deepcopy(current)
        altered['inputs'][0][field] = 'broken identity'
        try:
            compare_analyses(altered, previous, a.baseline_analysis, a.output / 'must_not_publish')
        except ValueError:
            pass
        else:
            raise AssertionError('identity mismatch accepted: ' + field)
    assert not (a.output / 'must_not_publish_comparison.json').exists()
    print('PASS: n=1 vs n=3/1 explicitly labeled, stdev absent, five frozen-input mismatch gates retained')


if __name__ == '__main__':
    main()
