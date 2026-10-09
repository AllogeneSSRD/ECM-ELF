"""Verify measured larger-D selection with a complete production curve."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [a.exe.resolve(),a.save.resolve(),a.profile.resolve()]
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path):sha(path) for path in paths}
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8'))
    samples = list(profile['ecm'].values())
    assert len(samples) >= 2 and all(s['carrier_exponent'] == 0 for s in samples)
    assert len({(s['target_bits'],s['b1'],s['b2']) for s in samples}) == 1
    best = min(samples,key=lambda s:s['median_seconds']+2*s['mad_seconds'])
    baseline = min(samples,key=lambda s:s['d'])
    assert best['d'] > baseline['d']
    ini = out/'bench.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    cmd = [str(a.exe.resolve()),'--ini',str(ini),'--save',str(a.save.resolve()),
           '--device',str(a.device),'--batch-mb',str(profile['policy']['batch_mb']),
           '--arena-mb',str(profile['policy']['arena_mb']),'--owner-budget-mb',str(profile['policy']['fold_mb']),
           '--b2',str(best['b2']),'--tune-profile',str(a.profile.resolve()),'--log-level','quiet']
    proc = subprocess.run(cmd+['--plan-only'],cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=120)
    (out/'plan.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode == 0
    records = [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]
    selection = next(x for x in records if x.get('type') == 'tune_selection')
    plan = next(x for x in records if x.get('type') == 'stage2_plan')
    assert selection['selected'] and selection['D'] == plan['D'] == best['d']
    assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
    result_file,log = out/'curve.jsonl',out/'curve.log'
    proc = subprocess.run(cmd+['--curves','1','--results',str(result_file),'--log',str(log)],
                          cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=900)
    (out/'driver.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode == 0
    result = json.loads(result_file.read_text(encoding='utf-8'))
    assert result['status'] == 'stage2_completed' and result['requested_D'] == 0
    assert result['tune_plan']['selected'] and result['tune_plan']['D'] == best['d']
    assert result['hits'] == result['bad_factors'] == 0
    text = log.read_text(encoding='utf-8')
    assert re.search(r'real_shape: D='+str(best['d'])+r'\b',text)
    assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
    assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b',text)
    assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b',text)
    wall = next(x for x in text.splitlines() if x.startswith('stage2_full_wall:'))
    total = float(re.search(r'\btotal=([0-9.]+)',wall).group(1))
    assert 'clean=1' in wall
    assert identities == {str(path):sha(path) for path in paths}
    report = dict(binary_sha256=sha(a.exe.resolve()), profile_sha256=sha(a.profile.resolve()), save_sha256=sha(a.save.resolve()),
                  target_bits=best['target_bits'], B1=best['b1'], B2=best['b2'], device=a.device,
                  baseline_d=baseline['d'], selected_d=best['d'], baseline_samples=baseline['seconds'],
                  selected_samples=best['seconds'], baseline_median=baseline['median_seconds'],
                  selected_median=best['median_seconds'], measured_reduction_fraction=1-best['median_seconds']/baseline['median_seconds'],
                  production_full_wall=total, production_worker_wall=result['seconds'], selection=selection,
                  joint_peak_bytes=plan['curve_workspace_memory']['peak_bytes'],
                  required_free_bytes=plan['curve_workspace_memory']['required_free_bytes'], arithmetic_bad=0,
                  total_scope='stage2_full_wall.total; Stage1/process/planning/publication excluded')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
