"""Verify measured larger-D or Mersenne-carrier selection with a complete curve."""
import argparse
import hashlib
import json
from pathlib import Path
import re
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
    p.add_argument('--expect-carrier', type=int,
                   help='Verify a paired arithmetic profile and require this winning exponent (>0).')
    p.add_argument('--target-bits', type=int, help='Select one width from a merged profile.')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [a.exe.resolve(),a.save.resolve(),a.profile.resolve()]
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path):sha(path) for path in paths}
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8'))
    samples = list(profile['ecm'].values())
    if a.target_bits is not None:
        samples = [sample for sample in samples if sample['target_bits'] == a.target_bits]
    assert len(samples) >= 2
    assert len({(s['target_bits'],s['b1'],s['b2']) for s in samples}) == 1
    best = min(samples,key=lambda s:s['median_seconds']+2*s['mad_seconds'])
    if a.expect_carrier is None:
        assert all(s['carrier_exponent'] == 0 for s in samples)
        baseline = min(samples,key=lambda s:s['d'])
        assert best['d'] > baseline['d']
    else:
        assert a.expect_carrier > 0
        assert best['carrier_exponent'] == a.expect_carrier
        ordinary = [s for s in samples if s['carrier_exponent'] == 0 and s['d'] == best['d']]
        assert len(ordinary) == 1, 'Carrier comparison requires an ordinary sample at the same D.'
        baseline = ordinary[0]
        assert best['median_seconds'] < baseline['median_seconds']
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
    assert selection['carrier_exponent'] == plan['carrier_exponent'] == best['carrier_exponent']
    assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
    if a.expect_carrier is not None:
        def plan_case(name, args, success=True):
            proc = subprocess.run(args+['--plan-only'],cwd=ROOT,capture_output=True,
                                  text=True,errors='replace',timeout=120)
            (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            assert (proc.returncode == 0) == success, name
            return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]

        fixed = plan_case('fixed_ordinary',cmd+['--carrier-exponent','0'])
        fixed_choice = next(x for x in fixed if x.get('type') == 'tune_selection')
        assert fixed_choice['selected'] and fixed_choice['carrier_exponent'] == 0
        # Same performance scope does not prove divisibility. This point is only
        # used for plan-only validation and is never executed as a Stage1 result.
        unrelated = (1 << (best['target_bits']-1))+3
        assert unrelated.bit_length() == best['target_bits']
        assert ((1 << a.expect_carrier)-1) % unrelated != 0
        unrelated_save = out/'unrelated_plan_only.save'
        unrelated_save.write_text('METHOD=ECM; PARAM=0; SIGMA=26; B1='+str(best['b1'])+
                                  '; N=0x'+format(unrelated,'x')+'; X=1; Z=1;\n',encoding='utf-8')
        unrelated_cmd = cmd.copy()
        unrelated_cmd[unrelated_cmd.index('--save')+1] = str(unrelated_save)
        unrelated_rows = plan_case('unrelated_target',unrelated_cmd)
        unrelated_choice = next(x for x in unrelated_rows if x.get('type') == 'tune_selection')
        assert unrelated_choice['selected'] and unrelated_choice['carrier_exponent'] == 0
        plan_case('invalid_explicit_carrier',unrelated_cmd+['--carrier-exponent',str(a.expect_carrier)],False)
    result_file,log = out/'curve.jsonl',out/'curve.log'
    proc = subprocess.run(cmd+['--curves','1','--results',str(result_file),'--log',str(log)],
                          cwd=ROOT,capture_output=True,text=True,errors='replace',timeout=900)
    (out/'driver.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode == 0
    result = json.loads(result_file.read_text(encoding='utf-8'))
    assert result['status'] == 'stage2_completed' and result['requested_D'] == 0
    assert result['tune_plan']['selected'] and result['tune_plan']['D'] == best['d']
    assert result['requested_carrier_exponent'] == 0
    assert result['carrier_exponent'] == result['tune_plan']['carrier_exponent'] == best['carrier_exponent']
    assert result['hits'] == result['bad_factors'] == 0
    text = log.read_text(encoding='utf-8')
    assert re.search(r'real_shape: D='+str(best['d'])+r'\b',text)
    assert re.search(r'stage2_modulus: target_bits='+str(best['target_bits'])+
                     r' carrier_bits='+str(best['arithmetic_bits'])+
                     r' carrier_exponent='+str(best['carrier_exponent'])+r'\b',text)
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
                  baseline_carrier=baseline['carrier_exponent'], selected_carrier=best['carrier_exponent'],
                  carrier_divisibility_gate_verified=a.expect_carrier is not None,
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
