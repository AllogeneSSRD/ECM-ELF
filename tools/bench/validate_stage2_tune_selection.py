"""Independently validate tune costs and D/carrier rankings on unseen B2 inputs.

Each eligible candidate receives one warmup and repeated full production curves.
The unforced driver also executes a curve at every holdout. Raw evidence stays
in a fresh output directory; no tune samples or qualification limits are edited.
This harness accepts literal integer N in a normalized PARAM0 save.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tomllib
import shutil
from stage2_tune_route_cost import predict_route
import analyze_stage2_tune_components as components
from analyze_stage2_tune_workload import embedded_ntt_samples, one_json, workload

ROOT = Path(__file__).resolve().parents[2]
TIME_ERROR_LIMIT = .08
RANK_LOSS_LIMIT = .05
SCOPE = ('target_bits', 'arithmetic_bits', 'carrier_exponent', 'modulus_kind', 'b1', 'd')


def predict(samples, b2, opted):
    """Exact measurements or independent NumPy route regression."""
    exact = next((s for s in samples if s['b2'] == b2), None)
    if exact:
        if not exact['fold_resident'] or not exact['frontier_resident']:
            return None
        return dict(model='measured_exact_scope_v1', seconds=exact['median_seconds'],
                    rank=exact['median_seconds']+2*exact['mad_seconds'])
    return predict_route(samples, b2, opted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--save', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--holdout-b2', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--min-candidates', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--training-plans', type=Path, help='Frozen case_*.plan.jsonl for format-4 component reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.min_candidates < 2 or args.timeout <= 0:
        parser.error('require repeats >=2, candidates >=2 and positive timeout')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [args.exe.resolve(), args.profile.resolve(), args.save.resolve(), Path(__file__).resolve(),
               Path(components.__file__).resolve(),ROOT/'tools/bench/analyze_stage2_tune_workload.py',
               ROOT/'tools/bench/stage2_tune_route_cost.py']
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path): sha(path) for path in sources}
    manifest_path=args.exe.resolve().parent/'build_manifest.json'
    dll=args.exe.resolve().parent/'gmp-10.dll'
    assert manifest_path.is_file() and dll.is_file(), 'actual build manifest and GMP DLL required'
    identities[str(manifest_path)]=sha(manifest_path);identities[str(dll)]=sha(dll)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    assert manifest['sha256'].lower()==sha(args.exe.resolve())
    closure=0
    for entry in manifest['sources']:
        name,separator,digest=entry.rpartition('=')
        path=ROOT/name
        if separator and len(digest)==64 and path.is_file():
            assert sha(path)==digest.lower(), ('source differs from binary',name)
            identities[str(path.resolve())]=digest.lower();closure+=1
    assert closure>=49, 'incomplete production source closure'
    assert args.profile.stat().st_size <= 64*1048576, 'profile exceeds native size limit'
    profile = tomllib.loads(args.profile.read_text(encoding='utf-8-sig'))
    assert 0 < len(profile['ecm']) <= 4096, 'profile exceeds native sample limit'
    with args.save.open(encoding='utf-8-sig') as stream:
        line = next(line for line in stream if line.strip())
    fields = dict(re.findall(r'(\w+)\s*=\s*([^;]+)', line))
    target = int(fields['N'].strip(), 0)
    b1 = int(fields['B1'])
    assert target > 3 and target%2 and target.bit_length() <= 16384 and b1 >= 2
    groups = {}
    for sample in profile['ecm'].values():
        if not sample['fold_resident'] or not sample['frontier_resident']:
            continue
        if sample['target_bits'] != target.bit_length() or sample['b1'] != b1:
            continue
        carrier = sample['carrier_exponent']
        assert 0 <= carrier <= 16384, 'invalid carrier exponent'
        if carrier and ((1 << carrier)-1) % target:
            continue
        if not carrier and sample['modulus_kind'] != ('mersenne' if (target+1)&target == 0 else 'generic'):
            continue
        key = tuple(sample[k] for k in SCOPE)
        groups.setdefault(key, []).append(sample)
    assert groups, 'profile does not cover this save'
    measurements=embedded_ntt_samples(profile)
    component_models={}
    if measurements:
        assert args.training_plans, 'format 4 requires independent frozen training plans'
        plan_samples={tuple(s[k] for k in ('target_bits','arithmetic_bits','carrier_exponent','b1','b2','d')):s
                      for group in groups.values() for s in group}
        records={};seen=set()
        for path in sorted(args.training_plans.glob('case_*.plan.jsonl')):
            plan=one_json(path)
            key=tuple(plan[k] for k in ('target_bits','bits','carrier_exponent','B1','B2','D'))
            if key not in plan_samples:
                continue
            assert key not in seen, 'duplicate component anchor plan'
            seen.add(key);identities[str(path.resolve())]=sha(path)
            sample=plan_samples[key]
            scope=tuple(sample[k] for k in SCOPE)
            try:
                reference=components.loop_reference(workload(plan),measurements)
                records.setdefault(scope,[]).append(dict(sample=sample,loop_reference_seconds=reference))
            except ValueError:
                pass  # Incomplete groups must stay on the existing qualified route model.
        assert seen==set(plan_samples), 'missing independent component training plans'
        for scope,group in groups.items():
            if len(records.get(scope,[]))!=len(group):
                continue
            try:
                model=components.train(records[scope])
                if model['qualified']:
                    component_models[scope]=model
            except ValueError:
                pass
    ini = output/'bench.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    policy = profile['policy']
    environment=dict(os.environ)
    for key,value in policy['environment'].items():
        assert type(value) is int and value>=0
        environment['NTT_'+key.upper()]=str(value)
    common = [str(args.exe.resolve()), '--ini', str(ini), '--save', str(args.save.resolve()),
              '--device', str(args.device), '--tune-profile', str(args.profile.resolve()),
              '--batch-mb', str(policy['batch_mb']), '--arena-mb', str(policy['arena_mb']),
              '--owner-budget-mb', str(policy['fold_mb']), '--log-level', 'quiet']
    report = dict(binary_sha256=identities[str(args.exe.resolve())],
                  profile_sha256=identities[str(args.profile.resolve())],
                  source_identities=identities, device=args.device, target_bits=target.bit_length(),
                  b1=b1, policy={k:v for k,v in policy.items() if k!='environment'},
                  environment={k:v for k,v in environment.items() if k.startswith('NTT_')},
                  repeats=args.repeats, warmups=1, time_error_limit=TIME_ERROR_LIMIT,
                  rank_loss_limit=RANK_LOSS_LIMIT, holdouts=[], complete=False,
                  total_scope='stage2_full_wall.total; Stage1/process/planning/publication excluded')

    def publish():
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    def run(name, b2, extra):
        proc = subprocess.run(common+['--b2', str(b2), *extra], cwd=ROOT, capture_output=True,
                              text=True, errors='replace', timeout=args.timeout,env=environment)
        (output/(name+'.console.log')).write_text(proc.stdout+proc.stderr, encoding='utf-8')
        assert proc.returncode == 0, (name, proc.returncode, proc.stderr)
        return [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]

    def curve(name, b2, extra):
        results, log = output/(name+'.jsonl'), output/(name+'.log')
        run(name, b2, [*extra, '--curves', '1', '--results', str(results), '--log', str(log)])
        row = json.loads(results.read_text(encoding='utf-8'))
        assert row['status'] == 'stage2_completed' and row['hits'] == row['bad_factors'] == 0
        text = log.read_text(encoding='utf-8')
        assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
        assert re.search(r'real_batched_folddevice: requested=1 enabled=1 fallback=none\b', text)
        assert re.search(r'scaled_frontier_device: requested=1 enabled=1\b', text)
        wall = next(line for line in text.splitlines() if line.startswith('stage2_full_wall:'))
        assert 'clean=1' in wall
        return float(re.search(r'\btotal=([0-9.]+)', wall).group(1)), row

    for path,digest in identities.items():
        target=output/'inputs'/digest/Path(path).name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    telemetry=(output/'telemetry.csv').open('wb')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used',
        '--format=csv','-lms','1000'],stdout=telemetry,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    publish()
    try:
        for index, b2 in enumerate(args.holdout_b2):
            assert b2 > b1 and all(s['b2'] != b2 for group in groups.values() for s in group)
            holdout = dict(b2=b2, candidates=[], ineligible=[], complete=False)
            report['holdouts'].append(holdout)
            for key, group in groups.items():
                d, carrier = key[-1], key[2]
                prediction = predict(group, b2, profile['profile'].get('prediction_model'))
                stem = f'holdout_{index}_d{d}_c{carrier}'
                extra = ['--d', str(d), '--carrier-exponent', str(carrier)]
                rows = run(stem+'_plan', b2, [*extra, '--plan-only'])
                choice = next(x for x in rows if x.get('type') == 'tune_selection')
                plan = next(x for x in rows if x.get('type') == 'stage2_plan')
                if key in component_models:
                    try:
                        result=components.predict(component_models[key],plan,measurements)
                        prediction=dict(model=components.MODEL,seconds=result['seconds'],rank=result['rank_seconds'])
                    except ValueError:
                        pass
                if not choice['selected']:
                    holdout['ineligible'].append(dict(d=d, carrier=carrier, reason=choice['reason']))
                    continue
                assert prediction is not None, 'native selected an independently ineligible group'
                assert choice['model'] == prediction['model']
                assert math.isclose(choice['estimated_seconds'], prediction['seconds'], rel_tol=1e-10)
                assert math.isclose(choice['rank_seconds'], prediction['rank'], rel_tol=1e-10)
                assert plan['D'] == d and plan['carrier_exponent'] == carrier and plan['B2'] == b2
                assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
                candidate = dict(d=d, carrier=carrier, prediction=prediction, selection=choice, seconds=[])
                holdout['candidates'].append(candidate)
                publish()
                for repeat in range(args.repeats+1):
                    seconds, row = curve(stem+f'_repeat{repeat}', b2, extra)
                    assert row['tune_plan']['D'] == d and row['carrier_exponent'] == carrier
                    if repeat:
                        candidate['seconds'].append(seconds)
                    else:
                        candidate['warmup_seconds'] = seconds
                    print(json.dumps(dict(b2=b2,d=d,carrier=carrier,repeat=repeat,seconds=seconds)), flush=True)
                    publish()
                actual = statistics.median(candidate['seconds'])
                candidate.update(actual_median=actual, relative_error=abs(actual-prediction['seconds'])/actual)
                publish()
            candidates = holdout['candidates']
            assert len(candidates) >= args.min_candidates, (b2,holdout['ineligible'])
            assert all(c['relative_error'] <= TIME_ERROR_LIMIT for c in candidates), (b2,candidates)
            rows = run(f'holdout_{index}_automatic_plan', b2, ['--plan-only'])
            choice = next(x for x in rows if x.get('type') == 'tune_selection')
            assert choice['selected']
            selected = next(c for c in candidates if c['d'] == choice['D'] and c['carrier'] == choice['carrier_exponent'])
            assert choice['model']==selected['prediction']['model']
            assert math.isclose(selected['prediction']['rank'], min(c['prediction']['rank'] for c in candidates), rel_tol=1e-10)
            fastest = min(c['actual_median'] for c in candidates)
            loss = selected['actual_median']/fastest-1
            holdout.update(automatic_selection=choice, rank_loss=loss)
            publish()
            assert loss <= RANK_LOSS_LIMIT, (b2,loss)
            seconds, row = curve(f'holdout_{index}_automatic_curve', b2, [])
            assert row['requested_D'] == 0 and row['requested_carrier_exponent'] == 0
            assert row['tune_plan']['D'] == selected['d'] and row['carrier_exponent'] == selected['carrier']
            error = abs(seconds-selected['prediction']['seconds'])/seconds
            holdout.update(automatic_seconds=seconds, automatic_relative_error=error, complete=True)
            publish()
            assert error <= TIME_ERROR_LIMIT, (b2,error)
        assert all(sha(Path(path))==digest for path,digest in identities.items())
        report['complete'] = True
        publish()
    except BaseException as exc:
        report['error'] = repr(exc)
        publish()
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        telemetry.close()
    print(json.dumps(dict(complete=True, holdouts=len(report['holdouts']),
                         maximum_rank_loss=max(h['rank_loss'] for h in report['holdouts']))))


if __name__ == '__main__':
    main()
