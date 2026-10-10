"""Freeze component predictions, then validate unseen B2 with four full-curve candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tomllib

from analyze_stage2_tune_components import LIMIT, MODEL, predict
from analyze_stage2_tune_workload import ntt_profile_set

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe','save','models','profile','evidence','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ntt-profile',type=Path,action='append',required=True)
    p.add_argument('--holdout-b2',type=int,nargs='+',required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--ntt-repeats',type=int,default=21)
    a = p.parse_args()
    if a.device < 0 or not 1 <= a.repeats <= 1000:
        p.error('invalid device/repeat count')
    out = a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {}

    def freeze(path):
        path = path.resolve()
        digest = sha(path)
        identities[str(path)] = digest
        target = out/'inputs'/digest/path.name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        return path

    exe,save = freeze(a.exe),freeze(a.save)
    manifest = json.loads(freeze(exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    assert sha(exe) == manifest['sha256'].lower()
    freeze(exe.parent/'gmp-10.dll')
    for entry in manifest['sources']:
        name,_,digest = entry.rpartition('=')
        path = ROOT/name
        if len(digest)==64 and path.is_file():
            assert sha(path)==digest.lower(),name
            freeze(path)
    for path in (Path(__file__),ROOT/'tools/bench/analyze_stage2_tune_components.py',
                 ROOT/'tools/bench/analyze_stage2_tune_workload.py',ROOT/'tools/bench/benchmark_stage2_ntt_workload.py'):
        freeze(path)
    data = tomllib.loads(freeze(a.models).read_text(encoding='utf-8-sig'))
    assert data['profile']['model']==MODEL and data['profile']['ranking_qualified'] is False
    models = list(data['component'].values())
    assert len(models)>=4 and all(m['qualified'] and m['max_loo_relative_error']<=LIMIT for m in models)
    full = tomllib.loads(freeze(a.profile).read_text(encoding='utf-8-sig'))
    paths = [freeze(path) for path in a.ntt_profile]
    measurements,qualified = ntt_profile_set(full,[tomllib.loads(path.read_text()) for path in paths])
    assert qualified
    policy = full['policy']
    env = dict(os.environ)
    env.update({'NTT_'+k.upper():str(v) for k,v in policy['environment'].items()})
    ini = out/'ecm.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    common = [str(exe),'--ini',str(ini),'--save',str(save),'--device',str(a.device),
        '--batch-mb',str(policy['batch_mb']),'--arena-mb',str(policy['arena_mb']),
        '--owner-budget-mb',str(policy['fold_mb']),'--log-level','quiet']
    state = dict(complete=False,identities=identities,holdouts=[],repeats=a.repeats,
                 warmups=1,time_error_limit=LIMIT,rank_loss_limit=.05,
                 timing_boundary='stage2_full_wall.total; Stage1/process/planning excluded')
    publish = lambda:(out/'state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')

    def run(name,args):
        state['current'] = dict(name=name,command=common+args)
        publish()
        with (out/(name+'.console.log')).open('w',encoding='utf-8') as log:
            proc = subprocess.Popen(common+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            state['current']['pid'] = proc.pid
            publish()
            code = proc.wait(timeout=1800)
        assert code==0,(name,code)
        return [json.loads(line) for line in (out/(name+'.console.log')).read_text().splitlines() if line.startswith('{')]

    publish()
    monitor_file = (out/'telemetry.csv').open('wb')
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used',
        '--format=csv','-lms','1000'],stdout=monitor_file,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        query_paths = []
        pending = []
        for index,b2 in enumerate(a.holdout_b2):
            assert all(s['b2']!=b2 for s in full['ecm'].values()),'holdout B2 is already an anchor'
            holdout = dict(b2=b2,candidates=[])
            state['holdouts'].append(holdout)
            for model in models:
                scope = model['scope']
                d,carrier = scope['d'],scope['carrier_exponent']
                stem = f'holdout_{index}_d{d}_c{carrier}'
                extra = ['--b2',str(b2),'--d',str(d),'--carrier-exponent',str(carrier)]
                rows = run(stem+'_plan',extra+['--plan-only'])
                plan = next(row for row in rows if row.get('type')=='stage2_plan')
                assert plan['curve_workspace_memory']['initial_free_snapshot_fits']
                path = out/(stem+'_plan.jsonl')
                path.write_text(json.dumps(plan)+'\n',encoding='utf-8')
                query_paths.append(path)
                candidate = dict(d=d,carrier=carrier,seconds=[])
                holdout['candidates'].append(candidate)
                pending.append((stem,extra,model,plan,candidate))
        command = [sys.executable,str(ROOT/'tools/bench/benchmark_stage2_ntt_workload.py'),
            '--exe',str(exe),'--profile',str(a.profile.resolve()),'--evidence',str(a.evidence.resolve()),
            '--device',str(a.device),'--repeats',str(a.ntt_repeats),'--memory-mb','1024',
            '--output',str(out/'ntt_queries')]
        for path in paths:command.extend(['--ntt-profile',str(path)])
        for path in query_paths:command.extend(['--additional-plan',str(path)])
        with (out/'ntt_queries.console.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        new = json.loads((out/'ntt_queries/state.json').read_text())
        assert new['complete']
        paths.extend(freeze(Path(path)) for path in new['profiles'])
        measurements,qualified = ntt_profile_set(full,[tomllib.loads(path.read_text()) for path in paths])
        assert qualified
        for stem,extra,model,plan,candidate in pending:
            candidate['prediction'] = predict(model,plan,measurements)
        # Immutable predictions are written before any full ECM timing is sampled.
        frozen_prediction = out/'predictions.json'
        frozen_prediction.write_text(json.dumps(state['holdouts'],indent=2)+'\n',encoding='utf-8')
        freeze(frozen_prediction)
        publish()
        for stem,extra,model,plan,candidate in pending:
            for repeat in range(a.repeats+1):
                name = stem+f'_repeat{repeat}'
                results,log = out/(name+'.jsonl'),out/(name+'.log')
                run(name,extra+['--curves','1','--results',str(results),'--log',str(log)])
                row = json.loads(results.read_text())
                assert row['status']=='stage2_completed' and row['hits']==row['bad_factors']==0
                text = log.read_text(encoding='utf-8')
                assert 'gmp_selftest_bad=0' in text and 'gmp_check_bad=0' in text
                assert re.search(r'real_batched_folddevice:.*requested=1 enabled=1\b',text)
                assert re.search(r'scaled_frontier_device:.*requested=1 enabled=1\b',text)
                wall = next(line for line in text.splitlines() if line.startswith('stage2_full_wall:'))
                assert 'clean=1' in wall
                seconds = float(re.search(r'\btotal=([0-9.]+)',wall).group(1))
                if repeat:candidate['seconds'].append(seconds)
                else:candidate['warmup_seconds']=seconds
                publish()
                print('component_curve:',stem,repeat,seconds,flush=True)
        for holdout in state['holdouts']:
            for candidate in holdout['candidates']:
                actual = statistics.median(candidate['seconds'])
                candidate.update(actual_median=actual,relative_error=abs(candidate['prediction']['seconds']-actual)/actual)
            winner = min(holdout['candidates'],key=lambda c:c['prediction']['rank_seconds'])
            loss = winner['actual_median']/min(c['actual_median'] for c in holdout['candidates'])-1
            holdout.update(rank_loss=loss,selected_d=winner['d'],selected_carrier=winner['carrier'],
                qualified=loss<=.05 and all(c['relative_error']<=LIMIT for c in holdout['candidates']))
        assert all(h['qualified'] for h in state['holdouts']),'component time/rank validation failed'
        assert all(sha(Path(path))==digest for path,digest in identities.items())
        state['complete'] = True
        publish()
        print('component_validation_complete: curves=',len(pending)*(a.repeats+1),flush=True)
    except Exception as error:
        state['failure'] = repr(error)
        publish()
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_file.close()


if __name__ == '__main__':
    main()
