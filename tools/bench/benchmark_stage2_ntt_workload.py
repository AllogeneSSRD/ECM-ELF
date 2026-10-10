"""Measure missing exact NTT shapes from frozen, fully measured ECM scopes."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

from analyze_stage2_tune_workload import (integer, ntt_batch_samples,
    ntt_policy_matches, ntt_profile_set, one_json, workload)

ROOT = Path(__file__).resolve().parents[2]


def missing_jobs(shapes, measured, capacity=64):
    if type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError('invalid slices list capacity')
    groups = defaultdict(list)
    for n, slices in sorted(set(shapes)-set(measured)):
        if integer(n, 8) > 1 << 27 or n & (n-1) or integer(slices, 1) > 65535:
            raise ValueError('unsupported NTT workload shape')
        groups[n].append(slices)
    return [(n, values[start:start+capacity]) for n, values in sorted(groups.items())
            for start in range(0, len(values), capacity)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'profile', 'evidence', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--ntt-profile', type=Path, action='append', default=[])
    p.add_argument('--additional-plan', type=Path, action='append', default=[],
                   help='native query plans for independent in-scope holdouts; no curve costs imported')
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--repeats', type=int, default=21)
    p.add_argument('--memory-mb', type=int, default=1024)
    a = p.parse_args()
    if a.device < 0 or not 1 <= a.repeats <= 1000 or not 1 <= a.memory_mb <= 1048576:
        p.error('invalid device/repeats/memory')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {}

    def freeze(path):
        path = path.resolve()
        digest = sha(path)
        sources[str(path)] = digest
        snapshot = out/'inputs'/digest/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        return path

    exe = freeze(a.exe)
    freeze(exe.parent/'gmp-10.dll')
    manifest = json.loads(freeze(exe.parent/'build_manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe) != manifest['sha256'].lower():
        raise ValueError('executable differs from build manifest')
    closure = 0
    for entry in manifest['sources']:
        name, _, digest = entry.rpartition('=')
        path = ROOT/name
        if len(digest) == 64 and path.is_file():
            if sha(path) != digest.lower():
                raise ValueError('current source differs from executable: '+name)
            freeze(path)
            closure += 1
    if closure < 49:
        raise ValueError('incomplete build source closure')
    freeze(Path(__file__))
    freeze(ROOT/'tools/bench/analyze_stage2_tune_workload.py')
    profile = tomllib.loads(freeze(a.profile).read_text(encoding='utf-8-sig'))
    if (profile['profile']['format'] not in (2,3,4) or profile['profile']['unit'] != 'full_stage2' or
            not profile['summary']['complete'] or profile['summary']['failed'] or
            profile['summary']['measured'] != len(profile['ecm'])):
        raise ValueError('complete measured ECM profile required')
    nt_profiles = [tomllib.loads(freeze(path).read_text(encoding='utf-8-sig')) for path in a.ntt_profile]
    measured, qualified = ntt_profile_set(profile, nt_profiles)
    if nt_profiles and not qualified:
        raise ValueError('base NTT data must match full ECM policies')
    expected = {(s['target_bits'],s['arithmetic_bits'],s['carrier_exponent'],s['b1'],s['b2'],s['d'])
                for s in profile['ecm'].values()}
    seen, shapes = set(), set()
    for path in sorted(a.evidence.glob('case_*.plan.jsonl')):
        plan = one_json(freeze(path))
        scope = tuple(plan[k] for k in ('target_bits','bits','carrier_exponent','B1','B2','D'))
        if scope not in expected:
            continue
        if scope in seen:
            raise ValueError('duplicate measured ECM scope plan')
        seen.add(scope)
        shapes.update((r['length'],r['slices']) for r in workload(plan))
    if seen != expected or len(expected) != len(profile['ecm']):
        raise ValueError('missing/duplicate full ECM scope plans')
    groups = {scope[:4]+scope[5:] for scope in expected}
    for path in a.additional_plan:
        plan = one_json(freeze(path))
        scope = tuple(plan[k] for k in ('target_bits','bits','carrier_exponent','B1','D'))
        if scope not in groups:
            raise ValueError('additional plan outside measured width/B1/D/arithmetic groups')
        shapes.update((r['length'],r['slices']) for r in workload(plan))
    jobs = missing_jobs(shapes, measured)
    state = dict(complete=False, identities=sources, requested_shapes=len(shapes),
                 additional_plans=len(a.additional_plan),
                 initial_missing=len(shapes-set(measured)), profiles=[], jobs=[])
    publish = lambda: (out/'state.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    publish()
    ini = out/'ecm.ini'
    ini.write_text('verbose=false\nstage2_debug_log=false\n', encoding='utf-8')
    env = dict(os.environ)
    for key, value in profile['policy']['environment'].items():
        integer(value)
        env['NTT_'+key.upper()] = str(value)
    policy = profile['policy']
    common = [str(exe),'--ini',str(ini),'--device',str(a.device),
              '--batch-mb',str(policy['batch_mb']),'--arena-mb',str(policy['arena_mb']),
              '--owner-budget-mb',str(policy['fold_mb']),'--tune','ntt',
              '--tune-repeats',str(a.repeats),'--tune-memory-mb',str(a.memory_mb)]
    with (out/'telemetry.csv').open('wb') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used',
            '--format=csv','-lms','1000'], stdout=telemetry, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for i, (n, slices) in enumerate(jobs):
                destination = out/f'ntt_{i}.toml'
                command = common+['--length-log2',str(n.bit_length()-1),
                    '--tune-slices',','.join(map(str,slices)),'--tune-file',str(destination)]
                state['current'] = dict(job=i, command=command)
                publish()
                with (out/f'ntt_{i}.log').open('w',encoding='utf-8') as log:
                    proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
                    state['current']['pid'] = proc.pid
                    publish()
                    code = proc.wait(timeout=1800)
                if code:
                    raise ValueError(f'NTT job {i} failed: {code}; partial evidence retained')
                data = tomllib.loads(destination.read_text(encoding='utf-8-sig'))
                if not ntt_policy_matches(profile,data):
                    raise ValueError('measured NTT device/policy mismatch')
                fresh = ntt_batch_samples(data)
                if set(fresh) != {(n,b) for b in slices}:
                    raise ValueError('requested workload shapes exceeded memory; increase explicit budget')
                measured, qualified = ntt_profile_set(profile, nt_profiles+[data])
                nt_profiles.append(data)
                state['profiles'].append(str(destination))
                state['jobs'].append(dict(length=n,slices=slices,measured=len(fresh)))
                publish()
                print(f'ntt_workload: job={i+1}/{len(jobs)} length={n} shapes={len(fresh)}',flush=True)
            if not shapes <= set(measured) or not qualified:
                raise ValueError('incomplete/unqualified workload NTT coverage')
            if not all(sha(Path(path)) == digest for path,digest in sources.items()):
                raise ValueError('input changed during benchmark')
            state.update(complete=True, measured_shapes=len(measured), missing_shapes=0)
            publish()
            print('ntt_workload_complete:',len(shapes),'required shapes',flush=True)
        except Exception as error:
            state['failure'] = repr(error)
            publish()
            raise
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == '__main__':
    main()
