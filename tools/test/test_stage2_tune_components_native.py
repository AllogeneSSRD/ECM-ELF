"""Compare native component costs with independent Python on frozen ECM plans.

CPU only: no CUDA context, allocation, benchmark or hardware changes. Inputs
are full ECM measurements and exact NTT measurements with matching policies.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
import analyze_stage2_tune_components as ref
from analyze_stage2_tune_workload import (PHASES, ntt_phase_references,
    ntt_profile_set, ntt_workload_features, one_json, table, workload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--plans', type=Path, required=True)
    p.add_argument('--ntt-profile', type=Path, action='append', required=True)
    p.add_argument('--query-plan', type=Path, action='append', default=[])
    p.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identities = {}

    def freeze(path):
        path = path.resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        identities[str(path)] = digest
        target = out/'inputs'/digest/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    profile_path = freeze(a.profile)
    profile = tomllib.loads(profile_path.read_text(encoding='utf-8-sig'))
    frozen_ntt=[freeze(path) for path in a.ntt_profile]
    ntt = [tomllib.loads(path.read_text(encoding='utf-8-sig')) for path in frozen_ntt]
    measured, qualified = ntt_profile_set(profile, ntt)
    if not qualified:
        raise ValueError('matching declared NTT policies required')
    if (not profile['summary']['complete'] or profile['summary']['failed'] or
            profile['summary']['measured'] != len(profile['ecm'])):
        raise ValueError('complete full ECM input required')
    samples = list(profile['ecm'].values())
    expected = {(s['target_bits'],s['arithmetic_bits'],s['carrier_exponent'],s['b1'],s['b2'],s['d']): i
                for i,s in enumerate(samples)}
    if len(expected) != len(samples):
        raise ValueError('duplicate ECM scope')
    plans = {}
    for path in sorted(a.plans.glob('case_*.plan.jsonl')):
        plan = one_json(path)
        scope = tuple(plan[k] for k in ('target_bits','bits','carrier_exponent','B1','B2','D'))
        if scope in expected:
            freeze(path)
            if expected[scope] in plans:
                raise ValueError('duplicate anchor plan')
            plans[expected[scope]] = plan
    if set(plans) != set(range(len(samples))):
        raise ValueError('missing anchor plan')
    cases = [(i,plans[i]) for i in range(len(samples))]
    cases += [(-1,one_json(freeze(path))) for path in a.query_plan]

    # Freeze the actual header closure that the CPU fixture compiles.
    seen = set()

    def closure(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        freeze(path)
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.startswith('#include "'):
                closure(path.parent/line.split('"')[1])

    fixture = ROOT/'tools/test/stage2_tune_components_fixture.cpp'
    closure(fixture)
    for path in [Path(__file__), Path(ref.__file__), ROOT/'tools/bench/analyze_stage2_tune_workload.py',
                 ROOT/'tools/bench/stage2_tune_route_cost.py']:
        freeze(path)
    gmp = ROOT/'third_party/gmp-zen3/dist'
    exe = out/'fixture.exe'
    cmd = out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(gmp/'include')+'" "'+str(fixture)+
        '" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+
        '" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n', encoding='utf-8')
    proc = subprocess.run(['cmd','/c',str(cmd)],capture_output=True,text=True,errors='replace',timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    shutil.copy2(gmp/'bin/gmp-10.dll',out/'gmp-10.dll')
    freeze(exe)
    freeze(out/'gmp-10.dll')
    gates = json.loads(subprocess.check_output([str(exe),'--selftest'],text=True))
    checks = 0

    def close(left,right):
        nonlocal checks
        checks += 1
        if not math.isclose(left,right,rel_tol=1e-7,abs_tol=1e-10):
            raise AssertionError((left,right))

    def invoke(name, current_cases=cases, measurements=measured, input_profile=profile_path):
        lines = [str(len(measurements))]
        lines.extend(f'{n} {batch} {seconds:.17g}' for (n,batch),seconds in sorted(measurements.items()))
        lines.append(str(len(current_cases)))
        for index,plan in current_cases:
            tree = plan['tree_workspace']
            shapes = plan['s4_memory']['shapes']
            physical = tree['physical_chunks']
            values = [index,plan['bits'],plan['P'],plan['I'],tree['batch_bytes'],
                plan['workspace_buffers'] if physical else 3,int(physical),tree['chunk_max'],
                plan['D'],plan['carrier_exponent'],plan['B2'],len(shapes)]
            lines.append(' '.join(map(str,values)))
            lines.extend(f"{s['operand']} {s['N']} {s['slots']}" for s in shapes)
        path = out/(name+'.txt')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        proc = subprocess.run([str(exe),str(input_profile),str(path)],capture_output=True,text=True,timeout=60)
        (out/(name+'.stderr')).write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        (out/(name+'.json')).write_text(proc.stdout,encoding='utf-8')
        return json.loads(proc.stdout)

    actual = invoke('native')
    groups = {}
    for (index,plan),observed in zip(cases,actual['cases']):
        rows = workload(plan)
        coverage,annotated = ntt_workload_features(rows,measured)
        assert not coverage['ntt_missing_batch_bins'] and observed['complete']
        phases = ntt_phase_references(annotated)
        assert observed['bins'] == [[PHASES.index(r['phase']),r['length'],r['slices'],r['calls'],r['pairs']] for r in rows]
        for phase,published in zip(PHASES,observed['phases']):
            assert published['calls'] == phases[phase]['total_calls'] and published['pairs'] == phases[phase]['total_pairs']
            assert not published['missing']
            close(published['seconds'],phases[phase]['reference_seconds'])
        if index >= 0:
            s = samples[index]
            groups.setdefault(tuple(s[k] for k in ref.SCOPE),[]).append(dict(sample=s,
                loop_reference_seconds=phases['gtrees']['reference_seconds']+phases['fold']['reference_seconds']))
    models = [ref.train(records) for records in groups.values()]
    assert len(actual['models']) == len(models)
    for native in actual['models']:
        matching = [m for m in models if m['scope']['d']==native['d'] and m['scope']['carrier_exponent']==native['carrier']]
        assert len(matching)==1
        model=matching[0]
        assert native['qualified'] == model['qualified'] and native['qualified']
        close(native['fixed_seconds'],model['fixed_seconds'])
        close(native['relative'],model['max_loo_relative_error'])
        close(native['absolute'],model['max_loo_absolute_error'])
        for x,y in zip(native['coefficients'],model['coefficients']):
            close(x,y)
    for (_,plan),native in zip(cases[len(samples):],actual['cases'][len(samples):]):
        matching = [m for m in models if m['scope']['d']==plan['D'] and m['scope']['carrier_exponent']==plan['carrier_exponent']]
        assert len(matching)==1 and native['predicted']
        prediction = ref.predict(matching[0],plan,measured)
        close(native['seconds'],prediction['seconds'])
        close(native['rank'],prediction['rank_seconds'])

    # A missing measured shape must not turn partial reference time into a model.
    missing = dict(measured)
    used = workload(cases[0][1])[0]
    del missing[used['length'],used['slices']]
    partial = invoke('missing_shape',measurements=missing)
    assert not partial['cases'][0]['complete']
    affected = {(samples[i]['d'],samples[i]['carrier_exponent']) for i,c in enumerate(partial['cases'][:len(samples)]) if not c['complete']}
    assert affected and all(not m['qualified'] for m in partial['models'] if (m['d'],m['carrier']) in affected)
    # Endpoints and a different batch policy are not eligible interpolation.
    boundary_cases=[]
    for index,plan in cases:
        if index==0:
            boundary_cases.append((-1,plan))
    modified=copy.deepcopy(cases[len(samples)][1] if a.query_plan else cases[0][1])
    modified['tree_workspace']['batch_bytes']+=1
    boundary_cases.append((-1,modified))
    boundaries=invoke('boundaries',current_cases=cases[:len(samples)]+boundary_cases)
    assert all(not c['predicted'] for c in boundaries['cases'][-2:])
    rejected_groups=0

    def profile_variant(name,change,reader_refuses=False):
        nonlocal rejected_groups
        changed=copy.deepcopy(profile)
        changed_sample=next(iter(changed['ecm'].values()))
        change(changed_sample)
        text=''.join(table(k,changed[k]) for k in ('profile','device'))
        text+=table('policy',{k:v for k,v in changed['policy'].items() if k!='environment'})
        text+=table('policy.environment',changed['policy']['environment'])
        text+=''.join(table('ecm.'+k,s) for k,s in changed['ecm'].items())
        text+=table('summary',changed['summary'])
        path=out/(name+'.toml')
        path.write_text(text,encoding='utf-8')
        if reader_refuses:
            proc=subprocess.run([str(exe),str(path),str(out/'native.txt')],capture_output=True,text=True,timeout=60)
            (out/(name+'.stderr')).write_text(proc.stderr,encoding='utf-8')
            assert proc.returncode and proc.stderr
        else:
            result=invoke(name,input_profile=path)
            affected=[m for m in result['models'] if m['d']==samples[0]['d'] and m['carrier']==samples[0]['carrier_exponent']]
            assert len(affected)==1 and not affected[0]['qualified']
        rejected_groups+=1

    profile_variant('nonresident',lambda s:s.update(fold_resident=0))
    profile_variant('unchecked',lambda s:s.update(checked=0),True)
    profile_variant('bad_arithmetic',lambda s:s.update(bad=1),True)
    profile_variant('unpaired',lambda s:s.pop('phase_accounting'),True)

    def noise(sample):
        for key,value in list(sample.items()):
            if isinstance(value,list) and (key=='seconds' or key.endswith('_samples')):
                sample[key]=[2*v for v in value]
            elif isinstance(value,(int,float)) and key.endswith('_seconds'):
                sample[key]=2*value

    profile_variant('noisy_group',noise)
    insufficient=invoke('insufficient',current_cases=cases[:6])
    assert insufficient['models'] and not any(m['qualified'] for m in insufficient['models'])
    duplicated=invoke('duplicate_anchor',current_cases=cases[:len(samples)]+[cases[0]])
    affected=[m for m in duplicated['models'] if m['d']==samples[0]['d'] and m['carrier']==samples[0]['carrier_exponent']]
    assert len(affected)==1 and not affected[0]['qualified']
    changed_cases=copy.deepcopy(cases[:len(samples)])
    changed_cases[0][1]['tree_workspace']['batch_bytes']+=1
    changed_policy=invoke('anchor_policy_mismatch',current_cases=changed_cases)
    affected=[m for m in changed_policy['models'] if m['d']==samples[0]['d'] and m['carrier']==samples[0]['carrier_exponent']]
    assert len(affected)==1 and not affected[0]['qualified']
    rejected_groups+=3
    # Native import and format-4 roundtrip, then the same estimator used by main.
    attached=subprocess.run([str(exe),'--attach',str(profile_path),*[str(p) for p in frozen_ntt]],
                            capture_output=True,text=True,timeout=60)
    if attached.returncode:
        raise RuntimeError(attached.stderr)
    extended=out/'component_profile.toml'
    extended.write_text(attached.stdout,encoding='utf-8')
    embedded=tomllib.loads(attached.stdout)
    assert embedded['profile']['format']==4 and embedded['profile']['component_model']==ref.MODEL
    assert embedded['summary']['ntt_measured']==len(measured)
    assert ntt_profile_set(embedded,[])==(measured,True)
    assert not any(token in attached.stdout for token in ('sha256','binary_hash','D:\\','build_manifest'))
    assert subprocess.check_output([str(exe),'--load',str(extended)],text=True).split()==[str(len(samples)),str(len(measured))]
    imported=invoke('embedded',input_profile=extended)
    assert all(m['embedded_qualified'] for m in imported['models'])
    for native,baseline in zip(imported['cases'][len(samples):],actual['cases'][len(samples):]):
        assert native['embedded_predicted'] and native['embedded_model']==ref.MODEL
        close(native['embedded_seconds'],baseline['seconds'])
    format_rejections=0

    def import_refuses(name,changed=None,inputs=None):
        nonlocal format_rejections
        paths=inputs
        if paths is None:
            text=''.join(table(k,changed[k]) for k in ('profile','device'))
            text+=table('policy',{k:v for k,v in changed['policy'].items() if k!='environment'})
            text+=table('policy.environment',changed['policy']['environment'])
            text+=''.join(table('ntt.'+length+'.'+batch,s) for length,batches in changed['ntt'].items() for batch,s in batches.items())
            text+=table('summary',changed['summary'])
            path=out/(name+'.toml')
            path.write_text(text,encoding='utf-8')
            paths=[path]
        failed=subprocess.run([str(exe),'--attach',str(profile_path),*[str(p) for p in paths]],capture_output=True,text=True,timeout=60)
        (out/(name+'.stderr')).write_text(failed.stderr,encoding='utf-8')
        assert failed.returncode and failed.stderr
        format_rejections+=1

    base_ntt=next(p for p in ntt if any(s['status']=='measured' for batch in p['ntt'].values() for s in batch.values()))
    for name,key,value in [('wrong_ntt_mask','gl_add_sub_mask',3),('wrong_ntt_driver','cuda_driver',1)]:
        changed=copy.deepcopy(base_ntt)
        changed['device'][key]=value
        import_refuses(name,changed)
    changed=copy.deepcopy(base_ntt)
    changed['policy']['environment']['s4_batch_mb']+=1
    import_refuses('wrong_ntt_environment',changed)
    for name,key,value in [('unchecked_ntt','bad',1),('wrong_words','verified_words_per_sample',1),
                           ('wrong_ntt_unit','unit','ecm'),('wrong_ntt_reference','reference_kind','unknown'),
                           ('wrong_ntt_median','median_seconds',100.),('wrong_ntt_rate','conv_iter_per_s',0.)]:
        changed=copy.deepcopy(base_ntt)
        record=next(s for batch in changed['ntt'].values() for s in batch.values() if s['status']=='measured')
        record[key]=value
        import_refuses(name,changed)
    changed=copy.deepcopy(base_ntt)
    changed['profile']['format']=1
    import_refuses('legacy_ntt_policy_missing',changed)
    changed=copy.deepcopy(base_ntt)
    changed['profile']['repeats']+=1
    import_refuses('wrong_ntt_repeats',changed)
    changed=copy.deepcopy(base_ntt)
    first_length=next(iter(changed['ntt']))
    del changed['ntt'][first_length][next(iter(changed['ntt'][first_length]))]
    import_refuses('incomplete_ntt_grid',changed)
    import_refuses('duplicate_ntt_input',inputs=[frozen_ntt[0],frozen_ntt[0]])
    for name,change in [('embedded_count',lambda p:p['summary'].update(ntt_measured=1)),
                        ('embedded_model',lambda p:p['profile'].update(component_model='unknown')),
                        ('embedded_policy',lambda p:p['ntt']['policy'].update(batch_bytes=1))]:
        changed=copy.deepcopy(embedded)
        change(changed)
        text=''.join(table(k,changed[k]) for k in ('profile','device'))
        text+=table('policy',{k:v for k,v in changed['policy'].items() if k!='environment'})
        text+=table('policy.environment',changed['policy']['environment'])
        text+=''.join(table('ecm.'+k,s) for k,s in changed['ecm'].items())
        text+=table('ntt.policy',changed['ntt']['policy'])
        text+=''.join(table('ntt.'+length+'.'+batch,s) for length,batches in changed['ntt'].items() if length!='policy' for batch,s in batches.items())
        text+=table('summary',changed['summary'])
        path=out/(name+'.toml')
        path.write_text(text,encoding='utf-8')
        failed=subprocess.run([str(exe),'--load',str(path)],capture_output=True,text=True,timeout=60)
        assert failed.returncode and failed.stderr
        format_rejections+=1
    for path,digest in identities.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, path
    result = dict(complete=True,anchors=len(samples),queries=len(a.query_plan),groups=len(models),
        bins=sum(len(c['bins']) for c in actual['cases']),numeric_checks=checks,selftest_gates=gates['gates'],
        missing_shape_refused=True,endpoints_refused=True,changed_policy_refused=True,
        rejected_group_or_reader_cases=rejected_groups,noisy_group_ineligible=True,
        native_ntt_import=True,format4_roundtrip=True,embedded_predictions_match=True,format_rejections=format_rejections,
        source_identities_match=True,production_ranking_changed=False,identities=identities)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='identities'}))


if __name__=='__main__':
    main()
