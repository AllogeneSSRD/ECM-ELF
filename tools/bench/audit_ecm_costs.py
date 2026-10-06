"""Audit exact-tree/G1 evidence, per-scope accuracy, and relative-value ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from bench_stage2_budget_scaling import fields
from ecm_cost_model import giant_work,predict,features,admits,kruppa_value,FEATURE_PROFILE
from calibrate_stage2_d import parse
from ecm_cost_coverage import check_study_coverage,check_blind_coverage,check_replay_labels,independent_scope_counts


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranking(rows,model):
    scopes={s['id']:s for s in model['stage2']};result=[]
    for bits in sorted({r['case']['bits'] for r in rows}):
        samples=[r for r in rows if r['case']['bits']==bits and not r['case']['kind'].endswith('_replay')]
        if not samples:continue
        grouped={}
        for row in samples:
            c=row['case'];key=(c['D'],c['B2'],c['owner_mb'],c['scope_id'])
            grouped.setdefault(key,[]).append(row)
        for batch in (1,12):
            t1=next(s['process_seconds_per_curve'] for s in model['stage1'] if s['bits']==bits and s['batch']==batch)
            actual={};predicted={}
            for key,runs in grouped.items():
                d,b2,owner,sid=key;s=scopes[sid];k=kruppa_value(s['B1'],b2)
                engine=statistics.median(r['actual']['full'] for r in runs)
                nominal=runs[0]['case']['prediction']['full']
                actual[key]=k/(t1+engine+s['cold_overhead_seconds'])
                predicted[key]=k/(t1+nominal+s['cold_overhead_seconds'])
            selected=max(predicted,key=predicted.get);best=max(actual,key=actual.get)
            result.append(dict(bits=bits,stage1_batch=batch,selected=list(selected),actual_best=list(best),
                selected_actual_score=actual[selected],best_actual_score=actual[best],
                selected_value_loss_percent=100*(1-actual[selected]/actual[best]),
                score_basis='engine_plus_frozen_cold_and_measured_stage1'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--blind',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study=json.loads(a.study.read_text());blind=json.loads(a.blind.read_text());model=json.loads(a.profile.read_text())
    if not study['complete'] or not blind['complete']:raise ValueError('Incomplete evidence')
    schedule=check_study_coverage(study)
    if 'stage1_provenance' in study:
        prior=study['stage1_provenance'];previous=json.loads(Path(prior['path']).read_text(encoding='utf-8'))
        if sha(prior['path'])!=prior['sha256'] or not previous['complete'] or prior['identity']!=previous['identity'] or prior['controls']!=previous['controls']:
            raise ValueError('Stage1 source provenance changed')
        if study['stage1']!=previous['stage1'] or prior['records']!=len(previous['stage1']) or study['identity']['stage1_sha256']!=previous['identity']['stage1_sha256']:
            raise ValueError('Stage1 observations were changed')
        if study['device']['uuid']!=previous['device']['uuid']:raise ValueError('Reused Stage1 hardware differs')
        for row in study['stage1']:
            cmd=row['command']
            if row['B1']!=1000 or row['torsion']!=1 or cmd[cmd.index('--exponent')+1]!='lcm' or cmd[cmd.index('-d')+1]!='1':
                raise ValueError('Reused Stage1 does not match the measured lcm/GPU1 contract')
    if 'prior_collection' in study:
        prior=study['prior_collection'];previous=json.loads(Path(prior['path']).read_text(encoding='utf-8'))
        if sha(prior['path'])!=prior['sha256'] or not previous['complete']:
            raise ValueError('Prior calibration provenance changed')
        if prior['identity']!=previous['identity'] or prior['controls']!=previous['controls']:
            raise ValueError('Prior calibration identity changed')
        for kind in ('stage1','stage2'):
            count=prior[kind+'_records']
            if count!=len(previous[kind]) or study[kind][:count]!=previous[kind]:
                raise ValueError('Prior observations were changed or removed')
    if model['schema']!=2 or model['feature_profile']!=FEATURE_PROFILE:raise ValueError('Unsupported cost version')
    if model['model_code_sha256']!=sha(Path(__file__).with_name('ecm_cost_model.py')):raise ValueError('Cost implementation changed')
    if 'coverage_code_sha256' in model and model['coverage_code_sha256']!=sha(Path(__file__).with_name('ecm_cost_coverage.py')):
        raise ValueError('Calibration coverage policy changed; refit and audit')
    if blind['profile_sha256']!=sha(a.profile) or model['source_sha256']!=sha(a.study):raise ValueError('Profile identity changed')
    if sha(a.blind.parent/'predictions.json')!=blind['prediction_sha256']:raise ValueError('Frozen predictions changed')
    frozen=json.loads((a.blind.parent/'predictions.json').read_text());minimum=model['chain_min']
    validation_cases=check_blind_coverage(frozen,blind['runs'])
    check_replay_labels(study,frozen['cases'])
    independent_counts=independent_scope_counts(frozen['cases'])
    for key in ('binary_sha256','model_code_sha256','shuffle_seed','build_manifest_sha256','tools'):
        if key in frozen and blind.get(key)!=frozen[key]:raise ValueError('Blind collector identity changed: '+key)
    for row in study['stage1']:
        c=row['command']
        if sha(Path(c[c.index('-save')+1]))!=row['save_sha256']:raise ValueError('Verified Stage1 save changed')
    leaves={};checks=coeffs=g1_count=root_count=0
    for row in study['stage2']+blind['runs']:
        text=Path(row['log']).read_text()
        if sha(row['log'])!=row['log_sha256']:raise ValueError('Raw log changed')
        if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):raise ValueError('Missing arithmetic check')
        if parse(text)!=row.get('phases',row.get('actual')):raise ValueError('Phase decomposition changed')
        c=row.get('case',row);f=row.get('features',c.get('features'))
        if f!=features(c['D'],c['B2'],c['bits'],minimum):raise ValueError('Feature identity mismatch')
        work=giant_work(f);seed=fields(text,'real_giant_seed')
        if int(seed['chunks'])!=work['chain_chunks']:raise ValueError('Actual giant route differs')
        gt=fields(text,'real_batched_gdevice')
        if (int(gt['pairs']),int(gt['groups']),int(gt['copies']))!=(f['g_tree_pairs'],f['gtrees_groups'],f['g_tree_copies']):raise ValueError('Tree schedule differs')
        if f['G']==1:
            g1_count+=1;desc=fields(text,'scaled_descent');root=int(desc['root_divisions'])
            if int(desc['root_inverse_reused'])!=0 or root!=int(f['I']==f['P']):raise ValueError('G1 inverse/root dispatch differs')
            root_count+=root
        else:
            desc=fields(text,'scaled_descent')
            if int(desc['root_inverse_reused'])!=1 or int(desc['root_divisions'])!=0:
                raise ValueError('Multiple-G cached inverse/root dispatch differs')
        key=(c['bits'],c['D'],c['B2']);leaf=fields(text,'descent_values')['hash']
        if key in leaves and leaves[key]!=leaf:raise ValueError('Path/repetition changed leaf fingerprint')
        leaves[key]=leaf;checks+=1;coeffs+=int(fields(text,'s4_multiply_stats')['gmp_checked'])
    validation=[]
    scopes={s['id']:s for s in model['stage2']}
    for row in blind['runs']:
        c=row['case'];s=scopes[c['scope_id']]
        if not admits(s,c['features']) or c not in frozen['cases']:raise ValueError('Blind case outside frozen scope')
        pred,_=predict(c['bits'],c['D'],c['B2'],s['rates'],minimum)
        if pred!=c['prediction']:raise ValueError('Frozen prediction differs from model')
    for s in model['stage2']:
        rows=[r for r in blind['runs'] if r['case']['scope_id']==s['id']]
        expected=sum(c['scope_id']==s['id'] for c in frozen['cases'])
        errors=[abs(r['engine_error_percent']) for r in rows]
        validation.append(dict(id=s['id'],bits=s['bits'],owner_mb=s['owner_mb'],regime=s['regime'],
            samples=len(rows),expected_samples=expected,max_abs_percent=max(errors) if errors else None,
            independent_samples=independent_counts[s['id']],
            passed=bool(s['usable'] and expected and independent_counts[s['id']] and len(rows)==expected and max(errors)<=10)))
    ready={s['id'] for s in validation if s['passed']}
    all_errors=[r['engine_error_percent'] for r in blind['runs']]
    independent=[r['engine_error_percent'] for r in blind['runs'] if not r['case']['kind'].endswith('_replay')]
    ranks=ranking(blind['runs'],model)
    accuracy_passed=bool(validation) and all(s['passed'] for s in validation)
    expected_ranks={(s['bits'],s['batch']) for s in model['stage1']}
    ranking_passed=({(r['bits'],r['stage1_batch']) for r in ranks}==expected_ranks and
                    all(r['selected_value_loss_percent']<=5 for r in ranks))
    result=dict(schema=2,passed=accuracy_passed and ranking_passed,integrity_passed=True,
        accuracy_passed=accuracy_passed,ranking_passed=ranking_passed,
        schedule_coverage=schedule,validated_case_count=validation_cases,
        study_stage1_batches=len(study['stage1']),
        independently_verified_stage1_points=sum(r['batch'] for r in study['stage1']),
        fitting_stage2=sum(r['kind']=='train' for r in study['stage2']),
        diagnostic_b2_stage2=sum(r['kind']=='holdout' for r in study['stage2']),
        validation_stage2=len(blind['runs']),blind_stage2=len(independent),replayed_stage2=len(all_errors)-len(independent),
        root_replays=sum(r['case']['kind']=='root_replay' for r in blind['runs']),
        shape_replays=sum(r['case']['kind']=='shape_replay' for r in blind['runs']),
        clean_curves=checks,independent_gmp_coefficients=coeffs,g1_curves=g1_count,g1_root_reductions=root_count,
        matched_leaf_groups=len(leaves),validation_error_percent=[min(all_errors),max(all_errors)],
        blind_error_percent=[min(independent),max(independent)],blind_max_abs_percent=max(map(abs,independent)),
        validation_scopes=validation,ranking=ranks,
        validated_ranking=ranking([r for r in blind['runs'] if r['case']['scope_id'] in ready],model),
        profile_sha256=sha(a.profile),study_sha256=sha(a.study),blind_sha256=sha(a.blind),auditor_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
