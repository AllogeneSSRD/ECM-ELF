"""Audit phase/holdout evidence and publish a compact portable validation summary."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from bench_stage2_budget_scaling import fields
from ecm_cost_model import giant_work,predict
from calibrate_stage2_d import parse


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--blind',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study=json.loads(a.study.read_text(encoding='utf-8'))
    blind=json.loads(a.blind.read_text(encoding='utf-8'));model=json.loads(a.profile.read_text(encoding='utf-8'))
    if not study['complete'] or not blind['complete']:raise ValueError('Incomplete evidence')
    if blind['profile_sha256']!=sha(a.profile) or model['source_sha256']!=sha(a.study):raise ValueError('Profile identity changed')
    if sha(a.blind.parent/'predictions.json')!=blind['prediction_sha256']:raise ValueError('Blind predictions changed')
    for row in study['stage1']:
        command=row['command'];save=Path(command[command.index('-save')+1])
        if sha(save)!=row['save_sha256']:raise ValueError('Verified Stage1 output changed')
    leaves={};checks=0;points=0
    for row in study['stage2']+blind['runs']:
        text=Path(row['log']).read_text(encoding='utf-8')
        if sha(row['log'])!=row['log_sha256']:raise ValueError('Raw log changed')
        if not all(t in text for t in ('gmp_selftest_bad=0','gmp_check_bad=0','pending=0','clean=1')):
            raise ValueError('Mandatory arithmetic check missing')
        phases=row.get('phases',row.get('actual'))
        if parse(text)!=phases:raise ValueError('Phase evidence mismatch')
        case=row.get('case',row);f=row.get('features',case['features'])
        seeds=fields(text,'real_giant_seed');work=giant_work(f)
        if int(seeds['chunks'])!=work['chain_chunks']:raise ValueError('Modeled giant route differs from runtime')
        key=(case['bits'],case['D'],case['B2']);leaf=fields(text,'descent_values')['hash']
        if key in leaves and leaves[key]!=leaf:raise ValueError('Path/rep changed leaf fingerprint')
        leaves[key]=leaf;checks+=1
        values=fields(text,'s4_multiply_stats');points+=int(values['gmp_checked'])
    ranks=[]
    for bits in study['controls']['bits']:
        rows=[r for r in blind['runs'] if r['case']['bits']==bits]
        grouped={}
        for row in rows:
            key=(row['case']['D'],row['case']['owner_mb']);grouped.setdefault(key,[]).append(row['actual']['full'])
        actual={key:statistics.median(vals) for key,vals in grouped.items()}
        predicted={}
        for scope in model['stage2']:
            if scope['bits']!=bits:continue
            for d in scope['d_values']:
                stages,_=predict(bits,d,rows[0]['case']['B2'],scope['rates']);predicted[(d,scope['owner_mb'])]=stages['full']
        selected=min(predicted,key=predicted.get);best=min(actual,key=actual.get)
        ranks.append(dict(bits=bits,selected=list(selected),actual_fastest=list(best),
            selected_median_seconds=actual[selected],fastest_median_seconds=actual[best],
            selected_slowdown_percent=100*(actual[selected]/actual[best]-1)))
    errors=[r['engine_error_percent'] for r in blind['runs']]
    # Arithmetic evidence and prediction readiness are separate. Never publish
    # a noisy scope merely because another width passes the accuracy target.
    validation=[]
    for scope in model['stage2']:
        rows=[r for r in blind['runs'] if r['case']['bits']==scope['bits'] and
              r['case']['owner_mb']==scope['owner_mb'] and r['case']['D'] in scope['d_values']]
        expected=len(scope['d_values'])*len({r['case']['rep'] for r in blind['runs']})
        maximum=max((abs(r['engine_error_percent']) for r in rows),default=float('inf'))
        validation.append(dict(bits=scope['bits'],owner_mb=scope['owner_mb'],
            samples=len(rows),expected_samples=expected,max_abs_percent=maximum if rows else None,
            passed=bool(scope['usable'] and len(rows)==expected and maximum<=10)))
    validated_ranks=[]
    for bits in study['controls']['bits']:
        owners={s['owner_mb'] for s in validation if s['bits']==bits and s['passed']}
        if not owners:continue
        rows=[r for r in blind['runs'] if r['case']['bits']==bits and r['case']['owner_mb'] in owners]
        grouped={}
        for row in rows:
            grouped.setdefault((row['case']['D'],row['case']['owner_mb']),[]).append(row['actual']['full'])
        actual={key:statistics.median(vals) for key,vals in grouped.items()}
        predicted={}
        for scope in model['stage2']:
            if scope['bits']!=bits or scope['owner_mb'] not in owners:continue
            for d in scope['d_values']:
                predicted[(d,scope['owner_mb'])]=predict(bits,d,rows[0]['case']['B2'],scope['rates'])[0]['full']
        selected=min(predicted,key=predicted.get);best=min(actual,key=actual.get)
        validated_ranks.append(dict(bits=bits,selected=list(selected),actual_fastest=list(best),
            selected_slowdown_percent=100*(actual[selected]/actual[best]-1)))
    result=dict(passed=True,study_stage1_batches=len(study['stage1']),
        independently_verified_stage1_points=sum(r['batch'] for r in study['stage1']),
        fitting_stage2=sum(r['kind']=='train' for r in study['stage2']),
        diagnostic_b2_stage2=sum(r['kind']=='holdout' for r in study['stage2']),blind_stage2=len(blind['runs']),
        clean_curves=checks,independent_gmp_coefficients=points,
        blind_error_percent=[min(errors),max(errors)],blind_max_abs_percent=max(map(abs,errors)),
        matched_leaf_groups=len(leaves),ranking=ranks,validation_scopes=validation,
        validated_ranking=validated_ranks,profile_sha256=sha(a.profile),
        study_sha256=sha(a.study),blind_sha256=sha(a.blind),auditor_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
