"""Fit separate bit-width/residency scopes; validate only on held-out B2 runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from ecm_cost_model import fit, predict, observed, PHASE_FEATURES, giant_work
from calibrate_stage2_d import parse


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_phase_medians(rows):
    """Estimate typical phase costs; retain every raw sample in the evidence.

    Identical feature values share repeated work (e.g. inverse/descent for a
    fixed P across B2 anchors). A median prevents one external wait from
    becoming an arithmetic rate for every shape. This is not a tail estimator.
    """
    samples=[observed(r) for r in rows]
    grouped={phase:{} for phase in (*PHASE_FEATURES,'giant')}
    for row,sample in zip(rows,samples):
        for phase,feature in PHASE_FEATURES.items():
            grouped[phase].setdefault(row['features'][feature],[]).append(sample[phase])
        w=giant_work(row['features']);key=(w['chain'],w['chain_chunks'],w['ladder'])
        grouped['giant'].setdefault(key,[]).append(sample['giant'])
    adjusted=[]
    for row in rows:
        phases=dict(row['phases'])
        for phase,feature in PHASE_FEATURES.items():
            value=statistics.median(grouped[phase][row['features'][feature]])
            if phase=='glue':phases['residual']=value;phases['name']=0
            else:phases[phase]=value
        w=giant_work(row['features']);phases['giant']=statistics.median(grouped['giant'][(w['chain'],w['chain_chunks'],w['ladder'])])
        adjusted.append(dict(row,phases=phases))
    return fit(adjusted)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-error-percent',type=float,default=10)
    p.add_argument('--estimator',choices=('least_squares','phase_medians'),default='least_squares')
    a=p.parse_args();study=json.loads(a.measurements.read_text(encoding='utf-8'))
    if not study.get('complete'): raise ValueError('Study has not completed')
    for row in study['stage2']:
        if sha(row['log'])!=row['log_sha256']: raise ValueError('Raw log changed')
        if parse(Path(row['log']).read_text(encoding='utf-8'))!=row['phases']:
            raise ValueError('Stored phases differ from raw log')
        if row['accounting']['version']!='2': raise ValueError('Cannot mix accounting versions')
    model=dict(schema=1,identity=study['identity'],device=study['device'],accounting_version=2,
        source_sha256=sha(a.measurements),source=str(a.measurements.resolve()),
        feature_profile=0,scope_model='per_width_per_owner_full_phase_v1',fit_estimator=a.estimator,
        benefit_model='kruppa_p95_v1',name_hits=study['controls']['name_hits'],stage1=[],stage2=[],
        fitter_sha256=sha(__file__),model_code_sha256=sha(Path(__file__).with_name('ecm_cost_model.py')))
    for bits in study['controls']['bits']:
        for batch in study['controls']['stage1_batch']:
            samples=[r for r in study['stage1'] if r['bits']==bits and r['batch']==batch and not r['warmup']]
            model['stage1'].append(dict(bits=bits,B1=1000,torsion=1,batch=batch,
                process_seconds_per_curve=statistics.median(r['amortized_process_seconds'] for r in samples),
                gpu_seconds_per_curve=statistics.median(r['amortized_gpu_seconds'] for r in samples),
                process_range=[min(r['amortized_process_seconds'] for r in samples),max(r['amortized_process_seconds'] for r in samples)],
                gpu_range=[min(r['amortized_gpu_seconds'] for r in samples),max(r['amortized_gpu_seconds'] for r in samples)],samples=len(samples)))
        for owner in (study['controls']['resident_mb'],0):
            train=[r for r in study['stage2'] if r['bits']==bits and r['owner_mb']==owner and r['kind']=='train']
            hold=[r for r in study['stage2'] if r['bits']==bits and r['owner_mb']==owner and r['kind']=='holdout']
            rates=(fit_phase_medians if a.estimator=='phase_medians' else fit)(train);checks=[]
            for row in hold:
                predicted,_=predict(bits,row['D'],row['B2'],rates)
                checks.append(dict(name=row['name'],actual=row['phases']['full'],predicted=predicted['full'],
                                   error_percent=100*(predicted['full']/row['phases']['full']-1)))
            usable=bool(checks) and max(abs(r['error_percent']) for r in checks)<=a.max_error_percent
            all_rows=train+hold
            cold=[r['process_seconds']-r['phases']['full'] for r in train]
            model['stage2'].append(dict(bits=bits,owner_mb=owner,owner_resident=owner>0,rates=rates,
                B1=1000,d_values=sorted({r['D'] for r in train}),
                b2_min=min(r['B2'] for r in all_rows),b2_max=max(r['B2'] for r in all_rows),
                p_min=min(r['features']['P'] for r in all_rows),p_max=max(r['features']['P'] for r in all_rows),
                g_min=min(r['features']['G'] for r in all_rows),g_max=max(r['features']['G'] for r in all_rows),
                cold_overhead_seconds=statistics.median(cold),cold_overhead_range=[min(cold),max(cold)],
                arena_mb=study['controls']['arena_mb'],training=len(train),holdout=checks,usable=usable,
                training_raw_error_percent=[100*(predict(bits,r['D'],r['B2'],rates)[0]['full']/r['phases']['full']-1) for r in train]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(model,indent=2),encoding='utf-8')
    print(json.dumps({'stage1_scopes':len(model['stage1']),'stage2_scopes':[
        dict(bits=r['bits'],owner=r['owner_mb'],usable=r['usable'],errors=[s['error_percent'] for s in r['holdout']])
        for r in model['stage2']]},indent=2))


if __name__=='__main__':main()
