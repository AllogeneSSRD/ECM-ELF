"""Offline Auto B2 prototype: joint B2/D/path ranking inside measured scopes.

Uses Prime95's relative K/(T1+T2), and verifies the winning geometry/live budget
with the actual CUDA plan-only API. Does not execute Stage2 or rewrite a queue.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from ecm_cost_model import kruppa_value,predict
from bench_stage2_budget_scaling import Nvml


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--stage2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage1-batch',type=int,default=1)
    p.add_argument('--stage1-seconds-per-curve',type=float)
    p.add_argument('--ratio-adjust',type=float,default=1)
    p.add_argument('--owner-budget-mb',type=int,default=640)
    p.add_argument('--arena-mb',type=int,default=4096)
    p.add_argument('--min-b2',type=int);p.add_argument('--max-b2',type=int)
    p.add_argument('--d',type=int,help='Optional fixed D')
    p.add_argument('--grid-points',type=int,default=33)
    a=p.parse_args()
    if a.grid_points<3 or a.arena_mb<1 or a.owner_budget_mb<0 or not math.isfinite(a.ratio_adjust) or a.ratio_adjust<=0:
        p.error('Invalid search/budget/ratio inputs')
    if a.stage1_seconds_per_curve is not None and (not math.isfinite(a.stage1_seconds_per_curve) or a.stage1_seconds_per_curve<=0):
        p.error('Stage1 seconds must be finite and positive')
    model=json.loads(a.profile.read_text(encoding='utf-8'))
    if model['schema']!=1 or model['accounting_version']!=2: raise ValueError('Unsupported profile')
    if sha(a.stage2)!=model['identity']['stage2_sha256']: raise ValueError('Profile/binary fingerprint mismatch')
    if sha(Path(__file__).with_name('ecm_cost_model.py'))!=model['model_code_sha256']:
        raise ValueError('Cost-model implementation changed; refit profile')
    if Nvml().name!=model['device']['name'] or model['device']['uuid']!='GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d':
        raise ValueError('Profile/device mismatch')
    text=a.save.read_text(encoding='utf-8').splitlines()[0]
    bits_match=re.search(r'\bN=\(2\^(\d+)-1\)',text)
    if not bits_match: raise ValueError('First profile supports exact Mersenne inputs only')
    bits=int(bits_match[1]);b1=int(re.search(r'\bB1=(\d+)',text)[1])
    scopes=[r for r in model['stage2'] if r['bits']==bits and r['B1']==b1 and r['usable'] and r['arena_mb']==a.arena_mb]
    if not scopes: raise ValueError('No validated bit-width/B1/arena scope')
    if a.stage1_seconds_per_curve is not None:
        t1=a.stage1_seconds_per_curve;t1_source='explicit_total_workflow_seconds_per_curve'
    else:
        s1=next((r for r in model['stage1'] if r['bits']==bits and r['B1']==b1 and r['batch']==a.stage1_batch and r['torsion']==1),None)
        if s1 is None: raise ValueError('No matching Stage1 batch scope; supply seconds explicitly')
        t1=s1['process_seconds_per_curve'];t1_source='measured_stage1_process_amortized_batch_'+str(a.stage1_batch)
    minimum=max(r['b2_min'] for r in scopes) if a.min_b2 is None else a.min_b2
    maximum=min(r['b2_max'] for r in scopes) if a.max_b2 is None else a.max_b2
    if not b1<minimum<=maximum: raise ValueError('Invalid B2 interval')
    grid={minimum,maximum}
    for i in range(a.grid_points):
        grid.add(round(math.exp(math.log(minimum)+(math.log(maximum)-math.log(minimum))*i/(a.grid_points-1))))
    # Evaluate integer neighbors of the measured giant path switch and G-tree boundaries.
    for scope in scopes:
        for d in scope['d_values']:
            from calibrate_stage2_d import phi
            points=phi(d)//2
            boundaries=[d*(32768-2)]
            boundaries.extend(d*(g*points-2) for g in range(scope['g_min'],scope['g_max']+1))
            for edge in boundaries:
                grid.update(x for x in (edge-1,edge,edge+1) if minimum<=x<=maximum)
    rows=[]
    for scope in scopes:
        for d in scope['d_values']:
            if a.d is not None and d!=a.d: continue
            for b2 in sorted(grid):
                if not scope['b2_min']<=b2<=scope['b2_max']: continue
                stages,f=predict(bits,d,b2,scope['rates'])
                if not scope['p_min']<=f['P']<=scope['p_max'] or not scope['g_min']<=f['G']<=scope['g_max']: continue
                if scope['owner_resident'] and f['owner_bytes']>a.owner_budget_mb*(1<<20): continue
                # A measured fallback is achievable by explicitly forcing owner budget zero.
                t2=a.ratio_adjust*(stages['full']+scope['cold_overhead_seconds'])
                k=kruppa_value(b1,b2)
                rows.append(dict(B2=b2,D=d,owner_resident=scope['owner_resident'],
                    owner_runtime_mb=a.owner_budget_mb if scope['owner_resident'] else 0,
                    owner_bytes=f['owner_bytes'] if scope['owner_resident'] else 0,
                    P=f['P'],I=f['I'],G=f['G'],fold_length=f['fold_ntt'],T1=t1,T2=t2,
                    phases=stages,cold_overhead=scope['cold_overhead_seconds'],K=k,score=k/(t1+t2)))
    if not rows: raise ValueError('No feasible calibrated candidate (no extrapolation permitted)')
    rows.sort(key=lambda r:(-r['score'],r['owner_bytes'],r['T2']))
    import os
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    selected=None;native=None
    for row in rows:
        env.update(NTT_FOLD_DEVICE_MAX_MB=str(row['owner_runtime_mb']),NTT_D_MODEL='0',NTT_NAME_HITS=str(model['name_hits']))
        cmd=[str(a.stage2.resolve()),'--save',str(a.save.resolve()),'--b2',str(row['B2']),
             '--d',str(row['D']),'--device','1','--arena-mb',str(a.arena_mb),'--plan-only']
        result=subprocess.run(cmd,capture_output=True,env=env,timeout=60)
        if result.returncode: raise RuntimeError('Native planning failed: '+result.stderr.decode(errors='replace'))
        native=next(json.loads(l) for l in result.stdout.decode().splitlines() if l.startswith('{'))
        if (native['P'],native['I'],native['G'],native['fold_length'])!=(row['P'],row['I'],row['G'],row['fold_length']):
            raise ValueError('Offline/native geometry mismatch')
        # This remains component admission, not a proven simultaneous VRAM peak.
        if native['arena_estimate_fits'] and (not row['owner_resident'] or native['owner_budget_fits']):
            selected=row;break
    if selected is None: raise ValueError('No candidate passed current native component admission')
    output=dict(schema=1,scope='offline_measured_scope_prototype',executed_curves=0,bits=bits,B1=b1,
        stage1_source=t1_source,profile_sha256=sha(a.profile),binary_sha256=sha(a.stage2),
        benefit_model='kruppa_p95_v1',search_interval=[minimum,maximum],candidate_count=len(rows),
        range_limited=selected['B2'] in (minimum,maximum),chosen=selected,native_plan=native,
        process_peak_guaranteed=False,name_hits=model['name_hits'],ranked=rows[:10])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:output[k] for k in ('scope','bits','B1','stage1_source','range_limited','chosen')},indent=2))


if __name__=='__main__':main()
