"""Reject stale giant-route calibration using native production plans (no curves)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from stage2_tune_route_cost import annotate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['exe','profile','save','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    paths=[a.exe.resolve(),a.profile.resolve(),a.save.resolve()]
    identities={str(f):sha(f) for f in paths}
    data=tomllib.loads(a.profile.read_text(encoding='utf-8-sig'))
    ini=out/'ecm.ini';ini.write_text('verbose=false\nstage2_debug_log=false\n',encoding='utf-8')
    common=[str(a.exe.resolve()),'--ini',str(ini),'--save',str(a.save.resolve()),'--device',str(a.device),
        '--batch-mb',str(data['policy']['batch_mb']),'--arena-mb',str(data['policy']['arena_mb']),
        '--owner-budget-mb',str(data['policy']['fold_mb']),'--plan-only','--log-level','quiet']
    b2=min(s['b2'] for s in data['ecm'].values())
    calls=0
    for mode in ['valid','chunk','minimum','force']:
        candidate=json.loads(json.dumps(data))
        for key,s in candidate['ecm'].items():
            candidate['ecm'][key]=annotate(s,s['giant_chunk_points']*(2 if mode=='chunk' else 1),
                s['giant_chain_min']+(1 if mode=='minimum' else 0),mode=='force')
            # Synthetic policy rewrites are no longer the original grid route.
            # Keep their source neutral so refusal reaches the runtime gate.
            if mode!='valid' and 'sampling_source' in s:candidate['ecm'][key]['sampling_source']='base'
        sections=[('profile',candidate['profile']),('device',candidate['device']),
            ('policy',{k:v for k,v in candidate['policy'].items() if k!='environment'}),
            ('policy.environment',candidate['policy']['environment'])]
        sections += [('ecm.'+k,s) for k,s in candidate['ecm'].items()]
        sections += [('summary',candidate['summary'])]
        profile=out/(mode+'.toml')
        profile.write_text(''.join('\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in f.items())
            for name,f in sections),encoding='utf-8')
        for automatic in [False,True]:
            extra=['--auto-b2','--stage1-seconds-per-curve','1','--auto-min-b2',str(b2),'--auto-max-b2',str(b2)] if automatic else ['--b2',str(b2)]
            proc=subprocess.run(common+['--tune-profile',str(profile),*extra],capture_output=True,text=True,errors='replace',timeout=120)
            (out/(mode+('_auto' if automatic else '_fixed')+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            calls+=1
            if mode=='valid':
                assert proc.returncode==0,(mode,automatic,proc.stderr)
                rows=[json.loads(s) for s in proc.stdout.splitlines() if s.startswith('{')]
                choice=next(s for s in rows if s.get('type') in ['tune_selection','stage2_auto_plan'])
                assert choice.get('selected',True) and choice['route_rejected']==0
            elif automatic:
                assert proc.returncode!=0 and 'current giant policy' in proc.stderr,(mode,proc.stdout,proc.stderr)
            else:
                assert proc.returncode==0,(mode,proc.stderr)
                choice=next(json.loads(s) for s in proc.stdout.splitlines() if s.startswith('{') and json.loads(s).get('type')=='tune_selection')
                assert not choice['selected'] and choice['reason']=='no_candidate_matches_current_giant_policy',choice
    assert all(sha(Path(k))==v for k,v in identities.items())
    result=dict(complete=True,calls=calls,full_curves=0,identities=identities,synthetic_policy_refusals=6)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
