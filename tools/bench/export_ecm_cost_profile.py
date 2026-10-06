"""Export a validated JSON phase model to the standalone binary's runtime format."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from ecm_cost_model import FEATURE_PROFILE,scope_id

RATE_KEYS=('baby','affine','ftree','gtrees','fold','descent','inv','accum','glue')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def real(value):
    if not math.isfinite(value) or value<0:raise ValueError('Profile rates must be finite/nonnegative')
    return format(value,'.17g')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--stage2',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model=json.loads(a.profile.read_text(encoding='utf-8'));audit=json.loads(a.audit.read_text(encoding='utf-8'))
    if model['schema']!=2 or model['accounting_version']!=2 or model['feature_profile']!=FEATURE_PROFILE or model['name_hits'] not in (0,1):
        raise ValueError('Unsupported phase model')
    if model['identity']['stage2_sha256']!=sha(a.stage2):raise ValueError('Profile/binary mismatch; measure the new binary first')
    if model['model_code_sha256']!=sha(Path(__file__).with_name('ecm_cost_model.py')):raise ValueError('Cost model changed; refit/validate')
    if audit.get('schema')!=2 or audit['profile_sha256']!=sha(a.profile) or audit['study_sha256']!=model['source_sha256']:
        raise ValueError('Audit/profile identity mismatch')
    if not audit.get('passed') or not audit.get('integrity_passed') or not audit.get('accuracy_passed') or not audit.get('ranking_passed'):
        raise ValueError('Complete scope accuracy and ranking gates are required')
    scopes=[s for s in model['stage2'] if s['usable']]
    if 'validation_scopes' in audit:
        passed={s['id'] for s in audit['validation_scopes'] if s['passed'] and
                s['samples']==s['expected_samples'] and s['max_abs_percent'] is not None and s['max_abs_percent']<=10}
        ranks={bits:all(r['selected_value_loss_percent']<=5 for r in audit['validated_ranking'] if r['bits']==bits)
               for bits in {r['bits'] for r in audit['validated_ranking']}}
        scopes=[s for s in scopes if s['id']==scope_id(s) and s['id'] in passed and ranks.get(s['bits'],False) and
                s['holdout'] and max(abs(r['error_percent']) for r in s['holdout'])<=10]
    else:raise ValueError('Missing per-scope validation')
    if not scopes:raise ValueError('No independently validated scopes')
    included={s['id'] for s in scopes}
    excluded=[dict(id=s['id'],bits=s['bits'],owner_mb=s['owner_mb']) for s in model['stage2'] if s['id'] not in included]
    if excluded:raise ValueError('Refusing partial publication: every declared scope must pass independent validation')
    if a.output.suffix!='.cprof' or a.output.resolve() in {a.profile.resolve(),a.audit.resolve(),a.stage2.resolve()}:
        raise ValueError('Use a separate .cprof output')
    r=subprocess.run([str(a.stage2.resolve()),'--cost-device-info','--device',str(a.device)],capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError('Native device query failed: '+r.stderr.decode(errors='replace'))
    info=json.loads(r.stdout)
    expected=model['device']['uuid'].removeprefix('GPU-').replace('-','')
    if info['uuid_hex']!=expected or info['fixed_mode']!=3 or info['outer_unroll_u']!=0:
        raise ValueError('Current native device/backend differs from measured profile')
    identity=[model['identity']['stage2_sha256'],info['uuid_hex'],info['major'],info['minor'],info['runtime'],info['driver'],
              info['fixed_mode'],info['outer_unroll_u'],2,model['name_hits'],FEATURE_PROFILE,model['chain_min'],sha(a.profile),sha(a.audit)]
    lines=['ECM_STAGE2_COST_PROFILE 2','identity '+' '.join(map(str,identity))]
    for s in model['stage1']:
        lines.append('stage1 '+' '.join(map(str,(s['bits'],s['B1'],s['batch'],s['torsion'])))+' '+real(s['process_seconds_per_curve']))
    for s in scopes:
        fields=[s['bits'],s['B1'],s['arena_mb'],int(s['owner_resident']),s['b2_min'],s['b2_max'],s['p_min'],s['p_max'],s['g_min'],s['g_max'],
                real(s['cold_overhead_seconds']),len(s['d_values']),*s['d_values']]
        rates=[s['rates'][key] for key in RATE_KEYS]+[s['rates']['giant'][key] for key in ('chain','chain_chunk','ladder')]+[
            s['rates'][key] for key in ('descent_local_inverse','gtrees_group','gtrees_copy_word','descent_root_reduce')]+[s['rates']['giant']['ladder_launch']]
        coverage=[int(s['rates']['giant_coverage'][key]) for key in ('chain','chain_chunk','ladder')]
        lines.append('scope '+' '.join(map(str,fields))+' '+' '.join(real(x) for x in rates)+' '+' '.join(map(str,coverage)))
    if not scopes:raise ValueError('No validated scopes')
    lines.append(f'END {len(model["stage1"])} {len(scopes)}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    partial=a.output.with_name(a.output.name+'.partial.'+str(os.getpid()))
    with partial.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    # Atomic publication; a reader's Windows file lock may reject replacement.
    os.replace(partial,a.output)
    print(json.dumps(dict(file=str(a.output),sha256=sha(a.output),stage1_scopes=len(model['stage1']),stage2_scopes=len(scopes),
        excluded_scopes=excluded,device=info),indent=2))


if __name__=='__main__':main()
