"""Merge completed Stage1 cost profiles after offline native validation.

No device queries, curves, fitting or extrapolation. Input measurements are
preserved per scope. Conflicting duplicates require explicit --replace-scopes.
Paths and hashes belong only to the separate evidence JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

SCOPE = ('target_bits','b1','batch','modulus_kind','exponent')
FIELDS = {
    'profile': {'format','algorithm_revision','unit','effort_level','repeats','warmups'},
    'device': {'uuid_hex','sm_major','sm_minor','cuda_runtime','cuda_driver'},
    'policy': {'algorithm','backend','param','requested_tpi','exp_cache'},
    'summary': {'complete','failed','measured'},
    'stage1': {'target_bits','modulus_kind','b1','batch','exponent','benchmark_kind',
               'benchmark_exponent','sigma_first','repeats','seconds','median_seconds',
               'mad_seconds','gpu_seconds','median_gpu_seconds','checked_curves',
               'hits','bad','container_bits','tpi'},
}


def check_fields(profile):
    # Native readers allow extra keys for forward compatibility. The merger
    # must not copy arbitrary identities or paths into a performance file.
    if set(profile) != set(FIELDS):
        raise ValueError('unsupported Stage1 profile sections')
    for section, allowed in FIELDS.items():
        rows = profile[section].values() if section == 'stage1' else [profile[section]]
        for row in rows:
            if set(row) - allowed:
                raise ValueError('unsupported Stage1 performance fields: '+section)


def merge(profiles, replace=False):
    if not profiles:
        raise ValueError('at least one Stage1 profile is required')
    for profile in profiles:
        check_fields(profile)
    first = profiles[0]
    fields = {k:v for k,v in first['profile'].items() if k != 'effort_level'}
    samples = {};replaced = identical = 0
    for profile in profiles:
        if ({k:v for k,v in profile['profile'].items() if k != 'effort_level'} != fields or
                profile['device'] != first['device'] or profile['policy'] != first['policy']):
            raise ValueError('Stage1 format/unit/repeats/device/policy mismatch')
        for sample in profile['stage1'].values():
            scope = tuple(sample[key] for key in SCOPE)
            if scope in samples:
                if samples[scope] == sample:
                    identical += 1;continue
                if not replace:
                    raise ValueError('conflicting Stage1 scope; use --replace-scopes to take the last complete record')
                replaced += 1
            samples[scope] = sample
    if not samples or len(samples)>4096:
        raise ValueError('merged Stage1 scope count exceeds 4096')
    header = dict(first['profile'])
    header['effort_level'] = max(p['profile']['effort_level'] for p in profiles)
    tables = [('profile',header),('device',first['device']),('policy',first['policy'])]
    tables += [('stage1.sample_'+str(i),samples[key]) for i,key in enumerate(sorted(samples))]
    tables += [('summary',dict(complete=1,failed=0,measured=len(samples)))]
    text = '# Merged completed Stage1 costs. Seconds per curve; no extrapolation.\n'
    for name,values in tables:
        text += '\n['+name+']\n'
        for key,value in values.items():
            text += key+' = '+json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
    if len(text.encode('utf-8'))>16*1048576:
        raise ValueError('merged Stage1 profile exceeds 16MiB')
    return text,dict(measured=len(samples),replaced_scopes=replaced,identical_scopes=identical)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2',type=Path,required=True,help='Offline native Stage1 profile validator')
    p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True,help='Destination performance TOML')
    p.add_argument('--evidence',type=Path,required=True,help='New ignored evidence directory')
    p.add_argument('--replace-scopes',action='store_true')
    a = p.parse_args()
    exe = a.stage2.resolve();sources = [path.resolve() for path in a.input]
    destination = a.output.resolve();out = a.evidence.resolve()
    if destination.suffix.lower()!='.toml' or destination==exe or destination in sources:
        p.error('output must be a TOML path different from executable and inputs')
    out.mkdir(parents=True,exist_ok=False)
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path):sha(path) for path in [exe,Path(__file__).resolve()]+sources}
    report = dict(complete=False,identities=identities,replace_scopes=a.replace_scopes,
                  gpu_queries=0,curves=0)
    def publish():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    publish()
    try:
        def validate(path,label):
            proc = subprocess.run([str(exe),'--check-stage1-tune-profile',str(path)],cwd=out,
                capture_output=True,text=True,errors='replace',timeout=60)
            (out/(label+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode:
                raise ValueError('native Stage1 profile rejected: '+label)
        profiles = []
        for i,path in enumerate(sources):
            validate(path,'input_'+str(i))
            if path.stat().st_size>16*1048576:
                raise ValueError('input exceeds 16MiB')
            profiles.append(tomllib.loads(path.read_text(encoding='utf-8-sig')))
            frozen = out/'inputs'/identities[str(path)]/path.name
            frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(path.read_bytes())
        text,stats = merge(profiles,a.replace_scopes)
        destination.parent.mkdir(parents=True,exist_ok=True)
        partial = destination.with_name(destination.name+f'.partial.{os.getpid()}')
        partial.write_text(text,encoding='utf-8')
        validate(partial,'merged')
        if any(sha(Path(path))!=expected for path,expected in identities.items()):
            raise ValueError('measurement input or validator changed; destination not replaced')
        os.replace(partial,destination)
        report.update(complete=True,output_sha256=sha(destination),**stats);publish()
    except BaseException as error:
        report['failure']=repr(error);publish();raise
    print(json.dumps({key:value for key,value in report.items() if key!='identities'}))


if __name__=='__main__':
    main()
