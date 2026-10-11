"""Check frozen benchmark imports without modifying inputs or running GPU work."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from import_stage2_benchmark_tune import import_study,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--condition-profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    path=args.study.resolve()/'measurements.json';data=json.loads(path.read_text(encoding='utf-8'))
    report=json.loads(args.analysis.read_text(encoding='utf-8'))
    assert report['complete'] and Path(report['study']).resolve()==path.parent
    conditions=tomllib.loads(args.condition_profile.read_text(encoding='utf-8-sig'))['condition'].values()
    condition=next(c for c in conditions if c['arena_mb']==data['memory']['arena_mb'] and
        c['fold_mb']==data['memory']['fold_mb'] and c['batch_mb']==data['memory']['batch_mb'])
    audit={str(Path(r['log'])):r for r in report['raw_artifact_hashes']}
    before=sha(path);samples=import_study(path,condition,audit)
    assert samples and sum(s['repeats'] for s in samples)==sum(r['category'] in ('timing','control') and
        not r['result']['hits'] for r in data['runs'])
    assert all(s['source']=='measured' and s['median_seconds']>0 and s['clean']==1 for s in samples)
    formal=next(i for i,r in enumerate(data['runs']) if r['category']=='timing')
    mutations={
        'incomplete':lambda d:d.update(complete=False),
        'failures':lambda d:d.update(failures=[{'reason':'fixture'}]),
        'binary':lambda d:d.update(executable_sha256='broken'),
        'dll':lambda d:d.update(dll_sha256='broken'),
        'save':lambda d:d['cases'][0].update(save_sha256='broken'),
        'log':lambda d:d['runs'][formal].update(log_sha256='broken'),
        'result':lambda d:d['runs'][formal].update(result_sha256='broken'),
        'identity':lambda d:d['runs'][formal]['result'].update(B1=999),
        'arithmetic':lambda d:d['runs'][formal]['s4'].update(gmp_check_bad='1'),
        'clean':lambda d:d['runs'][formal]['wall'].update(clean='0'),
    }
    rejected=[]
    def reject(name,document,cond,records):
        fixture=out/(name+'.json');fixture.write_text(json.dumps(document),encoding='utf-8')
        try:import_study(fixture,cond,records)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid import accepted: '+name)
    for name,change in mutations.items():
        altered=copy.deepcopy(data);change(altered);reject(name,altered,condition,audit)
    for name,key in [('budget','batch_mb'),('gpu','uuid_hex'),('backend','gl_fixed_mode')]:
        altered=copy.deepcopy(condition);altered[key]='broken';reject(name,data,altered,audit)
    altered=copy.deepcopy(audit)
    altered[str(Path(data['runs'][formal]['log']))]['debug_sha256']='broken'
    reject('debug',data,condition,altered)
    assert sha(path)==before
    receipt=dict(passed=True,imported=len(samples),rejected=rejected,manifest_sha256=before,
                 importer_sha256=sha(ROOT/'tools/bench/import_stage2_benchmark_tune.py'))
    (out/'report.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
