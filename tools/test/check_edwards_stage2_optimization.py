"""Independent completion/evidence audit for the GPU curve-family build matrix."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads((a.study/'plan.json').read_text())
    if not data['complete']:raise ValueError('unfinished study')
    if sha(a.study/'collector_snapshot.py')!=data['collector_sha256']:raise ValueError('collection source snapshot changed')
    proof=[]
    if any(c['carrier'] for c in data['cells']):
        witness=data['carrier_proof']
        if sha(witness['path'])!=witness['sha256']:raise ValueError('carrier proof changed')
        document=json.loads(Path(witness['path']).read_text())
        if sha(Path(witness['path']).parent/'collector_snapshot.py')!=document['test_sha256']:
            raise ValueError('carrier oracle source snapshot changed')
        if not document['complete'] or document['performance_samples']!=0 or document['binaries']!={k:v['sha256'] for k,v in data['binaries'].items()}:
            raise ValueError('invalid carrier proof identity')
        reused=data['reused_inputs'];source=Path(reused['path'])/'plan.json'
        if sha(source)!=reused['sha256'] or document['study_sha256']!=reused['sha256']:
            raise ValueError('carrier proof input provenance changed')
        source_plan=json.loads(source.read_text())
        for build,binary in data['binaries'].items():
            if sha(Path(source_plan['binaries'][build]['path']).parent/'gmp-10.dll')!=binary['runtime_files']['gmp-10.dll']:
                raise ValueError('carrier proof used a different runtime')
        proof=document['runs']
        for row in proof:
            for name,digest in row['files'].items():
                if sha(name)!=digest:raise ValueError('carrier oracle evidence changed')
            path=next(Path(f) for f in row['files'] if Path(f).name=='internal.jsonl')
            native=json.loads(path.read_text())
            identity=dict(N_hex=row['N_hex'],B1=row['b1'],B2=row['b2'],sigma=row['sigma'],
                          curve=row['model'],d=row['d'],carrier_exponent=row['carrier'])
            if any(native[key]!=value for key,value in identity.items()) or native['status']!='stage2_completed' or native['factors'] or native['bad_factors']:
                raise ValueError('carrier oracle native work mismatch')
            if row['build']=='candidate' and row['chain_points_checked']!=row['b2']//row['d']+2:
                raise ValueError('carrier point oracle coverage missing')
            if not all(int(v)>0 for v in (row['groups']['checked_groups'],row['groups']['checked_leaf_words'],
                    row['seed']['segment_checks'],row['scale']['checked_words'])):
                raise ValueError('missing carrier group/segment/scale checks')
            if int(row['s4']['gmp_selftest_bad']) or int(row['s4']['gmp_check_bad']):raise ValueError('carrier GMP failure')
    ids=set();observed=0
    for binary in data['binaries'].values():
        exe=Path(binary['path'])
        if sha(exe)!=binary['sha256']:raise ValueError('frozen binary changed')
        for name,digest in binary['runtime_files'].items():
            if sha(exe.parent/name)!=digest:raise ValueError('frozen runtime changed')
        for relative,digest in binary['identity']['sources'].items():
            if sha(exe.parent/'sources'/relative)!=digest:raise ValueError('frozen compiled source changed')
    for row in data['runs']:
        if row['id'] in ids:raise ValueError('duplicate invocation')
        ids.add(row['id'])
        if row.get('imported_from'):
            source=row['imported_from'];path=Path(source['study'])/'plan.json'
            if row['category']!='warmup' or sha(path)!=source['sha256']:raise ValueError('invalid warmup provenance')
            original=next(r for r in json.loads(path.read_text())['runs'] if r['id']==row['id'])
            if original['stats']!=row['stats'] or original['engine_seconds']!=row['engine_seconds']:
                raise ValueError('imported warmup changed')
            if {Path(f).name:h for f,h in original['files'].items()}!={Path(f).name:h for f,h in row['files'].items()}:
                raise ValueError('imported warmup evidence differs')
        for name,digest in row['files'].items():
            if sha(name)!=digest:raise ValueError('raw invocation evidence changed')
        result_path=next(Path(f) for f in row['files'] if Path(f).name=='internal.jsonl')
        result=json.loads(result_path.read_text());cell=next(c for c in data['cells'] if c['id']==row['cell'])
        case=data['cases'][str(cell['exponent'])]
        if result['N_hex']!=case['N_hex'] or result['sigma']!=case['sigma'] or result['B1']!=case['b1']:
            raise ValueError('curve/target identity mismatch')
        if result['B2']!=cell['b2'] or result['d']!=cell['d'] or result['curve']!=row['model'] or result['carrier_exponent']!=cell['carrier']:
            raise ValueError('Stage2 work changed')
        if result['factors'] or result['bad_factors'] or result['status']!='stage2_completed':raise ValueError('not complete NF')
        ledger=row['stats']['stage2_full_wall'];checks=row['stats']['s4_multiply_stats']
        if ledger['clean']!='1' or abs(float(ledger['total'])-float(ledger['init'])-float(ledger['main']))>.003:raise ValueError('unclosed wall ledger')
        if int(checks['gmp_selftest_bad']) or int(checks['gmp_check_bad']) or int(checks['gmp_checked'])<=0 or int(checks['gmp_selftest_cases'])<=0:
            raise ValueError('missing/failed GMP checks')
        observed+=int(checks['gmp_checked'])
    formal=warm=0
    for cell in data['cells']:
        case=data['cases'][str(cell['exponent'])]
        for model in ('ed','mont'):
            if sha(case['saves'][model])!=case['save_hashes'][model]:raise ValueError('Stage1 input changed')
            pair=[r for r in data['runs'] if r['cell']==cell['id'] and r['model']==model]
            if cell['carrier']:
                ordinary=next(c for c in data['cells'] if not c['carrier'] and c['exponent']==cell['exponent'] and c['b2']==cell['b2'] and c['d']==cell['d'])
                canonical=next(r for r in data['runs'] if r['cell']==ordinary['id'] and r['model']==model)['stats']['descent_values']
                witnesses=[r for r in proof if r['cell']==cell['id'] and r['model']==model]
                if len(witnesses)!=2 or {r['build'] for r in witnesses}!={'baseline','candidate'}:
                    raise ValueError('missing carrier witnesses')
                for witness in witnesses:
                    if any(witness['target'][key]!=canonical[key] for key in ('leaves','words','hash')):
                        raise ValueError('carrier/ordinary target-ring leaf mismatch')
                    identity=dict(N_hex=case['N_hex'],sigma=case['sigma'],b1=case['b1'],b2=cell['b2'],d=cell['d'],carrier=cell['carrier'])
                    if any(witness[key]!=value for key,value in identity.items()):raise ValueError('carrier oracle work changed')
                    if any(r['stats']['descent_values']!=witness['raw'] for r in pair if r['build']==witness['build']):
                        raise ValueError('carrier raw leaves differ from checked projection')
            elif len({json.dumps(r['stats']['descent_values'],sort_keys=True) for r in pair})!=1:
                raise ValueError('complete ordinary leaf fingerprint changed across builds/repetitions')
            for key in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'):
                if len({r['stats']['s4_multiply_stats'][key] for r in pair})!=1:
                    raise ValueError('arithmetic/check coverage changed: '+key)
            for build in ('baseline','candidate'):
                rows=[r for r in pair if r['build']==build]
                timed=[r for r in rows if r['category']=='timing'];preheat=[r for r in rows if r['category']=='warmup']
                if {r['repeat'] for r in timed}!=set(range(1,data['repeats']+1)) or len(timed)!=data['repeats'] or len(preheat)!=1:
                    raise ValueError('missing formal/warmup sample')
                formal+=len(timed);warm+=len(preheat)
    expected=len(data['cells'])*4*(data['repeats']+1)
    if len(ids)!=expected:raise ValueError('unexpected invocation count')
    report=dict(complete=True,study_sha256=sha(a.study/'plan.json'),audit_sha256=sha(__file__),
                total=len(ids),formal=formal,warmup=warm,gmp_checked=observed,
                purpose='offline mathematical and protocol validation; not server acceptance')
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print('PASS',len(ids),'complete NF calls;',formal,'formal samples; unchanged arithmetic coverage')


if __name__=='__main__':main()
