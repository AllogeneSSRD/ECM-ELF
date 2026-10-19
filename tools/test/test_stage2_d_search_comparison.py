"""Check the default-matrix comparison rules without a GPU or timed workload."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from compare_stage2_d_search import compare, repeat_totals, paired_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); before = {}
    for i in range(13):
        for bound in (1000, 10000):
            for engine in ('gpu', 'gpu-carrier'):
                key = ('case' + str(i), bound, engine)
                before[key] = dict(case=key[0], bits=100+i, exponent=200+i, B1=10, sigma=26,
                    B2=bound, nominal_B2=bound, engine=engine, N_hex='123', save_bytes=b'point',
                    seconds=10., min_seconds=9., max_seconds=11., process_seconds=12.,
                    D=30, P=4, carrier=0 if engine == 'gpu' else 200+i)
    after = copy.deepcopy(before); key = next(iter(after))
    assert len(compare(before, after)) == 52
    after[key].update(seconds=10.6)
    assert next(r for r in compare(before, after) if r['case'] == key[0] and r['B2'] == key[1] and r['engine'] == key[2])['retest_required']
    after[key].update(seconds=.4); before[key].update(seconds=.3)
    assert not any(r['retest_required'] for r in compare(before, after)), 'Tiny absolute delta treated as a clear regression'
    before[key].update(seconds=10.); after[key].update(seconds=10.4)
    assert not any(r['retest_required'] for r in compare(before, after)), 'Small relative delta treated as a clear regression'
    after[key]['N_hex'] = '456'
    try:
        compare(before, after)
    except ValueError:
        pass
    else:
        raise AssertionError('Different modulus accepted')
    after = copy.deepcopy(before); after.pop(key)
    try:
        compare(before, after)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete matrix accepted')
    runs = [dict(repeat=1, engine='gpu', engine_seconds=2), dict(repeat=1, engine='gpu-carrier', engine_seconds=3),
            dict(repeat=2, engine='gpu', engine_seconds=7), dict(repeat=2, engine='gpu-carrier', engine_seconds=8)]
    assert repeat_totals(dict(runs=runs), 'gpu') == [2., 7.]
    assert repeat_totals(dict(runs=runs), 'all') == [5., 15.]
    a.output.mkdir(parents=True, exist_ok=False)
    study=a.output/'paired';study.mkdir();plans={};schedule=[]
    for arm,version in [('baseline','0.38.4'),('candidate','0.39.2')]:
        folder=study/arm;folder.mkdir();exe=folder/'fake.exe';exe.write_bytes(arm.encode())
        plans[arm]=dict(program_version=version,repeats=1,exe=str(exe),runs=[dict(id=str(i)) for i in range(52)])
    for i in range(52):
        for arm in (('baseline','candidate') if i%2 else ('candidate','baseline')):schedule.append(dict(arm=arm,id=str(i)))
    record=dict(complete=True,failures=[],repeats=1,versions={k:v['program_version'] for k,v in plans.items()},
                exe_sha256={k:hashlib.sha256(Path(v['exe']).read_bytes()).hexdigest() for k,v in plans.items()},
                schedule=schedule,order='Adjacent versions; reverse per condition')
    def paired(data,baseline=None):
        (study/'pair.json').write_text(json.dumps(data),encoding='utf-8')
        return paired_protocol(study,baseline or study/'baseline',study/'candidate',plans['baseline'],plans['candidate'])
    assert paired(record)['retained_invocations']==104
    for field,value in [('complete',False),('failures',['failure']),('repeats',2),('schedule',schedule[:-1]),
                        ('versions',{'baseline':'wrong','candidate':'0.39.2'}),
                        ('exe_sha256',{'baseline':'wrong','candidate':record['exe_sha256']['candidate']})]:
        bad=copy.deepcopy(record);bad[field]=value
        try:paired(bad)
        except ValueError:pass
        else:raise AssertionError('Bad paired metadata accepted: '+field)
    bad=copy.deepcopy(record);bad['schedule'][0],bad['schedule'][2]=bad['schedule'][2],bad['schedule'][0]
    try:paired(bad)
    except ValueError:pass
    else:raise AssertionError('Nonadjacent schedule accepted')
    try:paired(record,study/'unrelated')
    except ValueError:pass
    else:raise AssertionError('Unrelated study accepted as paired')
    report = dict(complete=True, checks=17, exact_identity=True, full_scope=True,paired_metadata=True,
                  absolute_and_relative_retest_thresholds=True, CUDA_calls=0)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
