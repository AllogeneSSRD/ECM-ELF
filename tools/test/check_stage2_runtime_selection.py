"""Independently check receipt-derived statistics and absence of a dominated winner."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spread(values):
    m=statistics.median(values)
    return m,statistics.median(abs(x-m) for x in values)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.study.resolve();assert not a.output.exists()
    report=json.loads((folder/'result.json').read_text(encoding='utf-8'));assert report['complete']
    rows=[json.loads(s) for s in (folder/'internal.jsonl').read_text(encoding='utf-8').splitlines()]
    history={};ids=set();measured=0;observations=0
    for row in rows:
        assert row['status']=='stage2_completed' and not row['factors'] and not row['bad_factors']
        plan=row['tune_plan'];current={}
        for c in plan['runtime_candidates']:
            key=c['D'],c['carrier_exponent'];values=history.get(key,[])
            assert c['samples']==len(values),(key,c,len(values))
            if values:
                m,mad=spread(values)
                assert abs(m-c['median_seconds'])<1e-8 and abs(mad-c['mad_seconds'])<1e-8
                current[key]=(m,mad)
        if not plan['runtime_probe']:
            key=row['d'],row['carrier_exponent'];assert len(history[key])>=2
            m,mad=current[key];lower=m-2*1.4826*mad
            assert all(v+2*1.4826*d>=.98*lower for v,d in current.values()),'selected path is measurably dominated'
            measured+=1
        obs=row.get('tune_observation')
        if obs:
            s=obs['sample'];identifier=s['runtime_sample_id'];assert identifier not in ids;ids.add(identifier)
            assert s['clean']==1 and not s['bad'] and not s['hits'] and s['checked'] and s['selftest_cases']
            assert s['repeats']==1 and s['median_seconds']>0
            assert abs(s['init_seconds']+s['main_seconds']-s['median_seconds'])<1e-7
            assert s['b2']==row['B2'] and s['d']==row['d'] and s['carrier_exponent']==row['carrier_exponent']
            history.setdefault((s['d'],s['carrier_exponent']),[]).append(s['median_seconds']);observations+=1
    stats=[]
    for line in (folder/'engine.log').read_text(encoding='utf-8').splitlines():
        if line.startswith('s4_multiply_stats: '):
            s=dict(re.findall(r'(\w+)=(\S+)',line))
            assert int(s['gmp_checked'])>0 and int(s['gmp_selftest_cases'])>0
            assert int(s['gmp_check_bad'])==int(s['gmp_selftest_bad'])==0
            stats.append(s)
    assert len(stats)==len(rows)
    result=dict(complete=True,curves=len(rows),independent_observations=observations,
        checked_measured_choices=measured,receipt_statistics_match=True,no_measurably_dominated_choice=True,
        gmp_checked=sum(int(s['gmp_checked']) for s in stats),arithmetic_errors=0,
        raw_sha256={str(f.relative_to(folder)):sha(f) for f in folder.rglob('*') if f.is_file()},
        checker_sha256=sha(__file__))
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='raw_sha256'}))
if __name__=='__main__':main()
