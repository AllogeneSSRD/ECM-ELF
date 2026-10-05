"""Read-only provenance and semantic audit of the completed budget matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();d=json.loads((a.study/'measurements.json').read_text())
    assert len(d['runs'])==len(d['cases'])==d['passed'] and d['failed']==0
    assert len(d['sources'])==19 and sha(Path(d['exe']))==d['exe_sha256']
    for name,want in d['sources'].items():assert sha(a.sources/name)==want,name
    probe=d['shape_probe'];probe_path=Path(probe['exe'])
    assert sha(probe_path)==probe['sha256'].lower()
    for name,want in probe['sources'].items():assert want.lower()==d['sources'][name]
    for name,want in probe['generated_sources'].items():assert sha(probe_path.parent/name)==want.lower(),name
    for name,want in probe['local_sources'].items():assert sha(a.repo/name)==want.lower(),name
    driver='tools/bench/bench_stage2_budget_scaling.py'
    assert sha(a.study/'preparation_driver.py')==d['tool_hashes'][driver]
    histories=d['runtime_tool_hashes_history']
    for h in histories:
        assert sha(a.study/('runtime_driver_'+h[driver]+'.py'))==h[driver]
        for name,want in h.items():
            if name!=driver:assert sha(a.repo/name)==want,name
    for bits,save in d['saves'].items():
        sp=Path(save['path']);assert sha(sp)==save['sha256']
        line=sp.read_text();s=int(bits);n=(1<<s)-1
        x=int(re.search(r'\bX=(0x[0-9a-f]+)',line)[1],16)
        checksum=int(re.search(r'CHECKSUM=(\d+)',line)[1])
        gx=int(re.search(r'\bX=(0x[0-9a-fA-F]+)',(a.study/f'gmp_m{bits}.save').read_text())[1],16)
        assert x==gx and checksum==1000*26*n*x%4294967291
        assert hashlib.sha256(f'{x:x}'.encode()).hexdigest()==save['Q_sha256']
    gmp=Path(d['saves']['4423']['gmp_command'][0]);assert sha(gmp)==d['gmp_sha256']
    pairs={};logs={};missing=0
    for i,r in enumerate(d['runs']):
        c=r['case'];assert c==d['cases'][i]
        h=r.get('runtime_tool_hashes',histories[0]);assert h in histories
        log=a.study/(c['name']+'_engine.log');text=log.read_text(encoding='utf-8',errors='replace')
        assert all(t in text for t in ('clean=1','gmp_selftest_bad=0','gmp_check_bad=0','pending=0','fixed=3','point_arithmetic: xadd6=1'))
        assert f"point_mersenne_mode: requested=1 enabled=1 bits={c['bits']} nw={(c['bits']+63)//64}" in text
        assert 'baby_device: requested=1 enabled=1' in text
        assert int(r['s4']['gmp_selftest_bad'])==int(r['s4']['gmp_check_bad'])==int(r['result']['bad_factors'])==0
        assert int(r['oracle']['queued'])==int(r['oracle']['compared']) and int(r['oracle']['pending'])==0
        assert int(r['ntt']['big_peak_bytes'])<=c['big_mb']*(1<<20)
        p=c['plan']['P'];w=(c['bits']+63)//64;owner=8*w*(9*p+8)+48
        assert owner==c['plan']['owner_bytes']
        resident=owner<=c['fold_mb']*(1<<20)
        assert bool(int(r['fold']['enabled']))==resident and int(r['fold']['peak_bytes'])==(owner if resident else 0)
        assert r['fold']['fallback']==('none' if resident else 'budget')
        assert int(r['arena']['arena_overflow'])==0
        i_pts=c['B2']//c['plan']['D']+2
        assert int(r['shape']['giant_points'])==i_pts and int(r['shape']['num_poly_g'])==(i_pts+p-1)//p
        q=re.search(r'real_setup_Q_full: hex=([0-9a-f]+)',text)[1]
        assert hashlib.sha256(q.encode()).hexdigest()==d['saves'][str(c['bits'])]['Q_sha256']
        key=(c['bits'],c['B2'],c['plan']['D'])
        coverage=tuple(r['s4'][k] for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks'))
        coverage+=tuple(r['oracle'][k] for k in ('selected','queued','compared','samples','signature'))
        semantic=(r['leaf']['hash'],r['result']['factors'],coverage)
        if key in pairs:assert pairs[key]==semantic
        pairs[key]=semantic
        # Runtime logs and JSONL must agree; record hashes alone are insufficient.
        saved_result=json.loads((a.study/(c['name']+'.jsonl')).read_text().splitlines()[-1])
        assert saved_result==r['result']
        sampled=json.loads((a.study/(c['name']+'_gpu.json')).read_text())
        valid=[s['used'] for s in sampled['samples'] if s['used'] is not None]
        assert r['device_samples']['observed_max_used']==max(valid,default=None)
        missing+=sum(s['gpu'] is None for s in sampled['samples'])
        logs[log.name]=sha(log)
    if (a.study/'boundary_extension.json').exists():
        e=json.loads((a.study/'boundary_extension.json').read_text())
        assert sha(a.study/'matrix_measurements.json')==e['matrix_measurements_sha256']
        assert sha(a.study/'matrix_plan.json')==e['matrix_plan_sha256']
        assert e['cases']==d['cases'][48:]
    out=dict(passed=len(d['runs']),failed=0,source_count=19,unique_semantic_groups=len(pairs),
             runtime_versions=len(histories),missing_utilization_samples=missing,
             arithmetic_checks='All completed curves: GPU baby and exact-modulus point fold observed enabled; default mandatory checks; GMP bad0, oracle compared=queued, pending0, clean1; same(S,B2,D) leaf/factors/S4 work/GMP-oracle coverage match.',
             source_sha256=sha(Path(__file__)),measurements_sha256=sha(a.study/'measurements.json'),logs=logs,
             memory_scope='Big and owner payload independently bounded; arena overflow0; whole-card sampled maximum separately verified; no summed process peak.')
    a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='logs'},indent=2))


if __name__=='__main__':main()
