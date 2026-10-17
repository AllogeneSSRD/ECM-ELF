"""Audit benchmark completeness, exact workload identity and frozen evidence.

Print only aggregate results; account fields and result checksums stay private.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();plan=load(a.study/'plan.json')
    require(plan['prepared'],'Input preparation incomplete')
    require(plan['complete'] or a.allow_partial,'Measurement matrix incomplete')
    require(not plan['failures'],'Recorded measurement failures require review')
    require(len(plan['engines'])==len(set(plan['engines'])),'Duplicate engines')
    for binary in plan['binaries'].values():
        require(sha(binary['path'])==binary['sha256'],'Frozen binary changed')
    require(sha(a.study/'bin/gmp-10.dll')==plan['dll_sha256'],'Frozen GMP changed')
    cases={c['id']:c for c in plan['cases']}
    require(len(cases)==len(plan['cases']),'Duplicate input cases')
    for c in cases.values():
        n=int(c['N_hex'],16);cofactor=(1<<c['exponent'])-1
        for f in c['known']:
            f=int(f);require(f>1 and cofactor%f==0,'Known factor does not divide target');cofactor//=f
        require(cofactor==n and n.bit_length()==c['bits'],'Recovered modulus differs')
        require(sha(c['save'])==c['save_sha256'],'Verified save changed')
    expected=set()
    for ref in plan['prime95_rows']:
        if ref['curve_type']!='Edwards' or not ref['benchmark_eligible']:continue
        matches=[c for c in cases.values() if c['N_hex']==ref['N_hex'] and c['sigma']==int(ref['s']) and c['b1']==ref['b1']]
        require(len(matches)==1,'Ambiguous reference workload')
        c=matches[0]
        require(math.isclose(ref['s2_total_time'],sum(ref[k] for k in ('s2_init_time','s2_time','s2_gcd_time')),abs_tol=0.003),
                'Prime95 full Stage2 boundary differs')
        for engine in plan['engines']:
            for rep in range(1,plan['repeats']+1):expected.add(f'{c["id"]}_b{ref["b2"]}_{engine}_r{rep}')
    seen=set();ntt_counts=[];cpu_occupancy=[]
    for row in plan['runs']:
        require(row['id'] in expected and row['id'] not in seen,'Unexpected or duplicate invocation');seen.add(row['id'])
        prefix=Path(row['prefix']);c=cases[row['case']];n=int(c['N_hex'],16)
        require(sha(prefix.with_suffix('.jsonl'))==row['result_sha256'],'Raw result changed')
        frozen=a.study/('collector_'+row['script_sha256']+'.py')
        require(sha(frozen)==row['script_sha256'],'Frozen collector changed')
        result=load(prefix.with_suffix('.jsonl'))
        factors=list(map(int,result.get('factors',[])))
        require(all(1<f<n and n%f==0 for f in factors),'Invalid factor')
        require(row['full_sample']==(not factors),'Factor sample classification differs')
        require(row['engine_seconds']>0 and row['process_seconds']>0,'Nonpositive time')
        require(prefix.with_suffix('.driver.log').exists(),'Missing invocation log')
        if row['engine'].startswith('gpu'):
            require(result['N_hex']==c['N_hex'] and result['B1']==c['b1'] and result['B2']==row['B2'],'GPU workload differs')
            carrier=c['exponent'] if row['engine']=='gpu-carrier' else 0
            require(result['carrier_exponent']==carrier and not result['bad_factors'],'GPU route/result differs')
            phase=row['phase_seconds'];s4=row['s4']
            require(phase['clean']=='1' and abs(float(phase['init'])+float(phase['main'])-float(phase['total']))<=.003,'GPU timing ledger differs')
            require(int(s4['gmp_checked'])>0 and int(s4['gmp_selftest_cases'])>0 and not int(s4['gmp_check_bad']) and not int(s4['gmp_selftest_bad']),
                    'Missing or failed arithmetic checks')
            require(prefix.with_suffix('.telemetry.json').exists(),'Missing GPU telemetry')
        else:
            backend=row['engine'].removeprefix('cpu-')
            require(result['b1']==c['b1'] and result['b2']==row['B2'] and result['Edwards']['sigma']==c['sigma'],'CPU workload differs')
            require(result['exponent']==c['exponent'] and sorted(map(int,result['known-factors']))==sorted(map(int,c['known'])),
                    'CPU modulus metadata differs')
            require(result['stage2-engine']['backend']==backend and result['stage2-engine']['seconds']==row['engine_seconds'],
                    'CPU result backend/time differs')
            require(not result['errors']['invariant'] and (bool(factors) or result['status']=='NF'),'CPU completion/invariants differ')
            log=prefix.with_suffix('.driver.log').read_text(encoding='utf-8')
            require(f'backend={backend} threads={plan["thread_limit"]} memory_mib={plan["cpu_memory_mib"]}' in log,'CPU runtime configuration differs')
            counts=re.findall(r'Using (\d+) small primes for NTT',log)
            require(bool(counts)==(backend=='ntt'),'CPU NTT/plain execution differs from requested backend')
            ntt_counts.extend(map(int,counts));cpu_occupancy.append(row['parent_cpu_wall_ratio'])
            require(row['cpu_affinity_mask']==plan['cpu_affinity_mask'],'CPU affinity differs')
    require(not plan['complete'] or seen==expected,'Completed study lacks declared samples')
    report=dict(complete=plan['complete'],expected_invocations=len(expected),accepted_invocations=len(seen),
                missing_invocations=len(expected-seen),engine_counts=dict(Counter(r['engine'] for r in plan['runs'])),
                full_nf_samples=sum(r['full_sample'] for r in plan['runs']),factor_samples=sum(not r['full_sample'] for r in plan['runs']),
                cpu_ntt_small_prime_range=[min(ntt_counts),max(ntt_counts)] if ntt_counts else None,
                cpu_time_wall_range=[min(cpu_occupancy),max(cpu_occupancy)] if cpu_occupancy else None,
                study_sha256=sha(a.study/'plan.json'),audit_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    require(not a.output.exists(),'Audit output already exists')
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS frozen evidence and exact workloads:',len(seen),'of',len(expected),'invocations')


if __name__=='__main__':main()
