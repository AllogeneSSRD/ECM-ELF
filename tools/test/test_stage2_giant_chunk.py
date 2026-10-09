"""Audit whole-batch point budgets against finished native arithmetic evidence.

Reads completed same-binary matrices; launches no GPU work. Budget means X/Z
payload, not complete device usage. Timing rows must have the ledger disabled.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import freeze,sha,fields
from stage2_memory_ledger import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--matrix',type=Path,nargs='+',required=True)
    p.add_argument('--previous-check',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();identity=freeze(a.exe.resolve())
    for name,want in identity['sources'].items():
        if sha(ROOT/name)!=want:raise ValueError('current compiled source changed: '+name)
    previous=[]
    for path in a.previous_check:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not data['complete'] or data['mode']!='check':raise ValueError('previous leaf reference incomplete')
        for row in data['runs']:
            if sha(row['debug_log'])!=row['debug_sha256']:raise ValueError('previous raw leaf evidence changed')
        previous.append(data)
    verified=[]
    for path in a.matrix:
        data=json.loads(path.read_text(encoding='utf-8'))
        if not data['complete'] or data['identity']!=identity or data['comparison']!='giant-chunk':
            raise ValueError('matching completed point-budget matrix required')
        if sha(path.parent/'collector.py')!=data['tool_sha256'] or sha(data['input']['save'])!=data['input']['save_sha256']:
            raise ValueError('collector snapshot or save changed')
        rows=data['runs'];keys=('legacy_points','bounded_points')
        if data['mode']=='timing':
            if data['memory_ledger']:raise ValueError('instrumented rows cannot be formal timing')
            if [r['key'] for r in rows if r['category']=='timing']!=[keys[i] for i in (0,1,1,0,1,0,0,1)]:
                raise ValueError('formal timing sequence changed')
        control=rows[0]['environment']
        for row in rows:
            for name,want in row['environment'].items():
                if name!='NTT_GIANT_CHUNK_FLOOR' and control.get(name)!=want:
                    raise ValueError('non-point policy changed between arms')
            debug=Path(row['debug_log'])
            if sha(debug)!=row['debug_sha256'] or sha(row['log'])!=row['log_sha256']:
                raise ValueError('raw evidence changed')
            text=debug.read_text(encoding='utf-8');plan=fields(text,'giant_chunk_plan');done=fields(text,'giant_chunk_done')
            if plan!=row['point_plan'] or done!=row['point_done']:raise ValueError('retained point evidence differs')
            P,nw,budget,points=(int(plan[k]) for k in ('P','nw','budget_bytes','points'))
            floor=row['key']=='bounded_points';batch_bytes=16*nw*P
            k=budget//(16*nw)
            expected=P*max(1,k//P+(not floor and k%P!=0))
            if (points!=expected or plan['floor_requested']!=str(int(floor)) or
                int(plan['coordinate_cap_bytes'])!=16*nw*points or
                bool(int(plan['minimum_over_budget']))!=(batch_bytes>budget) or
                int(done['chunks'])!=int(done['points'])//points+(int(done['points'])%points!=0)):
                raise ValueError('whole-batch coordinate budget contract failed')
            if floor and batch_bytes<=budget and int(plan['coordinate_cap_bytes'])>budget:
                raise ValueError('floor exceeded its feasible coordinate budget')
            if any(int(row['coverage'][k]) for k in ('gmp_selftest_bad','gmp_check_bad')):
                raise ValueError('native arithmetic mismatch')
            if data['workspace_fixture'] and fields(text,'giant_chunk_check')!={'checks':'298','bad':'0'}:
                raise ValueError('point boundary/overflow fixture missing')
            if data['mode']=='check' and data['oracle'] and data['oracle']['unit']:
                if row['leaf']!={k:str(v) for k,v in data['oracle']['expected_leaf'].items()}:
                    raise ValueError('independent full target leaf oracle differs')
                if not data['projection_only']:
                    chains=[fields(line,'giant_chain_check') for line in text.splitlines()
                            if line.startswith('giant_chain_check:')]
                    if len(chains)!=int(done['chain_chunks']) or any(int(c['mismatches']) for c in chains):
                        raise ValueError('independent affine chain/ladder checks differ')
                    if chains and not int(row['giant_seed']['checked_words']):
                        raise ValueError('paired seed checks did not execute')
            for prior in previous:
                if (prior['input']['save_sha256'],prior['input']['D'],prior['input']['B2'])==(
                    data['input']['save_sha256'],data['input']['D'],data['input']['B2']) and data['mode']=='check':
                    if any(row['leaf']!=r['leaf'] for r in prior['runs']):
                        raise ValueError('complete target leaves differ from previous validated high-width build')
            entry=dict(matrix=str(path),name=row['name'],plan=plan,done=done)
            if data['memory_ledger']:
                ledger=parse(text)
                if ledger!=row['memory_ledger']:raise ValueError('allocation ledger representation changed')
                entry['peak_owned_bytes']=int(ledger['final']['peak_bytes'])
            verified.append(entry)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(complete=True,identity=identity,tool_sha256=sha(__file__),
        matrices={str(p):sha(p) for p in a.matrix},previous_checks={str(p):sha(p) for p in a.previous_check},
        verified=verified),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,matrices=len(a.matrix),runs=len(verified))))


if __name__=='__main__':main()
