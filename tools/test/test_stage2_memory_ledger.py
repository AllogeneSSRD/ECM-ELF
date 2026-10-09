"""Check allocator-ledger parsing, native lifecycle conservation and source coverage.

Consumes finished correctness matrices; performs no additional GPU work.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from bench_stage2_production import freeze,sha
from stage2_memory_ledger import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--matrix',type=Path,nargs='+',required=True)
    p.add_argument('--fallback-dir',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();identity=freeze(a.exe.resolve())
    def text(records,sites):
        return '\n'.join(['stage2_memory_ledger: '+' '.join(f'{k}={v}' for k,v in r.items()) for r in records]+
                         ['stage2_memory_site: '+' '.join(f'{k}={v}' for k,v in s.items()) for s in sites])
    first=dict(snapshot='before_baby',live_bytes=100,peak_bytes=100,allocations=0,frees=0,
        failed_allocations=0,unknown_frees=0,live_allocations=1,persistent_bytes=100,
        persistent_allocations=1,baseline_bytes=100,baseline_allocations=1,interval_peak_bytes=100,payload_only=1,version=1)
    final=first|dict(snapshot='final',peak_bytes=150,allocations=1,frees=1,interval_peak_bytes=150)
    records=[first,final]
    sites=[dict(snapshot=n,scope=scope,site='cache:1',bytes=100)
           for n in ('before_baby','final') for scope in ('live','interval_peak')]
    sites.extend([dict(snapshot='final',scope='interval_peak',site='scratch:2',bytes=50),
        dict(snapshot='final',scope='global_peak',site='cache:1',bytes=100),
        dict(snapshot='final',scope='global_peak',site='scratch:2',bytes=50)])
    parse(text(records,sites));unit_checks=1
    for key,value in [('unknown_frees',1),('persistent_bytes',99),('persistent_allocations',0),
                      ('baseline_allocations',0),('peak_bytes',149),('interval_peak_bytes',149),('version',2),('payload_only',0)]:
        bad=copy.deepcopy(records);bad[-1][key]=value
        try:parse(text(bad,sites))
        except ValueError:unit_checks+=1
        else:raise ValueError('parser accepted corrupt '+key)
    try:parse(text(records+[final],sites))
    except ValueError:unit_checks+=1
    else:raise ValueError('parser accepted duplicate final')
    # Verify the exact frozen source closure uses no device allocation API that
    # bypasses the instrumented cudaMalloc/free entry points. Runtime overhead
    # outside these source calls is explicitly outside the payload contract.
    sources=identity['sources'];assert 'src/cuda/stage2/device_memory_ledger.cuh' in sources
    allocation_api=r'\b(cuda(?:Malloc|Free)\w*|cuMem(?:Alloc|Free)\w*)\s*\('
    for name in sources:
        if name.endswith(('.cu','.cuh')):
            calls=set(re.findall(allocation_api,(ROOT/name).read_text(encoding='utf-8-sig')))
            if calls-{'cudaMalloc','cudaFree','cudaMallocHost','cudaFreeHost'}:
                raise ValueError('untracked device allocation API in '+name)
    verified=[]
    for matrix in a.matrix:
        data=json.loads(matrix.read_text(encoding='utf-8'))
        if not data['complete'] or data['identity']!=identity or not data['memory_ledger']:
            raise ValueError('native evidence incomplete or wrong identity')
        if data['memory_parser_sha256']!=sha(ROOT/'tools/bench/stage2_memory_ledger.py'):
            raise ValueError('native parser identity changed')
        for row in data['runs']:
            if sha(row['debug_log'])!=row['debug_sha256']:
                raise ValueError('raw evidence changed')
            native=parse(Path(row['debug_log']).read_text(encoding='utf-8'))
            if native!=row['memory_ledger'] or int(row['coverage']['gmp_check_bad']) or int(row['coverage']['gmp_selftest_bad']):
                raise ValueError('native lifecycle/arithmetic evidence differs')
            verified.append(dict(matrix=str(matrix),name=row['name'],final=native['final']))
    fallbacks=[]
    for folder in a.fallback_dir:
        metadata=json.loads((folder/'checks.json').read_text(encoding='utf-8'))
        if not metadata['complete'] or metadata['identity']!=identity:
            raise ValueError('fallback evidence incomplete or wrong identity')
        for row in metadata['rows']:
            debug=folder/(row['name']+'.debug.log')
            if sha(debug)!=row['debug_sha256']:raise ValueError('fallback raw evidence changed')
            native_text=debug.read_text(encoding='utf-8')
            if row['returncode']:
                if 'stage2_memory_ledger: snapshot=final' in native_text:
                    raise ValueError('fatal arithmetic fault unexpectedly finalized the curve')
                continue
            native=parse(native_text)
            fallbacks.append(dict(folder=str(folder),name=row['name'],final=native['final']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(complete=True,identity=identity,tool_sha256=sha(__file__),
        unit_checks=unit_checks,verified=verified,fallbacks=fallbacks),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(unit_checks=unit_checks,native_runs=len(verified),native_fallbacks=len(fallbacks),complete=True)))


if __name__=='__main__':main()
