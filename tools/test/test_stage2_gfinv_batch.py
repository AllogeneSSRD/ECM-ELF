"""Compare segment inverse batching with the original projective Stage2 path."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe', type=Path, default=repo/'build_cuda_cmake/stage2_tree_gpu.exe')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', type=int, default=1)
args = p.parse_args()
out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()): raise RuntimeError('Use a fresh output directory')
env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
env.update(NTT_S4_OLDTAIL='0', NTT_S4_PACK_DIRECT='1', NTT_S4_FLAT_DIRECT='1',
           NTT_S4_GROOT_ONLY='1', NTT_GROOT_DEVICE='1', NTT_GROOT_LEAF_STAGING='1',
           NTT_GROOT_COMPACT_RAW='1', NTT_SCALED_DESCENT='1', NTT_S4_OUTPUT_WINDOW='1',
           NTT_S4_CHUNK_OUTPUT='1', NTT_S4_CARRY_BATCH='0', NTT_S4_SAMPLE='96',
           NTT_S4_CHECK_EVERY='8', NTT_S4_ORACLE_ASYNC='1', NTT_NO_PROGRESS='1', NTT_NAME_MAX='1')
rows=[]

def run(name,n,b1=20,b2=1000,d=210,**settings):
    command=[str(args.exe.resolve()),'--real','--n-hex',format(n,'x'),'--sigma','26',
             '--b1',str(b1),'--b2',str(b2),'--d',str(d),'--device',str(args.device)]
    proc=subprocess.run(command,env=env|settings,capture_output=True,text=True,timeout=240,cwd=out)
    text=proc.stdout+proc.stderr; log=out/(name+'.log');log.write_text(text,encoding='utf-8')
    assert proc.returncode==0,(name,proc.returncode,str(log))
    assert not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]|slot_canonical_bad=[1-9]',text),(name,str(log))
    def stat(prefix):
        m=re.search(r'(?m)^'+prefix+r': (.*)',text);assert m,(name,prefix,str(log))
        return dict(re.findall(r'(\w+)=(\S+)',m[1]))
    inverse=stat('real_batched_gfinv')
    assert inverse['enabled']==settings.get('NTT_GFINV_BATCH','0')
    if settings.get('NTT_GFINV_BATCH_TEST')=='1':
        f=stat('gfinv_fixture');assert f['cases']=='90' and f['bad']=='0'
        assert all(int(f[k])>0 for k in ('checks','nonunits','group_failures','cache_hits'))
        assert re.search(r'stage2_full_wall:.*clean=0',text)
    if inverse['enabled']=='1' and int(inverse['requests']):
        assert int(inverse['groups'])>0 and int(inverse['segments'])>0
        assert int(inverse['good'])+int(inverse['nonunits'])==int(inverse['segments'])
        assert int(inverse['group_attempts'])+int(inverse['individual_attempts'])<=int(inverse['segments'])+int(inverse['groups'])
    root=stat('real_batched_groot'); leaf=stat('descent_values'); proj=stat('real_batched_projective')
    trace=stat('real_batched_carrytrace')
    stage=re.search(r'stage2:.*hits=(\d+) bad_factors=0 factors=(\S*) hit_primes=(\S*)',text);assert stage
    row=dict(case=name,log=str(log),inverse=inverse,projective=proj,root_hash=root['root_hash'],
             leaf_hash=leaf['hash'],leaves=leaf['leaves'],words=leaf['words'],root_words=root['root_words'],
             trace_words=trace['words'],trace_hash=trace['signature'],hits=stage[1],factors=stage[2],hit_primes=stage[3])
    rows.append(row); print('PASS',name,flush=True);return row

for bits,n in ((64,(1<<64)-59),(129,(1<<128)+1),(5261,(1<<5261)-1)):
    run(f'fixture_{bits}',n,NTT_GFINV_BATCH='1',NTT_GFINV_BATCH_TEST='1')

for name,n,b1,b2,d,chain in (
    ('composite',15,2,1000,210,'0'),('mixed',35,2,1000,210,'999999999'),
    ('frozen',(1<<128)+1,1000,1000000,210,'0'),
    ('tail',(1<<128)+1,1000,400000,2310,'0'),
    ('ladder',(1<<128)+1,20,114000,210,'999999999'),
    ('long',(1<<5261)-1,20,114000,210,'0')):
    pair=[]
    for batch in ('0','1'):
        pair.append(run(name+'_'+batch,n,b1,b2,d,NTT_GFINV_BATCH=batch,NTT_GIANT_CHAIN_MIN=chain,
                        NTT_S4_CARRY_TRACE='1',NTT_GROOT_DEVICE_CHECK='1'))
    for k in ('root_hash','root_words','leaf_hash','leaves','words','trace_words','trace_hash','hits','factors','hit_primes','projective'):
        assert pair[0][k]==pair[1][k],(name,k,pair[0][k],pair[1][k])
    assert pair[0]['inverse']['requests']==pair[1]['inverse']['requests']
    if name=='frozen':
        assert pair[1]['factors']=='59649589127497217' and pair[1]['hit_primes']=='114713'
    if name in ('composite','mixed'):
        assert int(pair[1]['inverse']['nonunits'])>0 and int(pair[1]['projective']['affine_fallback_points'])>0

# Corrupt a returned cached inverse; the independent GMP fixture must reject it.
command=[str(args.exe.resolve()),'--real','--n-hex','ffffffffffffffc5','--sigma','26',
         '--b1','20','--b2','1000','--d','210','--device',str(args.device)]
proc=subprocess.run(command,env=env|dict(NTT_GFINV_BATCH='1',NTT_GFINV_BATCH_TEST='1',NTT_GFINV_BATCH_TEST_BAD='1'),
                    capture_output=True,text=True,timeout=240,cwd=out)
text=proc.stdout+proc.stderr;(out/'fixture_fault.log').write_text(text,encoding='utf-8')
assert proc.returncode!=0 and 'segment inverse GMP mismatch' in text
rows.append(dict(case='fixture_fault',fault_caught=True));print('PASS fixture_fault',flush=True)
summary=dict(passed=len(rows),failed=0,binary_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),runs=rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(f'TOTAL {len(rows)} passed / 0 failed',flush=True)
