"""Gate reuse of baby unit/GCD proofs for Stage2 small primes."""
import argparse, hashlib, importlib.util, json, math, os, re, subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',type=int,default=1)
a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
assert not any(out.iterdir()),'Use a fresh output directory'
spec=importlib.util.spec_from_file_location('ref',repo/'tools/stat/suyama_mont_ref.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_OLDTAIL='0',NTT_S4_PACK_DIRECT='1',NTT_S4_FLAT_DIRECT='1',NTT_GROOT_DEVICE='1',
 NTT_SCALED_DESCENT='1',NTT_S4_OUTPUT_WINDOW='1',NTT_S4_CHUNK_OUTPUT='1',NTT_GFINV_BATCH='1',
 NTT_FOLD_FLAT='1',NTT_GFINV_SEG_EXACT='1',NTT_GIANT_SEED_DEVICE='1',NTT_NO_PROGRESS='1',
 NTT_NAME_MAX='1',NTT_S4_CARRY_TRACE='1',NTT_GIANT_CHAIN_MIN='0')
rows=[]
def primes(lo,hi):
 return [v for v in range(max(2,lo+1),hi+1) if all(v%d for d in range(2,math.isqrt(v)+1))]
def invoke(name,n,b1,b2,d,sigma=26,curves=1,**env):
 proc=subprocess.run([str(exe),'--real','--n-hex',format(n,'x'),'--sigma',str(sigma),
  '--b1',str(b1),'--b2',str(b2),'--d',str(d),'--device',str(a.device),'--curves',str(curves)],
  env=base|env,capture_output=True,text=True,cwd=out,timeout=600)
 text=proc.stdout+proc.stderr;(out/(name+'.log')).write_text(text,encoding='utf-8')
 return proc.returncode,text

def run(name,n,b1,b2,d,sigma=26,curves=1,**env):
 rc,text=invoke(name,n,b1,b2,d,sigma,curves,**env)
 assert rc==0 and not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]',text),(name,rc)
 def stat(prefix):
  lines=re.findall(r'(?m)^'+prefix+r': (.*)',text);assert len(lines)==curves,(name,prefix,len(lines))
  return [dict(re.findall(r'(\w+)=(\S+)',v)) for v in lines]
 small=stat('small_prime_reuse');stage=stat('stage2');root=stat('real_batched_groot')
 leaf=stat('descent_values');carry=stat('real_batched_carrytrace');proj=stat('real_batched_projective')
 ps=primes(b1,min(b2,d//2));covered=[p for p in ps if math.gcd(p,d)==1]
 reuse=env.get('NTT_SMALL_PRIME_REUSE')=='1';matched=reuse and env.get('NTT_SMALL_PRIME_CACHE_STALE','0')=='0'
 for s in small:
  assert int(s['primes'])==len(ps) and s['bad']=='0'
  assert s['requested']==str(int(reuse)) and s['matched']==str(int(matched))
  r=len(covered) if matched else 0
  assert int(s['reused'])==r and int(s['fallback'])==len(ps)-r
  assert int(s['checked'])==(r if env.get('NTT_SMALL_PRIME_CHECK')=='1' else 0)
  assert int(s['avoided_h2d_bytes'])==8*r and int(s['avoided_d2h_bytes'])==16*((n.bit_length()+63)//64)*r
  assert int(s['avoided_montmuls'])==sum(13*p.bit_length()-6 for p in covered if matched)
  assert (int(s['cache_bytes'])>0)==reuse
 # Independent Python ladder: nonunit gcds include g=N, which must not be reported as a factor.
 extra=int(env.get('NTT_STAGE1_EXTRA','1'));q=ref.stage1(sigma,b1,n,torsion=extra)['x']
 _,a24,_,_=ref.suyama_curve(sigma,n)
 gcds=[math.gcd(ref.ladder(p,q,1,a24,n)[1],n) for p in ps]
 expected=[(p,g) for p,g in zip(ps,gcds) if 1<g<n]
 for st in stage:
  hits=[int(x) for x in st.get('hit_primes','').split(',') if x]
  assert hits[:len(expected)]==[p for p,g in expected],(name,expected,hits)
  fs={int(x) for x in st.get('factors','').split(',') if x}
  assert {g for p,g in expected}<=fs and n not in fs
  assert st['bad_factors']=='0'
 signature=[(r['root_hash'],l['hash'],c['signature'],st.get('hits'),st.get('factors',''),st.get('hit_primes',''),pr)
  for r,l,c,st,pr in zip(root,leaf,carry,stage,proj)]
 # Existing root/carry statistics accumulate across --curves; compare those per ordinal
 # against the paired run, and require final leaves/factors/Gamma to agree within a run.
 assert all(v[1]==signature[0][1] and v[3:]==signature[0][3:] for v in signature)
 rows.append(dict(case=name,small=small,signature=signature,small_nonunit=sum(g!=1 for g in gcds),
  small_saturated=sum(g==n for g in gcds),independent_small_hits=expected))
 print('PASS',name,flush=True);return signature

for name,n,b1,b2,d,sigma,extra in [
 ('prime127',(1<<127)-1,20,40000,210,26,1),
 ('prime127_extra12',(1<<127)-1,20,40000,210,26,12),
 ('sigma27',(1<<127)-1,2,5000,210,27,1),
 ('missing_D_primes',(1<<127)-1,2,5000,210,26,1),
 ('cached_factor',10403,2,400,22,26,1),
 ('fallback_factor',10403,2,400,210,26,1),
 ('cached_saturated',103,2,400,22,26,1),
 ('fallback_saturated',103,2,400,210,26,1),
 ('no_small',(1<<127)-1,120,40000,210,26,1),
 ('bounded_B2',(1<<127)-1,2,19,210,26,1),
 ('frozen_small_factor',(1<<128)+1,1000,1000000,300300,26,1),
 ('wide4423',(1<<4423)-1,20,114000,210,26,12),
 ('wide5261',(1<<5261)-1,20,114000,210,26,1)]:
 pair=[run(name+'_'+mode,n,b1,b2,d,sigma,NTT_SMALL_PRIME_REUSE=mode,
  NTT_SMALL_PRIME_CHECK=mode,NTT_STAGE1_EXTRA=str(extra),NTT_S4_MERSENNE='1') for mode in ('0','1')]
 assert pair[0]==pair[1],name
repeat=[run('repeated_same_Q_'+mode,(1<<127)-1,20,40000,210,curves=2,NTT_SMALL_PRIME_REUSE=mode,NTT_SMALL_PRIME_CHECK=mode) for mode in ('0','1')]
assert repeat[0]==repeat[1]
# Exact input mismatch must retain all small points on the original path.
run('stale_key',(1<<127)-1,20,40000,210,NTT_SMALL_PRIME_REUSE='1',NTT_SMALL_PRIME_CHECK='1',NTT_SMALL_PRIME_CACHE_STALE='1')
# Unit proof also works when S4 is disabled and the tree uses its host reference.
run('host_S4',(1<<127)-1,20,1000,210,NTT_SMALL_PRIME_REUSE='1',NTT_SMALL_PRIME_CHECK='1',NTT_S4_OFF='1',NTT_SCALED_DESCENT='0')
rc,text=invoke('poison',(1<<127)-1,20,1000,210,NTT_SMALL_PRIME_REUSE='1',NTT_SMALL_PRIME_CHECK='1',NTT_SMALL_PRIME_TEST_BAD='1')
assert rc!=0 and 'small-prime baby GCD proof mismatch' in text
rows.append(dict(case='poison',fault_caught=True))
(out/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,runs=rows,
 binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print('TOTAL',len(rows),'passed / 0 failed',flush=True)
