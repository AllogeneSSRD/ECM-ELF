"""Gate resident giant coordinates, GPU Gamma groups and mixed device G leaves."""
import argparse, hashlib, json, math, os, re, subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
a=p.parse_args();exe=a.exe.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
base.update(NTT_S4_OLDTAIL='0',NTT_S4_MERSENNE='1',NTT_GROOT_DEVICE='1',NTT_SCALED_DESCENT='1',
 NTT_S4_OUTPUT_WINDOW='1',NTT_S4_CHUNK_OUTPUT='1',NTT_GFINV_BATCH='1',NTT_FOLD_FLAT='1',
 NTT_GIANT_SEED_DEVICE='1',NTT_GFINV_SEG_EXACT='1',NTT_SMALL_PRIME_REUSE='1',NTT_NAME_MAX='1',
 NTT_NO_PROGRESS='1',NTT_S4_CARRY_TRACE='1',NTT_GIANT_CHAIN_MIN='0')
rows=[]
def invoke(name,n,b1,b2,d,**env):
 r=subprocess.run([str(exe),'--real','--n-hex',format(n,'x'),'--sigma','26','--b1',str(b1),'--b2',str(b2),
  '--d',str(d),'--device',str(a.device)],env=base|env,capture_output=True,text=True,timeout=600,cwd=out)
 text=r.stdout+r.stderr;(out/(name+'.log')).write_text(text,encoding='utf-8');return r.returncode,text

def run(name,n,b1,b2,d,active=False,**env):
 rc,text=invoke(name,n,b1,b2,d,**env);assert rc==0 and not re.search(r'FATAL|MISMATCH|gmp_check_bad=[1-9]',text),(name,rc)
 def stat(key):
  m=re.search(r'(?m)^'+key+r': (.*)',text);assert m,(name,key);return dict(re.findall(r'(\w+)=(\S+)',m[1]))
 dl=stat('device_gleaf');proj=stat('real_batched_projective');leaf=stat('descent_values');root=stat('real_batched_groot');trace=stat('real_batched_carrytrace');stage=stat('stage2');seed=stat('real_giant_seed')
 if active:
  assert int(dl['chunks'])>0 and dl['fallback_chunks']=='0' and int(dl['device_trees'])>0
  assert int(dl['group_d2h_bytes'])==8*((n.bit_length()+63)//64)*int(dl['groups'])
  w=(n.bit_length()+63)//64;i=b2//d+2
  assert int(dl['device_leaf_words'])==2*w*i
  assert int(dl['avoided_leaf_h2d_bytes'])+8*int(dl['patch_words'])==16*w*i
  assert int(dl['avoided_point_d2h_bytes'])+int(dl['bad_point_d2h_bytes'])==16*w*i
  if env.get('NTT_DEVICE_GLEAF_CHECK')=='1':
   assert dl['checked_groups']==dl['groups'] and dl['checked_leaf_words']==dl['device_leaf_words']
  assert (int(dl['bad_groups'])>0)==(int(proj['affine_fallback_points'])>0),name
 else:assert dl['chunks']==dl['device_trees']=='0',name
 signature=(root['root_hash'],root['root_words'],leaf['hash'],leaf['words'],trace['signature'],trace['words'],
  stage['hits'],stage.get('factors',''),stage.get('hit_primes',''),proj)
 rows.append(dict(case=name,device_leaf=dl,signature=signature,seed=seed));print('PASS',name,flush=True);return signature

for name,n,b1,b2,d,env in [
 ('prime127',(1<<127)-1,20,40000,210,{}),
 ('single_leaf_tail',(1<<127)-1,20,4830,210,{'NTT_GROOT_DEVICE_CHECK':'1'}),
 ('blocking',(1<<127)-1,20,40000,210,{'NTT_S4_ASYNC':'0'}),
 ('partial_group',(1<<127)-1,20,1000,210,{}),
 ('group_boundary',(1<<127)-1,20,223650,2310,{'NTT_GROOT_DEVICE_CHECK':'1'}),
 ('prime127_extra12',(1<<127)-1,20,40000,210,{'NTT_STAGE1_EXTRA':'12'}),
 ('bad_segments',10403,2,4000,22,{}),
 ('saturated',103,2,4000,22,{}),
 ('frozen',(1<<128)+1,1000,1000000,210,{}),
 ('ladder_frozen',(1<<128)+1,1000,1000000,210,{'NTT_GIANT_LADDER':'1'}),
 ('individual_inverse',(1<<127)-1,20,13230,210,{'NTT_GFINV_BATCH':'0'}),
 ('host_seed',(1<<127)-1,20,40000,210,{'NTT_GIANT_SEED_DEVICE':'0'}),
 ('wide4423',(1<<4423)-1,20,114000,210,{'NTT_STAGE1_EXTRA':'12'}),
 ('wide5261',(1<<5261)-1,20,114000,210,{}),
 ('limb64',(1<<64)-1,2,5000,210,{}),
 ('limb65',(1<<65)-1,2,5000,210,{})]:
 pair=[run(name+'_'+mode,n,b1,b2,d,active=mode=='1',NTT_DEVICE_GLEAF=mode,
  NTT_DEVICE_GLEAF_CHECK=mode,NTT_GFINV_SEG_CHECK='1',**env) for mode in ('0','1')]
 assert pair[0]==pair[1],name
# Memory budget and incompatible backends select the complete existing host path.
for name,env in [ ('budget',{'NTT_DEVICE_GLEAF_MAX_MB':'0'}),
 ('host_groot',{'NTT_GROOT_DEVICE':'0'}),('nonexact',{'NTT_GFINV_SEG_EXACT':'0'})]:
 n=(1<<127)-1
 pair=[run(name+'_'+mode,n,20,40000,210,NTT_DEVICE_GLEAF=mode,**env) for mode in ('0','1')]
 assert pair[0]==pair[1],name
rc,text=invoke('poison',(1<<127)-1,20,1000,210,NTT_DEVICE_GLEAF='1',NTT_DEVICE_GLEAF_CHECK='1',NTT_DEVICE_GLEAF_TEST_BAD='1')
assert rc!=0 and 'device giant leaf CPU mismatch' in text
rows.append(dict(case='poison',fault_caught=True))
(out/'summary.json').write_text(json.dumps(dict(passed=len(rows),failed=0,runs=rows,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print('TOTAL',len(rows),'passed / 0 failed',flush=True)
