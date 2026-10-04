"""Device fold differential gates. GPU jobs run sequentially on the requested device."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--exe',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',type=int,default=1)
a=p.parse_args();exe=Path(a.exe).resolve();root=Path(a.output).resolve()
root.mkdir(parents=True,exist_ok=True)
base={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
base.update({k:'1' for k in ['NTT_FOLD_FLAT','NTT_GROOT_DEVICE','NTT_DEVICE_GLEAF','NTT_GFINV_BATCH',
    'NTT_GFINV_SEG_EXACT','NTT_GIANT_SEED_DEVICE','NTT_SMALL_PRIME_REUSE','NTT_SCALED_DESCENT',
    'NTT_S4_OUTPUT_WINDOW','NTT_S4_CHUNK_OUTPUT','NTT_S4_MERSENNE','NTT_NO_PROGRESS']})
base.update(NTT_NAME_MAX='1',NTT_S4_SAMPLE='96',NTT_S4_CHECK_EVERY='8')
checks=[]
def run(name,n,flags,b2=12000,fail=False):
    env=dict(base);env.update(flags)
    args=[str(exe),'--real','--n',str(n),'--sigma','26','--b1','1000',
        '--b2',str(b2),'--d','210','--device',str(a.device)]
    proc=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        encoding='utf-8',errors='replace')
    (root/(name+'.log')).write_text(proc.stdout,encoding='utf-8')
    if fail:
        assert proc.returncode!=0 and 'device fold GMP mismatch' in proc.stdout,(name,proc.stdout[-1500:])
        checks.append({'name':name,'exit':proc.returncode,'poison_rejected':True});return
    assert proc.returncode==0,(name,proc.stdout[-3000:])
    for bad in ['gmp_selftest_bad=','gmp_check_bad=','bad_factors=']:
        assert all(int(v)==0 for v in re.findall(bad+r'(\d+)',proc.stdout)),name
    leaf=re.search(r'descent_values:.*hash=(\d+)',proc.stdout)
    tail=re.search(r'stage2: algorithm=tree_gpu_batched .*?hits=(\d+) bad_factors=0 factors=([^ ]*) hit_primes=([^ ]*)',proc.stdout)
    fd=re.search(r'real_batched_folddevice: (.*)',proc.stdout)
    fields=dict(re.findall(r'(\w+)=([^ ]+)',fd[1]))
    checks.append({'name':name,'exit':0,'leaf_hash':leaf[1],'tail':tail.groups(),'folddevice':fields})
    return proc.stdout,checks[-1]
for bits,n in [('65',2**65-59),('127',2**127-1),('129',2**128+1)]:
    control,c=run('host_'+bits,n,{'NTT_FOLD_DEVICE':'0'})
    text,d=run('device_'+bits,n,{'NTT_FOLD_DEVICE':'1','NTT_FOLD_DEVICE_CHECK':'1','NTT_FOLD_DEVICE_TEST':'1'})
    assert 'fold_device_fixture: cases=30' in text
    assert d['leaf_hash']==c['leaf_hash'] and d['tail']==c['tail']
    assert d['folddevice']['enabled']=='1' and int(d['folddevice']['checked_words'])>0

n=2**128+1
text,c=run('frozen_host',n,{'NTT_FOLD_DEVICE':'0'},b2=1000000)
text,d=run('frozen_device',n,{'NTT_FOLD_DEVICE':'1','NTT_FOLD_DEVICE_CHECK':'1'},b2=1000000)
assert d['leaf_hash']==c['leaf_hash'] and d['tail']==c['tail']
assert 'factors=59649589127497217' in text
for name,flags,reason in [
    ('budget',{'NTT_FOLD_DEVICE_MAX_MB':'0'},'budget'),
    ('allocation',{'NTT_FOLD_DEVICE_ALLOC_FAIL':'1'},'allocation_fixture'),
    ('copied_pack',{'NTT_S4_PACK_DIRECT':'0'},'backend'),
    ('trace',{'NTT_S4_CARRY_TRACE':'1'},'backend')]:
    _,c=run('fallback_control_'+name,n,dict(flags,NTT_FOLD_DEVICE='0'))
    _,d=run('fallback_device_'+name,n,dict(flags,NTT_FOLD_DEVICE='1'))
    assert d['leaf_hash']==c['leaf_hash'] and d['tail']==c['tail']
    assert d['folddevice']['enabled']=='0' and d['folddevice']['fallback']==reason
for window in ['0','1']:
    _,c=run('window_host_'+window,n,{'NTT_FOLD_DEVICE':'0','NTT_S4_OUTPUT_WINDOW':window})
    _,d=run('window_device_'+window,n,{'NTT_FOLD_DEVICE':'1','NTT_S4_OUTPUT_WINDOW':window})
    assert d['leaf_hash']==c['leaf_hash'] and d['tail']==c['tail']
run('poison',n,{'NTT_FOLD_DEVICE':'1','NTT_FOLD_DEVICE_CHECK':'1','NTT_FOLD_DEVICE_TEST_BAD':'1'},fail=True)
summary={'exe':str(exe),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
    'device':a.device,'checks':checks,'passed':len(checks),'failed':0}
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'passed':len(checks),'failed':0,'sha256':summary['sha256']}))
