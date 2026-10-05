"""Check experimental small-root arithmetic and production-layout tiles against GMP."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--short-reduce',type=int,choices=(0,1),default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    repo=Path(__file__).resolve().parents[2];sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha==manifest['sha256'].lower()
        for name,want in manifest['sources'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want.lower(),name
    verify();env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')};checks={};groups=[];resources=[]
    env['NTT_GL_SHORT_REDUCE']=str(a.short_reduce)
    for mode,code in [('--check',0),('--fault',3)]:
        verify();r=subprocess.run([str(exe),str(a.device),mode],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(mode[2:]+'.log')).write_text(text,encoding='utf-8');verify()
        checks[mode+'_exit']=r.returncode==code;checks[mode+'_no_live']=bool(re.search(r'small_root_memory: requested_peak_bytes=\d+ live_bytes=0',text))
        checks[mode+'_configured']=f'ntt_gl_reduce_mode: device={a.device} short={a.short_reduce}' in text
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in re.findall(r'small_root_check: (.*)',text)]
        if mode=='--check':
            groups=rows
            expected={(str(m),g):(1,98544) for m in (1,2,3) for g in ('primitive',)};expected[('1','fallback')]=(1,2048)
            expected[('3','reduce128')]=(1,100064)
            for m in (0,1,2,3):
                for g in ('forward','inverse','roundtrip','readonly','guard'):expected[(str(m),g)]=(160,5440 if g=='guard' else 705856)
            checks['groups_complete']=len(rows)==len(expected) and len({(r['method'],r['group']) for r in rows})==len(expected)
            for (method,group),(cases,words) in expected.items():
                found=[r for r in rows if (r['method'],r['group'])==(method,group)]
                checks[f'{method}_{group}']=len(found)==1 and found[0]['cases']==str(cases) and found[0]['words']==str(words) and found[0]['bad']=='0'
            checks['all_gates_zero']='small_root_primitive_gate: fault=0 bad=0' in text and 'small_root_tile_gate: bad=0' in text
            resources=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in re.findall(r'small_root_resources: (.*)',text)]
            checks['resources_complete']=len(resources)==8 and {(r['method'],r['inverse']) for r in resources}=={(str(m),str(i)) for m in range(4) for i in range(2)}
            checks['no_local_spill']=len(resources)==8 and all(r['local']=='0' and int(r['max_blocks'])>0 for r in resources)
        else:checks['fault_rejected']='small_root_primitive_gate: fault=1 bad=1' in text
    result={'exe':str(exe),'sha256':sha,'manifest':manifest,'device':a.device,'short_reduce':a.short_reduce,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checks':checks,'groups':groups,'resources':resources,'passed':sum(checks.values()),'failed':sum(not x for x in checks.values())}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return int(result['failed']!=0)
if __name__=='__main__':raise SystemExit(main())
