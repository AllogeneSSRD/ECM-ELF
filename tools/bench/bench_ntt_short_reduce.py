"""Same-binary pure NTT convolution comparison, fixed outer shape policy, full output check."""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--sizes',type=int,nargs='+',default=[24,25,26,27]);a=p.parse_args()
    if any(k<16 or k>27 for k in a.sizes):raise ValueError('log2 N must be16..27')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    repo=Path(__file__).resolve().parents[2];sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha==manifest['sha256'].lower()
        for name,want in manifest['sources'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want.lower(),name
    verify();env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    data={'exe':str(exe),'sha256':sha,'manifest':manifest,'device':a.device,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'timing':'8 ABBA+BAAB runs,4/mode,1warm+3event samples/run; fill/plan/init/check excluded',
          'fixed_controls':{'NTT_FUSE_COOP_OUTER':'2','NTT_FUSE_WARP_TAIL':'1','NTT_FUSE_T':'12','NTT_FUSE_M':'4'},
          'GMP_sparse_convolution_all_N_words_checked':True,'runs':[]}
    (a.output/'provenance.json').write_text(json.dumps({k:v for k,v in data.items() if k!='runs'},indent=2),encoding='utf-8')
    for size in a.sizes:
        name=f'ntt_reduce_{size}';cmd=[str(exe),str(a.device),'--bench-reduce',str(size)]
        verify();started=time.monotonic();r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8');verify()
        assert r.returncode==0,(name,r.returncode,text[-1000:])
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in re.findall(r'ntt_reduce_bench: (.*)',text)]
        assert [int(r['short']) for r in rows]==[0,1,1,0,1,0,0,1] and [int(r['run']) for r in rows]==list(range(1,9)),name
        assert all(r['bad']=='0' and int(r['k'])==size and int(r['N'])==1<<size and r['policy']=='1' for r in rows),name
        assert len({(r['passes_fwd'],r['selected_M'],r['selected_coop']) for r in rows})==1,name
        modes=[int(x) for x in re.findall(r'ntt_gl_reduce_mode: device='+str(a.device)+r' short=(\d+)',text)]
        assert modes==[0,1,0,1,0,1],(name,modes)
        seconds={str(m):[float(r['seconds']) for r in rows if int(r['short'])==m] for m in (0,1)}
        means={m:statistics.mean(v) for m,v in seconds.items()}
        item={'name':name,'command':cmd,'seconds':seconds,'means':means,'gain_percent':100*(means['0']-means['1'])/means['0'],
              'driver_seconds':time.monotonic()-started,'raw':rows}
        data['runs'].append(item);(a.output/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in item.items() if k not in ('raw','command','seconds')}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
