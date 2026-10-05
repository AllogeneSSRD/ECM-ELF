"""Serial same-binary ABBA+BAAB tile comparisons; all outputs checked outside event timing."""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--sizes',type=int,nargs='+',default=[24]);p.add_argument('--tile-bits',type=int,default=12)
    p.add_argument('--methods',type=int,nargs='+',choices=(1,2,3,4),default=[1,2,3,4]);p.add_argument('--operations',type=int,nargs='+',choices=(0,1,2),default=[0,1,2]);a=p.parse_args()
    if not 5<=a.tile_bits<=12 or any(not max(12,a.tile_bits)<=k<=26 for k in a.sizes):raise ValueError('t must be5..12 and k max(12,t)..26')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    fixed=manifest.get('gl_fixed_mode',-1)
    if fixed>=0 and 4 in a.methods:raise ValueError('Reducer A/B method4 requires a runtime build')
    repo=Path(__file__).resolve().parents[2];sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha==manifest['sha256'].lower()
        for name,want in manifest['sources'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want.lower(),name
    verify();env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    data={'exe':str(exe),'sha256':sha,'manifest':manifest,'device':a.device,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'timing':'8 ABBA+BAAB runs, 4/mode, one warm + three CUDA-event samples/run; init/fill/check/planning excluded','all_outputs_checked':True,'runs':[]}
    (a.output/'provenance.json').write_text(json.dumps({k:v for k,v in data.items() if k!='runs'},indent=2),encoding='utf-8')
    for size in a.sizes:
        for method in a.methods:
            for operation in a.operations:
                name=f'roots_{size}_{method}_{operation}';cmd=[str(exe),str(a.device),'--bench',str(size),str(a.tile_bits),str(operation),str(method)]
                verify();started=time.monotonic();r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                text=r.stdout.decode('utf-8',errors='replace');(a.output/(name+'.log')).write_text(text,encoding='utf-8');verify()
                assert r.returncode==0,(name,r.returncode,text[-1000:])
                if fixed>=0:
                    assert f'ntt_gl_reduce_mode: device={a.device} short={fixed&1} ptx={fixed>>1} fixed={fixed}' in text,name
                rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in re.findall(r'small_root_bench: (.*)',text)]
                assert [int(r['mode']) for r in rows]==[0,1,1,0,1,0,0,1] and [int(r['run']) for r in rows]==list(range(1,9)),name
                assert all(r['bad']=='0' and int(r['candidate'])==method and int(r['operation'])==operation and int(r['k'])==size and
                           int(r['N'])==1<<size and int(r['t'])==a.tile_bits and int(r['threads'])==512 for r in rows),name
                if fixed>=0:assert all(int(r['fixed'])==fixed for r in rows),name
                memory=re.search(r'small_root_memory: requested_peak_bytes=(\d+) live_bytes=0',text);assert memory,name
                seconds={str(m):[float(r['seconds']) for r in rows if int(r['mode'])==m] for m in (0,1)}
                means={m:statistics.mean(v) for m,v in seconds.items()}
                item={'name':name,'command':cmd,'seconds':seconds,'means':means,'gain_percent':100*(means['0']-means['1'])/means['0'],
                      'logical_peak_bytes':int(memory[1]),'driver_seconds':time.monotonic()-started,'raw':rows}
                data['runs'].append(item);(a.output/'measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in item.items() if k not in ('raw','command','seconds')}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
