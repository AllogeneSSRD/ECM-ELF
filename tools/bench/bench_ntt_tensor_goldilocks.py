"""Serial same-binary integer Tensor/CUDA probes; every output is checked outside CUDA event timing."""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--kind',choices=('matrix','tile','mixed'),default='tile')
    p.add_argument('--sizes',type=int,nargs='+',help='log2 total tile words (default 24); matrix uses log2 batches (default 16)')
    p.add_argument('--threads',type=int,nargs='+',default=[256]);p.add_argument('--operations',type=int,nargs='+',choices=(0,1,2))
    p.add_argument('--tile-bits',type=int,default=12);a=p.parse_args()
    if a.sizes is None:a.sizes=[16 if a.kind=='matrix' else 24]
    if a.operations is None:a.operations=[0] if a.kind=='mixed' else [0,1,2]
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    if a.kind=='mixed' and a.operations!=[0]:raise ValueError('Mixed probe has forward tasks only; use --operations 0')
    if a.kind!='matrix' and any(t not in (128,256,512) for t in a.threads):raise ValueError('Tile CTA must be 128/256/512')
    if a.kind=='matrix' and any(t not in (32,64,128,256) for t in a.threads):raise ValueError('Matrix CTA must be 32/64/128/256')
    if a.kind=='matrix' and any(k<0 or k>18 for k in a.sizes):raise ValueError('Matrix log2 batches must be 0..18')
    if a.kind!='matrix' and (not 6<=a.tile_bits<=12 or any(k<max(12,a.tile_bits) or k>26 for k in a.sizes)):
        raise ValueError('Tile bits must be 6..12, log2 N must be max(12,t)..26')
    exe=a.exe.resolve();manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    assert sha.lower()==manifest['sha256'].lower()
    source=Path(__file__).resolve();repo=source.parents[2]
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        for name,want in manifest['sources'].items():
            assert hashlib.sha256((repo/name).read_bytes()).hexdigest().lower()==want.lower(),name
    verify();env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    metadata={'exe':str(exe),'sha256':sha,'manifest':manifest,'kind':a.kind,'device':a.device,
        'timing':'one warm + three CUDA-event samples/run; 8 ABBA+BAAB runs, 4/mode; validation/init/planning excluded',
        'script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'all_outputs_checked':True,'runs':[]}
    if a.kind=='mixed':metadata['timing']='16 crossed runs, 4/mode; one warm + three joined CUDA-event pair samples/run; both independent tasks checked'
    (a.output/'provenance.json').write_text(json.dumps({k:v for k,v in metadata.items() if k!='runs'},indent=2),encoding='utf-8')
    for size in a.sizes:
        for threads in a.threads:
            for operation in a.operations:
                stem=f'{a.kind}_{size}_{threads}_{operation}'
                if a.kind=='matrix':cmd=[str(exe),str(a.device),'--bench',str(1<<size),str(threads),str(operation)]
                else:cmd=[str(exe),str(a.device),'--mixed-bench' if a.kind=='mixed' else '--tile-bench',str(size),str(a.tile_bits),str(threads),str(operation)]
                verify();start=time.monotonic()
                result=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                text=result.stdout.decode('utf-8',errors='replace');log=a.output/(stem+'.log');log.write_text(text,encoding='utf-8')
                assert result.returncode==0,(stem,result.returncode,text[-2000:])
                assert re.search(r'tc_gold_memory: requested_peak_bytes=\d+ live_bytes=0',text),stem
                label={'matrix':'bench','tile':'tile_bench','mixed':'mixed_bench'}[a.kind]
                rows=[dict(re.findall(r'(\w+)=([^\s]+)',r)) for r in re.findall('tc_gold_'+label+r': (.*)',text)]
                order=[0,2,3,1,1,3,2,0,1,3,2,0,0,2,3,1] if a.kind=='mixed' else [0,1,1,0,1,0,0,1]+([2] if a.kind=='matrix' else [])
                assert [int(r['mode']) for r in rows]==order and all(r['bad']=='0' for r in rows),stem
                assert [int(r['run']) for r in rows]==list(range(1,len(order)+1)),stem
                assert all(int(r['threads'])==threads for r in rows),stem
                if a.kind=='matrix':
                    assert all(int(r['batches'])==1<<size and int(r['words'])==128*(1<<size) and int(r['operation'])==operation for r in rows),stem
                else:
                    assert all(int(r['N'])==1<<size and int(r['t'])==a.tile_bits for r in rows),stem
                    assert all(int(r['tasks'])==2 for r in rows) if a.kind=='mixed' else all(int(r['operation'])==operation for r in rows),stem
                groups={str(mode):[float(r['seconds']) for r in rows if int(r['mode'])==mode] for mode in sorted(set(order))}
                means={mode:statistics.mean(values) for mode,values in groups.items()}
                peak=int(re.search(r'tc_gold_memory: requested_peak_bytes=(\d+)',text)[1]);verify()
                item={'name':stem,'command':cmd,'log':str(log.resolve()),'driver_seconds':time.monotonic()-start,
                    'seconds':groups,'means':means,'logical_peak_bytes':peak,'raw':rows}
                if a.kind!='mixed':item['tensor_gain_percent']=100*(means['0']-means['1'])/means['0']
                else:
                    item['cuda_parallel_gain_percent']=100*(means['0']-means['1'])/means['0']
                    item['mixed_parallel_vs_cuda_parallel_percent']=100*(means['1']-means['3'])/means['1']
                    item['mixed_overlap_gain_percent']=100*(means['2']-means['3'])/means['2']
                metadata['runs'].append(item)
                (a.output/'measurements.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in item.items() if k not in ('raw','command','log','seconds')},ensure_ascii=False),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
