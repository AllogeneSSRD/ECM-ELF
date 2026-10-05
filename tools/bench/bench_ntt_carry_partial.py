"""Immutable carry diagnostics gate and serial same-binary ABBA+BAAB timing."""
import argparse,hashlib,json,re,statistics,subprocess
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];root=a.build.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'));exe=root/'ntt_carry_partial_probe.exe'
    assert m['architecture']=='sm_89'
    def verify():
        assert digest(exe)==m['sha256'].lower()
        for name,want in m['sources'].items():assert digest(repo/name)==want.lower(),name
        for name,want in m['generated_sources'].items():assert digest(root/name)==want.lower(),name
        raw=(repo/'tools/bench/ntt_poly_probe.cu').read_bytes()
        begin=raw.index(b'__global__ void carry_residual_kernel(')
        rest=raw[begin:];marker=b'/*\r\n * One PARALLEL' if b'\r\n' in raw else b'/*\n * One PARALLEL'
        assert (root/'_src/tools/test/carry_reference.cuh').read_bytes()==rest[:rest.index(marker)]
        newline=b'\r\n' if b'\r\n' in raw else b'\n'
        begin=raw.index(b'template <int ROUNDS>'+newline+b'__device__ __forceinline__ unsigned long long carry_cone_value(')
        end=raw.index(b'/* ---- twiddle tables',begin)
        include=raw.find(b'#include "ntt_carry_partial.cuh"',begin,end)
        if include>=0:end=include
        assert (root/'_src/tools/test/carry_cone_reference.cuh').read_bytes()==raw[begin:end]
        for name in ('tools/bench/ntt_carry_partial.cuh','tools/test/ntt_carry_partial_probe.cu'):
            assert (root/'_src'/name).read_bytes()==(repo/name).read_bytes()
    def run(name,args):
        verify();r=subprocess.run([str(exe),str(a.device),*map(str,args)],capture_output=True,timeout=240)
        text=(r.stdout+r.stderr).decode('utf-8',errors='replace');(out/(name+'.log')).write_text(text,encoding='utf-8')
        verify();assert r.returncode==0,(name,r.returncode,text[-2000:])
        assert f'carry_partial_device: {a.device}' in text
        return text
    gate=run('gate',[]);line=re.search(r'carry_partial_gate: cases=(\d+) words=(\d+) bad=0 fault_propagated=1',gate)
    assert line and int(line[1])==180 and int(line[2])==657000
    resources=[dict(re.findall(r'(\w+)=([^\s]+)',row)) for row in re.findall(r'carry_partial_resource: (.*)',gate)]
    assert len(resources)==7 and all(r['local']=='0' and int(r['capacity'])>0 for r in resources)
    assert 'carry_cone_gate: configurations=12 shapes=18 modes=2 bad=0' in gate
    result=dict(manifest=m,driver_sha256=digest(Path(__file__)),device=a.device,gate_cases=180,gate_words=657000,
                resources=resources,comparisons=[],cone_comparisons=[],scope='Diagnostics or carry+diagnostics kernels; warm3 + 20 repeated launches/run, four runs/mode, no CI')
    for k,batch in ((12,1),(16,1),(16,64),(20,4),(24,1),(26,1),(27,1)):
        for poison in (0,1):
            name=f'k{k}_m{batch}_poison{poison}';text=run(name,['--bench',k,batch,poison])
            rows=[dict(re.findall(r'(\w+)=([^\s]+)',row)) for row in re.findall(r'carry_partial_bench: (.*)',text)]
            assert len(rows)==8 and [int(r['mode']) for r in rows]==[0,1,1,0,1,0,0,1]
            assert all(r['bad']=='0' and int(r['k'])==k and int(r['batch'])==batch and int(r['poison'])==poison for r in rows)
            assert all(int(r['scratch_bytes'])==8*(((1<<k)+255)//256)*batch for r in rows)
            means={str(u):statistics.mean(float(r['seconds']) for r in rows if int(r['mode'])==u) for u in (0,1)}
            result['comparisons'].append(dict(k=k,batch=batch,poison=poison,rows=rows,means=means,gain_percent=100*(1-means['1']/means['0'])))
            (out/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(name,means,'gain',result['comparisons'][-1]['gain_percent'],flush=True)
    shapes=((11,990,5),(12,495,5),(16,30,5),(17,15,5),(16,1,5),(16,64,5),(20,4,5),
            (24,1,5),(26,1,5),(27,1,5),(16,1,6),(20,4,6),(24,1,6),(26,1,6),(27,1,6))
    for k,batch,rounds in shapes:
        text=run(f'cone_k{k}_m{batch}_r{rounds}',['--bench-cone',k,batch,rounds])
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',row)) for row in re.findall(r'carry_cone_bench: (.*)',text)]
        assert len(rows)==8 and [int(r['mode']) for r in rows]==[0,1,1,0,1,0,0,1]
        assert all(r['bad']=='0' and int(r['k'])==k and int(r['batch'])==batch and int(r['rounds'])==rounds for r in rows)
        assert all(int(r['scratch_bytes'])==8*(((1<<k)+255)//256)*batch for r in rows)
        means={str(u):statistics.mean(float(r['seconds']) for r in rows if int(r['mode'])==u) for u in (0,1)}
        result['cone_comparisons'].append(dict(k=k,batch=batch,rounds=rounds,rows=rows,means=means,gain_percent=100*(1-means['1']/means['0'])))
        (out/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('cone',k,batch,means,'gain',result['cone_comparisons'][-1]['gain_percent'],flush=True)
    result.update(passed=2+len(result['comparisons'])+len(result['cone_comparisons']),failed=0)
    (out/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
