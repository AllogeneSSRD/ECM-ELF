"""Serial outer-loop compilation experiments with immutable source and GMP gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--sizes',type=int,nargs='+',default=[24,25,26,27])
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    assert all(16<=k<=27 for k in args.sizes)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not any(out.iterdir()),'Use a fresh output directory'
    builds={};origin=None
    for root in args.builds:
        root=root.resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
        width=manifest['unroll_u'];assert width in (0,1,2,4,8) and width not in builds
        assert manifest['gl_fixed_mode']==3 and manifest['architecture']=='sm_89'
        if manifest.get('integrated_schedule',False):assert width in (0,4) and manifest['compiled_outer_u']==width
        if origin is None:origin=manifest['sources']
        else:assert origin==manifest['sources'],'Variant original source differs'
        builds[width]=(root,manifest)
    assert 0 in builds and len(builds)>1
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    env['NTT_GL_SHIFT_SCALE']='0'
    result=dict(device=args.device,sizes=args.sizes,manifests={str(u):m for u,(_,m) in builds.items()},
        driver_sha256=digest(Path(__file__)),gates={},comparisons=[],
        timing='ABBA processes per size/candidate; 8 runs/process, warm+3 events/run, all L outputs checked')

    def save():
        (out/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

    def verify(width):
        root,manifest=builds[width];exe=root/'ntt_outer_ilp_probe.exe'
        assert digest(exe)==manifest['sha256'].lower()
        for relative,want in manifest['sources'].items():assert digest(repo/relative)==want.lower(),relative
        for relative,want in manifest['generated_sources'].items():
            path=root/relative;assert digest(path)==want.lower(),relative
            original=(repo/relative.removeprefix('_src/')).read_bytes()
            if relative.endswith('/ntt_coop_outer.cuh') and width and not manifest.get('integrated_schedule',False):
                needle=b'            for(int u=row;u<d;u+=ROWS) {'
                assert original.count(needle)==1
                newline=b'\r\n' if b'\r\n' in original else b'\n'
                original=original.replace(needle,f'#pragma unroll {width}'.encode()+newline+needle)
            if relative.endswith('/ntt_coop_outer_probe.cu'):
                needle=b'    CK(cudaSetDevice(device));';assert original.count(needle)==1
                newline=b'\r\n' if b'\r\n' in original else b'\n'
                ack=f'    std::printf("ntt_outer_variant: unroll_u={width} device=%d\\n",device);'.encode()
                if manifest.get('integrated_schedule',False):
                    ack+=newline+b'    std::printf("ntt_outer_compiled: unroll_u=%d\\n",NTT_OUTER_UNROLL_U);'
                original=original.replace(needle,needle+newline+ack)
            assert path.read_bytes()==original,'Unexpected source generation: '+relative
        return exe

    for width in builds:
        exe=verify(width);gate=out/('gate_u'+str(width))
        command=[sys.executable,str(repo/'tools/test/test_ntt_coop_outer.py'),'--exe',str(exe),
            '--device',str(args.device),'--short-reduce','1','--output',str(gate)]
        with (out/('gate_u'+str(width)+'_driver.log')).open('wb') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        verify(width)
        data=json.loads((gate/'summary.json').read_text(encoding='utf-8'))
        for name in ('gmp','fault','policy','gl_selftest'):
            text=(gate/(name+'.log')).read_text(encoding='utf-8')
            assert f'ntt_outer_variant: unroll_u={width} device={args.device}' in text
            if builds[width][1].get('integrated_schedule',False):assert f'ntt_outer_compiled: unroll_u={width}' in text
        assert data['sha256']==digest(exe)
        result['gates'][str(width)]=dict(exit=run.returncode,summary=data,accepted=run.returncode==0)
        save();print('GATE',width,data['passed'],data['failed'],flush=True)
    assert result['gates']['0']['accepted'],'Baseline gate failed'

    def sample(width,k,name):
        exe=verify(width)
        command=[str(exe),str(args.device),'--bench-fixed',str(k)]
        run=subprocess.run(command,env=env,capture_output=True,timeout=300)
        text=(run.stdout+run.stderr).decode('utf-8',errors='replace')
        (out/(name+'.log')).write_text(text,encoding='utf-8');verify(width)
        assert run.returncode==0,(name,run.returncode,text[-1000:])
        assert f'ntt_outer_variant: unroll_u={width} device={args.device}' in text
        if builds[width][1].get('integrated_schedule',False):assert f'ntt_outer_compiled: unroll_u={width}' in text
        assert f'ntt_gl_reduce_mode: device={args.device} short=1 ptx=1 fixed=3' in text
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in re.findall(r'ntt_fixed_bench: (.*)',text)]
        assert len(rows)==8 and [int(r['run']) for r in rows]==list(range(1,9))
        assert all(r['bad']=='0' and r['backend']=='3' and int(r['N'])==1<<k and int(r['k'])==k and r['policy']=='1' for r in rows)
        geometry={key:rows[0][key] for key in ('passes_fwd','selected_M','selected_coop')}
        assert all(all(r[key]==value for key,value in geometry.items()) for r in rows)
        return dict(name=name,unroll_u=width,command=command,geometry=geometry,
                    seconds=[float(r['seconds']) for r in rows],raw=rows)

    for width in builds:
        if not width or not result['gates'][str(width)]['accepted']:continue
        for k in args.sizes:
            rows=[sample(u,k,f'u{width}_k{k}_{i+1}_actual{u}') for i,u in enumerate((0,width,width,0))]
            assert all(row['geometry']==rows[0]['geometry'] for row in rows)
            means={str(u):statistics.mean(v for row in rows if row['unroll_u']==u for v in row['seconds']) for u in (0,width)}
            comparison=dict(unroll_u=width,k=k,geometry=rows[0]['geometry'],processes=rows,means=means,
                gain_percent=100*(1-means[str(width)]/means['0']))
            result['comparisons'].append(comparison);save()
            print('BENCH',width,k,means,'gain',comparison['gain_percent'],flush=True)
    result.update(passed=sum(g['accepted'] for g in result['gates'].values()),
        failed=sum(not g['accepted'] for g in result['gates'].values()),
        scope='Independent complete field convolutions, no ECM point or Stage1 work; resource-invalid candidates are excluded')
    save();return 0


if __name__=='__main__':raise SystemExit(main())
