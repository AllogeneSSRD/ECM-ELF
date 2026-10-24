"""One-process curve/parameter/width/sigma switching against frozen Stage1.

Uses private queues and shared functional caches only. These are not timing
samples; the observer must see exactly the six modules requested by eight jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from test_release_runtime import entry_names,saved_rows

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','observer','ptx-fixtures','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--build-root',type=Path,default=ROOT/'build_cuda_release')
    parser.add_argument('--device',type=int,default=1)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    observer=args.observer.resolve();derived=read(args.ptx_fixtures);assert derived['complete']
    inspector=Path(os.environ.get('CUDA_PATH_V13_3','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'))/'bin/cuobjdump.exe'
    binaries={'baseline':args.baseline.resolve()/'stage1/ecm_cuda.exe'}
    inputs={str(Path(__file__).resolve()):sha(__file__),str(observer):sha(observer),str(args.ptx_fixtures.resolve()):sha(args.ptx_fixtures)}
    for profile in ('cu11.8','cu13.3'):
        record=read(args.build_root/'components'/f'{profile}.stage1.json')
        binaries[profile]=Path(record['exe']);inputs[str(Path(record['manifest']).resolve())]=sha(record['manifest'])
    item=derived['components']['stage1'];assert item['source_exe_sha256']==sha(binaries['cu13.3'])
    binaries['cu13.3_ptx']=Path(item['exe']);assert sha(binaries['cu13.3_ptx'])==item['sha256']
    for path in binaries.values():
        inputs[str(path)]=sha(path);inputs[str(path.parent/'gmp-10.dll')]=sha(path.parent/'gmp-10.dll')
    tasks=[('m0_a_20.save',127),('m2_b_20.save',127),('m3_c_20.save',521),('e_d_20.save',127),
           ('m0_e_20.save',521),('e_f_20.save',521),('m0_g_20.save',127)]
    queue='[Worker #1]\n'+''.join(f'ECMSTAGE1=1,2,{bits},-1,"{name}",0,0,4\n' for name,bits in tasks)+'ECM2=1,2,127,-1,20,0,4,42\n'
    expected_modules={('m0',192),('m2',192),('m3',768),('ed',128),('m0',768),('ed',768)}
    result=dict(complete=False,inputs=inputs,device=args.device,cases={});reference=None
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    for label,binary in binaries.items():
        folder=out/label;folder.mkdir()
        private_queue=queue
        for name,_ in tasks:private_queue=private_queue.replace('"'+name+'"',json.dumps(str(folder/name)))
        (folder/'worktodo.txt').write_text(private_queue)
        (folder/'scratch').mkdir();config=folder/'fixture.toml'
        config.write_text('[shared]\nworktodo="worktodo.txt"\n[stage1]\ncurve="mont"\ngpu_param=0\ntarget="gpu"\n'
            f'device={args.device}\nsigma=26\nlcm="1"\nauto_stage2=0\nedwards_stage2_transfer=false\nckpt_seconds=0\nlog_file=""\n'+
            'tmp_dir='+json.dumps(str(folder/'scratch'))+'\nfinished='+json.dumps(str(folder/'finished.txt'))+'\n')
        env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA_FORCE_','CUDA_DISABLE_','CUDA_INJECTION','CUDA_MODULE_','CUDA_CACHE_','ECM_','NTT_'))}
        env.update(CUDA_MODULE_LOADING='LAZY',CUDA_CACHE_PATH=str(folder/'cache'))
        Path(env['CUDA_CACHE_PATH']).mkdir()
        if label!='baseline':
            env.update(CUDA_INJECTION64_PATH=str(observer),ECM_JIT_TRACE_DIR=str(folder))
            env['PATH']=str(observer.parent)+';'+env.get('PATH','')
        command=[str(binary),'--config',str(config),'--worker','1','--exp-cache','off']
        (folder/'invocation.json').write_text(json.dumps(dict(command=command,queue=private_queue,environment={k:v for k,v in env.items() if k.startswith(('CUDA_','ECM_'))}),indent=2))
        proc=subprocess.run(command,cwd=folder,env=env,capture_output=True,timeout=1200)
        (folder/'stdout.log').write_bytes(proc.stdout+proc.stderr)
        assert proc.returncode==0,(label,proc.returncode,proc.stdout[-1500:])
        assert not any(line.startswith(('ECMSTAGE1=','ECM2=')) for line in (folder/'worktodo.txt').read_text().splitlines())
        signatures={p.relative_to(folder).as_posix():saved_rows(p) for p in folder.rglob('*.save') if not p.name.endswith('.bak.save')}
        assert len(signatures)==8 and all(len(v)==4 for v in signatures.values()),(label,{k:len(v) for k,v in signatures.items()})
        if reference is None:reference=signatures
        else:assert signatures==reference,('Switching changed canonical points',label)
        observed=set();entries=[]
        if label!='baseline':
            assert 'cuptiSubscribe=0' in (folder/'injection.txt').read_text()
            assert not (folder/'dropped.txt').exists()
            modules=list(folder.glob('module_*.cubin'));assert len(modules)==6,(label,'module count',len(modules))
            assert len((folder/'modules.csv').read_text().splitlines())==6,('module reloaded',label)
            for module in modules:
                for entry in entry_names(inspector,module):
                    entries.append(entry)
                    match=re.search(r'ParamsILj(\d+)ELj\d+E',entry)
                    if match:observed.add(('ed',int(match[1])));continue
                    match=re.search(r'cgbn_params_tILj\d+ELj(\d+)E',entry);assert match,entry
                    param='m2' if 'suyama' in entry and 'ELb1E' in entry else 'm0' if 'suyama' in entry else 'm3'
                    observed.add((param,int(match[1])))
            assert observed==expected_modules,('Unselected module or missing requested module',label,observed)
        result['cases'][label]=dict(exe_sha256=sha(binary),jobs=8,saved_curves=sum(map(len,signatures.values())),modules=sorted(observed),entries=entries)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS switching',label,flush=True)
    assert inputs=={p:sha(p) for p in inputs},'Frozen switching input changed'
    result['complete']=True;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
