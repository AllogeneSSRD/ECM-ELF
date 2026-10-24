"""Frozen-input SASS/PTX, arithmetic and actually loaded module qualification.

Runs only in fresh private directories. Does not change clocks, drivers, queues,
global CUDA cache or user configuration. Instrumented scope runs and uninstrumented
timing samples are separate; absence of direct JIT records is never reported as zero.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/ecm_report'))
from result_format import normalize_result
from test_cuda_p3_runtime import stable
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def expression(bits):
    primes=[p for p in range(3,2000,2) if all(p%d for d in range(3,int(p**.5)+1,2))]
    delta=1
    while any((pow(2,bits-1,p)+delta)%p==0 for p in primes):delta+=2
    return f'(2^{bits-1}+{delta})'


def saved_rows(path):
    rows=[]
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if 'SIGMA=' not in line:continue
            fields=dict(part.strip().split('=',1) for part in line.split(';') if '=' in part)
            for key in ('WHO','TIME'):fields.pop(key,None)
            rows.append(fields)
    return rows


def entry_names(inspector,binary):
    output=subprocess.check_output([str(inspector),'--dump-elf-symbols',str(binary)],stderr=subprocess.STDOUT).decode(errors='replace')
    return re.findall(r'^STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+(\S+)',output,re.M)


def stage1_scope(names,curve,param,capacity):
    assert names,'No loaded entry points observed'
    for name in names:
        if curve=='ed':
            match=re.search(r'ParamsILj(\d+)ELj\d+E',name)
            assert match and int(match[1])==capacity,('unselected Edwards module',name,capacity)
        else:
            match=re.search(r'cgbn_params_tILj\d+ELj(\d+)E',name)
            assert match and int(match[1])==capacity,('unselected Montgomery width',name,capacity)
            if param in (0,2):
                assert 'kernel_double_add_suyama' in name and ('ELb1E' in name)==(param==2),('wrong param',name,param)
            else:assert 'kernel_double_add' in name and 'suyama' not in name,('wrong curve family',name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,default=ROOT/'build_cuda_release')
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--observer',type=Path,required=True)
    parser.add_argument('--inspector',type=Path,default=Path(os.environ.get('CUDA_PATH_V13_3',str(Path(os.environ.get('ProgramFiles','C:/Program Files'))/'NVIDIA GPU Computing Toolkit/CUDA/v13.3')))/'bin/cuobjdump.exe',help='Inspector compatible with the current driver-generated cubin format')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--profile',choices=['all','cu11.8','cu13.3'],default='all')
    parser.add_argument('--stage',choices=['all','stage1','stage2'],default='all')
    parser.add_argument('--fixtures',type=Path,help='Independent Stage2 fixtures.json, checked by save_sha256')
    parser.add_argument('--ptx-fixtures',type=Path,help='Explicit derived PTX-only test images; original SASS/default cases still use release EXEs')
    parser.add_argument('--tiers',choices=['all','representative','smoke'],default='all')
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--skip-sampling',action='store_true')
    parser.add_argument('--sampling-only',action='store_true')
    parser.add_argument('--allow-grouped-stage2',action='store_true',help='Explicitly test grouped Stage2 math/timing without claiming per-capacity JIT isolation')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profiles=['cu11.8','cu13.3'] if args.profile=='all' else [args.profile]
    observer=args.observer.resolve();baseline=args.baseline.resolve();build=args.build_root.resolve()
    inspector=args.inspector.resolve();assert inspector.is_file(),'Provide a cuobjdump compatible with the current driver'
    gpu=subprocess.check_output(['nvidia-smi','-i',str(args.device),'--query-gpu=name,uuid,driver_version,compute_cap,power.limit,clocks.current.sm','--format=csv']).decode()
    (out/'device.csv').write_text(gpu)
    collector=out/'collector.py';collector.write_bytes(Path(__file__).read_bytes())
    result=dict(complete=False,device=gpu,profiles=profiles,selection=dict(stage=args.stage,tiers=args.tiers,sampling_only=args.sampling_only),
        inputs={str(observer):sha(observer),str(inspector):sha(inspector),str(Path(__file__).resolve()):sha(__file__)},cases=[],samples=[],unverified=['Maxwell hardware','Pascal hardware','old drivers','native sm60','native sm80'])
    for stage,filename in [('stage1','ecm_cuda.exe'),('stage2','ecm_cuda_stage2.exe')]:
        path=baseline/stage/filename;result['inputs'][str(path)]=sha(path)
    binaries={}
    inspectors={}
    ptx_binaries={}
    stage2_layouts={}
    for profile in profiles:
        binaries[profile]={}
        for stage in (('stage1','stage2') if args.stage=='all' else (args.stage,)):
            record=load(build/'components'/f'{profile}.{stage}.json')
            manifest=load(record['manifest']);binary=Path(record['exe']).resolve()
            if stage=='stage2':
                layout=manifest.get('kernel_layout','isolated' if manifest.get('module_directory') else 'grouped')
                assert layout=='isolated' or args.allow_grouped_stage2,'Grouped Stage2 requires --allow-grouped-stage2; per-capacity JIT isolation is not its scope contract'
                stage2_layouts[profile]=layout
            assert sha(binary).lower()==manifest.get('executable_sha256',manifest.get('sha256')).lower()
            result['inputs'][str(binary)]=sha(binary);binaries[profile][stage]=binary
            for path in (Path(record['manifest']).resolve(),binary.parent/'gmp-10.dll'):
                result['inputs'][str(path)]=sha(path)
            inspectors[profile]=inspector
    result['stage2_kernel_layouts']=stage2_layouts
    if args.ptx_fixtures:
        derived=load(args.ptx_fixtures)
        assert derived['complete'] and derived['kind']=='derived_ptx_only_test_images'
        result['inputs'][str(args.ptx_fixtures.resolve())]=sha(args.ptx_fixtures)
        for stage,item in derived['components'].items():
            if stage not in binaries.get('cu13.3',{}):continue
            assert sha(binaries['cu13.3'][stage])==item['source_exe_sha256']
            binary=Path(item['exe']);assert sha(binary)==item['sha256']
            ptx_binaries[str(binaries['cu13.3'][stage])]=binary
            result['inputs'][str(binary)]=sha(binary)
    def persist():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    base_env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_','NTT_','CUDA_FORCE_','CUDA_DISABLE_','CUDA_INJECTION','CUDA_MODULE_','CUDA_CACHE_'))}
    base_env['NTT_TARGET_LEAF_HASH']='1'
    def invoke(binary,name,arguments,input=None,profile=None,mode='default',cache=None,observe=True):
        original_binary=binary
        if mode=='ptx':binary=ptx_binaries.get(str(binary),binary)
        folder=out/name;folder.mkdir(parents=True)
        config=folder/'fixture.toml'
        config.write_text('[stage1]\nauto_stage2=0\nedwards_stage2_transfer=false\n[stage2]\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
        env=base_env|{'CUDA_CACHE_MAXSIZE':str(1024*1024*1024)}
        if mode!='default':env['CUDA_MODULE_LOADING']='LAZY'
        if mode=='ptx':env['CUDA_FORCE_PTX_JIT']='1'
        elif mode=='sass':env['CUDA_DISABLE_PTX_JIT']='1'
        cache=Path(cache or folder/'cache');cache.mkdir(parents=True,exist_ok=True);env['CUDA_CACHE_PATH']=str(cache)
        if observe:
            env.update(CUDA_INJECTION64_PATH=str(observer),ECM_JIT_TRACE_DIR=str(folder))
            env['PATH']=str(observer.parent)+';'+env.get('PATH','')
        command=[str(binary),'--config',str(config),*[str(v).replace('$RUN',str(folder)) for v in arguments]]
        (folder/'invocation.json').write_text(json.dumps(dict(command=command,input=input,environment={k:v for k,v in env.items() if k.startswith(('CUDA_','ECM_'))}),indent=2))
        start=time.perf_counter()
        proc=subprocess.run(command,input=input.encode() if input else None,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,cwd=folder,timeout=1800)
        seconds=time.perf_counter()-start;text=proc.stdout.decode('utf-8','replace');(folder/'stdout.log').write_text(text,encoding='utf-8')
        debug=folder/'debug.log'
        if debug.is_file():text+='\n'+debug.read_text(encoding='utf-8-sig',errors='replace')
        assert proc.returncode==0,(name,proc.returncode,text[-2500:])
        names=[];loaded=[]
        if observe:
            assert 'cuptiSubscribe=0' in (folder/'injection.txt').read_text(),('observer unavailable',folder)
            assert not (folder/'dropped.txt').exists(),('trace dropped',folder)
            for cubin in sorted(folder.glob('module_*.cubin')):
                names+=entry_names(inspectors[profile],cubin)
                data=cubin.read_bytes();assert data[:4]==b'\x7fELF' and data[4]==2
                flags=struct.unpack_from('<I',data,48)[0]
                assert data[8] in (7,8),('unknown CUDA ELF ABI',cubin,data[8])
                arch=(flags>>8)&255 if data[8]>=8 else flags&255
                loaded.append(dict(file=cubin.name,sha256=sha(cubin),elf_flags=hex(flags),architecture=arch))
            if mode=='ptx' and names:
                # On this sm89 host the shipped sm60/sm80 SASS differs from a
                # driver-generated sm89 cubin. Never infer JIT from env alone.
                device_arch=int(float(next(csv.reader(gpu.splitlines()[1:]))[3])*10)
                assert all(item['architecture']==device_arch for item in loaded),('PTX request loaded another SASS image',name,loaded)
        direct=[]
        if (folder/'jit.csv').exists():
            direct=list(csv.reader((folder/'jit.csv').open()))
        operation_seconds={str(op):(sum((int(row[2])-int(row[1]))/1e9 for row in direct if int(row[0])==op) if direct else None) for op in (1,2,3)}
        for filename in ('engine.log','debug.log'):
            if (folder/filename).exists():text+='\n'+(folder/filename).read_text(encoding='utf-8',errors='replace')
        return folder,text,dict(wall_seconds=seconds,entries=names,loaded_modules=loaded,image_kind=('derived_ptx_only' if binary!=original_binary else 'release'),executed_sha256=sha(binary),jit_activities=direct,jit_operation_seconds=operation_seconds,
            jit_measurement='CUPTI: 1 cache-load, 2 cache-store, 3 compile' if direct else 'not separately observed; wall includes initialization and loading')
    tiers=[tuple(map(int,v)) for v in re.findall(r'^ECM_CUDA_TIER\((\d+),\d+,\d+,([01]),', (ROOT/'src/ecm/backend/cuda/dispatch/tiers.def').read_text(),re.M)]
    if args.tiers=='representative':tiers=[v for v in tiers if v[0] in (128,2560,10240)]
    elif args.tiers=='smoke':tiers=[v for v in tiers if v[0]==512]
    if args.stage=='stage2' or args.sampling_only:tiers=[]
    fixtures=[]
    for profile in profiles:
        if 'stage1' not in binaries[profile] or args.sampling_only:continue
        for curve in ('mont','ed'):
            _,_,measurement=invoke(binaries[profile]['stage1'],f'{profile}/metadata_{curve}',
                ['--gpu-info','--curve',curve,'-d',args.device],profile=profile,mode='ptx')
            assert not measurement['entries'] and not measurement['jit_activities'],('metadata triggered JIT',profile,curve)
            result['cases'].append(dict(stage='metadata',profile=profile,curve=curve,measurement=measurement));persist()
    for capacity,mont in tiers:
        models=[('ed_naf','ed',None),('ed_binary','ed',None)]+([('mont0','mont',0),('mont2','mont',2),('mont3','mont',3)] if mont else [])
        for model,curve,param in models:
            bits=capacity-(1 if curve=='ed' else 6);expr=expression(bits)
            arguments=['--curve',curve,'--target','gpu','-d',args.device,'-sigma',26,'-gpucurves',4,'--ckpt',0,'--exp-cache','off','--tmp-dir','$RUN/scratch','-save','$RUN/point.save']
            if param is not None:arguments+=['--gpu-param',param]
            if model=='ed_binary':base_env['ECM_EDWARDS_STAGE1_ALGO']='binary'
            else:base_env.pop('ECM_EDWARDS_STAGE1_ALGO',None)
            arguments += [20,0]
            reference,_,_=invoke(baseline/'stage1/ecm_cuda.exe',f'reference/{model}_{capacity}',arguments,expr+'\n',observe=False)
            expected=saved_rows(reference/'point.save');assert len(expected)==4,('reference did not finish four curves',model,capacity)
            if model in ('mont0','ed_naf') and capacity in (128,512,1024,2048,4096,8192,16384):
                fixtures.append(dict(name=f'{model}_{capacity}',save=reference/'point.save',capacity=next(v for v in (4,8,16,32,64,128,256) if v>=(bits+63)//64)))
            for profile in profiles:
                modes=['ptx','sass','default'] if profile=='cu13.3' else ['ptx','default']
                for mode in modes:
                    folder,_,measurement=invoke(binaries[profile]['stage1'],f'{profile}/{mode}/{model}_{capacity}',arguments,expr+'\n',profile,mode,out/'cache'/profile)
                    assert saved_rows(folder/'point.save')==expected,('point changed',profile,model,capacity,mode)
                    stage1_scope(measurement['entries'],curve,param,capacity)
                    result['cases'].append(dict(stage='stage1',profile=profile,mode=mode,model=model,capacity=capacity,measurement=measurement))
                    persist();print('PASS',profile,mode,model,capacity,flush=True)
    base_env.pop('ECM_EDWARDS_STAGE1_ALGO',None)
    if args.fixtures:
        result['inputs'][str(args.fixtures.resolve())]=sha(args.fixtures)
        for item in load(args.fixtures)['fixtures']:
            if not (item['name'].startswith('generic') or item['name']=='m16381'):continue
            path=Path(item['save']);assert sha(path)==item['save_sha256'];result['inputs'][str(path)]=sha(path)
            fixtures.append(dict(name=item['name'],save=path,capacity=next(v for v in (4,8,16,32,64,128,256) if v>=(item['bits']+63)//64)))
    if args.stage!='stage1' and not args.sampling_only:
        assert fixtures,'Stage2 requires generated Stage1 fixtures or --fixtures'
        for fixture in fixtures:
            arguments=['--save',fixture['save'],'--b2',13230,'--d',210,'--device',args.device,'--curves',4,'--carrier-exponent',0,'--short-calibration',0,'--tune-runtime',0,'--factor-only','--results','$RUN/result.jsonl','--log','$RUN/engine.log','--debug-log-file','$RUN/debug.log','--log-level','debug']
            reference,text,_=invoke(baseline/'stage2/ecm_cuda_stage2.exe','reference/stage2_'+fixture['name'],arguments,observe=False)
            expected=[stable(normalize_result(json.loads(line))) for line in (reference/'result.jsonl').read_text().splitlines()]
            leaves=re.findall(r'^target_descent_values: (.*)$',text,re.M)
            for profile in profiles:
                for mode in (['ptx','sass','default'] if profile=='cu13.3' else ['ptx','default']):
                    folder,text,measurement=invoke(binaries[profile]['stage2'],f'{profile}/{mode}/stage2_'+fixture['name'],arguments,profile=profile,mode=mode,cache=out/'cache'/profile)
                    actual=[stable(normalize_result(json.loads(line))) for line in (folder/'result.jsonl').read_text().splitlines()]
                    assert actual==expected and actual and all(not row['bad_factors'] for row in actual),('Stage2 result changed',profile,mode)
                    for token in ('mont_selftest: cases=2048 mismatches=0','gmp_check_bad=0','pending=0'):
                        assert token in text,(profile,fixture['name'],token)
                    assert leaves and re.findall(r'^target_descent_values: (.*)$',text,re.M)[-1:]==leaves[-1:],'Stage2 target leaves changed'
                    if stage2_layouts[profile]=='isolated':
                        for cubin_name in measurement['entries']:
                            if re.search(r'(?:s2g_|s4_reduce_kernel|s4_mersenne_fixture)',cubin_name):
                                capacity_match=re.search(r'ILi(\d+)E',cubin_name)
                                if capacity_match:assert int(capacity_match[1])==fixture['capacity'],('unselected Stage2 capacity',cubin_name)
                    assert measurement['entries'],'Stage2 scope was not observed'
                    result['cases'].append(dict(stage='stage2',profile=profile,mode=mode,fixture=fixture['name'],measurement=measurement));persist();print('PASS',profile,mode,fixture['name'],flush=True)
    if not args.skip_sampling:
        assert args.samples>=5,'At least five independent cold/warm samples are required'
        workloads=[]
        if args.stage!='stage2':
            arguments=['--curve','mont','--target','gpu','-d',args.device,'-sigma',26,'-gpucurves',4,'--gpu-param',0,'--ckpt',0,'--exp-cache','off','--tmp-dir','$RUN/scratch','-save','$RUN/point.save',20,0]
            bits=506 if args.tiers=='smoke' else 122
            workloads.append(('stage1',arguments,expression(bits)+'\n',f'mont0_input{bits}'))
        if args.stage!='stage1':
            assert fixtures,'Stage2 sampling needs --fixtures or generated Stage1 saves'
            fixture=fixtures[0]
            arguments=['--save',fixture['save'],'--b2',13230,'--d',210,'--device',args.device,'--curves',1,'--carrier-exponent',0,'--short-calibration',0,'--tune-runtime',0,'--factor-only','--results','$RUN/result.jsonl','--log','$RUN/engine.log','--log-level','debug']
            workloads.append(('stage2',arguments,None,fixture['name']))
        for stage,arguments,value,workload in workloads:
            for profile in profiles:
                # Timing with/without tracing has separate independent caches.
                for observed in (False,True):
                    for sample in range(args.samples):
                        cache=out/('jit-sampling-cache' if observed else 'sampling-cache')/stage/profile/str(sample)
                        for state in ('cold','warm'):
                            folder,_,measurement=invoke(binaries[profile][stage],f'sampling/{stage}/{profile}/{observed}/{sample}/{state}',arguments,value,profile,'ptx',cache,observe=observed)
                            result['samples'].append(dict(stage=stage,workload=workload,profile=profile,instrumented=observed,sample=sample,cache_state=state,binary_sha256=measurement['executed_sha256'],source_binary_sha256=sha(binaries[profile][stage]),measurement=measurement));persist()
                            print('SAMPLE',stage,profile,observed,sample,state,measurement['wall_seconds'],flush=True)
        result['timing_summary']={stage:{p:{s:dict(samples=len(v:=[r['measurement']['wall_seconds'] for r in result['samples'] if r['stage']==stage and r['profile']==p and r['cache_state']==s and not r['instrumented']]),median=statistics.median(v),minimum=min(v),maximum=max(v)) for s in ('cold','warm')} for p in profiles} for stage,_,_,_ in workloads}
        result['jit_summary']={stage:{p:{s:dict(samples=len(v:=[r['measurement']['jit_operation_seconds']['3'] for r in result['samples'] if r['stage']==stage and r['profile']==p and r['cache_state']==s and r['instrumented'] and r['measurement']['jit_operation_seconds']['3'] is not None]),compile_median_seconds=statistics.median(v) if v else None) for s in ('cold','warm')} for p in profiles} for stage,_,_,_ in workloads}
    assert result['inputs']=={p:sha(p) for p in result['inputs']},'Frozen input changed during validation'
    result['complete']=True;persist()


if __name__=='__main__':main()
