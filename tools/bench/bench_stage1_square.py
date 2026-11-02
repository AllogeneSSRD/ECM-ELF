"""Complete Stage1 square-backend A/B, with saved-point and independent checks.

Use separately built production executables to preserve each kernel's own
register allocation. All measured calls complete B1; no progress projection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from bench_edwards_gpu_stage1 import reference, validate
from bench_stage2_point_matrix import environment
from bench_stage2_version import check_gpu_processes,load,write,stamp
from gpu_telemetry import TelemetryRecorder


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(build,dest,repo,source_archive=None):
    manifest=load(build/'stage1_local_manifest.json')
    assert sha(build/'ecm_cuda.exe')==manifest['executable_sha256'].lower()
    for path,digest in manifest['objects'].items():assert sha(build/path)==digest.lower(),path
    dest.mkdir();(dest/'.gitignore').write_text('*\n')
    for source in [build/'ecm_cuda.exe',*build.glob('*.dll'),build/'stage1_local_manifest.json',
                   build/'profile.json',build/'CMakeCache.txt',build/'compile_commands.json']:
        shutil.copy2(source,dest/source.name)
    # Capture the source contract declared by the formal builder. Generated
    # modules may live under the build leaf and are kept with their identities.
    def source_path(name):return Path(name) if Path(name).is_absolute() else repo/name
    def label(path):return path.relative_to(repo).as_posix() if path.is_relative_to(repo) else path.name
    if source_archive is not None:
        # Explicit frozen baseline: verify every declared dependency from its
        # archive, not from a newer working tree. Exe/objects stay hash-checked.
        with zipfile.ZipFile(source_archive) as archive:
            for name,digest in manifest['source_hashes'].items():
                assert hashlib.sha256(archive.read(label(source_path(name)))).hexdigest()==digest.lower(),name
        shutil.copy2(source_archive,dest/'sources.zip')
    else:
        with zipfile.ZipFile(dest/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name,digest in manifest['source_hashes'].items():
                source=source_path(name);assert sha(source)==digest.lower(),source
                archive.write(source,label(source))
    return dict(exe_sha256=sha(dest/'ecm_cuda.exe'),sources_sha256=sha(dest/'sources.zip'),
        profile_sha256=manifest['profile_sha256'],files={p.name:sha(p) for p in dest.iterdir() if p.is_file()})


def main():
    sys.set_int_max_str_digits(0)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline-source-archive',type=Path,help='Explicit verified source snapshot for a frozen baseline')
    p.add_argument('--candidate-source-archive',type=Path)
    p.add_argument('--candidate-layout',type=int,choices=(0,1,2,3),default=0)
    p.add_argument('--input-plan',type=Path,default=Path('data/experiments/stage2_point_matrix_20261031/min1_full03/plan.json'))
    p.add_argument('--exponents',type=int,nargs='+',default=[4001,8011,16319])
    p.add_argument('--models',nargs='+',choices=('mont0','ed'),default=['mont0','ed'])
    p.add_argument('--b1',type=int,default=5000)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--sigma',type=int,default=20260922)
    p.add_argument('--process-policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert a.rounds>=3 and a.b1>=2
    repo=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
    allowed=load(a.process_policy)['allowed_processes']
    bins=out/'bin';bins.mkdir();identity={}
    for name,build,archive in [('baseline',a.baseline.resolve(),a.baseline_source_archive),('carrysave',a.candidate.resolve(),a.candidate_source_archive)]:
        cache=(build/'CMakeCache.txt').read_text(encoding='utf-8')
        assert re.search(r'^ECM_CGBN_SQUARE_BACKEND:STRING='+('0' if name=='baseline' else '1')+r'$',cache,re.M),build
        layout=re.search(r'^ECM_CGBN_SQUARE_LAYOUT:STRING=(\d+)$',cache,re.M)
        if name=='carrysave':assert int(layout[1] if layout else 0)==a.candidate_layout,build
        identity[name]=freeze(build,bins/name,repo,archive)
    plan=load(a.input_plan);cases=[c for c in plan['cases'] if c['exponent'] in a.exponents]
    assert {c['exponent'] for c in cases}==set(a.exponents)
    for c in cases:
        assert int(c['N_hex'],16).bit_length()==c['bits']
    shutil.copy2(__file__,out/'collector.py')
    write(out/'inputs.json',cases)
    config=out/'minimal.toml'
    config.write_text('[shared]\n[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = 0\n',encoding='utf-8')
    report=dict(complete=False,started_utc=stamp(),identity=identity,device=a.device,
        b1=a.b1,sigma=a.sigma,warmups=0,rounds=a.rounds,candidate_layout=a.candidate_layout,
        runs=[],failures=[],allowed_processes=allowed,
        timing='complete process wall; Mont GPU event includes preparation; Ed loop excludes dictionary; oracle excluded')
    write(out/'summary.json',report);signatures={};oracle={}
    try:
        for case in cases:
            n=int(case['N_hex'],16)
            for model in a.models:
                for round_index in range(a.rounds):
                    for backend in (['baseline','carrysave'] if round_index%2==0 else ['carrysave','baseline']):
                        folder=out/f'm{case["exponent"]}_{model}_{backend}_r{round_index}';folder.mkdir()
                        exe=bins/backend/'ecm_cuda.exe'
                        # Geometry is obtained from the actual worker; its SM
                        # count and compiled tiers must match between backends.
                        env=environment();env['ECM_GPU_STAGE1_ALGO']='ladder';env['ECM_EDWARDS_STAGE1_ALGO']='naf'
                        info_cmd=[str(exe),'--config',str(config),'--gpu-info','--curve','mont' if model=='mont0' else 'ed','--target','gpu','-d',str(a.device)]
                        if model=='mont0':info_cmd+=['--gpu-param','0']
                        info=subprocess.check_output(info_cmd,cwd=folder,env=env,text=True)
                        (folder/'gpu_info.log').write_text(info,encoding='utf-8')
                        fields=dict(re.findall(r'^(\w+)=(.*)$',info,re.M))
                        tiers=sorted(tuple(map(int,v)) for v in re.findall(r'^tier bits=(\d+) tpb=(\d+) tpi=(\d+)',info,re.M))
                        container,tpb,tpi=next(t for t in tiers if n.bit_length()+int(fields['carry_bits'])<=t[0])
                        curves=2*int(fields['sm_count'])*tpb//tpi
                        save=folder/'completed.save'
                        cmd=[str(exe),'--config',str(config),'--curve','mont' if model=='mont0' else 'ed','--target','gpu','-d',str(a.device),'-sigma',str(a.sigma),'-gpucurves',str(curves),'--ckpt','0','--exp-cache',str(out/'exp_cache'),'--tmp-dir',str(folder/'scratch'),'-save',str(save),'-v']
                        if model=='mont0':cmd+=['--gpu-param','0']
                        else:cmd+=['--naf-w','6']
                        cmd += [str(a.b1),'0']
                        write(folder/'command.json',cmd);(folder/'input.txt').write_text(str(n)+'\n')
                        print('start',folder.name,flush=True)
                        mon=TelemetryRecorder(a.device,folder/'telemetry.jsonl',interval=.1)
                        try:
                            check_gpu_processes(mon.monitor,out,folder/'before.json',allowed=allowed)
                            begin=time.perf_counter()
                            run=subprocess.run(cmd,input=str(n)+'\n',capture_output=True,text=True,cwd=folder,env=env,timeout=600)
                            wall=time.perf_counter()-begin
                            (folder/'run.log').write_text(run.stdout+run.stderr,encoding='utf-8')
                            write(folder/'exit.json',dict(returncode=run.returncode))
                            check_gpu_processes(mon.monitor,out,folder/'after.json',allowed=allowed)
                        finally:mon.close()
                        assert run.returncode==0,folder
                        rows=validate(save,n,a.b1,curves,a.sigma,model)
                        key=(case['exponent'],model)
                        if key not in signatures:
                            checks=[]
                            for index in sorted({0,len(rows)-1}):
                                expected=reference(n,rows[index][0],a.b1,model)
                                assert rows[index][1:]==expected,(folder,index)
                                checks.append(dict(index=index,sigma=rows[index][0],matched=True))
                            oracle[str(key)]=checks;signatures[key]=rows
                            write(folder/'oracle.json',checks)
                        else:assert rows==signatures[key],('entire saved batch differs',folder)
                        text=run.stdout+run.stderr
                        if model=='mont0':
                            actual=re.findall(r'CGBN<(\d+),\s*(\d+)> kernel, N is (\d+) bits \((\d+) blocks x (\d+) threads\)',text)
                            assert len(actual)==1 and tuple(map(int,actual[0]))==(tpi,container,n.bit_length(),2*int(fields['sm_count']),tpb),folder
                        else:
                            actual=re.findall(r'GPU Edwards geometry: container_bits=(\d+) tpi=(\d+) tpb=(\d+) blocks=(\d+)',text)
                            assert len(actual)==1 and tuple(map(int,actual[0]))==(container,tpi,tpb,2*int(fields['sm_count'])),folder
                        metrics=re.findall(r'GPU Edwards NAF:.*?loop_gpu_ms=([\d.]+)',text)
                        mont=re.findall(r'GPU stage1 returned:\s*0\s+gputime=([\d.]+) ms',text)
                        assert len(metrics if model=='ed' else mont)==1,folder
                        seconds=float((metrics if model=='ed' else mont)[0])/1000
                        report['runs'].append(dict(exponent=case['exponent'],bits=case['bits'],model=model,
                            backend=backend,round=round_index,container=container,tpi=tpi,tpb=tpb,curves=curves,
                            wall_seconds=wall,gpu_seconds=seconds,gpu_seconds_per_curve=seconds/curves,
                            save_sha256=sha(save),folder=str(folder),matched_full_batch=True))
                        report['oracle']=oracle;write(out/'summary.json',report)
                        print('done',folder.name,'wall',round(wall,4),'gpu',seconds,flush=True)
        report['aggregates']=[]
        for case in cases:
            for model in a.models:
                groups={}
                for backend in ('baseline','carrysave'):
                    runs=[r for r in report['runs'] if r['exponent']==case['exponent'] and r['model']==model and r['backend']==backend]
                    assert len(runs)==a.rounds
                    groups[backend]={k:statistics.median(r[k] for r in runs) for k in ('wall_seconds','gpu_seconds','gpu_seconds_per_curve')}
                report['aggregates'].append(dict(exponent=case['exponent'],model=model,groups=groups,
                    gpu_ratio=groups['carrysave']['gpu_seconds']/groups['baseline']['gpu_seconds']))
        report['complete']=True;report['finished_utc']=stamp()
    except Exception as error:
        report['failures'].append(str(error));raise
    finally:write(out/'summary.json',report)


if __name__=='__main__':main()
