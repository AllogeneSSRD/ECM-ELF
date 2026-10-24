"""Exercise resource lookup using the real release runtime object after relocation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/packaging'))
from opencl_release_payload import read, verify
from cuda_inspection_cache import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fields=read(build/'profile.json')['fields'];identity=verify(build,fields['gmp_root'])
    obj=build/'CMakeFiles/cgbn_opencl.dir/src/ecm/backend/opencl/runtime/runtime.cpp.obj'
    assert identity['compiler_inputs_sha256'][str(obj)]==sha(obj)
    library=Path(fields['opencl_root'])/'lib/x64/OpenCL.lib'
    if not library.exists():library=Path(fields['opencl_root'])/'lib/OpenCL.lib'
    command=[fields['compiler'],'/nologo','/std:c++17','/MD','/EHsc','/utf-8',
        '/I'+str(ROOT/'include'),'/I'+str(ROOT/'src'),'/I'+str(Path(fields['opencl_root'])/'include'),
        str(ROOT/'tools/test/opencl_resource_probe.cpp'),str(obj),str(library),'/Fe'+str(out/'probe.exe'),'/Fo'+str(out/'probe.obj')]
    batch=out/'build.cmd'
    batch.write_text('@echo off\ncall "'+fields['vcvars']+'" -vcvars_ver='+fields['toolset']+' >nul 2>&1\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(command)+'\n',encoding='utf-8')
    compiled=subprocess.run(['cmd','/d','/c',str(batch)],cwd=out,capture_output=True)
    (out/'build.log').write_bytes(compiled.stdout+compiled.stderr)
    assert compiled.returncode==0,compiled.stdout.decode(errors='replace')+compiled.stderr.decode(errors='replace')
    relocated=out/'portable space 中文';relocated.mkdir();shutil.copy2(out/'probe.exe',relocated/'probe.exe')
    shutil.copytree(build/'opencl_resources',relocated/'opencl_resources')
    cwd=out/'unrelated';cwd.mkdir();override=out/'override';override.mkdir()
    (override/'override.cl').write_text('explicit override\n',newline='\n')
    (override/'diagnostics').mkdir()
    (override/'diagnostics/mont.cl').write_text('explicit override\n',newline='\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_','CGBN_','CUDA_'))}
    env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    proc=subprocess.run([str(relocated/'probe.exe'),str(build/'opencl_resources'),str(override)],cwd=cwd,env=env,capture_output=True,timeout=30)
    (out/'probe.log').write_bytes(proc.stdout+proc.stderr)
    (out/'invocation.json').write_text(json.dumps(dict(returncode=proc.returncode,command=proc.args)),encoding='utf-8')
    assert proc.returncode==0,(proc.returncode,proc.stdout.decode(errors='replace')+proc.stderr.decode(errors='replace'))
    for name in ('ecm.exe','gmp-10.dll','libcrypto-3-x64.dll','libssl-3-x64.dll'):shutil.copy2(build/name,relocated/name)
    worker=subprocess.run([str(relocated/'ecm.exe'),'--version'],cwd=cwd,env=env,capture_output=True,timeout=20)
    (out/'worker-version.log').write_bytes(worker.stdout+worker.stderr)
    assert worker.returncode==0 and b'ECM-ELF' in worker.stdout,(worker.returncode,worker.stderr)
    result=dict(complete=True,runtime_object_sha256=sha(obj),worker_sha256=identity['executable_sha256'],
        resources=len(identity['resources']),relocated=str(relocated),unrelated_cwd=True,path_only_system32=True,
        relocated_worker_version=worker.stdout.decode(errors='replace').strip(),gpu_compute=False)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(proc.stdout.decode(errors='replace').strip())


if __name__=='__main__':main()
