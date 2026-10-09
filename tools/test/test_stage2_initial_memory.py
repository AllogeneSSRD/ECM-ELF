"""Extract native Montgomery/baby allocation sites and audit the initial model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def between(text,first,last):
    if text.count(first)!=1 or text.count(last)!=1:
        raise ValueError('native initialization anchor is not unique')
    return text[text.index(first):text.index(last)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vcvars',type=Path,default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'src/cuda/ecm_cuda_stage2.cu'
    baby=ROOT/'src/cuda/stage2/stage2_baby_host.cuh'
    mont=between(source.read_text(encoding='utf-8'),'static int mont_selftest(','static int xadd_selftest(')
    allocate=between(baby.read_text(encoding='utf-8'),'    BabyBuffer dn,dqx,dqz,da,dm,dj,dx,dz,tree,output,mask;',
                     '    CK(cudaMemcpy(dn.ptr')
    allocate=allocate.replace('tree.allocate(words*8)','tree.allocate(words*8+(mutate?8:0))')
    fragments=dict(NATIVE_MONT_ALLOC=between(mont,'    unsigned long long *da = nullptr, *db = nullptr,',
                                            '    CK(cudaMemcpy(da, ha.data()'),
                   NATIVE_MONT_FREE=between(mont,'    cudaFree(da); cudaFree(db); cudaFree(dn); cudaFree(dout);',
                                           '    unsigned long long bad = 0, first = 0;'),
                   NATIVE_BABY_ALLOC=allocate)
    template=ROOT/'tools/test/stage2_initial_memory_fixture.cpp'
    text=template.read_text(encoding='utf-8').replace('"../../src/core/ecm_stage2_initial_memory.h"',
        '"'+(ROOT/'src/core/ecm_stage2_initial_memory.h').as_posix()+'"')
    for key,value in fragments.items():text=text.replace('// @'+key+'@',value)
    cpp=out/'fixture.cpp';cpp.write_text(text,encoding='utf-8')
    exe=out/'fixture.exe';script=out/'compile.cmd'
    script.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(cpp)+'" /Fe:"'+str(exe)+
        '" /Fo:"'+str(out/'fixture.obj')+'"\n',encoding='utf-8')
    proc=subprocess.run(['cmd','/c',str(script)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    proc=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
    (out/'runtime.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    mutation=subprocess.run([str(exe),'--mutate'],capture_output=True,text=True,timeout=60)
    (out/'mutation.log').write_text(mutation.stdout+mutation.stderr,encoding='utf-8')
    if mutation.returncode==0:raise ValueError('wrong native tree extent was accepted')
    result=json.loads(proc.stdout)
    result['sources']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in
        [source,baby,template,Path(__file__),ROOT/'src/core/ecm_stage2_initial_memory.h']}
    result['native_extent_mutation_rejected']=True
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}))


if __name__=='__main__':main()
