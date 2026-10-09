"""CPU test of interleaved NTT/S4 lifetimes and joint-state compression."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vcvars',type=Path,default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=out/'fixture.exe';script=out/'compile.cmd'
    script.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+str(ROOT/'tools/test/stage2_workspace_memory_fixture.cpp')+
        '" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+'"\n',encoding='utf-8')
    proc=subprocess.run(['cmd','/c',str(script)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    proc=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
    (out/'runtime.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    result=json.loads(proc.stdout)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
