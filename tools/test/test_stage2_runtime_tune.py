"""Compile the production profile/statistics/selection seam; no CUDA calls."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vcvars',type=Path,default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'.gitignore').write_text('*\n',encoding='ascii')
    gmp=ROOT/'third_party/gmp-zen3/dist';exe=out/'fixture.exe'
    cmd=out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(ROOT/'include')+'" /I"'+str(ROOT/'src')+'" /I"'+str(gmp/'include')+'" "'+
        str(ROOT/'tools/test/stage2_runtime_tune_fixture.cpp')+'" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+
        '" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n',encoding='utf-8')
    r=subprocess.run(['cmd','/c',str(cmd)],capture_output=True,text=True,errors='replace',timeout=90)
    (out/'compile.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    for dll in (gmp/'bin').glob('*.dll'):shutil.copy2(dll,out/dll.name)
    r=subprocess.run([str(exe),str(out)],capture_output=True,text=True,errors='replace',timeout=120)
    (out/'fixture.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    result=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
