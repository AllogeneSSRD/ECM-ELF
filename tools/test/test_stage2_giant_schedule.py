"""Build/run production integer scheduling and independent brute-force checks."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
(out/'.gitignore').write_text('*\n',encoding='ascii')
repo=Path(__file__).resolve().parents[2]
vc=Path('C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat')
cmd=out/'build.cmd';exe=out/'fixture.exe'
cmd.write_text('@echo off\ncall "'+str(vc)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
    'cl /nologo /std:c++17 /O2 /EHsc /utf-8 "'+str(repo/'tools/test/stage2_giant_schedule_fixture.cpp')+
    '" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+'"\n',encoding='utf-8')
subprocess.run(['cmd','/d','/c',str(cmd)],check=True,cwd=out)
r=subprocess.run([str(exe)],capture_output=True,text=True)
(out/'result.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
print(r.stdout+r.stderr);r.check_returncode()
