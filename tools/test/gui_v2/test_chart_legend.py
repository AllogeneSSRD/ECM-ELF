"""Compile and run the real ImGui chart layout sweep without a GUI or GPU."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,default=Path('build_gui_v2_local'))
    p.add_argument('--expect-jump',action='store_true')
    p.add_argument('--vcvars',type=Path,default=Path('C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat'))
    a=p.parse_args();out=a.out.resolve();out.mkdir();lib=a.build_dir.resolve()/'ecm_gui_v2_backend.lib'
    command=f'''@echo off
call "{a.vcvars}" >nul 2>&1
if errorlevel 1 exit /b 1
cl /nologo /MD /std:c++17 /utf-8 /EHsc /DNOMINMAX /I"{ROOT/'third_party/imgui'}" "{ROOT/'tools/test/gui_v2/chart_legend_test.cpp'}" "{lib}" /Fe:"{out/'chart_test.exe'}" /Fo:"{out/'chart_test.obj'}" >"{out/'build.log'}" 2>&1
exit /b %errorlevel%
'''
    script=out/'compile.cmd';script.write_text(command,encoding='utf-8');subprocess.run(['cmd.exe','/d','/c',str(script)],cwd=ROOT,check=True)
    result=subprocess.run([str(out/'chart_test.exe'),*(['--expect-jump'] if a.expect_jump else [])],cwd=out,text=True,capture_output=True)
    (out/'stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    result.check_returncode();data=json.loads(result.stdout);(out/'result.json').write_text(json.dumps(data,indent=2));print(result.stdout.strip())
if __name__=='__main__':main()
