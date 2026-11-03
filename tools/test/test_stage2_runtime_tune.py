"""Compile the production profile/statistics/selection seam; no CUDA calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture',choices=('runtime','point-backends','bounds','phase-model'),default='runtime')
    p.add_argument('--profile',type=Path,help='Single-anchor phase-model prediction audit; no GPU calls')
    p.add_argument('--vcvars',type=Path,default=Path(r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'.gitignore').write_text('*\n',encoding='ascii')
    gmp=ROOT/'third_party/gmp-zen3/dist';exe=out/'fixture.exe'
    if a.profile and a.fixture!='phase-model':p.error('--profile requires --fixture phase-model')
    source={'point-backends':'stage2_point_backend_fixture.cpp','bounds':'stage2_runtime_bounds_fixture.cpp',
            'phase-model':'stage2_phase_prediction_fixture.cpp'}.get(a.fixture,'stage2_runtime_tune_fixture.cpp')
    cmd=out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(a.vcvars)+'" >nul 2>&1\nif errorlevel 1 exit /b 1\n'
        'cl /nologo /std:c++17 /EHsc /O2 /utf-8 /I"'+str(ROOT/'include')+'" /I"'+str(ROOT/'src')+'" /I"'+str(gmp/'include')+'" "'+
        str(ROOT/'tools/test'/source)+'" /Fe:"'+str(exe)+'" /Fo:"'+str(out/'fixture.obj')+
        '" /link /LIBPATH:"'+str(gmp/'lib')+'" gmp.lib\n',encoding='utf-8')
    r=subprocess.run(['cmd','/c',str(cmd)],capture_output=True,text=True,errors='replace',timeout=90)
    (out/'compile.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    for dll in (gmp/'bin').glob('*.dll'):shutil.copy2(dll,out/dll.name)
    r=subprocess.run([str(exe),str(out)]+([str(a.profile.resolve())] if a.profile else []),capture_output=True,text=True,errors='replace',timeout=120)
    (out/'fixture.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    result=json.loads(r.stdout)
    if a.fixture=='phase-model':
        files=[ROOT/'tools/test'/source,Path(__file__).resolve(),
               ROOT/'src/ecm/stage2/planning/width_prediction.h',ROOT/'src/ecm/stage2/planning/portable_model.h',
               ROOT/'src/ecm/stage2/planning/runtime_tune.h',ROOT/'src/ecm/stage2/planning/d_catalog.h']
        if a.profile:files.append(a.profile.resolve())
        identities={}
        for path in files:
            label=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
            identities[label]=hashlib.sha256(path.read_bytes()).hexdigest()
            target=out/'sources'/label if path.is_relative_to(ROOT) else out/'input_profile.toml'
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        result['input_sha256']=identities
        result['fixture_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=result['complete'],prediction_only=True,rows=len(result['rows']),
                         result=str(out/'result.json')) if a.profile else result))
if __name__=='__main__':main()
