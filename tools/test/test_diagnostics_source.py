"""Rebuild the distributed diagnostic source archive without GUI/worker trees."""
import argparse,hashlib,json,os,shutil,subprocess,zipfile
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['build','out','vcvars']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build=a.build.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    with zipfile.ZipFile(build/'diagnostics-source.zip') as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(source)
    rebuilt=out/'rebuilt';log=out/'build.log'
    run=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(source/'tools/build/test/build_diagnostics.ps1'),'-Build',str(rebuilt),'-GmpRoot',str(source/'build-dependencies/gmp'),'-VcVars',str(a.vcvars.resolve()),'-WithoutLocalPari'],capture_output=True,timeout=180)
    log.write_bytes(run.stdout+run.stderr);assert run.returncode==0,log
    command=out/'probe.cmd';command.write_text(f'@echo off\ncall "{a.vcvars.resolve()}" >nul 2>&1\ncmake --build "{rebuilt}" --target ecm_diagnostics_probe\n',encoding='utf-8')
    run=subprocess.run(['cmd.exe','/d','/c',str(command)],capture_output=True,timeout=120)
    with log.open('ab') as f:f.write(run.stdout+run.stderr)
    assert run.returncode==0,log
    # Use the separately distributed DLL/data with the newly compiled host.
    shutil.copy2(build/'libpari.dll',rebuilt/'libpari.dll');shutil.copytree(build/'pari-data',rebuilt/'pari-data')
    env=os.environ.copy();env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    result=subprocess.run([str(rebuilt/'ecm_diagnostics_probe.exe'),'pari','absent-gp.exe','group','3436193553215096358758503','1955774944','0'],env=env,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr;value=json.loads(result.stdout)
    assert value['backend']=='pari' and value['status']=='complete' and value['all_proven'],value
    assert value['group_order']=='3436193553215760556775100' and value['point_order']=='286349462767980046397925'
    (out/'result.json').write_text(json.dumps(dict(passed=True,source_archive_sha256=hashlib.sha256((build/'diagnostics-source.zip').read_bytes()).hexdigest(),no_gui_source_required=True,no_gp_required=True,analysis=value),indent=2),encoding='utf-8')
    print('PASS: distributed source archive independently rebuilds and computes with PARI DLL')
if __name__=='__main__':main()
