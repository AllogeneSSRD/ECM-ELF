"""Check complete release directories/ZIPs and relocate them away from the SDK.

Runtime smoke uses only a private copy, empty queues and private UI state. It
does not start compute workers, change clocks or touch the user's configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(folder,expected):
    actual={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Release payload hash/membership mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-only',action='store_true')
    args=parser.parse_args();root=args.release.resolve();manifest=read(args.manifest)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,manifest_sha256=sha(args.manifest),collector_sha256=sha(__file__),profiles={},limitations=['Same Windows installation and graphics driver; not a clean OS or old-driver qualification'])
    def persist():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    for profile,identity in manifest['profiles'].items():
        assert profile in ('cu11.8','cu13.3')
        folder=root/profile;files=identity['files_sha256'];verify(folder,files)
        for name in ('ecm_cuda.exe','ecm_cuda_stage2.exe','ecm_gui_v2.exe','ecm_group_order_ui.exe','ecm_diagnostics_host.exe','gmp-10.dll','ecm.toml','ecm.toml.example','worktodo.txt','source/release-source.zip'):
            assert name in files,('missing required payload',profile,name)
        assert sha(folder/'gmp-10.dll')==manifest['gmp_sha256']
        with zipfile.ZipFile(folder/'source/release-source.zip') as sources:
            assert not any(name.startswith('tools/test/') and any(part.startswith('_') for part in Path(name).parts[2:-1]) for name in sources.namelist()),'Private test output inside source archive'
            assert 'tools/build/build_release.ps1' in sources.namelist()
            assert '.gitignore' in sources.namelist()
        assert set(p.name for p in (folder/'assets').iterdir())=={'pet_ely.png','pet_ely.json','CLASSIC-ELY-NOTICE.md','ELYTerminal-LICENSE.txt'}
        assert not any('manifest' in Path(name).name.lower() and name.endswith('.json') for name in files)
        archive=root/f'{root.name}-{profile}-win-x64.zip'
        assert sha(archive)==identity['archive_sha256']
        assert archive.with_suffix('.zip.sha256').read_text().split()[0]==sha(archive)
        with zipfile.ZipFile(archive) as value:
            assert set(value.namelist())=={profile+'/'+name for name in files}
            for name,digest in files.items():assert hashlib.sha256(value.read(profile+'/'+name)).hexdigest()==digest
        # Negative identity checks operate on expected metadata only, never on
        # the release directory or installed dependencies.
        bad=dict(files);bad['gmp-10.dll']='0'*64
        try:verify(folder,bad)
        except ValueError:pass
        else:raise AssertionError('Substituted GMP identity accepted')
        bad=dict(files);bad.pop('ecm_cuda_stage2.exe')
        try:verify(folder,bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete worker membership accepted')
        record=dict(files=len(files),archive_sha256=sha(archive),negative_gmp_and_worker_checks=True)
        if not args.static_only:
            relocated=out/('portable space 中文 '+profile);shutil.copytree(folder,relocated)
            cwd=out/('unrelated '+profile);cwd.mkdir()
            env={k:v for k,v in os.environ.items() if not k.startswith(('ECM_','NTT_','CUDA_','VK_'))}
            env['PATH']=str(Path(os.environ['SystemRoot'])/'System32');env.pop('VULKAN_SDK',None)
            for worker in ('ecm_cuda.exe','ecm_cuda_stage2.exe'):
                command=[str(relocated/worker),'--version']
                proc=subprocess.run(command,cwd=cwd,env=env,capture_output=True,timeout=20)
                assert proc.returncode==0 and manifest['version'] in proc.stdout.decode(errors='replace'),(profile,worker,proc.returncode)
            request=cwd/'request.json';response=cwd/'response.json'
            request.write_text(json.dumps(dict(factor='1009',sigma='26',profile=0,backend=3,gp='')))
            proc=subprocess.run([str(relocated/'ecm_group_order_ui.exe'),str(request),str(response)],cwd=cwd,env=env,capture_output=True,timeout=20)
            assert proc.returncode==0,(profile,'GMP helper',proc.stderr)
            order=read(response)
            assert order['status']=='complete' and order['backend']=='gmp' and order['group_computed'] and order['point_minimality'],order
            config=relocated/'ecm.toml';before=sha(config);queue=sha(relocated/'worktodo.txt')
            log=cwd/'gui.log';startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
            proc=subprocess.run([str(relocated/'ecm_gui_v2.exe'),'--frames','12','--no-nvml','--no-pet','--ui-state',str(cwd/'ui.toml'),'--diagnostics',str(log)],cwd=cwd,env=env,startupinfo=startup,capture_output=True,timeout=30)
            assert proc.returncode==0,(profile,'GUI',proc.stderr)
            text=log.read_text(encoding='utf-8')
            assert 'frame: 1' in text and 'worker_processes=0' in text and 'config='+str(config) in text,text[-2000:]
            assert sha(config)==before and sha(relocated/'worktodo.txt')==queue
            record.update(relocated=str(relocated),path_only_system32=True,helper_status=order['status'],gui_no_workers=True)
        report['profiles'][profile]=record;persist();print('PASS bundle',profile,flush=True)
    report['complete']=True;persist()


if __name__=='__main__':main()
