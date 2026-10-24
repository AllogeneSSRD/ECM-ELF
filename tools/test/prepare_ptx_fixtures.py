"""Derive PTX-only test images from frozen modern release objects.

For a driver/runtime that ignores forcing PTX while compatible SASS exists,
these explicitly labelled test fixtures retain byte-identical embedded PTX.
Original objects, EXEs and build manifests are never overwritten or relabelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['stage1','stage2','both'],default='both')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    report=dict(complete=False,kind='derived_ptx_only_test_images',collector_sha256=sha(__file__),components={})
    for stage in (('stage1','stage2') if args.stage=='both' else (args.stage,)):
        record=read(args.build_root/'components'/f'cu13.3.{stage}.json')
        leaf=Path(record['build']);manifest_path=Path(record['manifest']);manifest=read(manifest_path)
        original=Path(record['exe']);source_sha=sha(original)
        assert source_sha.lower()==manifest.get('executable_sha256',manifest.get('sha256')).lower()
        cuda=Path(manifest['cuda_root']);inspector=cuda/'bin/cuobjdump.exe'
        folder=out/stage;folder.mkdir();objects=folder/'objects';objects.mkdir()
        frozen={str(original):source_sha,str(manifest_path):sha(manifest_path)}
        derived=[];bindings=[]
        device_objects={str((leaf/name).resolve()).lower() for name in manifest['objects']} if stage=='stage1' else set()
        if stage=='stage1':
            text=(leaf/'CMakeFiles/ecm_cuda.dir/objects1.rsp').read_text(encoding='utf-8-sig')
            source_objects=[Path(v.strip('"')) for v in shlex.split(text,posix=False)]
            source_objects=[v if v.is_absolute() else leaf/v for v in source_objects]
        else:source_objects=[Path(v['entry']['output']) for v in manifest['compile_units'].values()]
        for index,source in enumerate(source_objects):
            target=objects/f'{index:04d}_{source.name}';frozen[str(source)]=sha(source)
            if stage=='stage1':
                device_module=str(source.resolve()).lower() in device_objects
            else:
                device_module=any(Path(manifest['compile_units'][name]['entry']['output'])==source for name in manifest['cuda_sources'])
            listing=subprocess.check_output([str(inspector),'--list-ptx',str(source)],stderr=subprocess.STDOUT) if device_module else b''
            if b'.ptx' in listing:
                subprocess.run([str(cuda/'bin/nvprune.exe'),'--generate-code=arch=compute_75,code=compute_75',str(source),'-o',str(target)],check=True,capture_output=True)
                def program(path,side):
                    directory=folder/'ptx'/f'{index:04d}'/side;directory.mkdir(parents=True)
                    subprocess.run([str(inspector),'--extract-ptx','all',str(path.resolve())],cwd=directory,capture_output=True,check=True)
                    files=list(directory.glob('*.ptx'));assert len(files)==1,('Unexpected PTX membership',path)
                    return files[0].read_bytes()
                before=program(source,'original');after=program(target,'derived')
                assert before==after,('PTX changed',source)
                elf=subprocess.check_output([str(inspector),'--list-elf',str(target)],stderr=subprocess.STDOUT)
                assert b'.cubin' not in elf,('SASS remained',target)
                bindings.append(dict(source=str(source),source_sha256=frozen[str(source)],derived=str(target),derived_sha256=sha(target),ptx_sha256=hashlib.sha256(after).hexdigest()))
            else:shutil.copy2(source,target)
            derived.append(target)
        binary=folder/original.name
        if stage=='stage1':
            make=(leaf/'CMakeFiles/ecm_cuda.dir/build.make').read_text()
            command=re.search(r'^\t(.+vs_link_exe.+) @<<\r?\n([^\n]+)\r?\n<<',make,re.M)
            assert command,'CMake native link rule missing'
            response=folder/'link-objects.rsp';response.write_text(' '.join('"'+str(v)+'"' for v in derived))
            options=command[2]
            options=re.sub(r'/out:\S+','/out:"'+str(binary).replace('\\','/')+'"',options)
            options=re.sub(r'/implib:\S+','/implib:"'+str(folder/'worker.lib').replace('\\','/')+'"',options)
            options=re.sub(r'/pdb:\S+','/pdb:"'+str(folder/'worker.pdb').replace('\\','/')+'"',options)
            libraries=folder/'link-libraries.rsp';libraries.write_text(options)
            line=command[1].replace('@CMakeFiles\\ecm_cuda.dir\\objects1.rsp','@"'+str(response)+'"')+' @"'+str(libraries)+'"'
            (folder/'CMakeFiles/ecm_cuda.dir').mkdir(parents=True)
        else:
            profile=read(leaf/'profile.json');prefix=profile['fields']['option/link']
            assert prefix.startswith(manifest['compile_setup']+' && ')
            prefix=prefix[len(manifest['compile_setup']+' && '):]
            # nvcc emits an empty link fatbin for its real target even when all
            # input objects are PTX-only. The test link has only virtual 75.
            prefix=re.sub(r'\s+--generate-code=arch=compute_80,code=sm_80\b','',prefix)
            response=folder/'link-objects.rsp'
            response.write_text(' '.join('"'+str(v)+'"' for v in derived)+' -L "'+manifest['gmp_root']+'/lib" -lgmp -o "'+str(binary)+'"')
            import sys
            line='"'+sys.executable+'" "'+str(ROOT/'tools/build/internal/timed_nvcc.py')+'" --root "'+str(folder)+'" -- '+prefix+' --options-file "'+str(response)+'"'
        script=folder/'link.cmd'
        script.write_text('@echo off\n'+manifest['compile_setup']+'\nif errorlevel 1 exit /b 1\ncd /d "'+str(folder)+'"\n'+line+'\n')
        with (folder/'link.log').open('wb') as log:subprocess.run(['cmd.exe','/d','/c',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(leaf/'gmp-10.dll',folder/'gmp-10.dll')
        assert b'.cubin' not in subprocess.check_output([str(inspector),'--list-elf',str(binary)],stderr=subprocess.STDOUT)
        assert b'.sm_75.ptx' in subprocess.check_output([str(inspector),'--list-ptx',str(binary)],stderr=subprocess.STDOUT)
        assert frozen=={p:sha(p) for p in frozen},'Original release input changed'
        report['components'][stage]=dict(source_exe=str(original),source_exe_sha256=source_sha,source_manifest_sha256=sha(manifest_path),exe=str(binary),sha256=sha(binary),bindings=bindings,original_inputs=frozen)
        (out/'ptx-fixtures.json').write_text(json.dumps(report,indent=2)+'\n');print('PTX fixture',stage,flush=True)
    report['complete']=True;(out/'ptx-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
