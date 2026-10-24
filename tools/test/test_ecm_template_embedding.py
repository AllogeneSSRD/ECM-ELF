"""Compile the embedded TOML with the actual release host compiler; compare bytes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.build_manifest.read_text(encoding='utf-8-sig'))
    spec=importlib.util.spec_from_file_location('ecm_generator',ROOT/'tools/gen/generate_ecm_config.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    stress=('quote=" slash=\\ UTF8=中文 😀\n'*7000).encode()
    cases=[('actual',(ROOT/'config/ecm.toml.default').read_bytes(),(ROOT/'src/ecm/application/configuration/generated/ecm_toml_template.h').read_bytes()),
           ('large',stress,generator.template_header(stress.decode()).encode())]
    results=[]
    for name,expected,header in cases:
        folder=out/name;folder.mkdir();(folder/'template.h').write_bytes(header)
        source=folder/'probe.cpp'
        source.write_text('#include "template.h"\n#include <fstream>\nint main(int argc,char** argv){if(argc!=2)return 2;'
            'std::ofstream out(argv[1],std::ios::binary);out.write(ecm_config::default_toml,sizeof(ecm_config::default_toml)-1);return out?0:3;}\n')
        command=[manifest['host_compiler'],'/nologo','/std:c++17','/EHsc','/MD','/utf-8',str(source),'/Fe'+str(folder/'probe.exe'),'/Fo'+str(folder/'probe.obj')]
        batch=folder/'build.cmd';batch.write_text('@echo off\n'+manifest['compile_setup']+'\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(command)+'\n',encoding='utf-8')
        proc=subprocess.run(['cmd','/d','/c',str(batch)],cwd=folder,capture_output=True)
        (folder/'build.log').write_bytes(proc.stdout+proc.stderr)
        assert proc.returncode==0,(name,proc.returncode,proc.stdout.decode(errors='replace')+proc.stderr.decode(errors='replace'))
        subprocess.run([str(folder/'probe.exe'),str(folder/'output.toml')],cwd=folder,check=True)
        assert (folder/'output.toml').read_bytes()==expected,name
        results.append(dict(case=name,bytes=len(expected),byte_exact=True));print(f'PASS {name}: {len(expected)} bytes',flush=True)
    (out/'result.json').write_text(json.dumps(dict(complete=True,compiler=manifest['host_compiler'],host_version=manifest['host_version'],cases=results),indent=2),encoding='utf-8')


if __name__=='__main__':main()
