"""Freeze and distribute the shared diagnostic host, dependencies and sources."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
from gmp_payload import distribution
ROOT=Path(__file__).resolve().parents[2]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sources():
    roots=['src/ecm/diagnostics','src/ecm/group_order','src/ecm/curve','src/ecm/storage/results','include/ecm']
    files={p for root in roots for p in (ROOT/root).rglob('*') if p.is_file()}
    files.update(ROOT/p for p in ['LICENSE','tools/packaging/diagnostics_runtime.py','tools/packaging/gmp_payload.py','tools/build/test/build_diagnostics.ps1','tools/test/diagnostics_probe.cpp'])
    return sorted(files)
def input_identity(gmp,pari,without):
    value=dict(sources_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sources()},
        gmp={n:digest(gmp/n) for n in ['include/gmp.h','lib/gmp.lib','bin/gmp-10.dll']},pari={})
    if not without:
        assert (pari/'provenance.json').is_file(),'Prepare PARI using tools/build/test/prepare_gui_v2_pari.py first'
        provenance=json.loads((pari/'provenance.json').read_text(encoding='utf-8'))
        for name,sha in provenance['sha256'].items():assert digest(pari/name)==sha,f'PARI identity mismatch: {name}'
        value['pari']=dict(provenance['sha256'],provenance=digest(pari/'provenance.json'))
    return value
def freeze(binary,gmp,pari,without):
    folder=binary.parent;payload=[binary];inputs=input_identity(gmp,pari,without)
    assert inputs==json.loads((folder/'diagnostics-inputs.json').read_text(encoding='utf-8')),'Diagnostic inputs changed during compilation'
    catalog=inputs['sources_sha256']
    def copy(src,name):
        dst=folder/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
        payload.append(dst)
    copy(gmp/'bin/gmp-10.dll','gmp-10.dll')
    if without:
        assert not (folder/'libpari.dll').exists(),'Use a fresh directory without PARI'
    else:
        provenance=json.loads((pari/'provenance.json').read_text(encoding='utf-8'))
        for name,sha in provenance['sha256'].items():assert digest(pari/name)==sha,f'PARI identity mismatch: {name}'
        copy(pari/'runtime/libpari.dll','libpari.dll');copy(pari/'provenance.json','pari-provenance.json')
        for src in sorted((pari/'data/seadata').rglob('*')):
            if src.is_file():copy(src,'pari-data/'+src.relative_to(pari/'data').as_posix())
        copy(pari/'COPYING','licenses/PARI-GPL-2.0.txt')
    gmp_sources,gmp_licenses=distribution(gmp)
    for src,name in gmp_licenses:copy(src,'licenses/'+name)
    copy(ROOT/'LICENSE','licenses/ECM-ELF-LICENSE.txt')
    archive=folder/'diagnostics-source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
        for p in sources():out.write(p,p.relative_to(ROOT).as_posix())
        for p,name in gmp_sources:out.write(p,name)
        for name in ['include/gmp.h','lib/gmp.lib','bin/gmp-10.dll']:
            out.write(gmp/name,'build-dependencies/gmp/'+name)
        out.writestr('SOURCE-BUILD.txt','Rebuild the diagnostic host from this extracted source root on Windows x64:\n'
            'cmake -S src/ecm/diagnostics -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DECM_DIAGNOSTICS_LOCAL_PARI=OFF -DECM_DIAGNOSTICS_GMP_ROOT="<absolute-source-root>/build-dependencies/gmp"\n'
            'cmake --build build\n'
            'This builds without bundling PARI. To execute PARI, copy the distributed libpari.dll and pari-data/ beside the rebuilt host, or rebuild PARI from the included original archive.\n')
        if not without:
            out.write(pari/'pari-2.17.3.tar.gz','third_party/pari-2.17.3/pari-2.17.3.tar.gz')
            for p in (pari/'include').rglob('*'):
                if p.is_file():out.write(p,'third_party/pari-2.17.3/'+p.relative_to(pari).as_posix())
    payload.append(archive)
    assert inputs==input_identity(gmp,pari,without),'Diagnostic inputs changed during runtime packaging'
    manifest=dict(schema=1,default_backend='pari-then-gp',gmp_explicit_only=True,local_pari=not without,
        sources_sha256=catalog,gmp_prefix=str(gmp),gmp_header_sha256=digest(gmp/'include/gmp.h'),gmp_library_sha256=digest(gmp/'lib/gmp.lib'),
        files_sha256={p.relative_to(folder).as_posix():digest(p) for p in payload})
    (folder/'diagnostics_build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Shared diagnostics: {len(payload)} files, PARI={not without}')
def package(build,out):
    manifest=json.loads((build/'diagnostics_build_manifest.json').read_text(encoding='utf-8'))
    for name,sha in manifest['sources_sha256'].items():assert digest(ROOT/name)==sha,f'Diagnostic source changed: {name}'
    out.mkdir(parents=True,exist_ok=True)
    for name,sha in manifest['files_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        src=build/name;assert digest(src)==sha,f'Diagnostic runtime changed: {name}'
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    return list(manifest['files_sha256'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path);p.add_argument('--gmp',type=Path);p.add_argument('--pari',type=Path);p.add_argument('--without-pari',action='store_true');p.add_argument('--package-build',type=Path);p.add_argument('--freeze-inputs',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.freeze_inputs:
        value=json.dumps(input_identity(a.gmp.resolve(),a.pari.resolve(),a.without_pari),indent=2)
        if not a.freeze_inputs.exists() or a.freeze_inputs.read_text(encoding='utf-8')!=value:a.freeze_inputs.write_text(value,encoding='utf-8')
    elif a.package_build:
        for name in package(a.package_build.resolve(),a.out.resolve()):print(name)
    else:freeze(a.binary.resolve(),a.gmp.resolve(),a.pari.resolve(),a.without_pari)
if __name__=='__main__':main()
