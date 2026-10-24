"""Prepare the optional pinned local PARI runtime from an existing installation.

No installer or downloaded executable is run. The original upstream source
archive, configured headers, SEA tables and license remain alongside the DLL.
"""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE_SHA256='8d9c4fcd584c468d27e0f23c36836587284452094c4b1c404c20c4b810462dcb'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--installed',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'third_party/pari-2.17.3')
    a=p.parse_args();installed=a.installed.resolve();out=a.out.resolve();archive=a.archive.resolve()
    assert digest(archive)==SOURCE_SHA256,'Use the unmodified PARI 2.17.3 source archive'
    assert '2.17.3' in (installed/'include/pari/paricfg.h').read_text()
    assert 'LONG_IS_64BIT' in (installed/'include/pari/paricfg.h').read_text()
    out.mkdir(parents=True,exist_ok=True)
    target=out/'pari-2.17.3.tar.gz'
    if archive!=target:shutil.copy2(archive,target)
    with tarfile.open(target) as source:source.extractall(out/'source',filter='data')
    (out/'runtime').mkdir(exist_ok=True)
    shutil.copy2(installed/'libpari.dll',out/'runtime/libpari.dll')
    shutil.copytree(installed/'data/seadata',out/'data/seadata',dirs_exist_ok=True)
    shutil.copytree(installed/'include',out/'include',dirs_exist_ok=True)
    shutil.copy2(out/'source/pari-2.17.3/COPYING',out/'COPYING')
    paths=[out/'runtime/libpari.dll',target,out/'COPYING',*(out/'include').rglob('*'),*(out/'data').rglob('*')]
    manifest=dict(version='2.17.3',abi='Windows x64, 64-bit PARI words, single-threaded MinGW build',
        source_url='https://pari.math.u-bordeaux.fr/pub/pari/unix/pari-2.17.3.tar.gz',
        source_mirror='https://deb.debian.org/debian/pool/main/p/pari/pari_2.17.3.orig.tar.gz',
        source_sha256=SOURCE_SHA256,license='GPL-2.0-or-later',
        runtime_origin=str(installed/'libpari.dll'),
        boundary='Existing PARI 2.17.3 runtime and configured headers; upstream source supplied. Original binary build flags are not independently reconstructed.',
        sha256={path.relative_to(out).as_posix():digest(path) for path in sorted(paths) if path.is_file()})
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(root=str(out),runtime_sha256=digest(out/'runtime/libpari.dll'),files=len(manifest['sha256']))))
if __name__=='__main__':main()
