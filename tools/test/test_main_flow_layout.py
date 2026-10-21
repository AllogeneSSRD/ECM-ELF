"""Verify whole-file relocation, unchanged code, source order and include edges."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from source_layout import ROOT, LAYOUT, current_source, relocation_text

INCLUDE = re.compile(r'(?m)^\s*#\s*include\s*["<][^">\r\n]+[">]')


def frozen(name):
    return subprocess.check_output(['git','show',LAYOUT['baseline']+':'+name],cwd=ROOT).decode('utf-8-sig').replace('\r\n','\n')


def body(text):
    return INCLUDE.sub('#include <relocated>',text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    mapping=LAYOUT['moved'];assert len(mapping)==len(set(mapping.values()))==142
    hashes={}
    for old,new in mapping.items():
        assert not (ROOT/old).exists(),old
        path=current_source(old);after=path.read_text(encoding='utf-8-sig')
        assert body(frozen(old))==body(after),('non-include implementation change',old,new)
        original=subprocess.check_output(['git','show',LAYOUT['baseline']+':'+old],cwd=ROOT)
        assert original.startswith(b'\xef\xbb\xbf')==path.read_bytes().startswith(b'\xef\xbb\xbf'),('BOM changed',old)
        assert frozen(old)==relocation_text(old),('include target changed',old)
        hashes[new]=hashlib.sha256(path.read_bytes()).hexdigest()
    for old in LAYOUT['include_consumers']:
        path=current_source(old)
        assert body(frozen(old))==body(path.read_text(encoding='utf-8-sig')),old
        assert frozen(old)==relocation_text(old),('include target changed',old)
    assert not list((ROOT/'kernels/cuda').rglob('*.*'))
    assert not list((ROOT/'include/ecm/cuda').rglob('*.*'))
    assert not list((ROOT/'src/cuda').rglob('*.*'))
    # Actual include edges must resolve after relocation. SDK/third-party angle
    # includes retain their existing compiler search and are checked by builds.
    roots=[ROOT/p for p in ['include','src','src/core','src/cpu','src/ecm/backend/cuda/runtime','src/ecm/backend/cuda','third_party/cgbn/include','third_party/cgbn/include/cgbn']]
    edges=0
    for name in sorted(set(mapping.values())|set(LAYOUT['include_consumers'])-set(mapping)):
        path=ROOT/name
        for inc in re.finditer(r'(?m)^\s*#\s*include\s*"([^"\r\n]+)"',path.read_text(encoding='utf-8-sig')):
            assert any(p.is_file() for p in [path.parent/inc[1]]+[d/inc[1] for d in roots]),(name,inc[1])
            edges+=1
    old=frozen('tools/build/internal/cuda_sources.cmake')
    expected=old
    for src,dest in sorted(mapping.items(),key=lambda pair:-len(pair[0])):expected=expected.replace(src,dest)
    assert expected==(ROOT/'tools/build/internal/cuda_sources.cmake').read_text(encoding='utf-8-sig'),'TU membership/order changed'
    # All CMake source properties, including per-instance registers and object
    # dependencies, are preserved by the same exhaustive physical path mapping.
    expected=frozen('CMakeLists.txt')
    for src,dest in sorted(mapping.items(),key=lambda pair:-len(pair[0])):expected=expected.replace(src,dest)
    expected=expected.replace('src/cuda/common','src/ecm/backend/cuda/runtime').replace('kernels/cuda','src/ecm/backend/cuda')
    assert expected==(ROOT/'CMakeLists.txt').read_text(encoding='utf-8-sig'),'unapproved CMake policy change'
    result=dict(complete=True,baseline=LAYOUT['baseline'],moved_files=len(mapping),include_consumers=len(LAYOUT['include_consumers']),resolved_quote_edges=edges,tu_order_unchanged=True,source_properties_unchanged=True,inputs=hashes,gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key!='inputs'}))


if __name__=='__main__':main()
