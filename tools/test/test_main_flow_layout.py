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


def expected_text(relative):
    before=frozen(relative)
    edit=LAYOUT.get('version_edit',{})
    if relative==edit.get('file'):
        assert before.count('"'+edit['before']+'"')==1
        before=before.replace('"'+edit['before']+'"','"'+edit['after']+'"')
    for edit in LAYOUT.get('text_edits',[]):
        if relative==edit['file']:
            assert before.count(edit['before'])==edit.get('count',1),(relative,edit['before'])
            before=before.replace(edit['before'],edit['after'])
    return before


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    mapping=LAYOUT['moved'];assert len(mapping)==len(set(mapping.values()))==LAYOUT.get('expected_moved_files',142)
    hashes={}
    for old,new in mapping.items():
        assert not (ROOT/old).exists(),old
        path=current_source(old);after=path.read_text(encoding='utf-8-sig')
        assert body(expected_text(old))==body(after),('non-include implementation change',old,new)
        original=subprocess.check_output(['git','show',LAYOUT['baseline']+':'+old],cwd=ROOT)
        assert original.startswith(b'\xef\xbb\xbf')==path.read_bytes().startswith(b'\xef\xbb\xbf'),('BOM changed',old)
        assert expected_text(old)==relocation_text(old),('include target changed',old)
        hashes[new]=hashlib.sha256(path.read_bytes()).hexdigest()
    for old in LAYOUT['include_consumers']:
        path=current_source(old)
        assert body(expected_text(old))==body(path.read_text(encoding='utf-8-sig')),old
        assert expected_text(old)==relocation_text(old),('include target changed',old)
    for legacy in LAYOUT.get('empty_legacy_roots',['kernels/cuda','include/ecm/cuda','src/cuda']):
        assert not list((ROOT/legacy).rglob('*.*')),legacy
    # Actual include edges must resolve after relocation. SDK/third-party angle
    # includes retain their existing compiler search and are checked by builds.
    roots=[ROOT/p for p in ['include','src','src/core','src/cpu','src/ecm/backend/cuda/runtime','src/ecm/backend/cuda','third_party/cgbn/include','third_party/cgbn/include/cgbn',
                          'third_party/imgui','third_party/imgui/backends','kernels/opencl']]
    edges=0;generated_edges=[]
    for name in sorted(set(mapping.values())|set(LAYOUT['include_consumers'])-set(mapping)):
        path=ROOT/name
        for inc in re.finditer(r'(?m)^\s*#\s*include\s*"([^"\r\n]+)"',path.read_text(encoding='utf-8-sig')):
            if (name,inc[1])==('tools/test/stage2_s4_program_fixture.cpp','fixture.cpp'):
                # This existing CPU allocator test consumes an output generated
                # beside its native fixture; it never enters a production TU.
                producer=(ROOT/'tools/test/test_stage2_s4_memory.py').read_text(encoding='utf8')
                assert "generated = out/'fixture.cpp'" in producer
                assert "with_name('stage2_s4_program_fixture.cpp')" in producer
                assert '#include "fixture.cpp"' in frozen(name)
                generated_edges.append(name+':'+inc[1]);continue
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
    if LAYOUT.get('remove_legacy_include_dirs'):
        expected=re.sub(r'(?m)^\s*\$\{CMAKE_SOURCE_DIR\}/src/(?:core|cpu)\r?\n','',expected)
    for edit in LAYOUT.get('cmake_edits',[]):
        assert expected.count(edit['before'])==edit.get('count',1),edit['before']
        expected=expected.replace(edit['before'],edit['after'])
    assert expected==(ROOT/'CMakeLists.txt').read_text(encoding='utf-8-sig'),'unapproved CMake policy change'
    result=dict(complete=True,baseline=LAYOUT['baseline'],moved_files=len(mapping),include_consumers=len(LAYOUT['include_consumers']),resolved_quote_edges=edges,generated_consumer_edges=generated_edges,tu_order_unchanged=True,source_properties_unchanged=True,inputs=hashes,gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key!='inputs'}))


if __name__=='__main__':main()
