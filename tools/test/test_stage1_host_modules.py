"""Verify Stage1 host scalar/presentation extraction against frozen definitions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|\s+|.',re.S)


def normalize(text):
    return ''.join(t[0] for t in TOKEN.finditer(text) if not t[0].isspace() and not t[0].startswith(('//','/*')))


def body(text,name):
    # Keep string contents for equality, but mask them for brace matching.
    masked=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),text,flags=re.S)
    at=re.search(r'\b'+re.escape(name)+r'\s*\(',masked).end()
    start=masked.index('{',at);depth=0
    for i in range(start,len(masked)):
        depth+=(masked[i]=='{')-(masked[i]=='}')
        if depth==0:return normalize(text[start:i+1])
    raise AssertionError(name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',default='029a709');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frozen=lambda name:subprocess.check_output(['git','show',a.baseline+':'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    old_naf=frozen('src/ecm/core/stage1/ecm_edwards_naf.h')
    old_progress=frozen('src/ecm/platform/legacy/ecm_stage1_gpu_progress.h')
    scalar=read('src/ecm/stage1/scalar.cpp');progress=read('src/ecm/stage1/progress.cpp')
    checked={}
    for before,after,names in [(old_naf,scalar,['read','EcmEdwardsNaf','digit']),(old_progress,progress,['line','add','speed','per_curve','remaining'])]:
        for name in names:
            original,current=body(before,name),body(after,name)
            assert original==current,('host implementation changed',name)
            checked[name]=hashlib.sha256(current.encode()).hexdigest()
    original=frozen('src/ecm/core/stage1/ecm_stage1_exp.cpp')
    prefix=scalar[:scalar.index('uint32_t EcmEdwardsNaf::read')].replace('#include <stdexcept>','')
    without_includes=lambda text:re.sub(r'(?m)^\s*#\s*include[^\n]*','',text)
    assert normalize(without_includes(original))==normalize(without_includes(prefix)),'LCM implementation changed'
    nh=read('include/ecm/stage1/scalar.h');ph=read('include/ecm/stage1/progress.h')
    old_fields=old_naf.split('struct EcmEdwardsNaf {',1)[1].split('static uint32_t read',1)[0]
    new_fields=nh.split('struct EcmEdwardsNaf {',1)[1].split('static uint32_t read',1)[0]
    assert normalize(old_fields)==normalize(new_fields),'NAF data layout changed'
    old_fields=old_progress.split('class SpeedWindow {',1)[1].split('public:',1)[0]
    new_fields=ph.split('class SpeedWindow {',1)[1].split('public:',1)[0]
    assert normalize(old_fields)==normalize(new_fields),'speed-window data layout changed'
    for name,text in [('scalar',nh),('progress',ph)]:
        assert not re.search(r'__device__|__global__|cuda\w+_t|mpz_export\(|snprintf\(|std::isfinite\(',text),name
        assert not re.search(r'#\s*include\s*["<].*(?:cuda|src/)',text),name
    for name in ['src/ecm/core/stage1/ecm_edwards_naf.h','src/ecm/platform/legacy/ecm_stage1_gpu_progress.h']:
        assert not (ROOT/name).exists(),name
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    groups=json.loads(inventory.read_text())['groups']
    common=groups['STAGE1_COMMON']
    assert common.count('src/ecm/stage1/scalar.cpp')==1
    assert common.count('src/ecm/stage1/progress.cpp')==1
    cmake=read('CMakeLists.txt')
    assert not re.search(r'set_property\(SOURCE src/ecm/backend/cuda/stage1/montgomery/adapter\.cu APPEND PROPERTY OBJECT_DEPENDS',cmake)
    assert 'include/ecm/stage1/scalar.h' in cmake
    for name in ['build_edwards_gpu_workers.ps1','build_edwards_gpu_stage1_probe.ps1']:
        assert "@($inventory.groups.STAGE1_EDWARDS)+@('src/ecm/stage1/scalar.cpp')" in read('tools/build/test/'+name)
    assert "edwards_stage1/cuda/instances/'+$kernels[$Tier]" in read('tools/build/test/build_stage1_host_wait_probe.ps1')
    result=dict(complete=True,baseline=a.baseline,functions=checked,naf_layout_unchanged=True,speed_layout_unchanged=True,lcm_body_unchanged=True,new_implementation_tus=1,new_scalar_tus=0,cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'complete':True,'functions':len(checked),'data_layouts':2}))


if __name__=='__main__':main()
