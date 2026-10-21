"""Prove the complete CPU workflow moved without changing state or event order."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize,body

ROOT=Path(__file__).resolve().parents[2]


def function(text,name):
    masked=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),text,flags=re.S)
    def end(at,left,right):
        depth=0
        for i in range(at,len(masked)):
            depth+=(masked[i]==left)-(masked[i]==right)
            if not depth:return i
        raise AssertionError(name)
    for match in re.finditer(r'\b'+name+r'\s*\(',masked):
        at=masked.index('(',match.start());i=end(at,'(',')')+1
        while masked[i].isspace():i+=1
        if masked[i]!='{':continue
        start=text.rfind('\nstatic ',0,match.start())+1
        assert text[start:match.start()].startswith('static ')
        return text[start:end(i,'{','}')+1]
    raise AssertionError(name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=subprocess.check_output(['git','show','fff2301:src/ecm/application/cli/ecm_driver.cpp'],cwd=ROOT).decode('utf-8-sig')
    cpu=(ROOT/'src/ecm/backend/cpu/stage1/legacy_executor.cpp').read_text(encoding='utf-8-sig')
    prep=(ROOT/'src/ecm/service/stage1/preparation.cpp').read_text(encoding='utf8')
    driver=(ROOT/'src/ecm/application/cli/ecm_driver.cpp').read_text(encoding='utf-8-sig')
    names=['compute_batch_s','random_sigma_u64','stage1_save_prefix','stage1_prefixed_save','mont_format_save_name','fixed_sigma_batch_exceeds_prime95_range','ensure_dir','stage1_now_ms','stdout_is_tty_local']
    hashes={}
    for name in names:
        before,after=body(function(old,name),name),body(prep,name)
        assert before==after,name
        hashes[name]=hashlib.sha256(after.encode()).hexdigest()
    expected=old[old.index('static std::mutex g_stage1_out_mutex;'):old.index('#include "../../backend/cuda/stage1/edwards/legacy/ecm_edwards_gpu_driver.inl"')]
    for name in ['ensure_dir','mont_format_save_name','parse_affinity_spec','stage1_now_ms','stdout_is_tty_local']:
        text=function(old,name);assert expected.count(text)==1;expected=expected.replace(text,'',1)
    expected=function(old,'to_ifma_field_mode')+'\n'+function(old,'edwards_capture_save')+'\n'+expected
    expected=re.sub(r'(?m)^#include[^\n]*\n?','',expected)
    for name in ['run_edwards_stage1','run_mont_stage1']:expected=expected.replace('static int '+name+'(','int '+name+'(',1)
    actual=cpu[cpu.index('static int to_ifma_field_mode'):cpu.rindex('} // namespace ecm_cpu_stage1')]
    assert normalize(expected)==normalize(actual),'CPU body/state/preprocessor/order changed'
    assert function(old,'parse_affinity_spec')==function(driver,'parse_affinity_spec'),'affinity input parse changed'
    assert 'static int run_edwards_stage1(' not in driver and 'static int run_mont_stage1(' not in driver
    assert driver.count('return ecm_cpu_stage1::run_edwards_stage1(')==1
    assert driver.count('return ecm_cpu_stage1::run_mont_stage1(')==1
    assert not re.search(r'#\s*include[^\n]*(?:backend/cuda|CL/cl|cuda_runtime|cgbn)',cpu)
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    common=json.loads(inventory.read_text())['groups']['STAGE1_COMMON']
    for source in ['src/ecm/backend/cpu/stage1/legacy_executor.cpp','src/ecm/service/stage1/preparation.cpp']:assert common.count(source)==1
    cmake=(ROOT/'CMakeLists.txt').read_text(encoding='utf8')
    assert not re.search(r'set_source_files_properties\([^)]*legacy_executor\.cpp[^)]*(?:AVX512|mavx512)',cmake,re.S)
    result=dict(complete=True,baseline='fff2301',shared_functions=hashes,cpu_body_sha256=hashlib.sha256(normalize(actual).encode()).hexdigest(),cpu_state_and_order_unchanged=True,affinity_parse_retained_in_application=True,new_wrapper_classes=0,driver_lines=len(driver.splitlines()),cpu_lines=len(cpu.splitlines()),cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='shared_functions'}))


if __name__=='__main__':main()
