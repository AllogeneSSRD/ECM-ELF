"""Verify complete GPU host workflows and the unique execution dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize,body

ROOT=Path(__file__).resolve().parents[2]


def old(path):
    return subprocess.check_output(['git','show','c5f1db7:'+path],cwd=ROOT).decode('utf-8-sig')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    driver_path='src/ecm/application/cli/ecm_driver.cpp'
    previous=old(driver_path)
    execution=(ROOT/'src/ecm/service/stage1/execution.cpp').read_text(encoding='utf8')
    ed_path='src/ecm/backend/cuda/stage1/edwards/host_workflow.cpp'
    ed=(ROOT/ed_path).read_text(encoding='utf8')
    old_ed=old('src/ecm/backend/cuda/stage1/edwards/legacy/ecm_edwards_gpu_driver.inl')
    expected=body(previous,'run_stage1_impl')
    start=expected.index('out->ret=ECM_ERROR;')
    assert body(execution,'run_mont_gpu_stage1')==expected[start:-1].join(['{','}']), 'complete Montgomery host workflow changed'
    dispatch=body(execution,'execute_stage1')
    prefix=expected[1:start].replace('run_edwards_gpu_stage1(', 'ecm_cuda_edwards::run_stage1(')
    assert dispatch=='{'+prefix+'returnrun_mont_gpu_stage1(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);}', 'dispatch order changed'
    old_payload=old_ed[old_ed.index('namespace ed_gpu_driver {'):old_ed.rindex('#endif')]
    old_payload=old_payload.replace('static int run_edwards_gpu_stage1(', 'int ecm_cuda_edwards::run_stage1(',1)
    actual=ed[ed.index('namespace ed_gpu_driver {'):]
    assert normalize(old_payload)==normalize(actual),'Edwards state, callbacks or ordering changed'
    driver=(ROOT/driver_path).read_text(encoding='utf-8-sig')
    assert 'run_stage1_impl' not in driver and 'gpu_driver.inl' not in driver
    assert driver.count('ecm_stage1::execute_stage1,stage1_build_identity,edwards_get_naf_w')==1
    assert not (ROOT/'src/ecm/backend/cuda/stage1/edwards/legacy/ecm_edwards_gpu_driver.inl').exists()
    inventory=a.output/'inventory.json'
    subprocess.run(['cmake','-DECM_CUDA_SOURCES_OUTPUT='+str(inventory),'-P',str(ROOT/'tools/build/internal/cuda_sources.cmake')],check=True)
    groups=json.loads(inventory.read_text())['groups']
    assert groups['STAGE1_COMMON'].count('src/ecm/service/stage1/execution.cpp')==1
    assert groups['STAGE1_CUDA'].count(ed_path)==1 and ed_path not in groups['STAGE1_EDWARDS'], 'production host workflow leaked into low-level probes'
    cmake=(ROOT/'CMakeLists.txt').read_text(encoding='utf8')
    assert '_stage1_driver_headers' not in cmake
    result=dict(complete=True,baseline='c5f1db7',montgomery_host_body_sha256=hashlib.sha256(body(execution,'run_mont_gpu_stage1').encode()).hexdigest(),edwards_host_payload_sha256=hashlib.sha256(normalize(actual).encode()).hexdigest(),dispatch_order_unchanged=True,callbacks_and_checkpoint_order_unchanged=True,production_host_workflow_excluded_from_low_level_probes=True,driver_lines=len(driver.splitlines()),new_factory_classes=0,cuda_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
