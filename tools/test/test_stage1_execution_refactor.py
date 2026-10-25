"""Current Stage1 execution architecture and unchanged arithmetic inventory gate.

Runtime acceptance uses the task/file contracts, real input matrix, differential
saved endpoints and checkpoint compatibility tests. Approved host execution and
publication changes are explicitly enumerated; device arithmetic stays frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASELINE='6df207b'
APPROVED={
    'src/ecm/backend/cpu/stage1/legacy_executor.cpp',
    'src/ecm/backend/cpu/stage1/legacy_executor.h',
    'src/ecm/backend/cuda/stage1/execution.h',
    'src/ecm/backend/cuda/stage1/montgomery/workflow.cpp',
    'src/ecm/backend/cuda/stage1/montgomery/checkpoint.cpp',
    'src/ecm/backend/cuda/stage1/montgomery/checkpoint.h',
    'src/ecm/backend/cuda/stage1/montgomery/runtime.cu',
    'src/ecm/backend/cuda/stage1/edwards/workflow.cpp',
    'src/ecm/backend/opencl/stage1/montgomery/workflow.cpp',
    'src/ecm/backend/opencl/stage1/montgomery/workflow.h',
    'src/ecm/backend/opencl/stage1/montgomery/engine.cpp',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    old_names=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE],cwd=ROOT,text=True).splitlines())
    old=lambda n:subprocess.check_output(['git','show',BASELINE+':'+n],cwd=ROOT).decode('utf-8-sig').replace('\r\n','\n')
    read=lambda n:(ROOT/n).read_text(encoding='utf-8-sig')
    unchanged={};changed=[]
    for folder in ['src/ecm/backend','src/ecm/core']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix not in {'.cpp','.cu','.h','.cuh'}:continue
            name=path.relative_to(ROOT).as_posix()
            if name not in old_names:
                assert name=='src/ecm/backend/cuda/stage1/montgomery/construction.h',('unreviewed computation file',name)
                continue
            if old(name)!=read(name):
                assert name in APPROVED,('unreviewed arithmetic change',name)
                changed.append(name)
            else:unchanged[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(changed)==APPROVED
    # Verify the only edited .cu changes layout description and checkpoint status,
    # preserving the existing ladder launch and arithmetic implementation exactly.
    name='src/ecm/backend/cuda/stage1/montgomery/runtime.cu'
    expected=old(name).replace('#include "checkpoint.h"','#include "checkpoint.h"\n#include "construction.h"')
    edits=[('  const bool param0 = (gpu_param == 0);','  const auto layout=ladder_construction(uint32_t(gpu_param));\n  const bool param0 = layout.full_sigma;'),
           ('  const bool suyama_layout = param0 || param2;','  const bool suyama_layout = layout.words==7;'),
           ('const uint32_t words_per_curve = suyama_layout ? 7u : 5u;','const uint32_t words_per_curve = layout.words;'),
           ('const int      p1_word = suyama_layout ? 3 : 1;','const int      p1_word = int(layout.first_x);'),
           ('const int      p2_word = suyama_layout ? 5 : 3;','const int      p2_word = int(layout.second_x);'),
           ('      const int saved=ckpt_path.empty()?ECM_NO_FACTOR_FOUND:\n          save_checkpoint(ckpt_filename,&header,data,data_size);\n      if(saved!=ECM_NO_FACTOR_FOUND)',
            '      const bool saved=ckpt_path.empty() || save_checkpoint(ckpt_filename,&header,data,data_size);\n      if(!saved)')]
    for before,after in edits:
        assert expected.count(before)==1,before
        expected=expected.replace(before,after)
    expected=expected.replace('saved == ECM_NO_FACTOR_FOUND','saved')
    assert expected==read(name),'unexpected CUDA runtime change'
    assert read('config/ecm_options.json')==old('config/ecm_options.json'),'configuration changed'
    task=read('src/ecm/service/stage1/task.cpp')
    assert 'deps.execute(request, execution)' in task and 'publish_batch(' in task
    assert 'LegacyExecutionResult' not in task and 'prepare_failed' not in task
    for n in ['src/ecm/storage/save/publication.h','src/ecm/storage/save/publication.cpp']:
        assert 'Capture' not in read(n) and 'thread_local' not in read(n)
    assert 'execute_stage1' not in read('src/ecm/service/stage1/execution.cpp')
    for curve in ['montgomery','edwards']:
        workflow=read(f'src/ecm/backend/cuda/stage1/{curve}/workflow.cpp')
        assert 'publish_save(' not in workflow and 'append_save_lines' not in workflow
    subprocess.run(['python','tools/gen/generate_ecm_config.py','--check'],cwd=ROOT,check=True)
    result=dict(complete=True,baseline=BASELINE,unchanged_computation_files=len(unchanged),
                approved_host_files=sorted(changed),cuda_arithmetic_unchanged=True,
                existing_configuration_unchanged=True,implicit_capture_removed=True,
                source_hashes=unchanged,gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))

if __name__=='__main__':main()
