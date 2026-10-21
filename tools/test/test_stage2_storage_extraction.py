"""Verify exact Stage2 progress/receipt/SHA bodies, layout and caller order."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','b747d48:'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    before=old('src/ecm/storage/queue/ecm_stage2_queue_state.h');impl=read('src/ecm/storage/queue/stage2_progress.cpp')
    method=lambda s,n:body(re.sub(r'\)\s*const\b',')',s),n)
    names=['token','load','save','id','path','LegacyIdentity','result_written','finished_written']
    for name in names:assert method(before,name)==method(impl,name),name
    assert body(before,'import_legacy')==body(read('src/ecm/storage/queue/ecm_stage2_queue_state.h'),'import_legacy')
    public=read('include/ecm/storage/queue/stage2_progress.h')
    fields=lambda s:normalize('\n'.join(re.findall(r'^\s*(?:std::string|uint64_t|unsigned|bool)\b[^\n{]*;\s*$',s,re.M)))
    for name,first in [('State','bool load'),('Task','std::string id')]:
        section=lambda s:s[s.index('struct '+name+' {'):s.index(first,s.index('struct '+name+' {'))]
        assert fields(section(before))==fields(section(public)),name
    for name in ['sha256_stream','sha256_file','sha256_text']:
        assert body(old('src/ecm/platform/ecm_stage2_fingerprint.h'),name)==body(read('src/ecm/platform/fingerprint.cpp'),name),name
    main='src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp'
    expected=old(main).replace('#include "../../platform/ecm_stage2_fingerprint.h"','#include "ecm/platform/fingerprint.h"')
    assert normalize(expected)==normalize(read(main)),'task/child/replay/tune order changed'
    for header in ['include/ecm/storage/queue/stage2_progress.h','include/ecm/platform/fingerprint.h']:
        s=read(header);assert not re.search(r'#\s*include[^\n]*(?:windows|bcrypt|fstream|ecm_result_json)',s,re.I)
        assert 'inline ' not in s and 'CreateFile' not in s and 'MoveFileEx' not in s
    assert not (ROOT/'src/ecm/platform/ecm_stage2_fingerprint.h').exists()
    result=dict(complete=True,baseline='b747d48',storage_bodies_unchanged=8,sha_bodies_unchanged=3,
                legacy_import_unchanged=True,state_task_layout_unchanged=True,complete_driver_unchanged=True,public_os_headers=0,gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
