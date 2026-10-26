"""Architecture and unchanged computation gate for the approved host I/O refactor.

Runtime behavior is covered by stage1_host_io_contract and the real-worker input
matrix. Historical whole-main extraction equality is intentionally superseded:
input precedence and completion semantics changed with user approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from source_layout import frozen_source
ROOT=Path(__file__).resolve().parents[2]
BASELINE='8500a84'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    old=lambda name:frozen_source(BASELINE,name)
    # Current computation equality belongs to the complete domain inventory;
    # the original intake baseline still verifies default/option compatibility.
    subprocess.run(['python', 'tools/test/test_ecm_domain_layout.py', '--output',
                    str(a.output/'computation')], cwd=ROOT, check=True)
    previous=json.loads(old('config/ecm_options.json'))['options']
    current=json.loads(read('config/ecm_options.json'))['options']
    defaults=lambda options:{(o['owner'],o['key']):o['default'] for o in options}
    before,after=defaults(previous),defaults(current)
    assert all(after[key]==value for key,value in before.items()),'existing default changed'
    assert set(after)-set(before)=={('stage1','p95_sync_dir')}
    entry=read('src/main.cpp')
    for name in ['src/io/cli.cpp','src/io/config.cpp','src/io/worktodo.cpp','src/io/normalize.cpp','src/sync/p95.cpp']:
        assert (ROOT/name).is_file(),name
    for name in ['src/ecm/application/cli/ecm_driver.cpp','src/ecm/application/cli/queue_source.cpp',
                 'src/ecm/service/work/queue_workflow.cpp','include/ecm/service/work/queue_workflow.h']:
        assert not (ROOT/name).exists(),name
    assert 'NextQueueTask' not in entry and 'run_queue_workflow' not in entry
    assert entry.count('static int run_worktodo(')==1
    parser=read('src/io/cli.cpp')
    assert 'ecm_runtime_config()' not in parser and 'ecm_exp_cache_set_dir' not in parser
    assert 'fopen(' not in parser and 'ofstream' not in parser
    assert 'ecm_sync_save_files' not in read('src/io/worktodo.cpp')
    assert 'ecm_save::verified_snapshot' in entry and 'ecm_worktodo_store_completion' in entry
    subprocess.run(['python','tools/gen/generate_ecm_config.py','--check'],cwd=ROOT,check=True)
    result=dict(complete=True,baseline=BASELINE,complete_computation_inventory=True,
                existing_defaults_unchanged=len(before),new_options=['p95_sync_dir'],
                queue_callback_removed=True,parser_runtime_side_effects=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))

if __name__=='__main__':main()
