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

ROOT=Path(__file__).resolve().parents[2]
BASELINE='8500a84'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    old=lambda name:subprocess.check_output(['git','show',BASELINE+':'+name],cwd=ROOT).decode('utf-8-sig')
    checked={}
    publication_fixes=[]
    for folder in ['src/ecm/backend','src/ecm/core']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix not in {'.cpp','.cu','.h','.cuh'}:continue
            name=path.relative_to(ROOT).as_posix()
            previous=old(name)
            if name in {'src/ecm/backend/cuda/stage1/montgomery/workflow.cpp','src/ecm/backend/opencl/stage1/montgomery/workflow.cpp'}:
                # The independent M991/PARAM3 regression proves the former save
                # expression invalidated its own checksum after finding a factor.
                # Permit only this serialization fix; every computation byte stays.
                start=previous.index('/* The save must describe');end=previous.index('        bool wrote = false;',start)
                indent='            ' if '/cuda/' in name else '        '
                previous=previous[:start]+('/* All parametrizations save the actual computation modulus. Dividing\n'+indent+
                    '   out factors here would invalidate the stored X and checksum. */\n'+indent+
                    'const auto n_expr_save=ecm_save_modulus_expression(n_expr,N);\n')+previous[end:]
                previous=previous.replace('n_expr_param0','n_expr_save').replace('n_expr_stripped','n_expr_save')
                previous=previous.replace('\n        bool wrote = false;','\n'+indent+'bool wrote = false;',1)
                publication_fixes.append(name)
            assert previous==read(name),('computation changed beyond the approved save serialization correction',name)
            checked[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['src/ecm/service/stage1/execution.cpp','src/ecm/service/stage1/execution_contract.cpp',
                 'src/ecm/service/stage1/execution_contract.h','src/ecm/service/stage1/preparation.cpp',
                 'src/ecm/application/cli/group_order.cpp']:
        assert old(name)==read(name),name
        checked[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
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
    result=dict(complete=True,baseline=BASELINE,unchanged_computation_files=len(checked)-len(publication_fixes),save_serialization_fixes=publication_fixes,
                existing_defaults_unchanged=len(before),new_options=['p95_sync_dir'],
                queue_callback_removed=True,parser_runtime_side_effects=0,source_hashes=checked)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))

if __name__=='__main__':main()
