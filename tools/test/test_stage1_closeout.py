"""Verify the final private execution contract and complete PARI extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from test_stage1_host_modules import normalize
from test_result_publication_extraction import body

ROOT=Path(__file__).resolve().parents[2]
NAMES={'Stage1RunOptions':'Stage1Options','Stage1ExecutionOptions':'ExecutionOptions','LegacyStage1Result':'ExecutionResult',
       'Stage1TaskStatus':'TaskStatus','Stage1FailureSource':'FailureSource'}


def renamed(s):
    s=s.replace('legacy_execution.h','execution_contract.h')
    for a,b in NAMES.items():s=re.sub(r'\b'+a+r'\b',b,s)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','469663a:'+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/name).read_text(encoding='utf-8-sig')
    old_driver=old('src/ecm/application/cli/ecm_driver.cpp');driver=read('src/ecm/application/cli/ecm_driver.cpp')
    diagnostic=read('src/ecm/application/cli/diagnostics/group_order.cpp')
    helpers=['trim','mpz_to_dec_string','build_primes_up_to_B1','factor_by_small_primes','format_group_order_smooth',
             'normalize_gp_path','gp_executable_exists','resolve_gp_path','get_gp_executable','compute_group_order_pari']
    for name in helpers:assert body(old_driver,name)==body(diagnostic,name),name
    # The entire diagnostic block is preserved, including all output and old
    # group-order-bound limitations; loop control becomes function return only.
    start=old_driver.index('                if (print_group_order) {',old_driver.index('const bool cli_has_factors'))
    end=old_driver.index('\n                }\n            }\n        }',start)+len('\n                }')
    original=old_driver[start:end];inner=original[original.index('{')+1:original.rfind('}')]
    expected='void diagnostic() {'+inner.replace('continue;','return true;',1).replace('return 1;','return false;')+'return true;}'
    assert body(expected,'diagnostic')==body(diagnostic,'print_group_order_factor')
    call='if (print_group_order && !print_group_order_factor(result,i,go_param,go_gp_exe,go_primes))return 1;'
    restored=body(driver,'main').replace(normalize(call),normalize(original),1)
    assert restored==body(renamed(old_driver),'main'),'CLI main changed beyond diagnostic extraction'
    for name in ['parse_sigma64_arg','print_ecm_usage','ecm_wants_usage','stage1_build_identity','print_gpu_info']:
        assert body(renamed(old_driver),name)==body(driver,name),name
    assert len(re.findall(r'\bparse_sigma_arg\s*\(',old_driver))==1 and 'parse_sigma_arg' not in driver
    before=renamed(old('src/ecm/service/stage1/legacy_execution.h'))
    header=read('src/ecm/service/stage1/execution_contract.h');implementation=read('src/ecm/service/stage1/execution_contract.cpp')
    assert ': Stage1Options(options) {}' in before and ': Stage1Options(options) {}' in implementation
    method_body=lambda s,name:body(s.replace(' noexcept','').replace(' : Stage1Options(options)',''),name)
    for name in ['ExecutionOptions','ExecutionResult','operator=','finish_task','swap']:
        assert method_body(before,name)==method_body(implementation,name),name
    assert body(before[before.index('~ExecutionResult'):],'ExecutionResult')==body(implementation[implementation.index('~ExecutionResult'):],'ExecutionResult')
    fields=lambda s:normalize('\n'.join(re.findall(r'^\s*(?:(?:TaskStatus|FailureSource|int|bool|uint\d+_t|long long|mpz_t)\b[^\n]*[;])\s*$',s,re.M)))
    assert fields(before)==fields(header),'private result/progress data layout'
    assert 'struct TaskResult::State' not in header and 'struct TaskResult::State : ExecutionResult {};' in read('src/ecm/service/stage1/task.cpp')
    unchanged=['service/stage1/execution.cpp','service/stage1/execution.h','service/stage1/preparation.cpp',
               'backend/cpu/stage1/legacy_executor.cpp','backend/cpu/stage1/legacy_executor.h',
               'backend/cuda/stage1/edwards/host_workflow.cpp','backend/cuda/stage1/edwards/host_workflow.h',
               'service/work/queue_workflow.cpp','application/cli/queue_source.cpp','application/cli/stage1_input.cpp']
    for name in unchanged:assert normalize(renamed(old('src/ecm/'+name)))==normalize(read('src/ecm/'+name)),name
    result=dict(complete=True,baseline='469663a',diagnostic_helpers_unchanged=10,complete_diagnostic_unchanged=True,
                main_order_unchanged=True,execution_methods_unchanged=6,private_layout_unchanged=True,unchanged_implementation_units=len(unchanged),
                old_duplicate_aliases_removed=True,driver_lines=len(driver.splitlines()),gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
