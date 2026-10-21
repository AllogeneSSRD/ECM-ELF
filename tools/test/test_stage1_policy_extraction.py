"""Prove policy extraction against the pre-refactor runtime; emit a host oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from test_stage1_host_modules import normalize
from test_result_publication_extraction import body

ROOT=Path(__file__).resolve().parents[2]
FOLDER='src/ecm/backend/cuda/stage1/montgomery/'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=lambda name:subprocess.check_output(['git','show','0bdd483:'+FOLDER+name],cwd=ROOT).decode('utf-8-sig')
    read=lambda name:(ROOT/FOLDER/name).read_text(encoding='utf-8-sig')
    before=old('prac/prac_runtime.inl');after=read('prac/prac_runtime.inl');policy=read('host_policy.cpp')
    definitions=[];checked=[]
    specs=[('read_prac_policy','        int mode = prac','        std::vector<uint32_t> tiers;',
            'PracPolicy read_prac_policy(bool prac)',
            'return {mode,compact,outlined,single_add,single_compact,requested_tpi};',
            'const auto policy=read_prac_policy(prac); int mode=policy.mode; const bool compact=policy.compact,outlined=policy.outlined,single_add=policy.single_add,single_compact=policy.single_compact; const uint32_t requested_tpi=policy.requested_tpi;'),
           ('prac_constants_policy','        const char *constants = "none";','        kernel = cgbn_stage1_domain_dispatch',
            'const char *prac_constants_policy(bool prac,int &mode,uint32_t bits,uint32_t requested_tpi,mpz_srcptr N,size_t nbits)',
            'return constants;', 'const char *constants=prac_constants_policy(prac,mode,bits,requested_tpi,N,nbits);'),
           ('prac_plan_cache','            std::string cache = ecm_exp_cache_get_dir();','            plan = ',
            'std::string prac_plan_cache()','return cache;','const auto cache=prac_plan_cache();'),
           ('prac_target_ms','        double target_ms = 100;','        uint64_t completed_work',
            'double prac_target_ms(bool prac)','return target_ms;','const double target_ms=prac_target_ms(prac);'),
           ('prac_sample_limit','        double sample_limit = 0;','        while (h.next < total)',
            'double prac_sample_limit()','return sample_limit;','const double sample_limit=prac_sample_limit();')]
    restored=body(after,'cgbn_stage1_resident')
    for name,start,end,signature,ret,call in specs:
        snippet=before[before.index(start):before.index(end,before.index(start))]
        expected=signature+' {\n'+snippet+ret+'\n}'
        assert body(expected,name)==body(policy,name),name
        restored=restored.replace(normalize(call),normalize(snippet),1)
        definitions.append(expected);checked.append(name)
    assert restored==body(before,'cgbn_stage1_resident'),'read/dispatch/lifetime order'
    previous_window=old('prac/prac_window.inl')
    for name in ['prac_window_integer','prac_window_settings']:
        assert body(previous_window,name)==body(policy,name),name;checked.append(name)
    defs=previous_window[previous_window.index('static uint64_t prac_window_integer'):previous_window.index('static int prac_window_run')]
    definitions.append(defs.replace('static PracWindow prac_window_settings','PracWindow prac_window_settings'))
    old_adapter=old('adapter.cu');adapter=read('adapter.cu')
    start=old_adapter.index('  const char *algorithm = getenv');end=old_adapter.index('#if ECM_CUDA_ENABLE_PRAC',start)
    prefix=old_adapter[start:end];algorithm='const char *stage1_algorithm() {\n'+prefix.replace('return ECM_ERROR;','return nullptr;')+'}\nreturn algorithm && *algorithm ? algorithm : "ladder";\n}'
    assert body(algorithm,'stage1_algorithm')==body(policy,'stage1_algorithm');definitions.append(algorithm);checked.append('stage1_algorithm')
    original_call='const char *algorithm = stage1_algorithm(); if (!algorithm) return ECM_ERROR; if (strcmp(algorithm, "ladder") != 0) {'
    restored_adapter=body(adapter,'cgbn_ecm_stage1').replace(normalize(original_call),normalize(prefix),1)
    start=old_adapter.index('  const char *dump_env = getenv');end=old_adapter.index('  FILE *dump_file',start)
    dump=old_adapter[start:end]
    restored_adapter=restored_adapter.replace(normalize('uint32_t dump_enable = gpu_dump_enabled() ? 1u : 0u;'),normalize(dump),1)
    assert restored_adapter==body(old_adapter,'cgbn_ecm_stage1'),'ladder order'
    start=previous_window.index('        FILE *dump = fopen("window_q.csv"');end=previous_window.index('\n    }\n    *gputime',start)
    csv=previous_window[start:end]
    assert body('void write_prac_window_csv() {'+csv+'}','write_prac_window_csv')==body(read('host_io.cpp'),'write_prac_window_csv')
    restored_window=read('prac/prac_window.inl').replace('write_prac_window_csv(w,work,b1,sigma,curves,bits,data);',csv,1)
    assert body(restored_window,'prac_window_run')==body(previous_window,'prac_window_run'),'export/readback/CSV order'
    def enums(text):return [normalize('enum {'+x.split('};')[0]+'};') for x in text.split('enum {')[1:]]
    assert enums(read('prac/domain.h'))==enums(old('prac/launch.h'))+enums(old('prac/constants.h'))
    # Oracle bodies come from the frozen runtime, not the current implementation.
    oracle='#include "ecm/backend/cuda/stage1/montgomery/host_policy.h"\n#include "ecm/backend/cuda/runtime/cuda_ecm_shim.h"\n#include "ecm/storage/cache/ecm_stage1_exp_cache.h"\n#include <algorithm>\n#include <cmath>\n#include <cstdlib>\n#include <cstring>\n#include <stdexcept>\nnamespace baseline {\nusing ecm_cuda_mont_host::PracPolicy;using ecm_cuda_mont_host::PracWindow;\n'+'\n'.join(definitions)+'\n}\n'
    (a.output/'baseline_policy.inc').write_text(oracle,encoding='utf8')
    result=dict(complete=True,baseline='0bdd483',policy_bodies_unchanged=checked,ladder_order_unchanged=True,resident_order_unchanged=True,window_export_order_unchanged=True,domain_values_unchanged=True,oracle_sha256=hashlib.sha256(oracle.encode()).hexdigest(),gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
