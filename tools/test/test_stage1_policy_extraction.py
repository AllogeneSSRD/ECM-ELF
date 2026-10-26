"""Emit the immutable policy oracle and audit the current captured-input policy.

The host contract runs these original bodies against the real current policy.
The original extraction layout is no longer an assumption about current files.
"""
import argparse
import hashlib
import json
from pathlib import Path
from source_layout import frozen_source
from test_ecm_domain_layout import tokens

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FOLDER = 'src/ecm/backend/cuda/stage1/montgomery/'
CURRENT_FOLDER = 'src/ecm/stage1/montgomery/cuda/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old = lambda name: frozen_source('0bdd483', FROZEN_FOLDER+name)
    before = old('prac/prac_runtime.inl')
    definitions = []
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
    for name, start, end, signature, ret, _ in specs:
        snippet = before[before.index(start):before.index(end, before.index(start))]
        definitions.append(signature+' {\n'+snippet+ret+'\n}')
    previous_window = old('prac/prac_window.inl')
    first = previous_window.index('static uint64_t prac_window_integer')
    last = previous_window.index('static int prac_window_run')
    definitions.append(previous_window[first:last].replace('static PracWindow prac_window_settings',
                                                          'PracWindow prac_window_settings'))
    adapter = old('adapter.cu')
    first = adapter.index('  const char *algorithm = getenv')
    last = adapter.index('#if ECM_CUDA_ENABLE_PRAC', first)
    snippet = adapter[first:last]
    definitions.append('const char *stage1_algorithm() {\n'+snippet.replace('return ECM_ERROR;', 'return nullptr;')+
                       '}\nreturn algorithm && *algorithm ? algorithm : "ladder";\n}')
    oracle = ('#include "ecm/montgomery_stage1/cuda/host_policy.h"\n'
              '#include "ecm/cuda/runtime/cuda_ecm_shim.h"\n'
              '#include "io/exponent_cache.h"\n'
              '#include <algorithm>\n#include <cmath>\n#include <cstdlib>\n#include <cstring>\n#include <stdexcept>\n'
              'namespace baseline {\nusing ecm_cuda_mont_host::PracPolicy;using ecm_cuda_mont_host::PracWindow;\n'+
              '\n'.join(definitions)+'\n}\n')
    (a.output/'baseline_policy.inc').write_text(oracle, encoding='utf8')
    for name in ['host_policy.cpp', 'host_policy.h']:
        expected = frozen_source('6dad3d1', CURRENT_FOLDER+name)
        current = (ROOT/CURRENT_FOLDER/name).read_text(encoding='utf-8-sig')
        assert tokens(expected) == tokens(current), ('captured policy changed', name)
    report = dict(complete=True, oracle_baseline='0bdd483', migration_baseline='6dad3d1',
                  current_policy_tokens_unchanged=True, immutable_oracle_functions=8,
                  oracle_sha256=hashlib.sha256(oracle.encode()).hexdigest(), gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
