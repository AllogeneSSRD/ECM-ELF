#pragma once
#include "naf.h"

namespace ecm_cuda_stage1 {
struct EdwardsOptions {
    int device=0;
    unsigned window=0; // 0=binary, otherwise width-w NAF.
    int doubling=4;
    uint32_t chunk=0;
};
// One production boundary for validation, resolved geometry and error mapping.
// Borrowed state remains owned by the caller, including on failure.
RunResult execute_edwards_batch(mpz_srcptr,mpz_srcptr,mpz_t *,uint32_t,uint32_t &,
    const EdwardsOptions &,EcmEdwardsProgress,void *,EcmEdwardsNafMetrics *,EcmEdwardsPrepared);
}
