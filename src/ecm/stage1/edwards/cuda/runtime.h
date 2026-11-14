#pragma once
#include "naf.h"
#include <functional>

namespace ecm_cuda_stage1 {
using EdwardsSnapshot = std::function<void()>;
// The callback can pull canonical host coordinates for this slice while it runs.
using EdwardsSliceProgress = std::function<bool(uint32_t, const EdwardsSnapshot &)>;

struct EdwardsOptions {
    int device=0;
    unsigned window=0; // 0=binary, otherwise width-w NAF.
    int doubling=4;
    uint32_t chunk=0;
    bool allow_memory_downgrade=false;
    bool allow_binary_fallback=false;
    std::function<void(unsigned, unsigned, size_t, size_t)> memory_selection;
    unsigned memory_limit_mb=0;
};
// One production boundary for validation, resolved geometry and error mapping.
// Borrowed state remains owned by the caller, including on failure.
RunResult execute_edwards_batch(mpz_srcptr,mpz_srcptr,mpz_t *,uint32_t,uint32_t &,
    const EdwardsOptions &,const EdwardsSliceProgress &,void *,EcmEdwardsNafMetrics *,EcmEdwardsPrepared);
}
