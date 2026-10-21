#pragma once
#include "../runtime_contract.h"
#include <gmp.h>

namespace ecm_cuda_stage1 {
// The callback persists a proper construction factor immediately. It borrows
// the value only for the call; false aborts execution as a publication failure.
using FactorSink=bool (*)(uint32_t,mpz_srcptr,void *);
RunResult execute_mont_batch(mpz_t *,int *,mpz_srcptr,mpz_srcptr,uint32_t,uint64_t *,
    unsigned long,float *,int,int,uint64_t,uint32_t,FactorSink=nullptr,void *context=nullptr);
}
