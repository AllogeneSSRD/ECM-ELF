#pragma once
#include "../../../../io/stage1_checkpoint.h"

#include "../../cuda_runtime.h"
#include <gmp.h>
namespace ecm_cuda_mont_host {struct MontRequestedSpec;}

namespace ecm_cuda_stage1 {
// The callback persists a proper construction factor immediately. It borrows
// the value only for the call; false aborts execution as a publication failure.
using FactorSink=bool (*)(uint32_t,mpz_srcptr,void *);
// Invoked only for a fresh batch with no accepted checkpoint seed.
using SigmaGenerator=uint64_t (*)(uint32_t curves,int parametrization);
RunResult execute_mont_batch(ecm_stage1_checkpoint::Context& checkpoint, mpz_t *,int *,mpz_srcptr,mpz_srcptr,uint32_t,uint64_t *,
    unsigned long,float *,int,int,uint64_t,uint32_t,FactorSink=nullptr,void *context=nullptr,
    SigmaGenerator generate_sigma=nullptr,const ecm_cuda_mont_host::MontRequestedSpec *request=nullptr);
}
