#pragma once
#include "io/stage1_checkpoint.h"

#include "../host_math.h"
namespace ecm_cuda_stage1 {
int run_mont_resident(ecm_stage1_checkpoint::Context& checkpoint, mpz_t *,int *,mpz_srcptr,mpz_srcptr,uint32_t,uint64_t *,
    unsigned long,float *,uint64_t,uint32_t,bool,RunResult &,
    const ecm_cuda_mont_host::ConstructionResults &,SigmaGenerator,const ecm_cuda_mont_host::MontRequestedSpec &);
}
