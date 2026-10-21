#pragma once
#include "../../../service/stage1/execution_contract.h"

// Complete, synchronous curve workflows. The task owns publication ledgers,
// batch checkpoint lifetime and queue policy; these functions own computation
// and curve-specific save output. No CUDA SDK types cross this host boundary.
namespace ecm_cuda_stage1 {
int run_mont_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                   const std::string &,const ecm_stage1::ExecutionOptions &,
                   ecm_stage1::ExecutionResult *);
int run_edwards_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                      const std::string &,const ecm_stage1::ExecutionOptions &,
                      ecm_stage1::ExecutionResult *);
}
