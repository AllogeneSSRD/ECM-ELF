#pragma once
#include "../../../../service/stage1/execution_contract.h"
namespace ecm_opencl_stage1 {
int run_mont_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                   const std::string &,ecm_stage1::ExecutionOptions &,
                   ecm_stage1::ExecutionResult *);
}
