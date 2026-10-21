#pragma once
#include "../../../../service/stage1/legacy_execution.h"
namespace ecm_cuda_edwards {
int run_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
               const ecm_stage1::Stage1RunOptions &,ecm_stage1::LegacyStage1Result *);
}
