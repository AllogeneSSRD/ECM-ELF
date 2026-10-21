#pragma once
#include "../../../service/stage1/legacy_execution.h"

// Complete synchronous CPU execution/restore/save workflow. Result ownership
// remains with the task; worker threads finish before these calls return.
namespace ecm_cpu_stage1 {
int run_edwards_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                      const std::string &,ecm_stage1::Stage1ExecutionOptions &,ecm_stage1::LegacyStage1Result *);
int run_mont_stage1(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                   const std::string &,ecm_stage1::Stage1ExecutionOptions &,ecm_stage1::LegacyStage1Result *);
}
