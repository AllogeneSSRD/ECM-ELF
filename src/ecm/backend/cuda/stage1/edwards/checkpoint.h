#pragma once
#include "batch.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include <string>
namespace ed_gpu_driver {
std::string checkpoint_body(Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
bool save_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t,
                        bool,bool,bool,int);
ecm_stage1_checkpoint::RestoreResult restore_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t,
                        bool,bool,bool,int);
}
