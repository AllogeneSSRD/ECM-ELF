#pragma once
#include "batch.h"
#include "../../../../io/stage1_checkpoint.h"
#include <string>
namespace ed_gpu_driver {
std::string checkpoint_body(Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
bool save_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
ecm_stage1_checkpoint::RestoreResult decode_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t,
                        bool,bool,bool,int);
ecm_stage1_checkpoint::RestoreResult restore_checkpoint(ecm_stage1_checkpoint::Context& checkpoint, const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t,
                        bool,bool,bool,int);
}
