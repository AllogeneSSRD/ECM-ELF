#pragma once
#include "batch.h"
#include <string>
namespace ed_gpu_driver {
std::string checkpoint_body(Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
bool save_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t);
bool restore_checkpoint(const std::string &,Batch &,mpz_srcptr,mpz_srcptr,uint64_t,
                        bool,bool,bool,int);
}
