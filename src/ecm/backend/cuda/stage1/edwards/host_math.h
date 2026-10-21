#pragma once
#include "batch.h"
#include <string>
namespace ed_gpu_driver {
bool initialize(Batch &,mpz_srcptr,std::string &);
}
