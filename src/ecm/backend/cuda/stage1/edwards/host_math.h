#pragma once
#include "batch.h"
#include <string>
#include <functional>
namespace ed_gpu_driver {
using ConstructionSink=std::function<bool(uint32_t,mpz_srcptr)>;
bool initialize(Batch &,mpz_srcptr,std::string &,const ConstructionSink &publish={});
}
