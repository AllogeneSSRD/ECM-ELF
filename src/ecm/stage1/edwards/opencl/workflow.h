#pragma once
#include "../../execution.h"

namespace ecm_opencl_stage1 {
inline constexpr unsigned edwards_max_bits = 1024;
ecm_stage1::ComputationResult run_edwards_stage1(const ecm_stage1::Stage1Request&,
    ecm_stage1::ExecutionOptions&, ecm_stage1_checkpoint::Context&);
}
