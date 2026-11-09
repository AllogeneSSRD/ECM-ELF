#pragma once
#include "../../../execution.h"

namespace ecm_cuda_stage1 {
ecm_stage1::ComputationResult run_ntt_stage1(
    const ecm_stage1::Stage1Request&, ecm_stage1::ExecutionOptions&, ecm_stage1_checkpoint::Context&);
}
