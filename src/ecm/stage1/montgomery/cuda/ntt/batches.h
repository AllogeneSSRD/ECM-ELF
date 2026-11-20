#pragma once
#include "../../../execution.h"
#include <functional>

namespace ecm_cuda_stage1 {
ecm_stage1::ComputationResult run_ntt_batches(
    const ecm_stage1::Stage1Request&, const ecm_stage1::TaskDependencies&,
    ecm_stage1_checkpoint::Context&,
    const std::function<void(const ecm_stage1::CurveOutcome&, uint32_t)>& publish_factor);
}
