#pragma once
#include "execution_contract.h"

namespace ecm_stage1 {
#if defined(ECM_HAS_CUDA_STAGE1) && ECM_HAS_CUDA_STAGE1
inline constexpr bool defer_gpu_checkpoint_validation = true;
#else
inline constexpr bool defer_gpu_checkpoint_validation = false;
#endif
// Resolve the actual task route without executing a backend or changing input.
Stage1Entry select_stage1_entry(const Selection&);
} // namespace ecm_stage1
