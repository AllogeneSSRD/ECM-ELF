#pragma once
#include "execution_contract.h"

namespace ecm_stage1 {
#if defined(ECM_HAS_CUDA_STAGE1) && ECM_HAS_CUDA_STAGE1
inline constexpr bool defer_gpu_checkpoint_validation=true;
#else
inline constexpr bool defer_gpu_checkpoint_validation=false;
#endif
// The single execution dispatch consumed by run_stage1_task.
int execute_stage1(mpz_srcptr, double, double, uint32_t, const std::string &, bool,
                   const std::string &, ExecutionOptions &, ExecutionResult *);
}
