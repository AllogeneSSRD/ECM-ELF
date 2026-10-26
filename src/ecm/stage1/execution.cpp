#include "execution.h"
#include <stdexcept>
#if !ECM_HAS_CUDA_STAGE1
#include "montgomery/opencl/workflow.h"
#endif

namespace ecm_stage1 {

Stage1Entry select_stage1_entry(const Selection& selection) {
    if (selection.target == ExecutionTarget::Cpu)
        return selection.curve == CurveModel::Edwards ? ecm_cpu_stage1::run_edwards_stage1
                                                      : ecm_cpu_stage1::run_mont_stage1;
#if ECM_HAS_CUDA_STAGE1
    return selection.curve == CurveModel::Edwards ? ecm_cuda_stage1::run_edwards_stage1
                                                  : ecm_cuda_stage1::run_mont_stage1;
#else
    if (selection.curve == CurveModel::Edwards)
        throw std::invalid_argument("GPU Edwards requires the CUDA backend");
    return ecm_opencl_stage1::run_mont_stage1;
#endif
}
} // namespace ecm_stage1
