#include "workflow.h"
#include "device.h"
#include "../thread_workflow.h"

namespace ecm_opencl_stage1 {

ecm_stage1::ComputationResult run_edwards_stage1(const ecm_stage1::Stage1Request& request,
        ecm_stage1::ExecutionOptions& options, ecm_stage1_checkpoint::Context& checkpoint) {
    return ecm_edwards_thread::run_stage1<Device>(request, options, checkpoint);
}
} // namespace ecm_opencl_stage1
