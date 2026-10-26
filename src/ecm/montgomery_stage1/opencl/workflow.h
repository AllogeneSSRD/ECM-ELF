#pragma once
#include "../../stage1/execution.h"

// Own computation and checkpoint safe points; final file publication belongs to task.
namespace ecm_opencl_stage1 {
ecm_stage1::ComputationResult run_mont_stage1(const ecm_stage1::Stage1Request &,ecm_stage1::ExecutionOptions &, ecm_stage1_checkpoint::Context& checkpoint);
}
