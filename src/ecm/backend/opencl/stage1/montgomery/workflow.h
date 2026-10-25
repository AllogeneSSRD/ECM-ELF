#pragma once
#include "../../../../service/stage1/execution_contract.h"

// Own computation and checkpoint safe points; final file publication belongs to task.
namespace ecm_opencl_stage1 {
ecm_stage1::ComputationResult run_mont_stage1(const ecm_stage1::Stage1Request &,ecm_stage1::ExecutionOptions &);
}
