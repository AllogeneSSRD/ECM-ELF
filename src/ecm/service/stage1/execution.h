#pragma once
#include "execution_contract.h"

namespace ecm_stage1 {
// The single execution dispatch consumed by run_stage1_task.
int execute_stage1(mpz_srcptr, double, double, uint32_t, const std::string &, bool,
                   const std::string &, ExecutionOptions &, ExecutionResult *);
}
