#pragma once

#include "ecm/platform/logging.h"
#include "ecm/stage1/task.h"

namespace ecm_stage1 {

// The GUI reads this record from stdout. Each optional detail is a single
// key=value token and describes the plan this batch will actually execute.
inline void log_execution_details(CurveModel curve, unsigned gpu_param,
                                  const char* backend, const std::string& details) {
    const auto param = curve == CurveModel::Edwards ? std::string("e") : "m" + std::to_string(gpu_param);
    ecm_ts_fprintf(stdout, "Stage1 execution: curve=%s param=%s backend=%s%s%s\n",
                   curve_model_name(curve), param.c_str(), backend,
                   details.empty() ? "" : " ", details.c_str());
    std::fflush(stdout);
}

inline void log_execution_details(const Stage1Options& options, const char* backend,
                                  const std::string& details) {
    const auto& selection = options.execution.selection;
    log_execution_details(selection.curve, selection.gpu_param, backend, details);
}

} // namespace ecm_stage1
