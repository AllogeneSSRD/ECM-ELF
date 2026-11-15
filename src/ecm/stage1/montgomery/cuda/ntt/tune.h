#pragma once

#include "ecm/stage1/task.h"
#include <string>

namespace ecm_mersenne_ntt {

struct Plan;
std::string tune_shape_descriptor(const Plan& plan);

int tune_command(int device, const ecm_stage1::Stage1NttOptions&, unsigned p, bool exhaustive);
ecm_stage1::Stage1NttOptions select_tuned_plan(int device, unsigned p, unsigned curves,
                                             const ecm_stage1::Stage1NttOptions&);

} // namespace ecm_mersenne_ntt
