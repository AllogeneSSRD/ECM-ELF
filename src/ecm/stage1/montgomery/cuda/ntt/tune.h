#pragma once

#include "ecm/stage1/task.h"
#include <string>

namespace ecm_mersenne_ntt {

struct Plan;
struct Run;
std::string tune_shape_descriptor(const Plan& plan);

int tune_command(int device, const ecm_stage1::Stage1NttOptions&, unsigned p, bool exhaustive);
ecm_stage1::Stage1NttOptions select_tuned_plan(int device, unsigned p, unsigned curves,
                                             const ecm_stage1::Stage1NttOptions&, bool native_edwards = false);
unsigned select_production_batch(const ecm_stage1::Stage1Request&, unsigned remaining);
void record_production_sample(const ecm_stage1::Stage1Request&, const Plan&, const Run&,
                              unsigned valid, unsigned batch, double compute_seconds);

} // namespace ecm_mersenne_ntt
