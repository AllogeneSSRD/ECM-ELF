#pragma once
#include "../../arithmetic/cgbn/types.cuh"
#include "launch.h"
#include <vector>

void ecm_build_tier_list_default(std::vector<uint32_t> &);
void ecm_build_tier_list_param0(std::vector<uint32_t> &);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_dispatch(uint32_t,uint32_t *);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_dispatch(uint32_t,uint32_t *);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_dispatch(uint32_t,uint32_t *);
