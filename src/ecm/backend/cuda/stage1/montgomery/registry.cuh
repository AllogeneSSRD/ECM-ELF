#pragma once
#include "../../arithmetic/cgbn/types.cuh"
#include "launch.h"
#include <vector>

// Actual immutable launch geometry after selection/restored-layout validation.
struct MontResolvedKernel {
    cgbn_stage1_kernel_fn entry;
    uint32_t bits,tpi,tpb;
    size_t blocks,state_bytes;
};

void ecm_build_tier_list_default(std::vector<uint32_t> &);
void ecm_build_tier_list_param0(std::vector<uint32_t> &);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_dispatch(uint32_t,uint32_t *);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_dispatch(uint32_t,uint32_t *);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_dispatch(uint32_t,uint32_t *);
