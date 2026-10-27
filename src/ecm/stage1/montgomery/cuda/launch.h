#pragma once
// Include CGBN's public types before this interface. No project device templates.
#include <cstdint>
#ifndef ECM_TPB
#define ECM_TPB 128
#endif
#define CARRY_BITS 6
#ifndef ECM_CUDA_FULL_BUILD
#define IS_DEV_BUILD
#endif
#ifndef ECM_MERS_FOLD
#define ECM_MERS_FOLD 0
#endif
#ifndef ECM_NO_PARAM2
#define ECM_NO_PARAM2 0
#endif
const uint32_t TPB_DEFAULT=ECM_TPB;
typedef void (*cgbn_stage1_kernel_fn)(cgbn_error_report_t *, uint64_t, uint64_t,
                                      uint64_t, uint32_t*, uint32_t*, uint32_t,
                                      uint32_t, uint32_t);

void cgbn_stage1_launch(cgbn_stage1_kernel_fn kernel,dim3 grid,dim3 block,
    cgbn_error_report_t *report,uint64_t scalar_bits,uint64_t start,uint64_t count,
    uint32_t *scalar,uint32_t *data,uint32_t curves,uint32_t sigma,uint32_t np0);
