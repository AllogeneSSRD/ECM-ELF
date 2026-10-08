// Split heavy template instantiations into parallel compilation jobs.
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_11264_12288(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
    if (bits == 11264) { *tpi = cgbn_params_11264::TPI; return domain_kernel<cgbn_params_11264>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
    if (bits == 12288) { *tpi = cgbn_params_12288::TPI; return domain_kernel<cgbn_params_12288>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}

