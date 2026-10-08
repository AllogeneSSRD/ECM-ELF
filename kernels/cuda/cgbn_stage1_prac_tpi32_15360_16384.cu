// Split heavy template instantiations into parallel compilation jobs.
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_15360_16384(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
    if (bits == 15360) { *tpi = cgbn_params_15360::TPI; return domain_kernel<cgbn_params_15360>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
    if (bits == 16384) { *tpi = cgbn_params_16384::TPI; return domain_kernel<cgbn_params_16384>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}

