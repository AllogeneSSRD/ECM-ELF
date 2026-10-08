// Split heavy template instantiations into parallel compilation jobs.
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_9216_10240(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
    if (bits == 9216) { *tpi = cgbn_params_9216::TPI; return domain_kernel<cgbn_params_9216>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
    if (bits == 10240) { *tpi = cgbn_params_10240::TPI; return domain_kernel<cgbn_params_10240>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}

