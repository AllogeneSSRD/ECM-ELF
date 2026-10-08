// Split heavy template instantiations into parallel compilation jobs.
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_13312_14336(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
    if (bits == 13312) { *tpi = cgbn_params_13312::TPI; return domain_kernel<cgbn_params_13312>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
    if (bits == 14336) { *tpi = cgbn_params_14336::TPI; return domain_kernel<cgbn_params_14336>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}

