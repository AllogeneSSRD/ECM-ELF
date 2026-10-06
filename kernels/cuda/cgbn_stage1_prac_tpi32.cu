#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
    if (bits == 9216) { *tpi = cgbn_params_9216::TPI; return domain_kernel<cgbn_params_9216>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
    if (bits == 10240) { *tpi = cgbn_params_10240::TPI; return domain_kernel<cgbn_params_10240>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
    if (bits == 11264) { *tpi = cgbn_params_11264::TPI; return domain_kernel<cgbn_params_11264>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
    if (bits == 12288) { *tpi = cgbn_params_12288::TPI; return domain_kernel<cgbn_params_12288>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
    if (bits == 13312) { *tpi = cgbn_params_13312::TPI; return domain_kernel<cgbn_params_13312>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
    if (bits == 14336) { *tpi = cgbn_params_14336::TPI; return domain_kernel<cgbn_params_14336>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
    if (bits == 15360) { *tpi = cgbn_params_15360::TPI; return domain_kernel<cgbn_params_15360>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
    if (bits == 16384) { *tpi = cgbn_params_16384::TPI; return domain_kernel<cgbn_params_16384>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}
