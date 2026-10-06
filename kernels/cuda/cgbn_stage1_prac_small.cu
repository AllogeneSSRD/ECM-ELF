#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_small(uint32_t bits, uint32_t *tpi, int mode) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_128)
    if (bits == 128) { *tpi = cgbn_params_128::TPI; return domain_kernel<cgbn_params_128>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_192)
    if (bits == 192) { *tpi = cgbn_params_192::TPI; return domain_kernel<cgbn_params_192>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_256)
    if (bits == 256) { *tpi = cgbn_params_256::TPI; return domain_kernel<cgbn_params_256>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_384)
    if (bits == 384) { *tpi = cgbn_params_384::TPI; return domain_kernel<cgbn_params_384>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_small)
    if (bits == 512) { *tpi = cgbn_params_small::TPI; return domain_kernel<cgbn_params_small>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_768)
    if (bits == 768) { *tpi = cgbn_params_768::TPI; return domain_kernel<cgbn_params_768>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_medium)
    if (bits == 1024) { *tpi = cgbn_params_medium::TPI; return domain_kernel<cgbn_params_medium>(mode); }
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1280)
    if (bits == 1280) { *tpi = cgbn_params_1280::TPI; return domain_kernel<cgbn_params_1280>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1536)
    if (bits == 1536) { *tpi = cgbn_params_1536::TPI; return domain_kernel<cgbn_params_1536>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1792)
    if (bits == 1792) { *tpi = cgbn_params_1792::TPI; return domain_kernel<cgbn_params_1792>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2048)
    if (bits == 2048) { *tpi = cgbn_params_2048::TPI; return domain_kernel<cgbn_params_2048>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}
