#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi16(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
    if (bits == 2560) { *tpi = cgbn_params_2560::TPI; return domain_kernel<cgbn_params_2560>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3072)
    if (bits == 3072) { *tpi = cgbn_params_3072::TPI; return domain_kernel<cgbn_params_3072>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3584)
    if (bits == 3584) { *tpi = cgbn_params_3584::TPI; return domain_kernel<cgbn_params_3584>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4096)
    if (bits == 4096) { *tpi = cgbn_params_4096::TPI; return domain_kernel<cgbn_params_4096>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (bits == 4608) { *tpi = cgbn_params_4608::TPI; return domain_kernel<cgbn_params_4608>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5120)
    if (bits == 5120) { *tpi = cgbn_params_5120::TPI; return domain_kernel<cgbn_params_5120>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5632)
    if (bits == 5632) { *tpi = cgbn_params_5632::TPI; return domain_kernel<cgbn_params_5632>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6144)
    if (bits == 6144) { *tpi = cgbn_params_6144::TPI; return domain_kernel<cgbn_params_6144>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6656)
    if (bits == 6656) { *tpi = cgbn_params_6656::TPI; return domain_kernel<cgbn_params_6656>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7168)
    if (bits == 7168) { *tpi = cgbn_params_7168::TPI; return domain_kernel<cgbn_params_7168>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7680)
    if (bits == 7680) { *tpi = cgbn_params_7680::TPI; return domain_kernel<cgbn_params_7680>(mode); }
#endif
#endif
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_8192)
    if (bits == 8192) { *tpi = cgbn_params_8192::TPI; return domain_kernel<cgbn_params_8192>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}
