#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"

// Same field/container and formulas; only cooperative threads per curve differ.
// CGBN pads partial limbs internally for tiers not divisible by 32*TPI.
cgbn_stage1_kernel_fn cgbn_stage1_domain_alternate32(uint32_t bits, uint32_t *tpi, int mode) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
    if (bits == 2560) { *tpi = 32; return domain_kernel<cgbn_params_t<32, 2560>>(mode); }
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (bits == 4608) { *tpi = 32; return domain_kernel<cgbn_params_t<32, 4608>>(mode); }
#endif
#endif
    *tpi = 0; return nullptr;
}
