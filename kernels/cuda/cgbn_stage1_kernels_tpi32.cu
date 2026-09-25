/* cgbn_stage1_kernels_tpi32.cu — TPI=32 kernel instantiations (full build only). */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi32(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
    if (BITS == cgbn_params_9216::BITS) {
        *TPI_out = cgbn_params_9216::TPI;
        return kernel_double_add<cgbn_params_9216>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
    if (BITS == cgbn_params_10240::BITS) {
        *TPI_out = cgbn_params_10240::TPI;
        return kernel_double_add<cgbn_params_10240>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
    if (BITS == cgbn_params_11264::BITS) {
        *TPI_out = cgbn_params_11264::TPI;
        return kernel_double_add<cgbn_params_11264>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
    if (BITS == cgbn_params_12288::BITS) {
        *TPI_out = cgbn_params_12288::TPI;
        return kernel_double_add<cgbn_params_12288>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
    if (BITS == cgbn_params_13312::BITS) {
        *TPI_out = cgbn_params_13312::TPI;
        return kernel_double_add<cgbn_params_13312>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
    if (BITS == cgbn_params_14336::BITS) {
        *TPI_out = cgbn_params_14336::TPI;
        return kernel_double_add<cgbn_params_14336>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
    if (BITS == cgbn_params_15360::BITS) {
        *TPI_out = cgbn_params_15360::TPI;
        return kernel_double_add<cgbn_params_15360>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
    if (BITS == cgbn_params_16384::BITS) {
        *TPI_out = cgbn_params_16384::TPI;
        return kernel_double_add<cgbn_params_16384>;
    }
#endif

#endif
    return nullptr;
}
