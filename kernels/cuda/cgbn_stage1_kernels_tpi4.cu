/* cgbn_stage1_kernels_tpi4.cu — TPI=4 kernel instantiations (always compiled). */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi4(uint32_t BITS, uint32_t *TPI_out) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_128)
    if (BITS == cgbn_params_128::BITS) {
        *TPI_out = cgbn_params_128::TPI;
        return kernel_double_add<cgbn_params_128>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_192)
    if (BITS == cgbn_params_192::BITS) {
        *TPI_out = cgbn_params_192::TPI;
        return kernel_double_add<cgbn_params_192>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_256)
    if (BITS == cgbn_params_256::BITS) {
        *TPI_out = cgbn_params_256::TPI;
        return kernel_double_add<cgbn_params_256>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_384)
    if (BITS == cgbn_params_384::BITS) {
        *TPI_out = cgbn_params_384::TPI;
        return kernel_double_add<cgbn_params_384>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_small)
    if (BITS == cgbn_params_small::BITS) {
        *TPI_out = cgbn_params_small::TPI;
        return kernel_double_add<cgbn_params_small>;
    }
#endif

    return nullptr;
}
