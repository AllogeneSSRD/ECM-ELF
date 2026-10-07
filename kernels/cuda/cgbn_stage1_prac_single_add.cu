#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>
#include "cgbn_stage1_kernel.h"
#include "cgbn_stage1_prac_kernel.cuh"
#include "cgbn_stage1_prac_single_add.cuh"

// Separate instantiations keep this candidate off the already large TPI16 TU.
cgbn_stage1_kernel_fn cgbn_stage1_domain_single_add(uint32_t bits, uint32_t *tpi, int mode) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (bits == 4608) {
        *tpi = 16;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD)
            return kernel_suyama_domain<cgbn_params_t<16, 4608>, ECM_DOMAIN_PRAC_SINGLE_ADD>;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD_168)
            return kernel_suyama_domain<cgbn_params_t<16, 4608>, ECM_DOMAIN_PRAC_SINGLE_ADD_168>;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT)
            return kernel_suyama_domain<cgbn_params_t<16, 4608>, ECM_DOMAIN_PRAC_SINGLE_COMPACT>;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168)
            return kernel_suyama_domain<cgbn_params_t<16, 4608>, ECM_DOMAIN_PRAC_SINGLE_COMPACT_168>;
    }
#endif
    *tpi = 0; return nullptr;
}

// Explicit same-container TPI32 experiment; normal tier selection stays TPI16.
// Keep the old single-add candidate and other point variants off this path.
cgbn_stage1_kernel_fn cgbn_stage1_domain_single_compact32(uint32_t bits, uint32_t *tpi, int mode) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (bits == 4608) {
        *tpi = 32;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT)
            return kernel_suyama_domain<cgbn_params_t<32, 4608>, ECM_DOMAIN_PRAC_SINGLE_COMPACT>;
        if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168)
            return kernel_suyama_domain<cgbn_params_t<32, 4608>, ECM_DOMAIN_PRAC_SINGLE_COMPACT_168>;
    }
#endif
    *tpi = 0; return nullptr;
}
