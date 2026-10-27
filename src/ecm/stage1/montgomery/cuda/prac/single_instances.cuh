#pragma once
#include "../../../../cuda/arithmetic/cgbn/types.cuh"
#include "../arithmetic.cuh"
#include "cgbn_stage1_prac_kernel.cuh"
#include "cgbn_stage1_prac_single_add.cuh"

cgbn_stage1_kernel_fn ECM_PRAC_FACTORY(uint32_t bits, uint32_t *tpi, int mode) {
    if (bits != ECM_PRAC_BITS) {
        *tpi = 0;
        return nullptr;
    }
    *tpi = ECM_PRAC_TPI;
    using P = cgbn_params_t<ECM_PRAC_TPI, ECM_PRAC_BITS>;
#if ECM_PRAC_TPI == 16
    if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD)
        return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_SINGLE_ADD>;
    if (mode == ECM_DOMAIN_PRAC_SINGLE_ADD_168)
        return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_SINGLE_ADD_168>;
#endif
    if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT)
        return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_SINGLE_COMPACT>;
    if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168)
        return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_SINGLE_COMPACT_168>;
#if ECM_PRAC_TPI == 16
    if (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_128)
        return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_SINGLE_COMPACT_128>;
#endif
    *tpi = 0;
    return nullptr;
}
