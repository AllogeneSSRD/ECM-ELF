#pragma once
#include "../../../../cuda/arithmetic/cgbn/types.cuh"
#include "../arithmetic.cuh"
#include "cgbn_stage1_prac_kernel.cuh"

cgbn_stage1_kernel_fn ECM_PRAC_FACTORY(uint32_t bits, uint32_t *tpi, int mode) {
    if (bits != ECM_PRAC_BITS) {
        *tpi = 0;
        return nullptr;
    }
    *tpi = ECM_PRAC_TPI;
    using P = cgbn_params_t<ECM_PRAC_TPI, ECM_PRAC_BITS>;
    return domain_kernel<P>(mode);
}
