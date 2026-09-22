/* cgbn_stage1_kernels_tpi8.cu — TPI=8 kernel instantiations.
   768/1024 are compiled in both dev and full builds; 1280..2048 are full-only. */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi8(uint32_t BITS, uint32_t *TPI_out) {
    if (BITS == cgbn_params_768::BITS) {
        *TPI_out = cgbn_params_768::TPI;
        return kernel_double_add<cgbn_params_768>;
    }
    if (BITS == cgbn_params_medium::BITS) {
        *TPI_out = cgbn_params_medium::TPI;
        return kernel_double_add<cgbn_params_medium>;
    }
#ifndef IS_DEV_BUILD
    if (BITS == cgbn_params_1280::BITS) {
        *TPI_out = cgbn_params_1280::TPI;
        return kernel_double_add<cgbn_params_1280>;
    }
    if (BITS == cgbn_params_1536::BITS) {
        *TPI_out = cgbn_params_1536::TPI;
        return kernel_double_add<cgbn_params_1536>;
    }
    if (BITS == cgbn_params_1792::BITS) {
        *TPI_out = cgbn_params_1792::TPI;
        return kernel_double_add<cgbn_params_1792>;
    }
    if (BITS == cgbn_params_2048::BITS) {
        *TPI_out = cgbn_params_2048::TPI;
        return kernel_double_add<cgbn_params_2048>;
    }
#endif
    return nullptr;
}
