/* cgbn_stage1_kernels_tpi16.cu — TPI=16 kernel instantiations (full build only).
   2560..8192 at 256-bit intervals. */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
    if (BITS == cgbn_params_2560::BITS) {
        *TPI_out = cgbn_params_2560::TPI;
        return kernel_double_add<cgbn_params_2560>;
    }
    if (BITS == cgbn_params_2816::BITS) {
        *TPI_out = cgbn_params_2816::TPI;
        return kernel_double_add<cgbn_params_2816>;
    }
    if (BITS == cgbn_params_3072::BITS) {
        *TPI_out = cgbn_params_3072::TPI;
        return kernel_double_add<cgbn_params_3072>;
    }
    if (BITS == cgbn_params_3328::BITS) {
        *TPI_out = cgbn_params_3328::TPI;
        return kernel_double_add<cgbn_params_3328>;
    }
    if (BITS == cgbn_params_3584::BITS) {
        *TPI_out = cgbn_params_3584::TPI;
        return kernel_double_add<cgbn_params_3584>;
    }
    if (BITS == cgbn_params_3840::BITS) {
        *TPI_out = cgbn_params_3840::TPI;
        return kernel_double_add<cgbn_params_3840>;
    }
    if (BITS == cgbn_params_4096::BITS) {
        *TPI_out = cgbn_params_4096::TPI;
        return kernel_double_add<cgbn_params_4096>;
    }
    if (BITS == cgbn_params_4352::BITS) {
        *TPI_out = cgbn_params_4352::TPI;
        return kernel_double_add<cgbn_params_4352>;
    }
    if (BITS == cgbn_params_4608::BITS) {
        *TPI_out = cgbn_params_4608::TPI;
        return kernel_double_add<cgbn_params_4608>;
    }
    if (BITS == cgbn_params_4864::BITS) {
        *TPI_out = cgbn_params_4864::TPI;
        return kernel_double_add<cgbn_params_4864>;
    }
    if (BITS == cgbn_params_5120::BITS) {
        *TPI_out = cgbn_params_5120::TPI;
        return kernel_double_add<cgbn_params_5120>;
    }
    if (BITS == cgbn_params_5376::BITS) {
        *TPI_out = cgbn_params_5376::TPI;
        return kernel_double_add<cgbn_params_5376>;
    }
    if (BITS == cgbn_params_5632::BITS) {
        *TPI_out = cgbn_params_5632::TPI;
        return kernel_double_add<cgbn_params_5632>;
    }
    if (BITS == cgbn_params_5888::BITS) {
        *TPI_out = cgbn_params_5888::TPI;
        return kernel_double_add<cgbn_params_5888>;
    }
    if (BITS == cgbn_params_6144::BITS) {
        *TPI_out = cgbn_params_6144::TPI;
        return kernel_double_add<cgbn_params_6144>;
    }
    if (BITS == cgbn_params_6400::BITS) {
        *TPI_out = cgbn_params_6400::TPI;
        return kernel_double_add<cgbn_params_6400>;
    }
    if (BITS == cgbn_params_6656::BITS) {
        *TPI_out = cgbn_params_6656::TPI;
        return kernel_double_add<cgbn_params_6656>;
    }
    if (BITS == cgbn_params_6912::BITS) {
        *TPI_out = cgbn_params_6912::TPI;
        return kernel_double_add<cgbn_params_6912>;
    }
    if (BITS == cgbn_params_7168::BITS) {
        *TPI_out = cgbn_params_7168::TPI;
        return kernel_double_add<cgbn_params_7168>;
    }
    if (BITS == cgbn_params_7424::BITS) {
        *TPI_out = cgbn_params_7424::TPI;
        return kernel_double_add<cgbn_params_7424>;
    }
    if (BITS == cgbn_params_7680::BITS) {
        *TPI_out = cgbn_params_7680::TPI;
        return kernel_double_add<cgbn_params_7680>;
    }
    if (BITS == cgbn_params_7936::BITS) {
        *TPI_out = cgbn_params_7936::TPI;
        return kernel_double_add<cgbn_params_7936>;
    }
    if (BITS == cgbn_params_8192::BITS) {
        *TPI_out = cgbn_params_8192::TPI;
        return kernel_double_add<cgbn_params_8192>;
    }
#endif
    return nullptr;
}
