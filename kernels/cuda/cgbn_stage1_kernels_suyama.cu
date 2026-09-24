/* cgbn_stage1_kernels_suyama.cu — Suyama param0 kernel instantiations.
 *
 * One separate __global__ function per (TPI, BITS) pair, mirroring the param3 set
 * exactly: the two parametrizations differ in the per-bit arithmetic (full-width a24
 * and the difference x-coordinate instead of the 32-bit d and the constant 2), so
 * they are different kernels -- see docs/ECM_Montgomery_STAGE1.md §21 for the answer
 * to "can param0 and param3 share instantiations?".
 *
 * Grid (identical to cgbn_stage1_kernels_tpi*.cu):
 *   TPI=4  : 128, 192, 256, 384, 512                      always compiled
 *   TPI=8  : 768, 1024, 1280, 1536, 1792, 2048            always / full
 *   TPI=16 : 2560 ... 8192, 512 interval                  full build only
 *   TPI=32 : 9216 ... 16384, 512 interval                 full build only
 *
 * The dev build (IS_DEV_BUILD) keeps only the small tiers so that correctness runs on
 * small N stay fast to compile; anything the batch path can handle above ~1024 bits
 * needs the full build (ECM_CUDA_FULL_BUILD=ON).
 */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi4(uint32_t BITS, uint32_t *TPI_out) {
    if (BITS == cgbn_params_128::BITS) {
        *TPI_out = cgbn_params_128::TPI;
        return kernel_double_add_suyama<cgbn_params_128>;
    }
    if (BITS == cgbn_params_192::BITS) {
        *TPI_out = cgbn_params_192::TPI;
        return kernel_double_add_suyama<cgbn_params_192>;
    }
    if (BITS == cgbn_params_256::BITS) {
        *TPI_out = cgbn_params_256::TPI;
        return kernel_double_add_suyama<cgbn_params_256>;
    }
    if (BITS == cgbn_params_384::BITS) {
        *TPI_out = cgbn_params_384::TPI;
        return kernel_double_add_suyama<cgbn_params_384>;
    }
    if (BITS == cgbn_params_small::BITS) {
        *TPI_out = cgbn_params_small::TPI;
        return kernel_double_add_suyama<cgbn_params_small>;
    }
    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t BITS, uint32_t *TPI_out) {
    if (BITS == cgbn_params_768::BITS) {
        *TPI_out = cgbn_params_768::TPI;
        return kernel_double_add_suyama<cgbn_params_768>;
    }
    if (BITS == cgbn_params_medium::BITS) {
        *TPI_out = cgbn_params_medium::TPI;
        return kernel_double_add_suyama<cgbn_params_medium>;
    }
#ifndef IS_DEV_BUILD
    if (BITS == cgbn_params_1280::BITS) {
        *TPI_out = cgbn_params_1280::TPI;
        return kernel_double_add_suyama<cgbn_params_1280>;
    }
    if (BITS == cgbn_params_1536::BITS) {
        *TPI_out = cgbn_params_1536::TPI;
        return kernel_double_add_suyama<cgbn_params_1536>;
    }
    if (BITS == cgbn_params_1792::BITS) {
        *TPI_out = cgbn_params_1792::TPI;
        return kernel_double_add_suyama<cgbn_params_1792>;
    }
    if (BITS == cgbn_params_2048::BITS) {
        *TPI_out = cgbn_params_2048::TPI;
        return kernel_double_add_suyama<cgbn_params_2048>;
    }
#endif
    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
    if (BITS == cgbn_params_2560::BITS) {
        *TPI_out = cgbn_params_2560::TPI;
        return kernel_double_add_suyama<cgbn_params_2560>;
    }
    if (BITS == cgbn_params_3072::BITS) {
        *TPI_out = cgbn_params_3072::TPI;
        return kernel_double_add_suyama<cgbn_params_3072>;
    }
    if (BITS == cgbn_params_3584::BITS) {
        *TPI_out = cgbn_params_3584::TPI;
        return kernel_double_add_suyama<cgbn_params_3584>;
    }
    if (BITS == cgbn_params_4096::BITS) {
        *TPI_out = cgbn_params_4096::TPI;
        return kernel_double_add_suyama<cgbn_params_4096>;
    }
    if (BITS == cgbn_params_4608::BITS) {
        *TPI_out = cgbn_params_4608::TPI;
        return kernel_double_add_suyama<cgbn_params_4608>;
    }
    if (BITS == cgbn_params_5120::BITS) {
        *TPI_out = cgbn_params_5120::TPI;
        return kernel_double_add_suyama<cgbn_params_5120>;
    }
    if (BITS == cgbn_params_5632::BITS) {
        *TPI_out = cgbn_params_5632::TPI;
        return kernel_double_add_suyama<cgbn_params_5632>;
    }
    if (BITS == cgbn_params_6144::BITS) {
        *TPI_out = cgbn_params_6144::TPI;
        return kernel_double_add_suyama<cgbn_params_6144>;
    }
    if (BITS == cgbn_params_6656::BITS) {
        *TPI_out = cgbn_params_6656::TPI;
        return kernel_double_add_suyama<cgbn_params_6656>;
    }
    if (BITS == cgbn_params_7168::BITS) {
        *TPI_out = cgbn_params_7168::TPI;
        return kernel_double_add_suyama<cgbn_params_7168>;
    }
    if (BITS == cgbn_params_7680::BITS) {
        *TPI_out = cgbn_params_7680::TPI;
        return kernel_double_add_suyama<cgbn_params_7680>;
    }
    if (BITS == cgbn_params_8192::BITS) {
        *TPI_out = cgbn_params_8192::TPI;
        return kernel_double_add_suyama<cgbn_params_8192>;
    }
#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
    if (BITS == cgbn_params_9216::BITS) {
        *TPI_out = cgbn_params_9216::TPI;
        return kernel_double_add_suyama<cgbn_params_9216>;
    }
    if (BITS == cgbn_params_10240::BITS) {
        *TPI_out = cgbn_params_10240::TPI;
        return kernel_double_add_suyama<cgbn_params_10240>;
    }
    if (BITS == cgbn_params_11264::BITS) {
        *TPI_out = cgbn_params_11264::TPI;
        return kernel_double_add_suyama<cgbn_params_11264>;
    }
    if (BITS == cgbn_params_12288::BITS) {
        *TPI_out = cgbn_params_12288::TPI;
        return kernel_double_add_suyama<cgbn_params_12288>;
    }
    if (BITS == cgbn_params_13312::BITS) {
        *TPI_out = cgbn_params_13312::TPI;
        return kernel_double_add_suyama<cgbn_params_13312>;
    }
    if (BITS == cgbn_params_14336::BITS) {
        *TPI_out = cgbn_params_14336::TPI;
        return kernel_double_add_suyama<cgbn_params_14336>;
    }
    if (BITS == cgbn_params_15360::BITS) {
        *TPI_out = cgbn_params_15360::TPI;
        return kernel_double_add_suyama<cgbn_params_15360>;
    }
    if (BITS == cgbn_params_16384::BITS) {
        *TPI_out = cgbn_params_16384::TPI;
        return kernel_double_add_suyama<cgbn_params_16384>;
    }
#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}
