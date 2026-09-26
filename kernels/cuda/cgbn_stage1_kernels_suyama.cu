/* cgbn_stage1_kernels_suyama.cu -- suyama lookups for the small tiers (TPI=4: 128..512, TPI=8: 768..2048).
 *
 * The TPI=16 (2560..8192) and TPI=32 (9216..16384) lookups live in
 * cgbn_stage1_kernels_suyama_tpi16.cu / cgbn_stage1_kernels_suyama_tpi32.cu: they dominated the compile time of this file
 * (745 s / 711 s total, the critical path of the parallel build).  See
 * docs/ECM_CGBN_OPTIMIZATION.md 8.8. */

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
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_128)
    if (BITS == cgbn_params_128::BITS) {
        *TPI_out = cgbn_params_128::TPI;
        return kernel_double_add_suyama<cgbn_params_128>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_192)
    if (BITS == cgbn_params_192::BITS) {
        *TPI_out = cgbn_params_192::TPI;
        return kernel_double_add_suyama<cgbn_params_192>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_256)
    if (BITS == cgbn_params_256::BITS) {
        *TPI_out = cgbn_params_256::TPI;
        return kernel_double_add_suyama<cgbn_params_256>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_384)
    if (BITS == cgbn_params_384::BITS) {
        *TPI_out = cgbn_params_384::TPI;
        return kernel_double_add_suyama<cgbn_params_384>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_small)
    if (BITS == cgbn_params_small::BITS) {
        *TPI_out = cgbn_params_small::TPI;
        return kernel_double_add_suyama<cgbn_params_small>;
    }
#endif

    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t BITS, uint32_t *TPI_out) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_768)
    if (BITS == cgbn_params_768::BITS) {
        *TPI_out = cgbn_params_768::TPI;
        return kernel_double_add_suyama<cgbn_params_768>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_medium)
    if (BITS == cgbn_params_medium::BITS) {
        *TPI_out = cgbn_params_medium::TPI;
        return kernel_double_add_suyama<cgbn_params_medium>;
    }
#endif

#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1280)
    if (BITS == cgbn_params_1280::BITS) {
        *TPI_out = cgbn_params_1280::TPI;
        return kernel_double_add_suyama<cgbn_params_1280>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1536)
    if (BITS == cgbn_params_1536::BITS) {
        *TPI_out = cgbn_params_1536::TPI;
        return kernel_double_add_suyama<cgbn_params_1536>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1792)
    if (BITS == cgbn_params_1792::BITS) {
        *TPI_out = cgbn_params_1792::TPI;
        return kernel_double_add_suyama<cgbn_params_1792>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2048)
    if (BITS == cgbn_params_2048::BITS) {
        *TPI_out = cgbn_params_2048::TPI;
        return kernel_double_add_suyama<cgbn_params_2048>;
    }
#endif

#endif
    return nullptr;
}
