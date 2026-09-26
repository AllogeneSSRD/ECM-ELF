/* cgbn_stage1_kernels_suyama_tpi16.cu -- suyama lookup for TPI=16 (2560..8192 tiers), split out of cgbn_stage1_kernels_suyama.cu on 2026-09-25.
 *
 * Reason: these instantiations dominate the kernel compile time (the old combined TU
 * took 745 s for the suyama family and 711 s for param2, i.e. it WAS the critical path
 * of the parallel build).  The per-tier guards (ECM_TIERS_RESTRICTED / ECM_TIER_<bits>)
 * and the exported symbol name are unchanged, so the dispatcher in cgbn_stage1.cu and
 * the restricted-tier builds keep working as before.
 * See docs/ECM_CGBN_OPTIMIZATION.md 8.8. */

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

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
    if (BITS == cgbn_params_2560::BITS) {
        *TPI_out = cgbn_params_2560::TPI;
        return kernel_double_add_suyama<cgbn_params_2560>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3072)
    if (BITS == cgbn_params_3072::BITS) {
        *TPI_out = cgbn_params_3072::TPI;
        return kernel_double_add_suyama<cgbn_params_3072>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3584)
    if (BITS == cgbn_params_3584::BITS) {
        *TPI_out = cgbn_params_3584::TPI;
        return kernel_double_add_suyama<cgbn_params_3584>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4096)
    if (BITS == cgbn_params_4096::BITS) {
        *TPI_out = cgbn_params_4096::TPI;
        return kernel_double_add_suyama<cgbn_params_4096>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (BITS == cgbn_params_4608::BITS) {
        *TPI_out = cgbn_params_4608::TPI;
        return kernel_double_add_suyama<cgbn_params_4608>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5120)
    if (BITS == cgbn_params_5120::BITS) {
        *TPI_out = cgbn_params_5120::TPI;
        return kernel_double_add_suyama<cgbn_params_5120>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5632)
    if (BITS == cgbn_params_5632::BITS) {
        *TPI_out = cgbn_params_5632::TPI;
        return kernel_double_add_suyama<cgbn_params_5632>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6144)
    if (BITS == cgbn_params_6144::BITS) {
        *TPI_out = cgbn_params_6144::TPI;
        return kernel_double_add_suyama<cgbn_params_6144>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6656)
    if (BITS == cgbn_params_6656::BITS) {
        *TPI_out = cgbn_params_6656::TPI;
        return kernel_double_add_suyama<cgbn_params_6656>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7168)
    if (BITS == cgbn_params_7168::BITS) {
        *TPI_out = cgbn_params_7168::TPI;
        return kernel_double_add_suyama<cgbn_params_7168>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7680)
    if (BITS == cgbn_params_7680::BITS) {
        *TPI_out = cgbn_params_7680::TPI;
        return kernel_double_add_suyama<cgbn_params_7680>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_8192)
    if (BITS == cgbn_params_8192::BITS) {
        *TPI_out = cgbn_params_8192::TPI;
        return kernel_double_add_suyama<cgbn_params_8192>;
    }
#endif

#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}
