/* cgbn_stage1_kernels_suyama_tpi32.cu -- suyama lookup for TPI=32 (9216..16384 tiers), split out of cgbn_stage1_kernels_suyama.cu on 2026-09-25.
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

cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
    if (BITS == cgbn_params_9216::BITS) {
        *TPI_out = cgbn_params_9216::TPI;
        return kernel_double_add_suyama<cgbn_params_9216>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
    if (BITS == cgbn_params_10240::BITS) {
        *TPI_out = cgbn_params_10240::TPI;
        return kernel_double_add_suyama<cgbn_params_10240>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
    if (BITS == cgbn_params_11264::BITS) {
        *TPI_out = cgbn_params_11264::TPI;
        return kernel_double_add_suyama<cgbn_params_11264>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
    if (BITS == cgbn_params_12288::BITS) {
        *TPI_out = cgbn_params_12288::TPI;
        return kernel_double_add_suyama<cgbn_params_12288>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
    if (BITS == cgbn_params_13312::BITS) {
        *TPI_out = cgbn_params_13312::TPI;
        return kernel_double_add_suyama<cgbn_params_13312>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
    if (BITS == cgbn_params_14336::BITS) {
        *TPI_out = cgbn_params_14336::TPI;
        return kernel_double_add_suyama<cgbn_params_14336>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
    if (BITS == cgbn_params_15360::BITS) {
        *TPI_out = cgbn_params_15360::TPI;
        return kernel_double_add_suyama<cgbn_params_15360>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
    if (BITS == cgbn_params_16384::BITS) {
        *TPI_out = cgbn_params_16384::TPI;
        return kernel_double_add_suyama<cgbn_params_16384>;
    }
#endif

#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}
