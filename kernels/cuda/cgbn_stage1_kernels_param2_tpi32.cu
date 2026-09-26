/* cgbn_stage1_kernels_param2_tpi32.cu -- param2 lookup for TPI=32 (9216..16384 tiers), split out of cgbn_stage1_kernels_param2.cu on 2026-09-25.
 *
 * Reason: these instantiations dominate the kernel compile time (the old combined TU
 * took 745 s for the suyama family and 711 s for param2, i.e. it WAS the critical path
 * of the parallel build).  The per-tier guards (ECM_TIERS_RESTRICTED / ECM_TIER_<bits>)
 * and the exported symbol name are unchanged, so the dispatcher in cgbn_stage1.cu and
 * the restricted-tier builds keep working as before.
 * See docs/ECM_CGBN_OPTIMIZATION.md 8.8. */

/* cgbn_stage1_kernels_param2.cu -- param2 (gmp-ecm "batch 2", 6-torsion) instantiations.
 *
 * Generated from cgbn_stage1_kernels_suyama.cu (same grid, same 7-word buffer layout,
 * same kernel body) with the only difference that the per-bit step is instantiated with
 * CONST_DIFF = true: the ladder difference is the constant 2 instead of an affine xdiff,
 * so the differential addition drops one mont_mul (5M+4S instead of 6M+4S).  That is
 * exactly the shape gmp-ecm's get_curve_from_param2() produces -- it ends with
 * mpres_set_ui (x0, 2, n), parametrizations.c:373 -- with a full-width a24.
 *
 * Measured: 79.15 M curve-bits/s against param0's 71.52 M (M511, 8192 curves, B1=1e5,
 * RTX 4060), +10.7%, matching the 10/9 operator ratio.  See
 * docs/ECM_CGBN_OPTIMIZATION.md section 5.6.
 *
 * Grid (identical to cgbn_stage1_kernels_tpi*.cu):
 *   TPI=4  : 128, 192, 256, 384, 512                      always compiled
 *   TPI=8  : 768, 1024, 1280, 1536, 1792, 2048            always / full
 *   TPI=16 : 2560 ... 8192, 512 interval                  full build only
 *   TPI=32 : 9216 ... 16384, 512 interval                 full build only
 */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

#if ECM_NO_PARAM2
/* param2 kernels disabled at configure time (-DECM_NO_PARAM2=1, see CMakeLists.txt).
   Every lookup returns null, so the dispatcher in cgbn_stage1.cu reports
   "--gpu-param 2 ... not compiled into this binary" instead of failing to link. */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t BITS, uint32_t *TPI_out) { (void)BITS; (void)TPI_out; return nullptr; }

#else
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
    if (BITS == cgbn_params_9216::BITS) {
        *TPI_out = cgbn_params_9216::TPI;
        return kernel_double_add_suyama<cgbn_params_9216, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
    if (BITS == cgbn_params_10240::BITS) {
        *TPI_out = cgbn_params_10240::TPI;
        return kernel_double_add_suyama<cgbn_params_10240, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
    if (BITS == cgbn_params_11264::BITS) {
        *TPI_out = cgbn_params_11264::TPI;
        return kernel_double_add_suyama<cgbn_params_11264, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
    if (BITS == cgbn_params_12288::BITS) {
        *TPI_out = cgbn_params_12288::TPI;
        return kernel_double_add_suyama<cgbn_params_12288, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
    if (BITS == cgbn_params_13312::BITS) {
        *TPI_out = cgbn_params_13312::TPI;
        return kernel_double_add_suyama<cgbn_params_13312, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
    if (BITS == cgbn_params_14336::BITS) {
        *TPI_out = cgbn_params_14336::TPI;
        return kernel_double_add_suyama<cgbn_params_14336, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
    if (BITS == cgbn_params_15360::BITS) {
        *TPI_out = cgbn_params_15360::TPI;
        return kernel_double_add_suyama<cgbn_params_15360, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
    if (BITS == cgbn_params_16384::BITS) {
        *TPI_out = cgbn_params_16384::TPI;
        return kernel_double_add_suyama<cgbn_params_16384, true>;
    }
#endif

#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}

#endif /* !ECM_NO_PARAM2 */
