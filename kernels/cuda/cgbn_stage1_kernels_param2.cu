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

cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi4(uint32_t BITS, uint32_t *TPI_out) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_128)
    if (BITS == cgbn_params_128::BITS) {
        *TPI_out = cgbn_params_128::TPI;
        return kernel_double_add_suyama<cgbn_params_128, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_192)
    if (BITS == cgbn_params_192::BITS) {
        *TPI_out = cgbn_params_192::TPI;
        return kernel_double_add_suyama<cgbn_params_192, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_256)
    if (BITS == cgbn_params_256::BITS) {
        *TPI_out = cgbn_params_256::TPI;
        return kernel_double_add_suyama<cgbn_params_256, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_384)
    if (BITS == cgbn_params_384::BITS) {
        *TPI_out = cgbn_params_384::TPI;
        return kernel_double_add_suyama<cgbn_params_384, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_small)
    if (BITS == cgbn_params_small::BITS) {
        *TPI_out = cgbn_params_small::TPI;
        return kernel_double_add_suyama<cgbn_params_small, true>;
    }
#endif

    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi8(uint32_t BITS, uint32_t *TPI_out) {
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_768)
    if (BITS == cgbn_params_768::BITS) {
        *TPI_out = cgbn_params_768::TPI;
        return kernel_double_add_suyama<cgbn_params_768, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_medium)
    if (BITS == cgbn_params_medium::BITS) {
        *TPI_out = cgbn_params_medium::TPI;
        return kernel_double_add_suyama<cgbn_params_medium, true>;
    }
#endif

#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1280)
    if (BITS == cgbn_params_1280::BITS) {
        *TPI_out = cgbn_params_1280::TPI;
        return kernel_double_add_suyama<cgbn_params_1280, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1536)
    if (BITS == cgbn_params_1536::BITS) {
        *TPI_out = cgbn_params_1536::TPI;
        return kernel_double_add_suyama<cgbn_params_1536, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1792)
    if (BITS == cgbn_params_1792::BITS) {
        *TPI_out = cgbn_params_1792::TPI;
        return kernel_double_add_suyama<cgbn_params_1792, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2048)
    if (BITS == cgbn_params_2048::BITS) {
        *TPI_out = cgbn_params_2048::TPI;
        return kernel_double_add_suyama<cgbn_params_2048, true>;
    }
#endif

#endif
    return nullptr;
}

cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi16(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
    if (BITS == cgbn_params_2560::BITS) {
        *TPI_out = cgbn_params_2560::TPI;
        return kernel_double_add_suyama<cgbn_params_2560, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3072)
    if (BITS == cgbn_params_3072::BITS) {
        *TPI_out = cgbn_params_3072::TPI;
        return kernel_double_add_suyama<cgbn_params_3072, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3584)
    if (BITS == cgbn_params_3584::BITS) {
        *TPI_out = cgbn_params_3584::TPI;
        return kernel_double_add_suyama<cgbn_params_3584, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4096)
    if (BITS == cgbn_params_4096::BITS) {
        *TPI_out = cgbn_params_4096::TPI;
        return kernel_double_add_suyama<cgbn_params_4096, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (BITS == cgbn_params_4608::BITS) {
        *TPI_out = cgbn_params_4608::TPI;
        return kernel_double_add_suyama<cgbn_params_4608, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5120)
    if (BITS == cgbn_params_5120::BITS) {
        *TPI_out = cgbn_params_5120::TPI;
        return kernel_double_add_suyama<cgbn_params_5120, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5632)
    if (BITS == cgbn_params_5632::BITS) {
        *TPI_out = cgbn_params_5632::TPI;
        return kernel_double_add_suyama<cgbn_params_5632, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6144)
    if (BITS == cgbn_params_6144::BITS) {
        *TPI_out = cgbn_params_6144::TPI;
        return kernel_double_add_suyama<cgbn_params_6144, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6656)
    if (BITS == cgbn_params_6656::BITS) {
        *TPI_out = cgbn_params_6656::TPI;
        return kernel_double_add_suyama<cgbn_params_6656, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7168)
    if (BITS == cgbn_params_7168::BITS) {
        *TPI_out = cgbn_params_7168::TPI;
        return kernel_double_add_suyama<cgbn_params_7168, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7680)
    if (BITS == cgbn_params_7680::BITS) {
        *TPI_out = cgbn_params_7680::TPI;
        return kernel_double_add_suyama<cgbn_params_7680, true>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_8192)
    if (BITS == cgbn_params_8192::BITS) {
        *TPI_out = cgbn_params_8192::TPI;
        return kernel_double_add_suyama<cgbn_params_8192, true>;
    }
#endif

#else
    (void)BITS;
    (void)TPI_out;
#endif
    return nullptr;
}

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
