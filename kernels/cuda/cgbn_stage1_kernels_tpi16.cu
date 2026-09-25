/* cgbn_stage1_kernels_tpi16.cu — TPI=16 kernel instantiations (full build only).
   2560..8192 at **512-bit intervals**.

   A 256-bit grid (2560, 2816, 3072, 3328, ...) was tried earlier and reverted
   (2026-09-24): the finer grid gave no measurable throughput advantage while
   doubling the number of template instantiations and therefore the build time of
   every full CUDA build.  Sizes that fall between two 512 steps simply use the next
   step (e.g. a 3300-bit N uses the 3584 container, ~8% more work per multiply than
   an exact container would need) -- that is the trade this grid makes on purpose.

   The param0 (Suyama) variants live in cgbn_stage1_kernels_suyama.cu and follow the
   same 512 grid.
 */

#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t BITS, uint32_t *TPI_out) {
#ifndef IS_DEV_BUILD
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
    if (BITS == cgbn_params_2560::BITS) {
        *TPI_out = cgbn_params_2560::TPI;
        return kernel_double_add<cgbn_params_2560>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3072)
    if (BITS == cgbn_params_3072::BITS) {
        *TPI_out = cgbn_params_3072::TPI;
        return kernel_double_add<cgbn_params_3072>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3584)
    if (BITS == cgbn_params_3584::BITS) {
        *TPI_out = cgbn_params_3584::TPI;
        return kernel_double_add<cgbn_params_3584>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4096)
    if (BITS == cgbn_params_4096::BITS) {
        *TPI_out = cgbn_params_4096::TPI;
        return kernel_double_add<cgbn_params_4096>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
    if (BITS == cgbn_params_4608::BITS) {
        *TPI_out = cgbn_params_4608::TPI;
        return kernel_double_add<cgbn_params_4608>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5120)
    if (BITS == cgbn_params_5120::BITS) {
        *TPI_out = cgbn_params_5120::TPI;
        return kernel_double_add<cgbn_params_5120>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5632)
    if (BITS == cgbn_params_5632::BITS) {
        *TPI_out = cgbn_params_5632::TPI;
        return kernel_double_add<cgbn_params_5632>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6144)
    if (BITS == cgbn_params_6144::BITS) {
        *TPI_out = cgbn_params_6144::TPI;
        return kernel_double_add<cgbn_params_6144>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6656)
    if (BITS == cgbn_params_6656::BITS) {
        *TPI_out = cgbn_params_6656::TPI;
        return kernel_double_add<cgbn_params_6656>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7168)
    if (BITS == cgbn_params_7168::BITS) {
        *TPI_out = cgbn_params_7168::TPI;
        return kernel_double_add<cgbn_params_7168>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7680)
    if (BITS == cgbn_params_7680::BITS) {
        *TPI_out = cgbn_params_7680::TPI;
        return kernel_double_add<cgbn_params_7680>;
    }
#endif

#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_8192)
    if (BITS == cgbn_params_8192::BITS) {
        *TPI_out = cgbn_params_8192::TPI;
        return kernel_double_add<cgbn_params_8192>;
    }
#endif

#endif
    return nullptr;
}
