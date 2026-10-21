// Default PRAC geometry follows the shared CGBN inventory.
#include "../../../arithmetic/cgbn/types.cuh"
#include "../arithmetic.cuh"
#include "cgbn_stage1_prac_kernel.cuh"
cgbn_stage1_kernel_fn cgbn_stage1_domain_tpi32_13312_14336(uint32_t bits,uint32_t *tpi,int mode) {
#define ECM_CUDA_MONT_ONLY
#ifdef IS_DEV_BUILD
#define ECM_CUDA_MONT_DEV_ONLY
#endif
#define ECM_CUDA_TIER_MIN_BITS 13312
#define ECM_CUDA_TIER_MAX_BITS 14336
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if(bits==W){*tpi=T;return domain_kernel<cgbn_params_##A>(mode);}
#include "../../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_CUDA_TIER_MIN_BITS
#undef ECM_CUDA_TIER_MAX_BITS
#undef ECM_CUDA_MONT_DEV_ONLY
#undef ECM_CUDA_MONT_ONLY
    *tpi=0;return nullptr;
}
