#pragma once
#include "../../../backend/cuda/arithmetic/cgbn/types.cuh"
#include "../arithmetic.cuh"
#define ECM_CUDA_MONT_ONLY
#ifdef IS_DEV_BUILD
#define ECM_CUDA_MONT_DEV_ONLY
#endif
cgbn_stage1_kernel_fn ECM_MONT_FACTORY(uint32_t BITS,uint32_t *TPI_out) {
#if ECM_MONT_PARAM != 2 || !ECM_NO_PARAM2
#if ECM_MONT_PARAM == 3
#define ECM_MONT_KERNEL(A) kernel_double_add<cgbn_params_##A>
#elif ECM_MONT_PARAM == 0
#define ECM_MONT_KERNEL(A) kernel_double_add_suyama<cgbn_params_##A>
#else
#define ECM_MONT_KERNEL(A) kernel_double_add_suyama<cgbn_params_##A,true>
#endif
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) static_assert(cgbn_params_##A::TPB==MT,"Montgomery block geometry"); if(BITS==W){*TPI_out=T;return ECM_MONT_KERNEL(A);}
#include "../../../backend/cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_MONT_KERNEL
#endif
    return nullptr;
}
#undef ECM_CUDA_MONT_DEV_ONLY
#undef ECM_CUDA_MONT_ONLY
