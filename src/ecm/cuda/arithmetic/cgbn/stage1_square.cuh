#pragma once
#include "types.cuh"

// Build-time experiment: no runtime branch or added kernel arguments. Keep
// canonicalisation at the existing Stage1 call site exactly once.
#ifndef ECM_CGBN_SQUARE_BACKEND
#define ECM_CGBN_SQUARE_BACKEND 0
#endif
#if ECM_CGBN_SQUARE_BACKEND == 1
#include "square_redc.cuh"
#elif ECM_CGBN_SQUARE_BACKEND != 0
#error ECM_CGBN_SQUARE_BACKEND must be 0 (CGBN) or 1 (carry-save square/REDC)
#endif

namespace ecm_cgbn {
template<class Env>
__device__ __forceinline__ void stage1_mont_square_raw(const Env &env,
    typename Env::cgbn_t &r,const typename Env::cgbn_t &a,
    const typename Env::cgbn_t &n,uint32_t np0) {
#if ECM_CGBN_SQUARE_BACKEND == 1
    mont_square_carrysave<2,Env,false,false>(env,r,a,n,np0);
#else
    cgbn_mont_sqr(env,r,a,n,np0);
#endif
}
}
