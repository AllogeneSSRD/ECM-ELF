#pragma once
#include "../../../../src/ecm/cuda/arithmetic/cgbn/square_redc.cuh"

namespace ecm_cgbn {
// Isolated no-alias producer: carry-save REDC owns a local diagonal buffer.
// Finish the child-square join before making the cross-input temporaries live.
// Do not call this producer with an input aliasing diagonal._low.
template<unsigned Depth,class Env>
__device__ __forceinline__ void square_parts_join_first(const Env &env,
    typename Env::cgbn_wide_t &diagonal,typename SquareSplitTypes<Env>::CrossWide &cross,
    const typename Env::cgbn_t &a) {
    using S=SquareSplitTypes<Env>;
    using HC=typename S::HalfContext;using HE=typename S::HalfEnv;
    using XC=typename S::CrossContext;using XE=typename S::CrossEnv;
    HC hc(cgbn_no_checks);HE he(hc);
    typename HE::cgbn_t ha;typename HE::cgbn_wide_t part;
    #pragma unroll
    for(unsigned j=0;j<Env::LIMBS;++j)ha._limbs[j]=a._limbs[j];
    square_wide_split<Depth-1,HE,2>(he,part,ha);
    square_join_diagonal<2,Env>(diagonal,part);
    XC xc(cgbn_no_checks);XE xe(xc);typename XE::cgbn_t left,right;
    square_cross_inputs<2,Env>(left,right,a);
    cgbn_mul_wide(xe,cross,left,right);
}

// A probe-local specialization keeps this unqualified policy out of production
// headers. Mode 23 uses it only through mont_square_carrysave's separate local
// buffer. The host parser and other backends use that routine's CGBN fallback.
#if defined(__CUDA_ARCH__) && defined(XMP_WMAD) && ECM_SQUARE_TPI>=8
using SquareJoinFirstProbeEnv=ECM_SQUARE_TEST_ENV;
template<>
__device__ __forceinline__ void square_split_parts<2,SquareJoinFirstProbeEnv,4>(
    const SquareJoinFirstProbeEnv &env,typename SquareJoinFirstProbeEnv::cgbn_wide_t &diagonal,
    typename SquareSplitTypes<SquareJoinFirstProbeEnv>::CrossWide &cross,
    const typename SquareJoinFirstProbeEnv::cgbn_t &a) {
    square_parts_join_first<2>(env,diagonal,cross,a);
}
#endif
}
