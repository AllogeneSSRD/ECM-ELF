#pragma once
#include "../../../../src/ecm/cuda/arithmetic/cgbn/square.cuh"

namespace ecm_cgbn {
template<class Env,class HalfWide>
__device__ __forceinline__ void square_join_halves(typename Env::cgbn_t &out,const HalfWide &v) {
    constexpr unsigned T=Env::TPI,L=Env::LIMBS,J=L/2,H=T/2;
    const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
    constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
    #pragma unroll
    for(unsigned j=0;j<L;++j) {
        const unsigned source=2*(lane%H)+j/J;
        const auto lo=__shfl_sync(mask,v._low._limbs[j%J],source,T);
        const auto hi=__shfl_sync(mask,v._high._limbs[j%J],source,T);
        out._limbs[j]=lane<H?lo:hi;
    }
}
template<class Env> __host__ __device__ constexpr bool square_karatsuba_eligible() {
    return Env::LIMBS%2==0 && Env::BITS==Env::UNPADDED_BITS;
}
// Three half-size squares use the full group. |a0-a1| fits the half container,
// avoiding an enlarged sum operand: 2*a0*a1 = a0^2+a1^2-|a0-a1|^2.
// Depth is finite; no asymptotic speedup is claimed for a fixed-depth policy.
template<unsigned Depth,class Env>
__device__ __forceinline__ void square_wide_karatsuba(const Env &env,typename Env::cgbn_wide_t &r,
    const typename Env::cgbn_t &a) {
    if constexpr(!Depth || !square_karatsuba_eligible<Env>())cgbn_sqr_wide(env,r,a);
    else {
        constexpr unsigned T=Env::TPI,L=Env::LIMBS,J=L/2,H=T/2;
        using HC=cgbn_context_t<T,SquareParameters<Env>>;using HE=cgbn_env_t<HC,Env::BITS/2>;
        HC hc(cgbn_no_checks);HE he(hc);typename HE::cgbn_t left,right,difference;
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
        #pragma unroll
        for(unsigned j=0;j<J;++j) {
            const auto al=__shfl_sync(mask,a._limbs[j],lane/2,T),ah=__shfl_sync(mask,a._limbs[j+J],lane/2,T);
            const auto bl=__shfl_sync(mask,a._limbs[j],lane/2+H,T),bh=__shfl_sync(mask,a._limbs[j+J],lane/2+H,T);
            left._limbs[j]=(lane&1)?ah:al;right._limbs[j]=(lane&1)?bh:bl;
        }
        if(cgbn_sub(he,difference,left,right))cgbn_sub(he,difference,right,left);
        typename HE::cgbn_wide_t part;typename Env::cgbn_t middle,difference_square;
        square_wide_karatsuba<Depth-1>(he,part,difference);square_join_halves<Env>(difference_square,part);
        square_wide_karatsuba<Depth-1>(he,part,left);square_join_halves<Env>(r._low,part);
        square_wide_karatsuba<Depth-1>(he,part,right);square_join_halves<Env>(r._high,part);
        const int carry=cgbn_add(env,middle,r._low,r._high);
        const int borrow=cgbn_sub(env,middle,middle,difference_square);
        const unsigned top=unsigned(carry)-unsigned(borrow!=0);
        typename Env::cgbn_t aligned;
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const auto v=__shfl_sync(mask,middle._limbs[j],lane%H,T);
            aligned._limbs[j]=lane<H?0:v;
        }
        const int low_carry=cgbn_add(env,r._low,r._low,aligned);
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const auto v=__shfl_sync(mask,middle._limbs[j],lane%H+H,T);
            aligned._limbs[j]=lane<H?v:(lane==H && j==0?top:0);
        }
        cgbn_add(env,r._high,r._high,aligned);cgbn_add_ui32(env,r._high,r._high,unsigned(low_carry));
    }
}
template<unsigned Depth,class Env>
__device__ __forceinline__ void mont_square_karatsuba(const Env &env,typename Env::cgbn_t &r,
    const typename Env::cgbn_t &a,const typename Env::cgbn_t &n,uint32_t np0) {
    if constexpr(!Depth || !square_karatsuba_eligible<Env>())cgbn_mont_sqr(env,r,a,n,np0);
    else {
        typename Env::cgbn_wide_t wide;square_wide_karatsuba<Depth>(env,wide,a);
        cgbn_mont_reduce_wide(env,r,wide,n,np0);
    }
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}
}
