#pragma once
#include "types.cuh"
#include <type_traits>

namespace ecm_cgbn {
// Preserve the caller's code-generation policy in the smaller environments.
template<class Env> struct SquareParameters {
    using Parent=std::remove_cv_t<std::remove_reference_t<decltype(Env::_context)>>;
    static constexpr unsigned TPB=Env::TPB,MAX_ROTATION=Env::MAX_ROTATION,SHM_LIMIT=Env::SHM_LIMIT;
    static constexpr bool CONSTANT_TIME=Parent::CONSTANT_TIME;
};
// Public wide halves split at BITS; the Montgomery radix uses UNPADDED_BITS.
// Avoid changing either layout by falling back when CGBN inserts padding.
template<class Env> __host__ __device__ constexpr bool square_split_eligible() {
    return Env::TPI>=8 && Env::LIMBS%2==0 && Env::BITS==Env::UNPADDED_BITS;
}
// A finite-depth split square reuses CGBN's efficient register multiplication:
// a^2 = a0^2 + 2*a0*a1*2^(BITS/2) + a1^2*2^BITS.
// The two sub-squares run in disjoint half-sized groups; their cross product
// uses the full group. At depth d the product count is (1/2+1/2^(d+1))*N^2.
template<unsigned Depth,class Env>
__device__ __forceinline__ void square_wide_split(const Env &env,typename Env::cgbn_wide_t &r,
    const typename Env::cgbn_t &a) {
    if constexpr(!Depth || !square_split_eligible<Env>())cgbn_sqr_wide(env,r,a);
    else {
        constexpr unsigned T=Env::TPI,L=Env::LIMBS,H=T/2,J=L/2;
        using HC=cgbn_context_t<H,SquareParameters<Env>>;using HE=cgbn_env_t<HC,Env::BITS/2>;
        using XC=cgbn_context_t<T,SquareParameters<Env>>;using XE=cgbn_env_t<XC,Env::BITS/2>;
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
        // Each half-group already owns one contiguous half of a. Both child
        // squares run concurrently, with the parent's limbs-per-thread count.
        HC hc(cgbn_no_checks);HE he(hc);typename HE::cgbn_t ha;typename HE::cgbn_wide_t part;
        #pragma unroll
        for(unsigned j=0;j<L;++j)ha._limbs[j]=a._limbs[j];
        square_wide_split<Depth-1>(he,part,ha);
        XC xc(cgbn_no_checks);XE xe(xc);typename XE::cgbn_t left,right;
        // The full-group half-width product has half as many limbs per lane.
        // Capture both cross operands before writing r: a may alias r._low.
        #pragma unroll
        for(unsigned j=0;j<J;++j) {
            const auto al=__shfl_sync(mask,a._limbs[j],lane/2,T);
            const auto ah=__shfl_sync(mask,a._limbs[j+J],lane/2,T);
            const auto bl=__shfl_sync(mask,a._limbs[j],lane/2+H,T);
            const auto bh=__shfl_sync(mask,a._limbs[j+J],lane/2+H,T);
            left._limbs[j]=(lane&1)?ah:al;right._limbs[j]=(lane&1)?bh:bl;
        }
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const auto ll=__shfl_sync(mask,part._low._limbs[j],lane%H,T);
            const auto lh=__shfl_sync(mask,part._high._limbs[j],lane%H,T);
            const auto hl=__shfl_sync(mask,part._low._limbs[j],lane%H+H,T);
            const auto hh=__shfl_sync(mask,part._high._limbs[j],lane%H+H,T);
            r._low._limbs[j]=lane<H?ll:lh;r._high._limbs[j]=lane<H?hl:hh;
        }
        // Doubling the cross product needs one bit beyond its two halves.
        typename XE::cgbn_wide_t cross;cgbn_mul_wide(xe,cross,left,right);
        const unsigned lc=__shfl_sync(mask,cross._low._limbs[J-1],T-1,T)>>31;
        const unsigned hc_bit=__shfl_sync(mask,cross._high._limbs[J-1],T-1,T)>>31;
        cgbn_shift_left(xe,cross._low,cross._low,1);cgbn_shift_left(xe,cross._high,cross._high,1);
        if(!lane)cross._high._limbs[0]|=lc;
        typename Env::cgbn_t aligned;
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const auto v=__shfl_sync(mask,cross._low._limbs[j%J],2*(lane%H)+j/J,T);
            aligned._limbs[j]=lane<H?0:v;
        }
        // Insert the doubled cross product half a container above the low
        // square. Preserve both this carry and the extra doubling bit.
        const int carry=cgbn_add(env,r._low,r._low,aligned);
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const auto v=__shfl_sync(mask,cross._high._limbs[j%J],2*(lane%H)+j/J,T);
            aligned._limbs[j]=lane<H?v:(lane==H && j==0?hc_bit:0);
        }
        cgbn_add(env,r._high,r._high,aligned);cgbn_add_ui32(env,r._high,r._high,unsigned(carry));
    }
}
template<unsigned Depth,class Env>
__device__ __forceinline__ void mont_square_split(const Env &env,typename Env::cgbn_t &r,
    const typename Env::cgbn_t &a,const typename Env::cgbn_t &n,uint32_t np0) {
    if constexpr(!Depth || !square_split_eligible<Env>())cgbn_mont_sqr(env,r,a,n,np0);
    else {
        typename Env::cgbn_wide_t wide;square_wide_split<Depth>(env,wide,a);
        cgbn_mont_reduce_wide(env,r,wide,n,np0);
    }
    // Match the project's unique canonical-residue contract, not just a
    // congruent CGBN representative. r may alias a; n must remain separate.
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}

}
