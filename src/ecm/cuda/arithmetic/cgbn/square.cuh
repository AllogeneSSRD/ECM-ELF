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
template<unsigned Depth,class Env,unsigned Layout=0>
__device__ __forceinline__ void square_wide_split(const Env &env,typename Env::cgbn_wide_t &r,
    const typename Env::cgbn_t &a);

template<class Env> struct SquareSplitTypes {
    using HalfContext=cgbn_context_t<Env::TPI/2,SquareParameters<Env>>;
    using HalfEnv=cgbn_env_t<HalfContext,Env::BITS/2>;
    using CrossContext=cgbn_context_t<Env::TPI,SquareParameters<Env>>;
    using CrossEnv=cgbn_env_t<CrossContext,Env::BITS/2>;
    using CrossWide=typename CrossEnv::cgbn_wide_t;
};

// Layout 0 preserves the reference gathers. Layout 1 exchanges the off-diagonal
// blocks when joining child squares. Layout 2 also transposes the cross inputs;
// layout 3 computes the cross product first to change temporary live ranges.
template<unsigned Layout,class Env>
__device__ __forceinline__ void square_cross_inputs(
    typename SquareSplitTypes<Env>::CrossEnv::cgbn_t &left,
    typename SquareSplitTypes<Env>::CrossEnv::cgbn_t &right,
    const typename Env::cgbn_t &a) {
    constexpr unsigned T=Env::TPI,L=Env::LIMBS,H=T/2,J=L/2;
    const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
    constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
    #pragma unroll
    for(unsigned j=0;j<J;++j) {
        if constexpr(Layout<2) {
            const auto al=__shfl_sync(mask,a._limbs[j],lane/2,T);
            const auto ah=__shfl_sync(mask,a._limbs[j+J],lane/2,T);
            const auto bl=__shfl_sync(mask,a._limbs[j],lane/2+H,T);
            const auto bh=__shfl_sync(mask,a._limbs[j+J],lane/2+H,T);
            left._limbs[j]=(lane&1)?ah:al;right._limbs[j]=(lane&1)?bh:bl;
        } else {
            const auto exchange=lane<H?a._limbs[j+J]:a._limbs[j];
            const auto incoming=__shfl_xor_sync(mask,exchange,H,T);
            const auto low=lane<H?a._limbs[j]:incoming;
            const auto high=lane<H?incoming:a._limbs[j+J];
            const unsigned source=lane/2+H*(lane&1);
            left._limbs[j]=__shfl_sync(mask,low,source,T);
            right._limbs[j]=__shfl_sync(mask,high,source,T);
        }
    }
}

template<unsigned Layout,class Env,class HalfWide>
__device__ __forceinline__ void square_join_diagonal(typename Env::cgbn_wide_t &out,const HalfWide &part) {
    constexpr unsigned T=Env::TPI,L=Env::LIMBS,H=T/2;
    const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
    constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
    #pragma unroll
    for(unsigned j=0;j<L;++j) {
        if constexpr(!Layout) {
            const auto ll=__shfl_sync(mask,part._low._limbs[j],lane%H,T);
            const auto lh=__shfl_sync(mask,part._high._limbs[j],lane%H,T);
            const auto hl=__shfl_sync(mask,part._low._limbs[j],lane%H+H,T);
            const auto hh=__shfl_sync(mask,part._high._limbs[j],lane%H+H,T);
            out._low._limbs[j]=lane<H?ll:lh;out._high._limbs[j]=lane<H?hl:hh;
        } else {
            // Transpose a 2x2 block: exchange lower-group high with upper-group
            // low. The two on-diagonal blocks are already in their final lanes.
            const auto exchange=lane<H?part._high._limbs[j]:part._low._limbs[j];
            const auto incoming=__shfl_xor_sync(mask,exchange,H,T);
            out._low._limbs[j]=lane<H?part._low._limbs[j]:incoming;
            out._high._limbs[j]=lane<H?incoming:part._high._limbs[j];
        }
    }
}

// Keep the two diagonal squares and the unshifted cross product separate.
// This common producer lets REDC consume carry-save components without first
// assembling a canonical 2*BITS-wide square. Only eligible environments use it.
template<unsigned Depth,class Env,unsigned Layout=0>
__device__ __forceinline__ void square_split_parts(const Env &env,typename Env::cgbn_wide_t &diagonal,
    typename SquareSplitTypes<Env>::CrossWide &cross,const typename Env::cgbn_t &a) {
    static_assert(Depth && square_split_eligible<Env>(),"split parts require an eligible environment");
    static_assert(Layout<=3,"unknown split-square layout");
    constexpr unsigned L=Env::LIMBS;
    using S=SquareSplitTypes<Env>;
    using HC=typename S::HalfContext;using HE=typename S::HalfEnv;
    using XC=typename S::CrossContext;using XE=typename S::CrossEnv;
    XC xc(cgbn_no_checks);XE xe(xc);
    if constexpr(Layout==3) {
        typename XE::cgbn_t left,right;
        square_cross_inputs<Layout,Env>(left,right,a);
        cgbn_mul_wide(xe,cross,left,right);
    }
    HC hc(cgbn_no_checks);HE he(hc);typename HE::cgbn_t ha;typename HE::cgbn_wide_t part;
    #pragma unroll
    for(unsigned j=0;j<L;++j)ha._limbs[j]=a._limbs[j];
    square_wide_split<Depth-1,HE,Layout>(he,part,ha);
    if constexpr(Layout==3)square_join_diagonal<Layout,Env>(diagonal,part);
    else {
        typename XE::cgbn_t left,right;
        // Keep reference ordering and capture inputs before an aliased write.
        square_cross_inputs<Layout,Env>(left,right,a);
        square_join_diagonal<Layout,Env>(diagonal,part);
        cgbn_mul_wide(xe,cross,left,right);
    }
}

template<unsigned Depth,class Env,unsigned Layout>
__device__ __forceinline__ void square_wide_split(const Env &env,typename Env::cgbn_wide_t &r,
    const typename Env::cgbn_t &a) {
    // The wide API promises low-half aliasing. Probe-only no-alias producers
    // must not become callable here through a specialised parts producer.
    static_assert(Layout<=3,"wide square requires an alias-safe layout");
    if constexpr(!Depth || !square_split_eligible<Env>())cgbn_sqr_wide(env,r,a);
    else {
        constexpr unsigned T=Env::TPI,L=Env::LIMBS,H=T/2,J=L/2;
        using S=SquareSplitTypes<Env>;
        using XC=typename S::CrossContext;using XE=typename S::CrossEnv;
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        constexpr unsigned bits=0xffffffffu>>(32-T);const unsigned mask=bits<<base;
        XC xc(cgbn_no_checks);XE xe(xc);typename S::CrossWide cross;
        square_split_parts<Depth,Env,Layout>(env,r,cross,a);
        // Doubling the cross product needs one bit beyond its two halves.
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
template<unsigned Depth,class Env,unsigned Layout=0>
__device__ __forceinline__ void mont_square_split(const Env &env,typename Env::cgbn_t &r,
    const typename Env::cgbn_t &a,const typename Env::cgbn_t &n,uint32_t np0) {
    if constexpr(!Depth || !square_split_eligible<Env>())cgbn_mont_sqr(env,r,a,n,np0);
    else {
        typename Env::cgbn_wide_t wide;square_wide_split<Depth,Env,Layout>(env,wide,a);
        cgbn_mont_reduce_wide(env,r,wide,n,np0);
    }
    // Match the project's unique canonical-residue contract, not just a
    // congruent CGBN representative. r may alias a; n must remain separate.
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}

}
