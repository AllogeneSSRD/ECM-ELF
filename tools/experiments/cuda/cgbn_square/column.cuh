#pragma once
#include "../../../../src/ecm/cuda/arithmetic/cgbn/square.cuh"

namespace ecm_cgbn {
// One private shared slice per CGBN group. Skewing every 32 words avoids
// broadcasting distinct input words into the same bank at power-of-two strides.
template<class Env> __host__ __device__ constexpr unsigned square_scratch_words() {
    constexpr unsigned n=Env::TPI*Env::LIMBS;
    return Env::LIMBS<2?0:n+(n+31)/32;
}
__device__ __forceinline__ unsigned square_shared_index(unsigned i){return i+(i>>5);}

template<unsigned L>
__device__ __forceinline__ unsigned square_add_tail(uint32_t (&x)[L],uint64_t tail) {
    static_assert(L>=2,"A column carry can span two words");
    uint64_t v=uint64_t(x[0])+uint32_t(tail);x[0]=uint32_t(v);
    v=uint64_t(x[1])+uint32_t(tail>>32)+(v>>32);x[1]=uint32_t(v);
    unsigned carry=unsigned(v>>32);
    #pragma unroll
    for(unsigned j=2;j<L;++j){v=uint64_t(x[j])+carry;x[j]=uint32_t(v);carry=unsigned(v>>32);}
    return carry;
}
template<unsigned L>
__device__ __forceinline__ void square_add_bit(uint32_t (&x)[L],unsigned carry) {
    #pragma unroll
    for(unsigned j=0;j<L;++j){const auto old=x[j];x[j]=old+carry;carry=x[j]<old;}
}
__device__ __forceinline__ uint32_t square_finish_column(uint64_t sum,unsigned high,
    uint64_t diagonal,uint64_t &carry) {
    high=(high<<1)|unsigned(sum>>63);sum<<=1;
    auto old=sum;sum+=diagonal;high+=sum<old;
    old=sum;sum+=carry;high+=sum<old;
    carry=(uint64_t(high)<<32)|(sum>>32);return uint32_t(sum);
}

// Exact 2*BITS product in CGBN's public wide layout. Each lane owns a low
// column k and a high column k+n: together they have n/2 (+/-1) distinct
// products. This avoids leaving half a warp idle on triangular diagonals.
// Scratch is group-private; only full CGBN groups participate in each sync.
template<class Env>
__device__ __forceinline__ void square_wide(const Env &env,typename Env::cgbn_wide_t &r,
    const typename Env::cgbn_t &a,uint32_t *scratch) {
    if constexpr(Env::LIMBS<2) {
        cgbn_sqr_wide(env,r,a); // Tiny containers cannot hold a two-word tail.
    } else {
        constexpr unsigned T=Env::TPI,L=Env::LIMBS,N=T*L,A=Env::BITS/32;
        static_assert(N<=1024,"96-bit column accumulator bound");
        const unsigned lane=threadIdx.x&(T-1),base=(threadIdx.x&31)&~(T-1);
        constexpr unsigned group_bits=0xffffffffu>>(32-T);
        const unsigned mask=group_bits<<base;
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const unsigned k=lane*L+j;
            scratch[square_shared_index(k)]=k<A?a._limbs[j]:0;
        }
        __syncwarp(mask);
        uint32_t low[L],high[L];uint64_t carry_low=0,carry_high=0;
        #pragma unroll
        for(unsigned j=0;j<L;++j) {
            const unsigned k=lane*L+j,lower_count=(k+1)/2;
            const unsigned terms=lower_count+(N-k-1)/2;
            uint64_t sum=0,lo=0;unsigned top=0,lo_top=0;
            #pragma nounroll
            for(unsigned q=0;q<terms;++q) {
                const bool lower=q<lower_count;
                const unsigned i=lower?q:k+1+q-lower_count;
                const unsigned z=(lower?k:N+k)-i;
                const uint64_t product=uint64_t(scratch[square_shared_index(i)])*
                    scratch[square_shared_index(z)];
                // A single accumulator crosses the low/high boundary once.
                // Snapshot its prefix, then subtract it after the loop; avoid
                // selecting and routing two 96-bit accumulators per product.
                const uint64_t old=sum;sum+=product;top+=sum<old;
                if(q+1==lower_count){lo=sum;lo_top=top;}
            }
            const uint64_t hi=sum-lo;
            const unsigned hi_top=top-lo_top-(sum<lo);
            uint64_t diagonal_low=0,diagonal_high=0;
            if(!(k&1)) {
                const uint64_t x=scratch[square_shared_index(k/2)];
                const uint64_t y=scratch[square_shared_index((N+k)/2)];
                diagonal_low=x*x;diagonal_high=y*y;
            }
            low[j]=square_finish_column(lo,lo_top,diagonal_low,carry_low);
            high[j]=square_finish_column(hi,hi_top,diagonal_high,carry_high);
        }
        // Local columns assumed zero incoming carry. First add each preceding
        // segment's known tail, then resolve the remaining single-bit carries
        // over all 2*T segments, including the low/high boundary.
        uint64_t incoming_low=__shfl_up_sync(mask,carry_low,1,T);
        uint64_t incoming_high=__shfl_up_sync(mask,carry_high,1,T);
        const uint64_t last_low=__shfl_sync(mask,carry_low,T-1,T);
        if(!lane){incoming_low=0;incoming_high=last_low;}
        const unsigned gl=square_add_tail(low,incoming_low),gh=square_add_tail(high,incoming_high);
        unsigned pl=0xffffffffu,ph=0xffffffffu;
        #pragma unroll
        for(unsigned j=0;j<L;++j){pl&=low[j];ph&=high[j];}
        uint64_t generate=uint64_t((__ballot_sync(mask,gl!=0)>>base)&group_bits)|
            (uint64_t((__ballot_sync(mask,gh!=0)>>base)&group_bits)<<T);
        uint64_t propagate=uint64_t((__ballot_sync(mask,pl==0xffffffffu)>>base)&group_bits)|
            (uint64_t((__ballot_sync(mask,ph==0xffffffffu)>>base)&group_bits)<<T);
        #pragma unroll
        for(unsigned d=1;d<2*T;d*=2){generate|=propagate&(generate<<d);propagate&=propagate<<d;}
        const uint64_t incoming_bits=generate<<1;
        square_add_bit(low,unsigned((incoming_bits>>lane)&1));
        square_add_bit(high,unsigned((incoming_bits>>(T+lane))&1));
        __syncwarp(mask); // All reads of a are complete before scratch reuse.
        #pragma unroll
        for(unsigned j=0;j<L;++j)r._low._limbs[j]=lane*L+j<A?low[j]:0;
        if constexpr(A==N) {
            #pragma unroll
            for(unsigned j=0;j<L;++j)r._high._limbs[j]=high[j];
        } else {
            // Public wide halves split at BITS, not UNPADDED_BITS. Reuse the
            // input slice for the two half gathers; no extra shared owner.
            #pragma unroll
            for(unsigned j=0;j<L;++j)scratch[square_shared_index(lane*L+j)]=low[j];
            __syncwarp(mask);
            #pragma unroll
            for(unsigned j=0;j<L;++j) {
                const unsigned k=lane*L+j,z=A+k;
                r._high._limbs[j]=k<A && z<N?scratch[square_shared_index(z)]:0;
            }
            __syncwarp(mask);
            #pragma unroll
            for(unsigned j=0;j<L;++j)scratch[square_shared_index(lane*L+j)]=high[j];
            __syncwarp(mask);
            #pragma unroll
            for(unsigned j=0;j<L;++j) {
                const unsigned k=lane*L+j,z=A+k;
                if(k<A && z>=N)r._high._limbs[j]=scratch[square_shared_index(z-N)];
            }
        }
        __syncwarp(mask);
    }
}

// Canonical Montgomery output with the original CGBN radix, including padding.
// Finish the wide product before writing r, so r==a is supported.
template<class Env>
__device__ __forceinline__ void mont_square(const Env &env,typename Env::cgbn_t &r,
    const typename Env::cgbn_t &a,const typename Env::cgbn_t &n,uint32_t np0,uint32_t *scratch) {
    typename Env::cgbn_wide_t wide;square_wide(env,wide,a,scratch);
    cgbn_mont_reduce_wide(env,r,wide,n,np0);
    if(cgbn_compare(env,r,n)>=0)cgbn_sub(env,r,r,n);
}

}
