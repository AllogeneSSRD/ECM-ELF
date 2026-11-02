#pragma once
#include "../stage2_scalar.cuh"

// One warp per coefficient. Shared remainders replace dynamically indexed
// per-thread local arrays; each quotient still has the original exact estimate.
__device__ __forceinline__ unsigned s4_warp_prefix_in(unsigned generates,unsigned propagates,unsigned carry,int lane) {
    const unsigned blockers=(~propagates)&((1u<<lane)-1u);
    return blockers?((generates>>(31-__clz(blockers)))&1u):carry;
}

template<bool HIGH>
__device__ __forceinline__ unsigned s4_warp_sub_product(unsigned long long *t,const unsigned long long *ns,
    int nw,int j,unsigned long long q,int lane) {
    unsigned carry=0;
    for(int base=0;base<=nw;base+=32) {
        const int i=base+lane;const bool live=i<=nw;
        const auto a=live?t[j+i]:0ull;
        unsigned long long b=0;
        if constexpr(HIGH) {if(i>0 && i<=nw)b=__umul64hi(q,ns[i-1]);}
        else {if(i<nw)b=q*ns[i];}
        const auto diff=a-b;
        const auto g=__ballot_sync(0xffffffffu,a<b),p=__ballot_sync(0xffffffffu,a==b);
        const auto incoming=s4_warp_prefix_in(g,p,carry,lane);
        if(live)t[j+i]=diff-incoming;
        const unsigned out=(a<b) || (a==b && incoming);
        carry=__shfl_sync(0xffffffffu,out,31);
    }
    return carry;
}

__device__ __forceinline__ unsigned s4_warp_add_divisor(unsigned long long *t,const unsigned long long *ns,
    int nw,int j,int lane) {
    unsigned carry=0;
    for(int base=0;base<=nw;base+=32) {
        const int i=base+lane;const bool live=i<=nw;
        // Padding must propagate, rather than kill, the final carry.
        const auto a=live?t[j+i]:~0ull,b=i<nw?ns[i]:0ull,sum=a+b;
        const auto g=__ballot_sync(0xffffffffu,sum<a),p=__ballot_sync(0xffffffffu,sum==~0ull);
        const auto incoming=s4_warp_prefix_in(g,p,carry,lane);
        if(live)t[j+i]=sum+incoming;
        const unsigned out=(sum<a) || (sum==~0ull && incoming);
        carry=__shfl_sync(0xffffffffu,out,31);
    }
    return carry;
}

// Match the scalar OR-based digit packer, including oversized digit bits at
// the previous word boundary. Canonical production digits are the common case.
__device__ __forceinline__ unsigned long long s4_warp_packed_word(const unsigned long long *d,
    unsigned long long slots,int bpw,int limb) {
    if(limb<0 || (unsigned long long)limb>=(slots*bpw+63)/64)return 0;
    const auto start=(unsigned long long)limb*64,first=start/bpw;
    const int offset=int(start%bpw);
    unsigned long long value=offset?d[first]>>offset:0;
    if(!offset && limb && bpw<64)value=d[first-1]>>bpw;
    for(auto k=first+(offset!=0);k<slots && k*(unsigned)bpw<start+64;++k)
        value|=d[k]<<(k*(unsigned)bpw-start);
    return value;
}

template<int NW>
__global__ void s4_warp_reduce_kernel(const unsigned long long *digits,unsigned long long n,
    int bpw,unsigned long long slot_words,unsigned long long out_slots,unsigned long long nbatch,
    int nw,int div_shift,unsigned long long div_recip,unsigned long long first,unsigned long long count,
    unsigned long long w,unsigned long long *out,unsigned long long *bad,unsigned long long slot_bits) {
    extern __shared__ unsigned long long scratch[];
    // The original kernel validates ALL source slots, including coefficients
    // omitted from the output window. Compact output mapping cannot drop this.
    const auto top_bits=slot_bits-(slot_words-1)*(unsigned long long)bpw;
    for(auto raw=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;raw<out_slots*nbatch;
        raw+=gridDim.x*(unsigned long long)blockDim.x) {
        const auto batch=raw/out_slots,k=raw%out_slots;
        if((k+1)*slot_words>n || (top_bits<64 && (digits[batch*n+(k+1)*slot_words-1]>>top_bits)!=0))
            if(bad)atomicAdd(bad,1ull);
    }
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    auto *ns=scratch,*t=ns+nw+warp*(2*nw+4);
    for(int i=threadIdx.x;i<nw;i+=blockDim.x)ns[i]=g_s2g_device_constants.divisor[i];
    for(int i=lane;i<2*nw+4;i+=32)t[i]=0;
    __syncthreads(); // All groups, including the final inactive ones, participate.
    const auto gid=(blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x)/32;
    if(gid>=count*nbatch)return;
    const auto batch=gid/count,k=first+gid%count;
    if(k>=out_slots || (k+1)*slot_words>n) {if(!lane && bad)atomicAdd(bad,1ull);return;}
    const auto *d=digits+batch*n+k*slot_words;
    const int limbs=int((slot_words*(unsigned long long)bpw+63)/64);
    for(int i=lane;i<limbs+(div_shift!=0);i+=32) {
        const auto a=s4_warp_packed_word(d,slot_words,bpw,i);
        t[i]=div_shift?(a<<div_shift)|(s4_warp_packed_word(d,slot_words,bpw,i-1)>>(64-div_shift)):a;
    }
    __syncwarp();
    int live=limbs+(div_shift!=0);
    if(!lane)while(live && !t[live-1])--live;
    live=__shfl_sync(0xffffffffu,live,0);
    for(int j=live-nw;j>=0;--j) {
        unsigned error=0;unsigned long long q=0;
        if(!lane) {
            const auto u1=t[j+nw],u0=t[j+nw-1],top=ns[nw-1];
            error=u1>top;
            q=u1==top?~0ull:s2g_udiv_2by1(u1,u0,top,div_recip);
        }
        q=__shfl_sync(0xffffffffu,q,0);error=__shfl_sync(0xffffffffu,error,0);
        if(error) {if(!lane && bad)atomicAdd(bad,1ull);for(auto i=(unsigned long long)lane;i<w;i+=32)out[gid*w+i]=0;return;}
        // q*N = low products + the high products shifted by one limb. Each
        // pass uses an exact warp borrow prefix. Their sum fits nw+1 limbs,
        // so at most one whole-window underflow can occur across both passes.
        unsigned under=s4_warp_sub_product<false>(t,ns,nw,j,q,lane);__syncwarp();
        under|=s4_warp_sub_product<true>(t,ns,nw,j,q,lane);__syncwarp();
        for(int correction=0;under && correction<2;++correction) {
            under=!s4_warp_add_divisor(t,ns,nw,j,lane);__syncwarp();
        }
        if(under) {if(!lane && bad)atomicAdd(bad,1ull);for(auto i=(unsigned long long)lane;i<w;i+=32)out[gid*w+i]=0;return;}
    }
    for(unsigned long long i=lane;i<w;i+=32)
        out[gid*w+i]=i<(unsigned)nw?(div_shift?(t[i]>>div_shift)|(i+1<(unsigned)nw?t[i+1]<<(64-div_shift):0):t[i]):0;
}
