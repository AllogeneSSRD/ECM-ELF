#pragma once
// Fixed 256-thread blocks. Every partial has an exclusive writer; no block-level
// global counter contention. The final kernel retains the old additive/max
// counters, including nonzero verdicts from earlier deferred chunks.
__device__ __forceinline__
void carry_partial_store(unsigned int bad,unsigned int mx,unsigned int *partial)
{
    for(int off=16;off;off>>=1){bad+=__shfl_down_sync(0xffffffffu,bad,off);mx=max(mx,__shfl_down_sync(0xffffffffu,mx,off));}
    __shared__ unsigned int counts[8],heights[8];
    if((threadIdx.x&31)==0){counts[threadIdx.x/32]=bad;heights[threadIdx.x/32]=mx;}
    __syncthreads();
    if(threadIdx.x<32){
        bad=threadIdx.x<8 ? counts[threadIdx.x] : 0;mx=threadIdx.x<8 ? heights[threadIdx.x] : 0;
        for(int off=16;off;off>>=1){bad+=__shfl_down_sync(0xffffffffu,bad,off);mx=max(mx,__shfl_down_sync(0xffffffffu,mx,off));}
        if(threadIdx.x==0){const auto j=2ull*((unsigned long long)blockIdx.y*gridDim.x+blockIdx.x);partial[j]=bad;partial[j+1]=mx;}
    }
}

__global__ __launch_bounds__(256)
void carry_partial_kernel(const unsigned long long *c,unsigned long long n,int bpw,
                          unsigned int *partial,unsigned long long stride)
{
    const auto i=blockIdx.x*256ull+threadIdx.x;
    unsigned long long v=0;if(i<n)v=c[(size_t)blockIdx.y*stride+i];
    carry_partial_store(i<n && v>=(1ull<<bpw),v ? 64-__clzll((long long)v) : 0,partial);
}

// carry_cone_value is the original helper from ntt_poly_probe.cu. Inactive tail
// threads still participate in the block reduction; live is the exact prefix
// mask used by the original early-return cone.
template<int ROUNDS>
__global__ __launch_bounds__(256)
void carry_cone_check_kernel(const unsigned long long *c,unsigned long long *cout,
                            unsigned long long n,int bpw,unsigned long long stride,
                            unsigned int *partial)
{
    const auto i=blockIdx.x*256ull+threadIdx.x;
    const auto sl=(size_t)blockIdx.y*stride;c+=sl;cout+=sl;
    const unsigned int live=__ballot_sync(0xffffffffu,i<n),lane=threadIdx.x&31;
    unsigned long long value=0;
    if(i<n){
        const auto mask=(1ull<<bpw)-1;
        const auto x=carry_cone_value<ROUNDS>(c,i,bpw);
        const unsigned int stop=__ballot_sync(live,x!=mask),generate=__ballot_sync(live,x>mask);
        unsigned int incoming=0;
        if(lane==0){auto j=i;while(j){const auto prev=carry_cone_value<ROUNDS>(c,--j,bpw);if(prev!=mask){incoming=prev>mask;break;}}}
        incoming=__shfl_sync(live,incoming,0);
        const unsigned int preceding=stop&((1u<<lane)-1u);
        if(preceding)incoming=(generate>>(31-__clz(preceding)))&1u;
        value=(x+incoming)&mask;cout[i]=value;
    }
    carry_partial_store(i<n && value>=(1ull<<bpw),value ? 64-__clzll((long long)value) : 0,partial);
}

__global__ __launch_bounds__(256)
void carry_partial_finish_kernel(const unsigned int *partial,unsigned long long blocks,
                                 unsigned long long *out)
{
    partial+=2ull*blockIdx.x*blocks;out+=2ull*blockIdx.x;
    unsigned long long bad=0;unsigned int mx=0;
    for(unsigned long long j=threadIdx.x;j<blocks;j+=256){bad+=partial[2*j];mx=max(mx,partial[2*j+1]);}
    for(int off=16;off;off>>=1){bad+=__shfl_down_sync(0xffffffffu,bad,off);mx=max(mx,__shfl_down_sync(0xffffffffu,mx,off));}
    __shared__ unsigned long long counts[8];__shared__ unsigned int heights[8];
    if((threadIdx.x&31)==0){counts[threadIdx.x/32]=bad;heights[threadIdx.x/32]=mx;}
    __syncthreads();
    if(threadIdx.x<32){
        bad=threadIdx.x<8 ? counts[threadIdx.x] : 0;mx=threadIdx.x<8 ? heights[threadIdx.x] : 0;
        for(int off=16;off;off>>=1){bad+=__shfl_down_sync(0xffffffffu,bad,off);mx=max(mx,__shfl_down_sync(0xffffffffu,mx,off));}
        if(threadIdx.x==0){if(bad)atomicAdd(out,bad);atomicMax(out+1,(unsigned long long)mx);}
    }
}
