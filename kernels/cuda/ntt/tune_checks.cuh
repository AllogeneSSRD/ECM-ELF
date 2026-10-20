#pragma once
#include "ecm/cuda/ntt_launch.h"
namespace stage2_tune {
__global__ void seed(unsigned long long *a,unsigned long long *b,unsigned long long n,unsigned long long slices) {
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n*slices) {
        const auto position=i%n,slice=i/n;
        a[i]=position==0 ? (slice?slice-1:GL_P-1) : position==1 ? 0x8000000000000000ull : position==2 ? GL_P-2 : 0;
        b[i]=position==0 ? 3 : position==1 ? 5 : position==2 ? 7 : 0;
    }
}
__global__ void verdict(const unsigned long long *a,unsigned long long n,
                        unsigned long long slices,const unsigned long long *want,unsigned long long *bad) {
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n*slices && a[i]!=(i%n<5 ? want[(i/n)*5+i%n] : 0))atomicAdd(bad,1ull);
}
}
void ecm_ntt_tune_seed(EcmCudaLaunch launch,unsigned long long *a,unsigned long long *b,unsigned long long n,unsigned long long slices) {
    stage2_tune::seed<<<launch.grid,launch.block,launch.shared,launch.stream>>>(a,b,n,slices);
}
void ecm_ntt_tune_verdict(EcmCudaLaunch launch,const unsigned long long *a,unsigned long long n,unsigned long long slices,const unsigned long long *want,unsigned long long *bad) {
    stage2_tune::verdict<<<launch.grid,launch.block,launch.shared,launch.stream>>>(a,n,slices,want,bad);
}
