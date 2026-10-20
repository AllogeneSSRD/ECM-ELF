#pragma once
#include "ecm/cuda/stage2_launch.h"
template<int NW>
__global__ void s2g_plain_scale_kernel(unsigned long long *h, size_t count,
    const unsigned long long *factor_mont, const unsigned long long *n,
    unsigned long long ninv, int nw)
{
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)s2g_mont_mul<NW>(h+i*nw,h+i*nw,factor_mont,n,ninv,nw);
}

template<int NW>
void ecm_launch_s2g_plain_scale_kernel(EcmCudaLaunch cfg, unsigned long long *h, size_t count, const unsigned long long *factor_mont, const unsigned long long *n, unsigned long long ninv, int nw) {
    s2g_plain_scale_kernel<NW><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(h,count,factor_mont,n,ninv,nw);
}
