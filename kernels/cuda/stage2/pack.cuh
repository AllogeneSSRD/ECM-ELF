#pragma once
#include "ecm/cuda/stage2_launch.h"


#include "../shared/pack_index.cuh"

namespace stage2_pack {
template<bool GATHER,bool REVERSE>
__global__ void pack_zero_kernel(const unsigned long long *src,const unsigned long long *offsets,
    unsigned nc,unsigned W,unsigned ndig,unsigned bpw,unsigned sw,unsigned reciprocal,
    unsigned N,unsigned long long first_slice,unsigned long long *dst)
{
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=N)return;
    const unsigned long long slice=first_slice+blockIdx.y;
    unsigned long long value=0;
    if((unsigned long long)j<(unsigned long long)nc*sw) {
        unsigned digit=0;const unsigned coef=quotient(j,sw,reciprocal,digit);
        if(digit<ndig) {
            const unsigned source_coef=REVERSE ? nc-1-coef : coef;
            const auto *c=src+(GATHER ? offsets[slice] : slice*(unsigned long long)nc*W)+
                (unsigned long long)source_coef*W;
            const unsigned bit=digit*bpw,w=bit>>6,shift=bit&63;
            value=c[w]>>shift;
            if(shift+bpw>64 && w+1<W)value|=c[w+1]<<(64-shift);
            value&=(1ull<<bpw)-1;
        }
    }
    dst[slice*(unsigned long long)N+j]=value;
}
}

__global__ void s4_pack_batch_kernel(const unsigned long long *src, int S, int bpw,
                                     unsigned long long slot_words, unsigned long long N,
                                     unsigned long long ds, unsigned long long ma, int W,
                                     unsigned long long *dst)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= ds * ma) return;
    const unsigned long long s = gid / ma, i = gid - s * ma;
    const unsigned long long *c = src + gid * (unsigned long long)W;  /* slice stride = ma*W */
    unsigned long long *d = dst + s * N + i * slot_words;
    const unsigned long long mask = (bpw >= 64) ? ~0ull : ((1ull << bpw) - 1ull);
    const int ndig = (S + bpw - 1) / bpw;
    for (int k = 0; k < ndig; ++k) {
        const int bit = k * bpw;
        const int w = bit >> 6, sh = bit & 63;
        unsigned long long v = c[w] >> sh;
        if (sh + bpw > 64 && (w + 1) < W) v |= c[w + 1] << (64 - sh);   /* bits >= S stay zero */
        d[k] = v & mask;
    }
}

__global__ void s4_pack_gather_kernel(const unsigned long long *src,
    const unsigned long long *offsets, size_t m, size_t nc, int W, int bits,
    int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(gid>=m*nc)return;
    const size_t slice=gid/nc,coef=gid%nc;
    const auto *c=src+offsets[slice]+coef*W;
    auto *d=dst+slice*N+coef*sw;
    const unsigned long long mask=(1ull<<bpw)-1;
    for(int k=0;k<(bits+bpw-1)/bpw;++k) {
        const int bit=k*bpw,w=bit>>6,shift=bit&63;
        unsigned long long v=c[w]>>shift;
        if(shift+bpw>64 && w+1<W)v|=c[w+1]<<(64-shift);
        d[k]=v&mask;
    }
}

__global__ void s4_pack_gather_reverse_kernel(const unsigned long long *src,
    const unsigned long long *offsets,size_t m,size_t nc,int W,int bits,
    int bpw,unsigned long long sw,unsigned long long N,unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(gid>=m*nc)return;
    const size_t slice=gid/nc,coef=gid%nc;
    const auto *c=src+offsets[slice]+(nc-1-coef)*W;
    auto *d=dst+slice*N+coef*sw;
    const unsigned long long mask=(1ull<<bpw)-1;
    for(int k=0;k<(bits+bpw-1)/bpw;++k) {
        const int bit=k*bpw,w=bit>>6,shift=bit&63;
        unsigned long long v=c[w]>>shift;
        if(shift+bpw>64 && w+1<W)v|=c[w+1]<<(64-shift);
        d[k]=v&mask;
    }
}

namespace stage2_pack {
template<bool GATHER,bool REVERSE>
void ecm_launch_pack_zero_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, unsigned nc, unsigned W, unsigned ndig, unsigned bpw, unsigned sw, unsigned reciprocal, unsigned N, unsigned long long first_slice, unsigned long long *dst) {
    pack_zero_kernel<GATHER,REVERSE><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,offsets,nc,W,ndig,bpw,sw,reciprocal,N,first_slice,dst);
}
}

void ecm_launch_s4_pack_batch_kernel(EcmCudaLaunch cfg, const unsigned long long *src, int S, int bpw, unsigned long long slot_words, unsigned long long N, unsigned long long ds, unsigned long long ma, int W, unsigned long long *dst) {
    s4_pack_batch_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,S,bpw,slot_words,N,ds,ma,W,dst);
}

void ecm_launch_s4_pack_gather_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, size_t m, size_t nc, int W, int bits, int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst) {
    s4_pack_gather_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,offsets,m,nc,W,bits,bpw,sw,N,dst);
}

void ecm_launch_s4_pack_gather_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, size_t m, size_t nc, int W, int bits, int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst) {
    s4_pack_gather_reverse_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,offsets,m,nc,W,bits,bpw,sw,N,dst);
}
