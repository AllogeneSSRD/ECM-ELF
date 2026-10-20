#pragma once
#include "ecm/cuda/stage2_launch.h"

// The production scalar chain body is preserved verbatim.
template <int NW, bool XADD6=false>
__global__ void s2g_chain_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
                                 const unsigned long long *ddx, const unsigned long long *ddz,
                                 const unsigned long long *dsx, const unsigned long long *dsz,
                                 const unsigned long long *esx, const unsigned long long *esz,
                                 unsigned long long npts, unsigned long long per_block,
                                 unsigned long long blocks,
                                 unsigned long long *out_x, unsigned long long *out_z,
                                 size_t seed_stride)
{
    const unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= blocks) return;
    const unsigned long long start = t * per_block;
    if (start >= npts) return;
    const unsigned long long end = ((start + per_block) < npts) ? (start + per_block) : npts;
    unsigned long long xa[NW], za[NW], xb[NW], zb[NW], xc[NW], zc[NW];
    for (int j = 0; j < nw; ++j) {
        xa[j] = dsx[(size_t)t * seed_stride * nw + j];
        za[j] = dsz[(size_t)t * seed_stride * nw + j];
        xb[j] = esx[(size_t)t * seed_stride * nw + j];
        zb[j] = esz[(size_t)t * seed_stride * nw + j];
    }
    for (int j = 0; j < nw; ++j) {
        out_x[(size_t)start * nw + j] = xa[j];
        out_z[(size_t)start * nw + j] = za[j];
    }
    if (end > start + 1)
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)(start + 1) * nw + j] = xb[j];
            out_z[(size_t)(start + 1) * nw + j] = zb[j];
        }
    for (unsigned long long k = start + 2; k < end; ++k) {
        /* x_k = x_{k-1} + x_1, with the difference x_{k-2}: p = x_{k-1}, q = x_D, diff = x_{k-2} */
        s2g_xadd<NW,XADD6>(xc, zc, xb, zb, ddx, ddz, xa, za, dn, ninv, nw);
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)k * nw + j] = xc[j];
            out_z[(size_t)k * nw + j] = zc[j];
            xa[j] = xb[j];
            za[j] = zb[j];
            xb[j] = xc[j];
            zb[j] = zc[j];
        }
    }
}


// Probe-only buffer rotation; no production instantiation.
template <int NW, bool XADD6=false, bool ROTATE=false>
__global__ void s2g_chain_rotate_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
    const unsigned long long *ddx, const unsigned long long *ddz,
    const unsigned long long *dsx, const unsigned long long *dsz,
    const unsigned long long *esx, const unsigned long long *esz,
    unsigned long long npts, unsigned long long per_block, unsigned long long blocks,
    unsigned long long *out_x, unsigned long long *out_z, size_t seed_stride)
{
    const unsigned long long t=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(t>=blocks)return;
    const auto start=t*per_block, end=start+per_block<npts?start+per_block:npts;
    unsigned long long xa[NW],za[NW],xb[NW],zb[NW],xc[NW],zc[NW];
    for(int j=0;j<nw;++j){
        xa[j]=dsx[(size_t)t*seed_stride*nw+j];za[j]=dsz[(size_t)t*seed_stride*nw+j];
        xb[j]=esx[(size_t)t*seed_stride*nw+j];zb[j]=esz[(size_t)t*seed_stride*nw+j];
        out_x[(size_t)start*nw+j]=xa[j];out_z[(size_t)start*nw+j]=za[j];
        if(end>start+1){out_x[(size_t)(start+1)*nw+j]=xb[j];out_z[(size_t)(start+1)*nw+j]=zb[j];}
    }
    auto *ax=xa,*az=za,*bx=xb,*bz=zb,*cx=xc,*cz=zc;
    for(auto k=start+2;k<end;++k){
        s2g_xadd<NW,XADD6>(cx,cz,bx,bz,ddx,ddz,ax,az,dn,ninv,nw);
        for(int j=0;j<nw;++j){
            out_x[(size_t)k*nw+j]=cx[j];out_z[(size_t)k*nw+j]=cz[j];
            if constexpr(!ROTATE){ax[j]=bx[j];az[j]=bz[j];bx[j]=cx[j];bz[j]=cz[j];}
        }
        if constexpr(ROTATE){auto *tx=ax,*tz=az;ax=bx;az=bz;bx=cx;bz=cz;cx=tx;cz=tz;}
    }
}

template<int NW,bool XADD6>
void ecm_launch_s2g_chain_kernel(EcmCudaLaunch cfg, const unsigned long long *dn, unsigned long long ninv, int nw, const unsigned long long *ddx, const unsigned long long *ddz, const unsigned long long *dsx, const unsigned long long *dsz, const unsigned long long *esx, const unsigned long long *esz, unsigned long long npts, unsigned long long per_block, unsigned long long blocks, unsigned long long *out_x, unsigned long long *out_z, size_t seed_stride) {
    s2g_chain_kernel<NW,XADD6><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(dn,ninv,nw,ddx,ddz,dsx,dsz,esx,esz,npts,per_block,blocks,out_x,out_z,seed_stride);
}
