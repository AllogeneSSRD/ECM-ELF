#pragma once
#include "ecm/cuda/stage2_launch.h"
__device__ __forceinline__ void s2g_mac(unsigned long long x, unsigned long long y,
                                        unsigned long long z, unsigned long long c,
                                        unsigned long long &lo, unsigned long long &hi)
{
    const unsigned long long p0 = x * y;
    const unsigned long long p1 = __umul64hi(x, y);
    unsigned long long s = p0 + z;
    unsigned long long k = (s < p0) ? 1ull : 0ull;
    s += c;
    k += (s < c) ? 1ull : 0ull;
    lo = s;
    hi = p1 + k;
}

__device__ __forceinline__ unsigned long long s2g_sub_n(unsigned long long *r,
                                                        const unsigned long long *n, int nw)
{
    unsigned long long borrow = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long d = r[i] - n[i];
        const unsigned long long b1 = (r[i] < n[i]) ? 1ull : 0ull;
        const unsigned long long d2 = d - borrow;
        const unsigned long long b2 = (d < borrow) ? 1ull : 0ull;
        r[i] = d2;
        borrow = b1 + b2;
    }
    return borrow;
}

__device__ __forceinline__ bool s2g_ge_n(const unsigned long long *a,
                                         const unsigned long long *n, int nw)
{
    for (int i = nw - 1; i >= 0; --i)
        if (a[i] != n[i]) return a[i] > n[i];
    return true;                                          /* equal counts as >= */
}

template <int NW>
__device__ __forceinline__ void s2g_addmod(unsigned long long *r, const unsigned long long *a,
                                           const unsigned long long *b,
                                           const unsigned long long *n, int nw)
{
    unsigned long long c = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long s1 = a[i] + b[i];
        const unsigned long long c1 = (s1 < a[i]) ? 1ull : 0ull;
        const unsigned long long s2 = s1 + c;
        const unsigned long long c2 = (s2 < c) ? 1ull : 0ull;
        r[i] = s2;
        c = c1 + c2;
    }
    if (c != 0 || s2g_ge_n(r, n, nw)) s2g_sub_n(r, n, nw);   /* a+b < 2N: one subtraction */
}

template <int NW>
__device__ __forceinline__ void s2g_submod(unsigned long long *r, const unsigned long long *a,
                                           const unsigned long long *b,
                                           const unsigned long long *n, int nw)
{
    unsigned long long borrow = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long d = a[i] - b[i];
        const unsigned long long b1 = (a[i] < b[i]) ? 1ull : 0ull;
        const unsigned long long d2 = d - borrow;
        const unsigned long long b2 = (d < borrow) ? 1ull : 0ull;
        r[i] = d2;
        borrow = b1 + b2;
    }
    if (borrow) {
        unsigned long long c = 0;
        for (int i = 0; i < nw; ++i) {
            const unsigned long long s1 = r[i] + n[i];
            const unsigned long long c1 = (s1 < r[i]) ? 1ull : 0ull;
            const unsigned long long s2 = s1 + c;
            const unsigned long long c2 = (s2 < c) ? 1ull : 0ull;
            r[i] = s2;
            c = c1 + c2;
        }
    }
}

__device__ __constant__ S2GDeviceConstants g_s2g_device_constants={};
#include "point_mersenne.cuh"

template <int NW>
__device__ __forceinline__ void s2g_mont_mul(unsigned long long *r,
                                             const unsigned long long *a,
                                             const unsigned long long *b,
                                             const unsigned long long *n,
                                             unsigned long long ninv, int nw)
{
    unsigned long long t[2 * NW + 2];
    for (int i = 0; i < 2 * nw + 2; ++i) t[i] = 0;
    for (int i = 0; i < nw; ++i) {
        unsigned long long c = 0;
        const unsigned long long bi = b[i];
        for (int j = 0; j < nw; ++j) {
            unsigned long long lo, hi;
            s2g_mac(a[j], bi, t[i + j], c, lo, hi);
            t[i + j] = lo;
            c = hi;
        }
        t[i + nw] = c;                 /* untouched before this iteration (SOS invariant) */
    }
    unsigned long long out[NW];
    const int mersenne_bits=g_s2g_device_constants.point_mersenne_bits;
    if(mersenne_bits){s2g_mersenne_mont_reduce<NW>(r,t,n,nw,mersenne_bits,out);return;}
    for (int i = 0; i < nw; ++i) {
        const unsigned long long m = t[i] * ninv;
        unsigned long long c = 0;
        for (int j = 0; j < nw; ++j) {
            unsigned long long lo, hi;
            s2g_mac(m, n[j], t[i + j], c, lo, hi);
            t[i + j] = lo;
            c = hi;
        }
        int k = i + nw;
        while (c != 0) {
            const unsigned long long s = t[k] + c;
            c = (s < c) ? 1ull : 0ull;
            t[k] = s;
            ++k;
        }
    }
    for (int i = 0; i < nw; ++i) out[i] = t[nw + i];
    if (t[2 * nw] != 0 || s2g_ge_n(out, n, nw)) s2g_sub_n(out, n, nw);
    for (int i = 0; i < nw; ++i) r[i] = out[i];
}

template <int NW>
__device__ __forceinline__ void s2g_xdbl(unsigned long long *rx, unsigned long long *rz,
                                         const unsigned long long *px, const unsigned long long *pz,
                                         const unsigned long long *a24, const unsigned long long *n,
                                         unsigned long long ninv, int nw)
{
    unsigned long long t1[NW], t2[NW], t3[NW], t4[NW];
    s2g_addmod<NW>(t1, px, pz, n, nw);
    s2g_submod<NW>(t2, px, pz, n, nw);
    s2g_mont_mul<NW>(t3, t1, t1, n, ninv, nw);            /* (X+Z)^2 */
    s2g_mont_mul<NW>(t4, t2, t2, n, ninv, nw);            /* (X-Z)^2 */
    s2g_mont_mul<NW>(rx, t3, t4, n, ninv, nw);            /* X2 = (X+Z)^2 (X-Z)^2 */
    s2g_submod<NW>(t1, t3, t4, n, nw);                    /* 4XZ */
    s2g_mont_mul<NW>(t2, a24, t1, n, ninv, nw);
    s2g_addmod<NW>(t2, t2, t4, n, nw);                    /* a24*4XZ + (X-Z)^2 */
    s2g_mont_mul<NW>(rz, t1, t2, n, ninv, nw);
}

template <int NW>
__device__ __forceinline__ void s2g_halfmod(unsigned long long *r,
                                            const unsigned long long *a,
                                            const unsigned long long *n, int nw)
{
    const bool odd=(a[0]&1ull)!=0;
    unsigned long long carry=0;
    for(int i=0;i<nw;++i) {
        const auto v=a[i], add=odd ? n[i] : 0ull;
        const auto s=v+add, c1=(s<v) ? 1ull : 0ull;
        const auto u=s+carry, c2=(u<s) ? 1ull : 0ull;
        r[i]=u; carry=c1+c2;
    }
    for(int i=nw-1;i>=0;--i) {
        const auto v=r[i];
        r[i]=(v>>1)|(carry<<63); carry=v&1ull;
    }
}

template <int NW, bool XADD6=false>
__device__ __forceinline__ void s2g_xadd(unsigned long long *rx, unsigned long long *rz,
                                         const unsigned long long *px, const unsigned long long *pz,
                                         const unsigned long long *qx, const unsigned long long *qz,
                                         const unsigned long long *dx, const unsigned long long *dz,
                                         const unsigned long long *n, unsigned long long ninv,
                                         int nw)
{
    unsigned long long a[NW], b[NW], tx[NW], tz[NW];
    if(XADD6) {
        // u=(Xp+Zp)*(Xq-Zq), v=(Xp-Zp)*(Xq+Zq), all in Montgomery images.
        s2g_addmod<NW>(a,px,pz,n,nw); s2g_submod<NW>(b,px,pz,n,nw);
        s2g_addmod<NW>(tx,qx,qz,n,nw); s2g_submod<NW>(tz,qx,qz,n,nw);
        s2g_mont_mul<NW>(a,a,tz,n,ninv,nw);
        s2g_mont_mul<NW>(b,b,tx,n,ninv,nw);
        s2g_addmod<NW>(tx,a,b,n,nw); s2g_submod<NW>(tz,a,b,n,nw);
        // Halves recover the old coordinate scale exactly; the Z sign is squared.
        s2g_halfmod<NW>(tx,tx,n,nw); s2g_halfmod<NW>(tz,tz,n,nw);
        s2g_mont_mul<NW>(tx,tx,tx,n,ninv,nw);
        s2g_mont_mul<NW>(tz,tz,tz,n,ninv,nw);
        s2g_mont_mul<NW>(a,dz,tx,n,ninv,nw);
        s2g_mont_mul<NW>(b,dx,tz,n,ninv,nw);
        // Delay both writes, including when the output aliases the difference.
        for(int i=0;i<nw;++i) {rx[i]=a[i];rz[i]=b[i];}
        return;
    }
    s2g_mont_mul<NW>(a, px, qx, n, ninv, nw);
    s2g_mont_mul<NW>(b, pz, qz, n, ninv, nw);
    s2g_submod<NW>(a, a, b, n, nw);                /* X_P X_Q - Z_P Z_Q */
    s2g_mont_mul<NW>(a, a, a, n, ninv, nw);
    s2g_mont_mul<NW>(tx, dz, a, n, ninv, nw);
    s2g_mont_mul<NW>(a, px, qz, n, ninv, nw);
    s2g_mont_mul<NW>(b, pz, qx, n, ninv, nw);
    s2g_submod<NW>(a, a, b, n, nw);                /* X_P Z_Q - Z_P X_Q */
    s2g_mont_mul<NW>(a, a, a, n, ninv, nw);
    s2g_mont_mul<NW>(tz, dx, a, n, ninv, nw);
    for (int i = 0; i < nw; ++i) { rx[i] = tx[i]; rz[i] = tz[i]; }
}

template <int NW, bool XADD6=false>
__device__ void s2g_ladder(unsigned long long k, const unsigned long long *px,
                           const unsigned long long *pz, const unsigned long long *a24,
                           const unsigned long long *n, unsigned long long ninv, int nw,
                           const unsigned long long *mone,          /* Montgomery image of 1 */
                           unsigned long long *rx, unsigned long long *rz)
{
    if (k == 0) {                                              /* identity (1 : 0) */
        for (int i = 0; i < nw; ++i) { rx[i] = mone[i]; rz[i] = 0; }
        return;                                                /* affine_x then gives 0 */
    }
    int top = 63;
    while (((k >> top) & 1ull) == 0) --top;
    unsigned long long r0x[NW], r0z[NW], r1x[NW], r1z[NW];
    for (int i = 0; i < nw; ++i) { r0x[i] = px[i]; r0z[i] = pz[i]; }
    s2g_xdbl<NW>(r1x, r1z, px, pz, a24, n, ninv, nw);
    for (int i = top - 1; i >= 0; --i) {
        if (((k >> i) & 1ull) != 0) {
            s2g_xadd<NW,XADD6>(r0x, r0z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r1x, r1z, r1x, r1z, a24, n, ninv, nw);
        } else {
            s2g_xadd<NW,XADD6>(r1x, r1z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r0x, r0z, r0x, r0z, a24, n, ninv, nw);
        }
    }
    for (int i = 0; i < nw; ++i) { rx[i] = r0x[i]; rz[i] = r0z[i]; }
}

template <int NW, bool XADD6=false>
__global__ void s2g_ladder_kernel(const unsigned long long *n, unsigned long long ninv, int nw,
                                  const unsigned long long *qx, const unsigned long long *qz,
                                  const unsigned long long *a24, const unsigned long long *mone,
                                  const unsigned long long *js, int npts,
                                  unsigned long long *out_x, unsigned long long *out_z, bool normal_output)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npts) return;
    unsigned long long rx[NW], rz[NW], one[NW], nx[NW], nz[NW];
    /* GRID-STRIDE: the launcher bounds the number of threads per launch (a ladder is a long
       serial chain, and a multi-second kernel is killed by the display driver's watchdog), so
       one launch may have to cover more points than it has threads.  The stride version is the
       same arithmetic, point by point, with a different work assignment. */
    for (int i = t; i < npts; i += gridDim.x * blockDim.x) {
        s2g_ladder<NW,XADD6>(js[i], qx, qz, a24, n, ninv, nw, mone, rx, rz);
        if(normal_output) {
            for (int j = 0; j < NW; ++j) one[j] = 0;
            one[0] = 1;
            s2g_mont_mul<NW>(nx, rx, one, n, ninv, nw);
            s2g_mont_mul<NW>(nz, rz, one, n, ninv, nw);
        }
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)i * nw + j] = normal_output ? nx[j] : rx[j];
            out_z[(size_t)i * nw + j] = normal_output ? nz[j] : rz[j];
        }
    }
}

template <int NW>
__global__ void s2g_mont_test_kernel(const unsigned long long *a, const unsigned long long *b,
                                     const unsigned long long *n, unsigned long long ninv,
                                     int nw, int cases, unsigned long long *out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cases) return;
    unsigned long long r[NW];
    /* the caller's arrays are packed with stride nw (not NW: NW is only the template's
       upper bound, chosen from the word-count dispatch) */
    s2g_mont_mul<NW>(r, a + (size_t)t * nw, b + (size_t)t * nw, n, ninv, nw);
    for (int i = 0; i < nw; ++i) out[(size_t)t * nw + i] = r[i];
}

template <int NW, bool XADD6>
__global__ void s2g_xadd_test_kernel(const unsigned long long *input,
                                     const unsigned long long *n, unsigned long long ninv,
                                     int nw,int cases,unsigned long long *output)
{
    const int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=cases*5) return;
    const int c=t/5, alias=t%5;
    const auto *s=input+(size_t)c*6*nw;
    unsigned long long px[NW],pz[NW],qx[NW],qz[NW],dx[NW],dz[NW],tx[NW],tz[NW];
    for(int i=0;i<nw;++i) {px[i]=s[i];pz[i]=s[nw+i];qx[i]=s[2*nw+i];
                          qz[i]=s[3*nw+i];dx[i]=s[4*nw+i];dz[i]=s[5*nw+i];}
    auto *rx=alias==1 || alias==4 ? px : alias==2 ? qx : alias==3 ? dx : tx;
    auto *rz=alias==1 ? pz : alias==2 || alias==4 ? qz : alias==3 ? dz : tz;
    s2g_xadd<NW,XADD6>(rx,rz,px,pz,qx,qz,dx,dz,n,ninv,nw);
    auto *out=output+(size_t)t*3*nw;
    for(int i=0;i<nw;++i) {out[i]=rx[i];out[nw+i]=rz[i];}
    // Out-of-place here; candidate xADD also exercises in-place modular halves.
    s2g_halfmod<NW>(out+2*nw,s,n,nw);
}

template <int NW, bool XADD6>
__global__ void s2g_seed_pair_kernel(const unsigned long long *n,unsigned long long ninv,int nw,
                                    const unsigned long long *hx,const unsigned long long *hz,
                                    const unsigned long long *a24,const unsigned long long *mone,
                                    unsigned long long clo,unsigned long long npts,
                                    unsigned long long per_block,unsigned long long blocks,
                                    unsigned long long offset,unsigned long long count,
                                    unsigned long long *ox,unsigned long long *oz)
{
    const unsigned long long local=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(local>=count)return;
    const unsigned long long b=offset+local,k=clo+b*per_block;
    unsigned long long ax[NW],az[NW],bx[NW],bz[NW];
    if(k==0) {
        for(int j=0;j<nw;++j){ax[j]=mone[j];az[j]=0;bx[j]=hx[j];bz[j]=hz[j];}
    } else {
        int top=63;while(((k>>top)&1ull)==0)--top;
        for(int j=0;j<nw;++j){ax[j]=hx[j];az[j]=hz[j];}
        s2g_xdbl<NW>(bx,bz,hx,hz,a24,n,ninv,nw);
        for(int bit=top-1;bit>=0;--bit) {
            if((k>>bit)&1ull) {
                s2g_xadd<NW,XADD6>(ax,az,ax,az,bx,bz,hx,hz,n,ninv,nw);
                s2g_xdbl<NW>(bx,bz,bx,bz,a24,n,ninv,nw);
            } else {
                s2g_xadd<NW,XADD6>(bx,bz,ax,az,bx,bz,hx,hz,n,ninv,nw);
                s2g_xdbl<NW>(ax,az,ax,az,a24,n,ninv,nw);
            }
        }
    }
    const bool single=b*per_block+1>=npts;
    for(int j=0;j<nw;++j) {
        ox[(size_t)(2*b)*nw+j]=ax[j];oz[(size_t)(2*b)*nw+j]=az[j];
        ox[(size_t)(2*b+1)*nw+j]=single?ax[j]:bx[j];
        oz[(size_t)(2*b+1)*nw+j]=single?az[j]:bz[j];
        if(b==0){ox[(size_t)(2*blocks)*nw+j]=hx[j];oz[(size_t)(2*blocks)*nw+j]=hz[j];}
    }
}

template <int NW>
__global__ void s2g_segprod_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
                                   const unsigned long long *dz, unsigned long long npts,
                                   unsigned long long seg, unsigned long long nseg,
                                   unsigned long long *out, const unsigned long long *fix)
{
    const unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    const unsigned long long start = s * seg;
    if (start >= npts) return;
    const unsigned long long end = ((start + seg) < npts) ? (start + seg) : npts;
    unsigned long long p[NW], tmp[NW];
    for (int i = 0; i < nw; ++i) p[i] = dz[(size_t)start * nw + i];
    for (unsigned long long j = start + 1; j < end; ++j) {
        s2g_mont_mul<NW>(tmp, p, dz + (size_t)j * nw, dn, ninv, nw);
        for (int i = 0; i < nw; ++i) p[i] = tmp[i];
    }
    const size_t m=(size_t)(end-start);
    if(fix && m>1) {
        s2g_mont_mul<NW>(tmp,p,fix+m*nw,dn,ninv,nw);
        for(int i=0;i<nw;++i)p[i]=tmp[i];
    }
    for (int i = 0; i < nw; ++i) out[(size_t)s * nw + i] = p[i];
}

__global__ void s2g_projective_leaf_kernel(const unsigned long long *n,int nw,
    const unsigned long long *x,const unsigned long long *z,size_t first,size_t count,
    unsigned long long *out)
{
    const size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(k>=count)return;
    const auto *xx=x+(first+k)*nw,*zz=z+(first+k)*nw;
    bool zero=true;for(int j=0;j<nw;++j)if(xx[j])zero=false;
    unsigned long long borrow=0;
    for(int j=0;j<nw;++j) {
        const unsigned long long t=n[j]-xx[j],b1=n[j]<xx[j],v=t-borrow,b2=t<borrow;
        borrow=b1|b2;out[(2*k)*nw+j]=zero?0:v;out[(2*k+1)*nw+j]=zz[j];
    }
}

template <int NW>
__global__ void s2g_block_prod_kernel(const unsigned long long *vals, int nvals, int nw,
                                      int per_block, const unsigned long long *n,
                                      unsigned long long ninv, unsigned long long *out)
{
    const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int nb = (nvals + per_block - 1) / per_block;
    if (b >= nb) return;
    const int lo = b * per_block;
    int hi = lo + per_block;
    if (hi > nvals) hi = nvals;
    unsigned long long acc[NW];
    for (int i = 0; i < nw; ++i) acc[i] = vals[(size_t)lo * nw + i];
    for (int t = lo + 1; t < hi; ++t)
        s2g_mont_mul<NW>(acc, acc, vals + (size_t)t * nw, n, ninv, nw);
    for (int i = 0; i < nw; ++i) out[(size_t)b * nw + i] = acc[i];
}

template<int NW,bool XADD6>
void ecm_launch_s2g_ladder_kernel(EcmCudaLaunch cfg, const unsigned long long *n, unsigned long long ninv, int nw, const unsigned long long *qx, const unsigned long long *qz, const unsigned long long *a24, const unsigned long long *mone, const unsigned long long *js, int npts, unsigned long long *out_x, unsigned long long *out_z, bool normal_output) {
    s2g_ladder_kernel<NW,XADD6><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,ninv,nw,qx,qz,a24,mone,js,npts,out_x,out_z,normal_output);
}

template<int NW>
void ecm_launch_s2g_mont_test_kernel(EcmCudaLaunch cfg, const unsigned long long *a, const unsigned long long *b, const unsigned long long *n, unsigned long long ninv, int nw, int cases, unsigned long long *out) {
    s2g_mont_test_kernel<NW><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,b,n,ninv,nw,cases,out);
}

template<int NW,bool XADD6>
void ecm_launch_s2g_xadd_test_kernel(EcmCudaLaunch cfg, const unsigned long long *input, const unsigned long long *n, unsigned long long ninv, int nw, int cases, unsigned long long *output) {
    s2g_xadd_test_kernel<NW,XADD6><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(input,n,ninv,nw,cases,output);
}

template<int NW,bool XADD6>
void ecm_launch_s2g_seed_pair_kernel(EcmCudaLaunch cfg, const unsigned long long *n, unsigned long long ninv, int nw, const unsigned long long *hx, const unsigned long long *hz, const unsigned long long *a24, const unsigned long long *mone, unsigned long long clo, unsigned long long npts, unsigned long long per_block, unsigned long long blocks, unsigned long long offset, unsigned long long count, unsigned long long *ox, unsigned long long *oz) {
    s2g_seed_pair_kernel<NW,XADD6><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,ninv,nw,hx,hz,a24,mone,clo,npts,per_block,blocks,offset,count,ox,oz);
}

template<int NW>
void ecm_launch_s2g_segprod_kernel(EcmCudaLaunch cfg, const unsigned long long *dn, unsigned long long ninv, int nw, const unsigned long long *dz, unsigned long long npts, unsigned long long seg, unsigned long long nseg, unsigned long long *out, const unsigned long long *fix) {
    s2g_segprod_kernel<NW><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(dn,ninv,nw,dz,npts,seg,nseg,out,fix);
}

void ecm_launch_s2g_projective_leaf_kernel(EcmCudaLaunch cfg, const unsigned long long *n, int nw, const unsigned long long *x, const unsigned long long *z, size_t first, size_t count, unsigned long long *out) {
    s2g_projective_leaf_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,nw,x,z,first,count,out);
}

template<int NW>
void ecm_launch_s2g_block_prod_kernel(EcmCudaLaunch cfg, const unsigned long long *vals, int nvals, int nw, int per_block, const unsigned long long *n, unsigned long long ninv, unsigned long long *out) {
    s2g_block_prod_kernel<NW><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(vals,nvals,nw,per_block,n,ninv,out);
}
