#pragma once
#include "../../../../src/ecm/stage2/cuda/launch.h"
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
#include "giant_point_mersenne.cuh"

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
