#pragma once
#include <cuda_runtime.h>
__host__ __device__ __forceinline__ void s2g_mul64(unsigned long long a, unsigned long long b,
                                                   unsigned long long &lo, unsigned long long &hi)
{
#if defined(__CUDA_ARCH__)
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    /* host code has no 128-bit type here (MSVC), so the high half comes from the 32-bit split:
       a*b = p11*2^64 + K*2^32 + p00 with K = p01 + p10, and floor((K*2^32 + p00)/2^64) =
       (K>>32) + ((K&m32) + (p00>>32))>>32 -- the two carry terms below. */
    const unsigned long long m32 = 0xffffffffull;
    const unsigned long long a0 = a & m32, a1 = a >> 32, b0 = b & m32, b1 = b >> 32;
    const unsigned long long p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const unsigned long long ks = (p01 & m32) + (p10 & m32);
    const unsigned long long lo_mid = ks + (p00 >> 32);
    lo = (lo_mid << 32) | (p00 & m32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (lo_mid >> 32);
#endif
}

__host__ __device__ __forceinline__ unsigned long long s2g_udiv_2by1(unsigned long long u1,
                                                                    unsigned long long u0,
                                                                    unsigned long long d,
                                                                    unsigned long long v)
{
    unsigned long long lo = 0, hi = 0;
    s2g_mul64(v, u1, lo, hi);
    unsigned long long q = hi + u1;
    for (int guard = 0; guard < 8; ++guard) {
        unsigned long long plo = 0, phi = 0;
        s2g_mul64(q, d, plo, phi);
        const unsigned long long rlo = u0 - plo;
        const unsigned long long b1 = (u0 < plo) ? 1ull : 0ull;
        const unsigned long long rhi = u1 - phi - b1;
        if ((u1 < phi) || (u1 == phi && b1 != 0)) { --q; continue; }   /* estimate too high */
        if (rhi != 0 || rlo >= d) { ++q; continue; }                   /* still >= d: step up */
        return q;                                                      /* exact */
    }
    return q;
}

template <int NW>
__host__ __device__ __forceinline__ int s4_div_rem(unsigned long long *t, int limbs,
    const unsigned long long *ns, int nw, int shift, unsigned long long recip,
    unsigned long long *out)
{
    for (int i = 0; i < nw; ++i) out[i] = 0;
    if (shift != 0) {
        unsigned long long carry = 0;
        for (int i = 0; i < limbs; ++i) {
            const unsigned long long v = t[i];
            t[i] = (v << shift) | carry;
            carry = v >> (64 - shift);
        }
        t[limbs++] = carry;
    }
    while (limbs > 0 && t[limbs - 1] == 0) --limbs;
    t[limbs] = 0;                        /* high sentinel, including the shift == 0 case */
    int repairs = 0;
    const unsigned long long top = ns[nw - 1];
    for (int j = limbs - nw; j >= 0; --j) {
        const unsigned long long u1 = t[j + nw], u0 = t[j + nw - 1];
        if (u1 > top) return -1;
        const unsigned long long q = (u1 == top) ? ~0ull
                                                  : s2g_udiv_2by1(u1, u0, top, recip);
        unsigned long long carry = 0;
        for (int i = 0; i < nw; ++i) {
            unsigned long long lo, hi;
            s2g_mul64(q, ns[i], lo, hi);
            const unsigned long long sub = lo + carry;
            hi += (sub < lo) ? 1ull : 0ull;
            const unsigned long long v = t[j + i];
            t[j + i] = v - sub;
            carry = hi + ((v < sub) ? 1ull : 0ull);
        }
        bool under = t[j + nw] < carry;
        t[j + nw] -= carry;
        for (int correction = 0; under && correction < 2; ++correction) {
            unsigned long long c = 0;
            for (int i = 0; i < nw; ++i) {
                const unsigned long long v = t[j + i], sum = v + ns[i];
                const unsigned long long next = sum + c;
                c = ((sum < v) || (next < sum)) ? 1ull : 0ull;
                t[j + i] = next;
            }
            const unsigned long long v = t[j + nw];
            t[j + nw] = v + c;
            under = (t[j + nw] >= v);    /* wrapping the negative high word ends the borrow */
            ++repairs;
        }
        if (under) return -1;
    }
    for (int i = 0; i < nw; ++i)
        out[i] = (shift == 0) ? t[i]
            : ((t[i] >> shift) | (i + 1 < nw ? (t[i + 1] << (64 - shift)) : 0ull));
    return repairs;
}

template <int NW>
__host__ __device__ __forceinline__ int s4_mersenne_rem(unsigned long long *t, int live,
    int nw, int bits, unsigned long long *out)
{
    const int q=bits/64,b=bits%64;
    const unsigned long long mask=b ? ((1ull<<b)-1) : ~0ull;
    while(live>0 && t[live-1]==0)--live;
    int folds=0;
    while(live>nw || (b && live==nw && (t[nw-1]>>b)!=0)) {
        const int len=(live-q)>nw ? live-q : nw;
        unsigned long long carry=0;
        for(int i=0;i<len;++i) {
            const int src=i+q;
            const unsigned long long h0=src<live ? t[src] : 0ull;
            const unsigned long long h1=(b && src+1<live) ? t[src+1] : 0ull;
            const unsigned long long hi=b ? ((h0>>b)|(h1<<(64-b))) : h0;
            const unsigned long long lo=i<nw ? (i==nw-1 ? (t[i]&mask) : t[i]) : 0ull;
            const unsigned long long sum=lo+hi,next=sum+carry;
            carry=((sum<lo)||(next<sum)) ? 1ull : 0ull;
            t[i]=next;
        }
        live=len;
        if(carry)t[live++]=carry;
        while(live>0 && t[live-1]==0)--live;
        ++folds;
    }
    bool equal=true;
    for(int i=0;i<nw;++i) {
        out[i]=i<live ? t[i] : 0ull;
        if(out[i]!=(i==nw-1 ? mask : ~0ull))equal=false;
    }
    if(equal)for(int i=0;i<nw;++i)out[i]=0; // N itself is zero, canonical [0,N)
    return folds;
}

__host__ __device__ static inline unsigned long long groot_mix64(unsigned long long x) {
    x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return x^(x>>31);
}
