#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "../../ntt/parameters.h"
namespace {
#include "goldilocks_reduce.cuh"
#ifdef __CUDACC__
#include "goldilocks_ptx.cuh"
#include "goldilocks_sub.cuh"
#if NTT_GL_FIXED_MODE == -1
__device__ __constant__ volatile unsigned int ntt_gl_short_reduce_device=0;
#endif
#endif
__host__ __device__ inline unsigned long long gl_reduce(unsigned long long lo,
                                                        unsigned long long hi)
{
#ifdef __CUDA_ARCH__
#if NTT_GL_FIXED_MODE == 3
    return gl_reduce128_ptx(lo,hi);
#elif NTT_GL_FIXED_MODE == 1
    return gl_reduce128_short(lo,hi);
#elif NTT_GL_FIXED_MODE == -1
    const auto mode=ntt_gl_short_reduce_device;
    if(mode&2u)return gl_reduce128_ptx(lo,hi);
    if(mode&1u)return gl_reduce128_short(lo,hi);
#endif
#endif
    for (int iter = 0; iter < 4; ++iter) {
        /* exact: lo + hi*(2^32-1) = lo + (hi<<32) - hi, split into (lo', hi') */
        const unsigned long long hl = hi << 32;
        const unsigned long long hh = hi >> 32;
        /* s1 = lo + hl, carry c1 */
        const unsigned long long s1 = lo + hl;
        const unsigned long long c1 = (s1 < lo) ? 1ull : 0ull;
        /* s2 = s1 - hi, borrow b */
        const unsigned long long s2 = s1 - hi;
        const unsigned long long b = (s1 < hi) ? 1ull : 0ull;
        /* hi' = hh (from hl's high half) + c1 - b ; hh < 2^32, so this fits and stays
           in [0, 2^33] */
        lo = s2;
        hi = hh + c1 - b;
    }
    if (lo >= GL_P) lo -= GL_P;
    if (lo >= GL_P) lo -= GL_P;
    if (lo >= GL_P) lo -= GL_P;
    return lo;
}
__host__ __device__ inline unsigned long long gl_mod(unsigned long long x)
{
    return gl_reduce(x, 0);
}
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_mul(unsigned long long a, unsigned long long b)
{
    return gl_reduce(a * b, __umul64hi(a, b));
}
#endif
static inline void mul_64x64_hi_lo(unsigned long long a, unsigned long long b,
                                   unsigned long long &lo, unsigned long long &hi)
{
    const unsigned long long al = a & 0xFFFFFFFFull, ah = a >> 32;
    const unsigned long long bl = b & 0xFFFFFFFFull, bh = b >> 32;
    const unsigned long long p0 = al * bl;
    const unsigned long long p1 = al * bh;         /* high word p1>>32 */
    const unsigned long long p2 = ah * bl;         /* high word p2>>32 */
    const unsigned long long s1 = p0 + (p1 << 32);
    const unsigned long long c1 = (s1 < (p1 << 32)) ? 1ull : 0ull;
    const unsigned long long s2 = s1 + (p2 << 32);
    const unsigned long long c2 = (s2 < (p2 << 32)) ? 1ull : 0ull;
    lo = s2;
    hi = ah * bh + (p1 >> 32) + (p2 >> 32) + c1 + c2;
}
static inline unsigned long long gl_mul_host(unsigned long long a, unsigned long long b)
{
    unsigned long long lo = 0, hi = 0;
    mul_64x64_hi_lo(a, b, lo, hi);
    return gl_reduce(lo, hi);
}
__host__ __device__ inline unsigned long long gl_add(unsigned long long a, unsigned long long b)
{
    unsigned long long s = a + b;
    if (s < a || s >= GL_P) s -= GL_P;
    return s;
}
__host__ __device__ inline unsigned long long gl_sub(unsigned long long a, unsigned long long b)
{
    return (a >= b) ? (a - b) : (a + (GL_P - b));
}
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_mod_dev(unsigned long long x)
{
    return gl_reduce(x, 0);
}
#endif
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_sub_dev(unsigned long long a, unsigned long long b)
{
    return gl_sub_canonical_ptx(a, b);
}
#endif
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_add_dev(unsigned long long a, unsigned long long b)
{
    return gl_add(a, b);
}
#endif
static inline unsigned long long gl_pow_host(unsigned long long a, unsigned long long e)
{
    unsigned long long r = 1 % GL_P;
    a %= GL_P;
    while (e) {
        if (e & 1ull) r = gl_mul_host(r, a);
        a = gl_mul_host(a, a);
        e >>= 1;
    }
    return r;
}
static inline unsigned long long gl_add_host(unsigned long long a, unsigned long long b)
{
    unsigned long long t = a + b;
    if (t < a || t >= GL_P) t -= GL_P;
    return t;
}
static inline unsigned long long gl_sub_host(unsigned long long a, unsigned long long b)
{
    return (a >= b) ? (a - b) : (a + (GL_P - b));
}
inline int bitlen_u64(unsigned long long v)
{
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}
inline unsigned long long bits_of(const unsigned long long *w, unsigned long long nw,
                                  unsigned long long bit, int nbits)
{
    const unsigned long long wi = bit >> 6;
    const int off = (int)(bit & 63ull);
    unsigned long long v = (wi < nw) ? w[(size_t)wi] : 0ull;
    v >>= off;
    if (off + nbits > 64) {
        const unsigned long long nxt = (wi + 1 < nw) ? w[(size_t)(wi + 1)] : 0ull;
        v |= nxt << (64 - off);
    }
    if (nbits < 64) v &= ((1ull << nbits) - 1ull);
    return v;
}
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_twiddle(unsigned long long omega,
                                                         unsigned long long e)
{
    if (e == 0) return 1ull;                     /* omega^0: the first stage's j = 0 */
    unsigned long long r = omega;
    int top = 63;
    while (((e >> top) & 1ull) == 0) --top;      /* e >= 1 always here */
    for (int i = top - 1; i >= 0; --i) {
        r = gl_mul(r, r);
        if ((e >> i) & 1ull) r = gl_mul(r, omega);
    }
    return r;
}
#endif
#ifdef __CUDACC__
__device__ __forceinline__ unsigned long long gl_mul_mod_n(unsigned long long a,
                                                           unsigned long long b,
                                                           unsigned long long n)
{
    const unsigned long long ap = gl_mod_dev(a);
    const unsigned long long bp = gl_mod_dev(b);
    unsigned long long e = gl_mul(ap, bp);
    if (e >= n) e -= n;
    if (e >= n) e -= n;
    return e;
}
#endif
static inline unsigned long long gl_mul_mod_n_host(unsigned long long a, unsigned long long b,
                                                   unsigned long long n)
{
    const unsigned long long ap = gl_reduce(a, 0);
    const unsigned long long bp = gl_reduce(b, 0);
    unsigned long long e = gl_mul_host(ap, bp);
    if (e >= n) e -= n;
    if (e >= n) e -= n;
    return e;
}
#ifdef __CUDACC__
__device__ __forceinline__ void ntt_stage(unsigned long long *a, unsigned long long half,
                                          unsigned long long exp_step,
                                          unsigned long long omega, unsigned long long n)
{
    const unsigned long long stride = 2 * half;
    const unsigned long long groups = n / stride;
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= groups * half) return;
    const unsigned long long g = tid / half;
    const unsigned long long j = tid - g * half;
    const unsigned long long i0 = g * stride + j;
    const unsigned long long i1 = i0 + half;
    const unsigned long long tw = gl_twiddle(omega, gl_mul_mod_n(j, exp_step, n));
    const unsigned long long u = a[i0];
    const unsigned long long v = a[i1];
    a[i0] = gl_add_dev(u, v);
    a[i1] = gl_mul(gl_sub_dev(u, v), tw);
}
#endif
#ifdef __CUDACC__
__device__ __forceinline__ void dit_stage(unsigned long long *a, unsigned long long half,
                                          unsigned long long exp_step,
                                          unsigned long long omega, unsigned long long n)
{
    const unsigned long long stride = 2 * half;
    const unsigned long long groups = n / stride;
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= groups * half) return;
    const unsigned long long g = tid / half;
    const unsigned long long j = tid - g * half;
    const unsigned long long i0 = g * stride + j;
    const unsigned long long i1 = i0 + half;
    const unsigned long long tw = gl_twiddle(omega, gl_mul_mod_n(j, exp_step, n));
    const unsigned long long u = a[i0];
    const unsigned long long v = gl_mul(a[i1], tw);
    a[i0] = gl_add_dev(u, v);
    a[i1] = gl_sub_dev(u, v);
}
#endif
}
