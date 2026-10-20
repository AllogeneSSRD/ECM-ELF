#pragma once
#include "goldilocks.cuh"
__device__ __forceinline__ void tile_radix4_block(unsigned long long *sm, unsigned long long TILE,
                                                 int st, const unsigned long long *tbl)
{
    const unsigned long long half = 1ull << st;      /* the LARGER half of the pair */
    const unsigned long long h2 = half >> 1;         /* the smaller one */
    const unsigned long long *twA = tbl + (half - 1);        /* larger-half stage */
    const unsigned long long *twB = tbl + (h2 - 1);          /* smaller-half stage */
    const unsigned long long units = TILE >> 2;
    for (unsigned long long m = threadIdx.x; m < units; m += blockDim.x) {
        const unsigned long long base = (m >> (st - 1)) << (st + 1);
        const unsigned long long u = m & (h2 - 1);
        const unsigned long long a = sm[base + u];
        const unsigned long long b = sm[base + u + half];
        const unsigned long long c = sm[base + h2 + u];
        const unsigned long long d = sm[base + h2 + u + half];
        /* stage st (half): (base+u, base+u+half) with j = u; (base+h2+u, base+h2+u+half)
           with j = (base+h2+u) mod half = h2+u  -- DIF multiplies AFTER the subtraction */
        const unsigned long long a1 = gl_add_dev(a, b);
        const unsigned long long b1 = gl_mul(gl_sub_dev(a, b), twA[u]);
        const unsigned long long c1 = gl_add_dev(c, d);
        const unsigned long long d1 = gl_mul(gl_sub_dev(c, d), twA[h2 + u]);
        /* stage st-1 (h2): (base+u, base+h2+u) with j = u  and  (base+2*h2+u, base+3*h2+u)
           with j = u  -- BOTH pairs of the smaller stage share the same twiddle */
        const unsigned long long tw = twB[u];
        sm[base + u] = gl_add_dev(a1, c1);
        sm[base + h2 + u] = gl_mul(gl_sub_dev(a1, c1), tw);
        sm[base + 2 * h2 + u] = gl_add_dev(b1, d1);
        sm[base + 3 * h2 + u] = gl_mul(gl_sub_dev(b1, d1), tw);
    }
}

__device__ __forceinline__ void tile_radix4_block_inv(unsigned long long *sm,
                                                      unsigned long long TILE, int st,
                                                      const unsigned long long *tbl)
{
    const unsigned long long h2 = 1ull << st;        /* the smaller half (stage st) */
    const unsigned long long half = h2 << 1;         /* the larger half (stage st+1) */
    const unsigned long long *twA = tbl + (h2 - 1);          /* stage st */
    const unsigned long long *twB = tbl + (half - 1);        /* stage st+1 */
    const unsigned long long units = TILE >> 2;
    for (unsigned long long m = threadIdx.x; m < units; m += blockDim.x) {
        const unsigned long long base = (m >> st) << (st + 2);
        const unsigned long long u = m & (h2 - 1);
        const unsigned long long a = sm[base + u];
        const unsigned long long b = sm[base + 2 * h2 + u];
        const unsigned long long c = sm[base + h2 + u];
        const unsigned long long d = sm[base + 3 * h2 + u];
        /* stage st: pairs (base+u, base+h2+u) and (base+2h2+u, base+3h2+u), both j = u */
        const unsigned long long w1 = twA[u];
        const unsigned long long cv = gl_mul(c, w1);
        const unsigned long long a1 = gl_add_dev(a, cv);
        const unsigned long long c1 = gl_sub_dev(a, cv);
        const unsigned long long dv = gl_mul(d, w1);
        const unsigned long long b1 = gl_add_dev(b, dv);
        const unsigned long long d1 = gl_sub_dev(b, dv);
        /* stage st+1: (base+u, base+2h2+u) with j = u, (base+h2+u, base+3h2+u) with j = h2+u */
        const unsigned long long bv = gl_mul(b1, twB[u]);
        const unsigned long long dv2 = gl_mul(d1, twB[h2 + u]);
        sm[base + u] = gl_add_dev(a1, bv);
        sm[base + 2 * h2 + u] = gl_sub_dev(a1, bv);
        sm[base + h2 + u] = gl_add_dev(c1, dv2);
        sm[base + 3 * h2 + u] = gl_sub_dev(c1, dv2);
    }
}

template <bool INVERSE>
__device__ __forceinline__ void tile_radix2_stage(unsigned long long *sm, unsigned long long TILE,
                                                  int st, const unsigned long long *tbl)
{
    const unsigned long long half = 1ull << st;
    const unsigned long long *tw = tbl + (half - 1);
    for (unsigned long long m = threadIdx.x; m < TILE / 2; m += blockDim.x) {
        const unsigned long long g = m >> st;
        const unsigned long long j = m - (g << st);
        const unsigned long long i0 = (g << (st + 1)) + j;
        const unsigned long long i1 = i0 + half;
        const unsigned long long uu = sm[i0], vv = sm[i1];
        const unsigned long long w0 = __ldg(tw + j);
        if (INVERSE) {
            const unsigned long long w = gl_mul(vv, w0);
            sm[i0] = gl_add_dev(uu, w);
            sm[i1] = gl_sub_dev(uu, w);
        } else {
            sm[i0] = gl_add_dev(uu, vv);
            sm[i1] = gl_mul(gl_sub_dev(uu, vv), w0);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ unsigned long long tile_warp_twiddle(unsigned long long v,
                                                              unsigned long long w, int st)
{
    if(st==0) return v; // the generated table's sole stage-0 entry is always 1
    if(st==1) {
        if(w==1) return v;
        if(w==(1ull<<48)) return gl_reduce(v<<48,v>>16);
        if(w==GL_P-(1ull<<48)) return gl_sub_dev(0,gl_reduce(v<<48,v>>16));
    }
    return gl_mul(v,w);
}

template <bool INVERSE>
__device__ __forceinline__ unsigned long long tile_warp5(unsigned long long v,
                                                        const unsigned long long *tbl)
{
    const unsigned int lane=threadIdx.x & 31;
    #pragma unroll
    for(int q=0;q<5;++q) {
        const int st=INVERSE ? q : 4-q;
        const unsigned int half=1u<<st, j=lane & (half-1);
        const bool upper=(lane & half)!=0;
        const unsigned long long w=__ldg(tbl+(half-1)+j);
        if(INVERSE && upper) v=tile_warp_twiddle(v,w,st);
        const auto peer=__shfl_xor_sync(0xffffffffu,v,half);
        if(INVERSE) v=upper ? gl_sub_dev(peer,v) : gl_add_dev(v,peer);
        else v=upper ? tile_warp_twiddle(gl_sub_dev(peer,v),w,st) : gl_add_dev(v,peer);
    }
    return v;
}

template <bool INVERSE>
__device__ __forceinline__ void tile_warp6_pair(unsigned long long &lo, unsigned long long &hi,
                                              const unsigned long long *tbl)
{
    const unsigned int lane=threadIdx.x&31;
    #pragma unroll
    for(int q=0;q<6;++q) {
        const int st=INVERSE ? q : 5-q;
        const unsigned int half=1u<<st, j=lane&(half-1);
        const auto w=__ldg(tbl+(half-1)+j);
        const auto u=lo, v=INVERSE ? tile_warp_twiddle(hi,w,st) : hi;
        lo=gl_add_dev(u,v);
        hi=INVERSE ? gl_sub_dev(u,v) : tile_warp_twiddle(gl_sub_dev(u,v),w,st);
        if(INVERSE ? st<5 : st>0) {
            const unsigned int mask=1u<<(INVERSE ? st : st-1);
            const bool lower=(lane&mask)==0;
            const auto peer=__shfl_xor_sync(0xffffffffu,lower ? hi : lo,mask);
            lo=lower ? lo : peer; hi=lower ? peer : hi;
        }
    }
}
