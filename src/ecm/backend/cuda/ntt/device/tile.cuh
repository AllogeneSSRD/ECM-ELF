#pragma once
#include "../launch.h"
#include "../../arithmetic/goldilocks/goldilocks.cuh"
#include "../../arithmetic/goldilocks/goldilocks_tile.cuh"
__global__ void build_tile_table_kernel(unsigned long long *tbl, int k, int t,
                                        unsigned long long root)
{
    const unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= (1ull << t)) return;
    int s = 0;
    while ((1ull << (s + 1)) <= idx + 1ull) ++s;
    if (s >= k) { tbl[idx] = 1ull; return; }
    const unsigned long long j = idx + 1ull - (1ull << s);
    tbl[idx] = gl_twiddle(root, j << (k - 1 - s));
}

void ecm_launch_build_tile_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, int k, int t, unsigned long long root) {
    build_tile_table_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(tbl,k,t,root);
}

__global__ void build_pass_table_kernel(unsigned long long *tbl, unsigned long long sz,
                                        int shift, unsigned long long root)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= sz) return;
    tbl[i] = gl_twiddle(root, i << shift);
}

void ecm_launch_build_pass_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, unsigned long long sz, int shift, unsigned long long root) {
    build_pass_table_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(tbl,sz,shift,root);
}

__global__ void build_radix_table_kernel(unsigned long long *tbl, int m, unsigned long long root,
                                         int rev)
{
    const int R = 1 << m;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R - 1) return;
    int i = 0;
    int off = 0;
    for (;;) {
        const int d = rev ? (1 << i) : (1 << (m - 1 - i));
        if (idx < off + d) break;
        off += d;
        ++i;
    }
    const int u = idx - off;
    const int sh = rev ? (m - 1 - i) : i;
    tbl[idx] = gl_twiddle(root, (unsigned long long)u << sh);
}

void ecm_launch_build_radix_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, int m, unsigned long long root, int rev) {
    build_radix_table_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(tbl,m,root,rev);
}

template <int M>
__global__ void outer_fwd_kernel(unsigned long long *a, unsigned long long n, int L,
                                 const unsigned long long *tbl, const unsigned long long *D,
                                 unsigned long long grp_per_blk, unsigned long long stride)
{
    a += (size_t)blockIdx.y * stride;
    constexpr int R = 1 << M;
    const unsigned long long S = n >> (L + M);      /* group offset count == stride */
    const unsigned long long MB = n >> L;           /* block size */
    const unsigned long long groups = n >> M;
    __shared__ unsigned long long sD[R - 1];
    for (int i = threadIdx.x; i < R - 1; i += blockDim.x) sD[i] = D[i];
    __syncthreads();
    const unsigned long long g0 = blockIdx.x * grp_per_blk;
    const unsigned long long g1 = (groups < g0 + grp_per_blk) ? groups : (g0 + grp_per_blk);
    for (unsigned long long g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        const unsigned long long blk = g / S;
        const unsigned long long v = g - blk * S;
        unsigned long long *p0 = a + blk * MB + v;
        unsigned long long x[R];
#pragma unroll
        for (int i = 0; i < R; ++i) x[i] = p0[(unsigned long long)i * S];
        unsigned long long tb = tbl[v];             /* omega^(v*2^L), squared per stage */
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const int d = 1 << (M - 1 - i);
#pragma unroll
            for (int u = 0; u < d; ++u) {
                const unsigned long long w = gl_mul(tb, sD[(R - (R >> i)) + u]);
#pragma unroll
                for (int b = 0; b < R / (2 * d); ++b) {
                    const int i1 = b * 2 * d + u, i2 = i1 + d;
                    const unsigned long long uu = x[i1], vv = x[i2];
                    x[i1] = gl_add_dev(uu, vv);
                    x[i2] = gl_mul(gl_sub_dev(uu, vv), w);
                }
            }
            tb = gl_mul(tb, tb);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) p0[(unsigned long long)i * S] = x[i];
    }
}

template<int M>
void ecm_launch_outer_fwd_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride) {
    outer_fwd_kernel<M><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,n,L,tbl,D,grp_per_blk,stride);
}
template <int M>
__global__ void outer_inv_kernel(unsigned long long *a, unsigned long long n, int s0,
                                 const unsigned long long *tq, const unsigned long long *dw,
                                 unsigned long long grp_per_blk, unsigned long long stride)
{
    a += (size_t)blockIdx.y * stride;
    constexpr int R = 1 << M;
    const unsigned long long S = 1ull << s0;
    const unsigned long long MB = S << M;
    const unsigned long long groups = n >> M;
    __shared__ unsigned long long sDW[R - 1];
    for (int i = threadIdx.x; i < R - 1; i += blockDim.x) sDW[i] = dw[i];
    __syncthreads();
    const unsigned long long g0 = blockIdx.x * grp_per_blk;
    const unsigned long long g1 = (groups < g0 + grp_per_blk) ? groups : (g0 + grp_per_blk);
    for (unsigned long long g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        const unsigned long long blk = g >> s0;
        const unsigned long long base = g - (blk << s0);
        unsigned long long *p0 = a + blk * MB + base;
        unsigned long long x[R];
#pragma unroll
        for (int i = 0; i < R; ++i) x[i] = p0[(unsigned long long)i << s0];
        /* q = omega_inv^(base*2^L); stage i needs q^(2^(m-1-i)) -> squaring chain downwards */
        unsigned long long tbv[M];
        tbv[M - 1] = tq[base];
#pragma unroll
        for (int i = M - 2; i >= 0; --i) tbv[i] = gl_mul(tbv[i + 1], tbv[i + 1]);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const int d = 1 << i;
#pragma unroll
            for (int u = 0; u < d; ++u) {
                const unsigned long long w = gl_mul(tbv[i], sDW[(d - 1) + u]);
#pragma unroll
                for (int b = 0; b < R / (2 * d); ++b) {
                    const int i1 = b * 2 * d + u, i2 = i1 + d;
                    const unsigned long long uu = x[i1];
                    const unsigned long long vv = gl_mul(x[i2], w);
                    x[i1] = gl_add_dev(uu, vv);
                    x[i2] = gl_sub_dev(uu, vv);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) p0[(unsigned long long)i << s0] = x[i];
    }
}

template<int M>
void ecm_launch_outer_inv_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride) {
    outer_inv_kernel<M><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,n,s0,tq,dw,grp_per_blk,stride);
}
template <bool INVERSE, bool WARP_TAIL=false>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WARP_TAIL ? 3 : 1)
void tile_kernel(unsigned long long *a, const unsigned long long *b,
                            unsigned long long n, int k, int t,
                            const unsigned long long *tbl, unsigned long long n_scale,
                            unsigned long long stride, bool shift_scale=false)
{
    const size_t sl = (size_t)blockIdx.y * stride;
    a += sl; if (b) b += sl;
    extern __shared__ unsigned long long sm[];
    const unsigned long long TILE = 1ull << t;
    const unsigned long long base = blockIdx.x * TILE;
    for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
        sm[i] = a[base + i];
    __syncthreads();
    if (INVERSE) {
        /* pointwise product + 1/N, both elementwise: fusing them here saves two passes */
        if(shift_scale && gl_scale_is_inverse_pow2(n_scale,k)) {
            for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
                sm[i] = gl_scale_inverse_pow2(gl_mul(sm[i], b[base + i]), k);
        } else {
            for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
                sm[i] = gl_mul(gl_mul(sm[i], b[base + i]), n_scale);
        }
        __syncthreads();
    }
    if (WARP_TAIL && t>=5) {
        const int wt=t>=6 ? 6 : 5;
        if (!INVERSE) {
            int st=t-1;
            for(;st>=wt+1;st-=2) {
                tile_radix4_block(sm,TILE,st,tbl);
                __syncthreads();
            }
            if(st==wt) tile_radix2_stage<false>(sm,TILE,wt,tbl);
            /* No shared store or barrier after the warp tail: each lane owns its output. */
            if(wt==6) {
                for(unsigned long long i=(threadIdx.x>>5)*64+(threadIdx.x&31);i<TILE;i+=2*blockDim.x) {
                    auto lo=sm[i], hi=sm[i+32];
                    tile_warp6_pair<false>(lo,hi,tbl);
                    const unsigned long long dst=(i&~63ull)+2*(threadIdx.x&31);
                    a[base+dst]=lo; a[base+dst+1]=hi;
                }
            } else for(unsigned long long i=threadIdx.x;i<TILE;i+=blockDim.x)
                a[base+i]=tile_warp5<false>(sm[i],tbl);
        } else {
            if(wt==6) {
                for(unsigned long long i=(threadIdx.x>>5)*64+(threadIdx.x&31);i<TILE;i+=2*blockDim.x) {
                    const unsigned long long src=(i&~63ull)+2*(threadIdx.x&31);
                    auto lo=sm[src], hi=sm[src+1];
                    tile_warp6_pair<true>(lo,hi,tbl);
                    /* All input words of this warp have reached registers before any
                       lane overwrites their permuted shared locations. */
                    __syncwarp();
                    sm[i]=lo; sm[i+32]=hi;
                }
            } else for(unsigned long long i=threadIdx.x;i<TILE;i+=blockDim.x)
                sm[i]=tile_warp5<true>(sm[i],tbl);
            __syncthreads();
            int st=wt;
            for(;st+1<t;st+=2) {
                tile_radix4_block_inv(sm,TILE,st,tbl);
                __syncthreads();
            }
            if(st<t) tile_radix2_stage<true>(sm,TILE,st,tbl);
            for(unsigned long long i=threadIdx.x;i<TILE;i+=blockDim.x)
                a[base+i]=sm[i];
        }
        return;
    }
    int q = 0;
    if (!INVERSE) {
        /* DIF runs half = t-1 down to 0: pair (t-1,t-2), (t-3,t-4), ... */
        for (; q + 1 < t; q += 2) {
            tile_radix4_block(sm, TILE, t - 1 - q, tbl);
            __syncthreads();
        }
        if (q < t) tile_radix2_stage<false>(sm, TILE, 0, tbl);
    } else {
        /* DIT runs half = 0 up to t-1: pair (0,1), (2,3), ... */
        for (; q + 1 < t; q += 2) {
            tile_radix4_block_inv(sm, TILE, q, tbl);
            __syncthreads();
        }
        if (q < t) tile_radix2_stage<true>(sm, TILE, q, tbl);
    }
    for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
        a[base + i] = sm[i];
}

template<bool INVERSE,bool WARP_TAIL>
void ecm_launch_tile_kernel(EcmCudaLaunch cfg, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale) {
    tile_kernel<INVERSE,WARP_TAIL><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,b,n,k,t,tbl,n_scale,stride,shift_scale);
}
