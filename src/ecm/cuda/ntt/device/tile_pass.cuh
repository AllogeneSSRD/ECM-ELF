#pragma once
#include "../../arithmetic/goldilocks/goldilocks_tile.cuh"

template<bool WarpTail>
__device__ __forceinline__ void ntt_tile_finish_forward(unsigned long long* out,
    unsigned long long* sm, unsigned long long tile_size, int stage, int tile_bits,
    const unsigned long long* twiddles)
{
    if (WarpTail && tile_bits >= 5) {
        const int warp_bits = tile_bits >= 6 ? 6 : 5;
        for (; stage > warp_bits; stage -= 2) {
            tile_radix4_block(sm, tile_size, stage, twiddles);
            __syncthreads();
        }
        if (stage == warp_bits) tile_radix2_stage<false>(sm, tile_size, stage, twiddles);
        if (warp_bits == 6) {
            for (unsigned long long i = (threadIdx.x >> 5) * 64 + (threadIdx.x & 31);
                 i < tile_size; i += 2 * blockDim.x) {
                auto lo = sm[i], hi = sm[i + 32];
                tile_warp6_pair<false>(lo, hi, twiddles);
                const unsigned long long dst = (i & ~63ull) + 2 * (threadIdx.x & 31);
                out[dst] = lo;
                out[dst + 1] = hi;
            }
        } else {
            for (unsigned long long i = threadIdx.x; i < tile_size; i += blockDim.x)
                out[i] = tile_warp5<false>(sm[i], twiddles);
        }
        return;
    }
    for (; stage > 0; stage -= 2) {
        tile_radix4_block(sm, tile_size, stage, twiddles);
        __syncthreads();
    }
    if (stage == 0) tile_radix2_stage<false>(sm, tile_size, stage, twiddles);
    for (unsigned long long i = threadIdx.x; i < tile_size; i += blockDim.x)
        out[i] = sm[i];
}

__device__ __forceinline__ void ntt_tile_finish_inverse(unsigned long long* out,
    unsigned long long* sm, unsigned long long tile_size, int first_stage, int tile_bits,
    const unsigned long long* twiddles) {
    int stage = first_stage;
    for (; stage + 1 < tile_bits; stage += 2) {
        tile_radix4_block_inv(sm, tile_size, stage, twiddles);
        __syncthreads();
    }
    if (stage < tile_bits) tile_radix2_stage<true>(sm, tile_size, stage, twiddles);
    for (unsigned long long i = threadIdx.x; i < tile_size; i += blockDim.x)
        out[i] = sm[i];
}

// Operates on a ready shared-memory tile and writes the same transform layout
// for in-place NTTs and Stage1 products loaded from immutable spectra.
template<bool INVERSE, bool WARP_TAIL>
__device__ __forceinline__ void ntt_tile_pass(unsigned long long* out, unsigned long long* sm,
    unsigned long long TILE, int t, const unsigned long long* tbl) {
    if constexpr (!INVERSE) {
        ntt_tile_finish_forward<WARP_TAIL>(out, sm, TILE, t - 1, t, tbl);
        return;
    }
    if (WARP_TAIL && t>=5) {
        const int wt=t>=6 ? 6 : 5;
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
        ntt_tile_finish_inverse(out, sm, TILE, wt, t, tbl);
        return;
    }
    ntt_tile_finish_inverse(out, sm, TILE, 0, t, tbl);
}
