#pragma once
#include "../launch.h"
#include "tile_pass.cuh"

template<bool ShiftScale>
__global__ __launch_bounds__(FUSE_TILE_THREADS, 3)
void ntt_inverse_tile_from_pairs(unsigned long long* a, const unsigned long long* b,
    unsigned long long stride, int k, int t, const unsigned long long* twiddles,
    unsigned long long scale) {
    extern __shared__ unsigned long long sm[];
    const unsigned width = 1u << t;
    const unsigned long long base = (unsigned long long)blockIdx.y * stride +
                                    (unsigned long long)blockIdx.x * width;
    const unsigned lane = threadIdx.x & 31;
    for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < width; i += 2 * blockDim.x) {
        const auto pair = base + i + lane;
        const auto left = *reinterpret_cast<const ulonglong2*>(a + pair);
        const auto right = *reinterpret_cast<const ulonglong2*>(b + pair);
        auto lo = gl_mul(left.x, right.x), hi = gl_mul(left.y, right.y);
        if constexpr (ShiftScale) {
            lo = gl_scale_inverse_pow2(lo, k);
            hi = gl_scale_inverse_pow2(hi, k);
        } else {
            lo = gl_mul(lo, scale);
            hi = gl_mul(hi, scale);
        }
        tile_warp6_pair<true>(lo, hi, twiddles);
        sm[i] = lo;
        sm[i + 32] = hi;
    }
    __syncthreads();
    ntt_tile_finish_inverse(a + base, sm, width, 6, t, twiddles);
}

template<bool ShiftScale>
void ecm_launch_ntt_inverse_tile_from_pairs(EcmCudaLaunch cfg, unsigned long long* a,
    const unsigned long long* b, unsigned long long stride, int k, int t,
    const unsigned long long* twiddles, unsigned long long scale) {
    ntt_inverse_tile_from_pairs<ShiftScale><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(
        a,b,stride,k,t,twiddles,scale);
}

template<int T, bool Packed>
__device__ __forceinline__ void ntt_short_forward_in_place(unsigned long long* plane,
    const unsigned long long* twiddles) {
    constexpr unsigned width = 1u << T;
    // The weighted packer has already executed the first DIF stage.
    constexpr int first = T - 1 - int(Packed);
    for (int stage = first; stage > 6; stage -= 2) {
        tile_radix4_block(plane, width, stage, twiddles);
        __syncthreads();
    }
    if constexpr (first % 2 == 0) tile_radix2_stage<false>(plane, width, 6, twiddles);
    const unsigned lane = threadIdx.x & 31;
    for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < width; i += 2 * blockDim.x) {
        auto lo = plane[i], hi = plane[i + 32];
        tile_warp6_pair<false>(lo, hi, twiddles);
        const unsigned pair = i + lane;
        __syncwarp();
        plane[pair] = lo;
        plane[pair + 1] = hi;
    }
    __syncthreads();
}

template<int T, bool ShiftScale, bool Packed>
__global__ __launch_bounds__(FUSE_TILE_THREADS, 2)
void ntt_short_convolution(unsigned long long* a, unsigned long long* b,
    const unsigned long long* fwd, const unsigned long long* inv, unsigned long long scale) {
    extern __shared__ unsigned long long shared[];
    constexpr unsigned width = 1u << T;
    auto* sa = shared;
    auto* sb = shared + width;
    a += (unsigned long long)blockIdx.x * width;
    b += (unsigned long long)blockIdx.x * width;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        sa[i] = a[i];
        sb[i] = b[i];
    }
    __syncthreads();
    ntt_short_forward_in_place<T,Packed>(sa, fwd);
    ntt_short_forward_in_place<T,Packed>(sb, fwd);
    const unsigned lane = threadIdx.x & 31;
    for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < width; i += 2 * blockDim.x) {
        const unsigned pair = i + lane;
        const auto rhs0 = sb[pair], rhs1 = sb[pair + 1];
        // Keep B's transformed representation for the ordinary caller contract.
        b[pair] = rhs0;
        b[pair + 1] = rhs1;
        auto lo = gl_mul(sa[pair], rhs0), hi = gl_mul(sa[pair + 1], rhs1);
        if constexpr (ShiftScale) {
            lo = gl_scale_inverse_pow2(lo, T);
            hi = gl_scale_inverse_pow2(hi, T);
        } else {
            lo = gl_mul(lo, scale);
            hi = gl_mul(hi, scale);
        }
        tile_warp6_pair<true>(lo, hi, inv);
        __syncwarp();
        sa[i] = lo;
        sa[i + 32] = hi;
    }
    __syncthreads();
    ntt_tile_finish_inverse(a, sa, width, 6, T, inv, T % 2 == 0);
}

template<int T, bool ShiftScale, bool Packed>
void ecm_launch_ntt_short_convolution(EcmCudaLaunch cfg, unsigned long long* a,
    unsigned long long* b, const unsigned long long* fwd, const unsigned long long* inv,
    unsigned long long scale) {
    ntt_short_convolution<T,ShiftScale,Packed><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(
        a,b,fwd,inv,scale);
}
