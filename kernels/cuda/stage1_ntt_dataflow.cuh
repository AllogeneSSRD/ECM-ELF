#pragma once

#include "../../src/ecm/cuda/arithmetic/goldilocks/goldilocks.cuh"
#include "../../src/ecm/cuda/ntt/device/tile_pass.cuh"

template<bool Weighted, bool Modulus = false, class Shape, class Digit>
__device__ __forceinline__ unsigned long long stage1_ntt_pack_digit(
    const uint32_t* row, Shape s, const Digit* weighted, unsigned j)
{
    using U = unsigned long long;
    if (j >= s.digits) return 0;
    const auto digit = Weighted ? weighted[j] : Digit{1, j * s.bpw, s.bpw};
    const U mask = (1ull << digit.width) - 1;
    if constexpr (Modulus) {
        return gl_mul(mask, digit.weight);
    } else {
        const unsigned word = digit.bit / 32, offset = digit.bit % 32;
        U v = row[word];
        if (word + 1 < s.words) v |= U(row[word + 1]) << 32;
        v = (v >> offset) & mask;
        if constexpr (Weighted) v = gl_mul(v, digit.weight);
        return v;
    }
}

// A forward plan without outer passes has exactly one tile per spectrum.
template<bool Weighted, bool WarpTail, class Shape, class Indices, class Digit>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WarpTail ? 3 : 1)
void stage1_ntt_packed_forward(const uint32_t* state, unsigned long long* out,
    Shape s, Indices in, const Digit* weighted, int tile, const unsigned long long* twiddles)
{
    using U = unsigned long long;
    extern __shared__ U sm[];
    const auto row = size_t(in.x[blockIdx.y / s.curves]) * s.curves + blockIdx.y % s.curves;
    const auto* source = state + row * s.words;
    for (unsigned i = threadIdx.x; i < s.length; i += blockDim.x)
        sm[i] = stage1_ntt_pack_digit<Weighted>(source, s, weighted, i);
    __syncthreads();
    ntt_tile_pass<false, WarpTail>(out + size_t(blockIdx.y) * s.length, sm, s.length, tile, twiddles);
}

// AA + (M - BB) in the weighted digit basis. The modulus spectrum has
// stride zero across the batch. Digits stay nonnegative before convolution.
__device__ __forceinline__ unsigned long long stage1_ntt_loose_difference(
    unsigned long long aa, unsigned long long bb, unsigned long long modulus)
{
    return gl_add_dev(aa, gl_sub_dev(modulus, bb));
}
