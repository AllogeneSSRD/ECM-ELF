#pragma once

#include "../../src/ecm/cuda/arithmetic/goldilocks/goldilocks.cuh"
#include "../../src/ecm/cuda/ntt/device/tile_pass.cuh"

template<bool Weighted, bool Modulus = false, class Shape, class Digit, class Read>
__device__ __forceinline__ unsigned long long stage1_ntt_pack_digit_from(
    Read read, Shape s, const Digit* weighted, unsigned j)
{
    using U = unsigned long long;
    if (j >= s.digits) return 0;
    const auto digit = Weighted ? weighted[j] : Digit{1, j * s.bpw, s.bpw};
    const U mask = (1ull << digit.width) - 1;
    if constexpr (Modulus) {
        return gl_mul(mask, digit.weight);
    } else {
        const unsigned word = digit.bit / 32, offset = digit.bit % 32;
        U v = read(word);
        if (word + 1 < s.words) v |= U(read(word + 1)) << 32;
        v = (v >> offset) & mask;
        if constexpr (Weighted) v = gl_mul(v, digit.weight);
        return v;
    }
}

template<bool Weighted, bool Modulus = false, class Shape, class Digit>
__device__ __forceinline__ unsigned long long stage1_ntt_pack_digit(
    const uint32_t* row, Shape s, const Digit* weighted, unsigned j)
{
    return stage1_ntt_pack_digit_from<Weighted,Modulus>(
        [=](unsigned word) { return row[word]; }, s, weighted, j);
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

// The inverse warp consumes adjacent even/odd coefficients per lane. All
// spectrum rows and pair offsets are 16-byte aligned for this path.
template<bool MultiplyAdd, bool Loose, class Inputs>
__device__ __forceinline__ ulonglong2 stage1_ntt_product_pair(const Inputs& inputs,
    const unsigned long long* left, const unsigned long long* right,
    size_t offset, unsigned modulus_index, bool last, unsigned long long scale)
{
    using U = unsigned long long;
    const auto load = [](const U* ptr) { return *reinterpret_cast<const ulonglong2*>(ptr); };
    ulonglong2 a, bb{};
    if constexpr (Loose) {
        if (last) {
            bb = load(inputs.bb + offset);
            const auto aa = load(inputs.aa + offset);
            const auto modulus = load(inputs.modulus + modulus_index);
            a = make_ulonglong2(stage1_ntt_loose_difference(aa.x, bb.x, modulus.x),
                                stage1_ntt_loose_difference(aa.y, bb.y, modulus.y));
            *reinterpret_cast<ulonglong2*>(inputs.cached_e + offset) = a;
        } else a = load(left);
    } else a = load(left);
    const auto b = left == right ? a : load(right);
    U lo = gl_mul(a.x, b.x), hi = gl_mul(a.y, b.y);
    if constexpr (MultiplyAdd) {
        if (last) {
            if constexpr (!Loose) bb = load(inputs.bb + offset);
            lo = gl_add_dev(lo, bb.x);
            hi = gl_add_dev(hi, bb.y);
        }
    }
    return scale == 1 ? make_ulonglong2(lo, hi)
                      : make_ulonglong2(gl_mul(lo, scale), gl_mul(hi, scale));
}
