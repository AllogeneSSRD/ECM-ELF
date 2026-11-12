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

template<bool WarpTail, bool InPlace, class Read>
__device__ __forceinline__ void stage1_ntt_forward_from(Read read, unsigned long long* out,
    unsigned long long* sm, int tile, const unsigned long long* twiddles)
{
    using U = unsigned long long;
    const unsigned lane = threadIdx.x;
    const unsigned size = 1u << tile;
    if (tile == 1) {
        if (lane == 0) {
            const auto a = read(0), b = read(1);
            out[0] = gl_add_dev(a, b);
            out[1] = gl_sub_dev(a, b);
        }
        return;
    }
    if constexpr (WarpTail) {
        if (tile == 5) {
            const auto v = tile_warp5<false>(read(lane), twiddles);
            out[lane] = v;
            return;
        }
        if (tile == 6) {
            U lo = read(lane), hi = read(lane + 32);
            tile_warp6_pair<false>(lo, hi, twiddles);
            if constexpr (InPlace) __syncwarp();
            *reinterpret_cast<ulonglong2*>(out + 2 * lane) = make_ulonglong2(lo, hi);
            return;
        }
        if (tile == 7) {
            for (unsigned j = lane; j < 64; j += blockDim.x) {
                const auto a = read(j), b = read(j + 64);
                sm[j] = gl_add_dev(a, b);
                sm[j + 64] = gl_mul(gl_sub_dev(a, b), twiddles[63 + j]);
            }
            __syncthreads();
            ntt_tile_finish_forward<true>(out, sm, size, 5, tile, twiddles);
            return;
        }
    }
    const unsigned quarter = size >> 2, half = quarter << 1;
    const auto* first = twiddles + half - 1;
    const auto* second = twiddles + quarter - 1;
    auto* destination = tile == 2 ? out : sm;
    for (unsigned j = lane; j < quarter; j += blockDim.x) {
        const auto a = read(j), b = read(j + quarter), c = read(j + half), d = read(j + 3 * quarter);
        const auto x = tile_radix4_dif(a, b, c, d, first[j], first[quarter + j], second[j]);
        destination[j] = x.x0;
        destination[j + quarter] = x.x1;
        destination[j + half] = x.x2;
        destination[j + 3 * quarter] = x.x3;
    }
    if (tile == 2) return;
    __syncthreads();
    ntt_tile_finish_forward<WarpTail>(out, sm, size, tile - 3, tile, twiddles);
}

template<bool WarpTail>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WarpTail ? 3 : 1)
void stage1_ntt_forward_tile(unsigned long long* data, size_t stride, int tile,
    const unsigned long long* twiddles)
{
    extern __shared__ unsigned long long sm[];
    auto* values = data + size_t(blockIdx.y) * stride + (size_t(blockIdx.x) << tile);
    stage1_ntt_forward_from<WarpTail,true>(
        [=](unsigned j) { return values[j]; }, values, sm, tile, twiddles);
}

template<bool Weighted, bool WarpTail, class Shape, class Indices, class Digit>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WarpTail ? 3 : 1)
void stage1_ntt_packed_forward_registers(const uint32_t* state, unsigned long long* out,
    Shape s, Indices in, const Digit* weighted, int tile, const unsigned long long* twiddles)
{
    extern __shared__ unsigned long long sm[];
    const auto row = size_t(in.x[blockIdx.y / s.curves]) * s.curves + blockIdx.y % s.curves;
    const auto* source = state + row * s.words;
    stage1_ntt_forward_from<WarpTail,false>(
        [=](unsigned j) { return stage1_ntt_pack_digit<Weighted>(source, s, weighted, j); },
        out + size_t(blockIdx.y) * s.length, sm, tile, twiddles);
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

// Full inverse tiles of at most 32 points. A subgroup reads adjacent pairs
// and writes the two output halves without visiting shared memory.
template<int Bits, bool MultiplyAdd, bool Loose, class Inputs>
__global__ void stage1_ntt_short_inverse(unsigned long long* out,
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const __grid_constant__ Inputs products,
#else
    const Inputs products,
#endif
    unsigned curves, const unsigned long long* twiddles, unsigned long long scale)
{
    constexpr unsigned length = 1u << Bits;
    constexpr unsigned half = length / 2;
    const unsigned op = blockIdx.y;
    const unsigned curve = (blockIdx.x * blockDim.x + threadIdx.x) / half;
    const unsigned mask = __ballot_sync(__activemask(), curve < curves);
    if (curve >= curves) return;
    const unsigned lane = threadIdx.x & (half - 1);
    const size_t base = size_t(curve) * length;
    const unsigned pair = 2 * lane;
    const auto index = base + pair;
    auto v = stage1_ntt_product_pair<MultiplyAdd, Loose>(
        products, products.left[op] + index, products.right[op] + index, index,
        pair, op + 1 == gridDim.y, scale);
    tile_warp6_pair<true, Bits>(v.x, v.y, twiddles, mask);
    auto* row = out + (size_t(op) * curves + curve) * length;
    row[lane] = v.x;
    row[lane + half] = v.y;
}
