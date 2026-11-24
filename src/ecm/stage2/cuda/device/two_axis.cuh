#pragma once

#include "../../../cuda/ntt/device/tile_pass.cuh"
#include "../../../cuda/ntt/parameters.h"
#include "../../../stage1/cuda/ntt/weighted.cuh"

namespace stage2_two_axis {

using Word = unsigned long long;
namespace mersenne = ecm_mersenne_ntt;

// If t = zeta * (b-c), the other branches are (a-c)+t and (a-b)-t.
// This uses zeta^2 = -1-zeta, so both branches share one field multiply.
__device__ __forceinline__ void radix3(Word& a, Word& b, Word& c, Word zeta) {
    const auto t = gl_mul(zeta, gl_sub(b, c));
    const auto u = gl_add(a, gl_add(b, c));
    const auto z = gl_sub(gl_sub(a, b), t);
    b = gl_add(gl_sub(a, c), t);
    c = z;
    a = u;
}

template<bool Radix3>
__global__ void transpose_forward(const Word* input, Word* output, unsigned ly,
                                  unsigned k, unsigned input_rows, unsigned frequencies,
                                  const Word* twiddle, Word zeta) {
    constexpr unsigned planes = Radix3 ? 3 : 1;
    __shared__ Word tile[planes][32][33];
    const unsigned frequency = blockIdx.x * 32 + threadIdx.x;
    const unsigned j = blockIdx.y * 32 + threadIdx.y;
    for (unsigned offset = 0; offset < 32; offset += 8) {
        const unsigned row = j + offset;
        if (row >= k || frequency >= frequencies) continue;
        auto read = [&](unsigned index) -> Word {
            return index < input_rows ? input[size_t(index) * ly + frequency] : 0;
        };
        auto a = read(row);
        if constexpr (Radix3) {
            auto b = read(row + k), c = read(row + 2 * k);
            radix3(a, b, c, zeta);
            tile[1][threadIdx.y + offset][threadIdx.x] = gl_mul(b, twiddle[row]);
            tile[2][threadIdx.y + offset][threadIdx.x] = gl_mul(c, twiddle[k + row]);
        }
        tile[0][threadIdx.y + offset][threadIdx.x] = a;
    }
    __syncthreads();
    const unsigned col_out = blockIdx.y * 32 + threadIdx.x;
    const unsigned row_out = blockIdx.x * 32 + threadIdx.y;
    for (unsigned offset = 0; offset < 32; offset += 8) {
        if (col_out >= k || row_out + offset >= frequencies) continue;
        auto* out = output + size_t(row_out + offset) * planes * k + col_out;
        for (unsigned plane = 0; plane < planes; ++plane)
            out[plane * k] = tile[plane][threadIdx.x][threadIdx.y + offset];
    }
}

template<bool Radix3>
__global__ void transpose_inverse(const Word* input, Word* output, unsigned ly,
                                  unsigned k, unsigned output_rows, unsigned frequencies,
                                  const Word* twiddle, Word zeta, unsigned first = 0) {
    constexpr unsigned planes = Radix3 ? 3 : 1;
    __shared__ Word tile[planes][32][33];
    const unsigned j = blockIdx.x * 32 + threadIdx.x;
    const unsigned frequency = blockIdx.y * 32 + threadIdx.y;
    const unsigned used = first ? k : min(k, output_rows);
    for (unsigned offset = 0; offset < 32; offset += 8) {
        if (j >= used || frequency + offset >= frequencies) continue;
        const auto* row = input + size_t(frequency + offset) * planes * k + j;
        auto a = row[0];
        if constexpr (Radix3) {
            auto b = gl_mul(row[k], twiddle[j]), c = gl_mul(row[2 * k], twiddle[k + j]);
            // The factor 1/3 is combined with the coefficient-axis deweighting.
            radix3(a, b, c, zeta);
            tile[1][threadIdx.y + offset][threadIdx.x] = b;
            tile[2][threadIdx.y + offset][threadIdx.x] = c;
        }
        tile[0][threadIdx.y + offset][threadIdx.x] = a;
    }
    __syncthreads();
    const unsigned col_out = blockIdx.y * 32 + threadIdx.x;
    const unsigned row_out = blockIdx.x * 32 + threadIdx.y;
    for (unsigned offset = 0; offset < 32; offset += 8) {
        if (col_out >= frequencies || row_out + offset >= used) continue;
        for (unsigned plane = 0; plane < planes; ++plane) {
            const unsigned row = plane * k + row_out + offset;
            if (row >= first && row - first < output_rows)
                output[size_t(row - first) * ly + col_out] = tile[plane][threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ __launch_bounds__(FUSE_TILE_THREADS)
void pack_forward(const Word* input, Word* output, unsigned words, unsigned length,
                  unsigned bits, const mersenne::WeightedDigit* digits, const Word* table,
                  unsigned reverse_rows = 0) {
    extern __shared__ Word shared[];
    const unsigned row_index = reverse_rows ? reverse_rows - 1 - blockIdx.x : blockIdx.x;
    const auto* row = input + size_t(row_index) * words;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        const auto digit = digits[i];
        const unsigned limb = digit.bit / 64, offset = digit.bit % 64;
        auto value = row[limb] >> offset;
        if (offset + digit.width > 64) value |= row[limb + 1] << (64 - offset);
        shared[i] = gl_mul(value & ((Word{1} << digit.width) - 1), digit.weight);
    }
    __syncthreads();
    ntt_tile_pass<false, true>(output + size_t(blockIdx.x) * length, shared, length, bits, table);
}

constexpr unsigned carry_threads = 256;

// This is the same cyclic carry transfer as the Stage1 weighted kernel.
// A thread owns consecutive limbs, then receives the prefix of its interval.
__global__ __launch_bounds__(carry_threads)
void normalize(uint32_t* output, const Word* sums, unsigned words, unsigned p, unsigned output_stride = 0) {
    __shared__ unsigned warps[carry_threads / 32];
    const unsigned pitch = output_stride ? output_stride : words;
    auto* dest = output + size_t(blockIdx.x) * pitch;
    const auto* row = sums + size_t(blockIdx.x) * words;
    const unsigned tail_bits = p - 32 * (words - 1);
    const uint32_t tail_mask = uint32_t((Word{1} << tail_bits) - 1);
    const unsigned stride = (words + carry_threads - 1) / carry_threads;
    const unsigned begin = min(threadIdx.x * stride, words), end = min(begin + stride, words);
    auto mask_for = [=](unsigned i) { return i + 1 == words ? tail_mask : ~uint32_t{0}; };
    unsigned tag = 1;
    for (unsigned i = begin; i < end; ++i) {
        const auto mask = mask_for(i);
        const auto v = mersenne::weighted_limb(row, words, tail_bits, i, mask);
        if (v != mask) tag = v > mask ? 2 : 0;
    }
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned stop = __ballot_sync(0xffffffffu, tag != 1);
    const unsigned gen = __ballot_sync(0xffffffffu, tag == 2);
    auto last = [=](unsigned mask) { return mask ? ((gen >> (31 - __clz(mask))) & 1u) * 2 : 1u; };
    if (lane == 31) warps[warp] = last(stop);
    __syncthreads();
    unsigned prefix = 1, whole = 1;
    for (unsigned w = 0; w < carry_threads / 32; ++w) {
        const unsigned transfer = warps[w];
        if (transfer == 1) continue;
        if (w < warp) prefix = transfer;
        whole = transfer;
    }
    const auto lane_prefix = last(stop & ((1u << lane) - 1));
    if (lane_prefix != 1) prefix = lane_prefix;
    unsigned carry = prefix == 2 || (prefix == 1 && whole == 2);
    for (unsigned i = begin; i < end; ++i) {
        const auto mask = mask_for(i);
        const auto v = mersenne::weighted_limb(row, words, tail_bits, i, mask) + carry;
        dest[i] = whole == 1 ? 0 : uint32_t(v) & mask;
        carry = v > mask;
    }
    for (unsigned i = words + threadIdx.x; i < pitch; i += blockDim.x) dest[i] = 0;
}

} // namespace stage2_two_axis
