#pragma once

#include "../../src/ecm/cuda/ntt/device/radix5.cuh"

namespace ecm_mersenne_ntt {

using ecm_cuda_ntt::Radix5Word;
using ecm_cuda_ntt::Radix5Constants;
using ecm_cuda_ntt::radix5_constants;
using ecm_cuda_ntt::radix5_dft;
using ecm_cuda_ntt::radix5_plane;

template<bool Pfa>
__device__ __forceinline__ void radix5_inverse_column(Radix5Word (&x)[5], const Radix5Word* row,
                                                     unsigned width, unsigned column,
                                                     const Radix5Word* twiddles, Radix5Constants c) {
    x[0] = row[column];
#pragma unroll
    for (unsigned i = 1; i < 5; ++i) {
        const auto value = row[i * width + column];
        x[i] = Pfa ? value : gl_mul(value, twiddles[(i - 1) * width + column]);
    }
    radix5_dft(x, c);
}

template<bool Pfa, bool Coalesced = false>
__global__ void radix5_inverse(Radix5Word* data, unsigned width, const Radix5Word* twiddles,
                               Radix5Constants c, unsigned rotation_step) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    auto* row = data + size_t(blockIdx.y) * 5 * width;
    Radix5Word x[5];
    radix5_inverse_column<Pfa>(x, row, width, j, twiddles, c);
    row += j;
    const unsigned rotation = Pfa ? (j * rotation_step) % 5 : 0;
    if constexpr (Coalesced) {
        // Each thread owns one column. The shared permutation keeps the
        // global stores contiguous and never exchanges values between threads.
        extern __shared__ Radix5Word local[];
        auto* column = local + threadIdx.x;
#pragma unroll
        for (unsigned i = 0; i < 5; ++i)
            column[radix5_plane(i, rotation) * blockDim.x] = x[i];
#pragma unroll
        for (unsigned i = 0; i < 5; ++i) row[i * width] = column[i * blockDim.x];
    } else {
#pragma unroll
        for (unsigned i = 0; i < 5; ++i)
            row[(Pfa ? radix5_plane(i, rotation) : i) * width] = x[i];
    }
}

} // namespace ecm_mersenne_ntt
