#pragma once

// Borrowed forward twiddles for the highest DIF bit. With no outer pass,
// fine points at the dense last stage of the full-length tile root table.
struct Stage2NttPackWeights {
    const unsigned long long *fine = nullptr, *coarse = nullptr;
    unsigned shift = 0;
};

#ifdef __CUDACC__
#include "../../src/ecm/cuda/arithmetic/goldilocks/goldilocks.cuh"

__device__ __forceinline__ unsigned long long stage2_ntt_weight_digit(
    unsigned long long value, unsigned digit, Stage2NttPackWeights weights)
{
    if (!value) return 0;
    if (!weights.coarse) return gl_mul(value, weights.fine[digit]);
    const auto mask=(1u<<weights.shift)-1;
    const auto root=gl_mul(weights.fine[digit&mask],weights.coarse[digit>>weights.shift]);
    return gl_mul(value,root);
}
#endif
