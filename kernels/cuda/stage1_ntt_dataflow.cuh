#pragma once

#include "../../src/ecm/cuda/arithmetic/goldilocks/goldilocks.cuh"

// AA + (M - BB) in the weighted digit basis. The modulus spectrum has
// stride zero across the batch. Digits stay nonnegative before convolution.
__device__ __forceinline__ unsigned long long stage1_ntt_loose_difference(
    unsigned long long aa, unsigned long long bb, unsigned long long modulus)
{
    return gl_add_dev(aa, gl_sub_dev(modulus, bb));
}
