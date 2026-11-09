#pragma once

#include "ecm/cuda/arithmetic/goldilocks/goldilocks.cuh"

// Standalone Stage2 experiment. Each output pair holds either one zero-padded
// linear input or the cyclic and root-twisted inputs for two half-length NTTs.
namespace ecm::stage2::ntt_twist {

template<bool Twisted>
__global__ void pack(unsigned long long* a, unsigned long long* b,
                     const unsigned long long* input_a, const unsigned long long* input_b,
                     const unsigned long long* roots, unsigned long long half_length)
{
    const auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= half_length) return;
    const auto input = blockIdx.y * half_length + i;
    const auto low = 2 * blockIdx.y * half_length + i;
    const auto high = low + half_length;
    const auto u = input_a[input];
    const auto v = input_b[input];
    a[low] = u;
    b[low] = v;
    if constexpr (Twisted) {
        const auto weight = roots[i];
        a[high] = gl_mul(u, weight);
        b[high] = gl_mul(v, weight);
    } else {
        a[high] = 0;
        b[high] = 0;
    }
}

__device__ __forceinline__ unsigned long long half_field(unsigned long long x)
{
    return (x >> 1) + ((x & 1) ? (GL_P + 1) / 2 : 0);
}

__global__ void merge(unsigned long long* a, const unsigned long long* inverse_roots,
                      unsigned long long half_length)
{
    const auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= half_length) return;
    const auto low = 2 * blockIdx.y * half_length + i;
    const auto high = low + half_length;
    const auto cyclic = a[low];
    const auto negacyclic = gl_mul(a[high], inverse_roots[i]);
    a[low] = half_field(gl_add(cyclic, negacyclic));
    a[high] = half_field(gl_sub(cyclic, negacyclic));
}

} // namespace ecm::stage2::ntt_twist
