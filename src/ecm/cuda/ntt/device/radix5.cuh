#pragma once

#include "../../arithmetic/goldilocks/goldilocks.cuh"

namespace ecm_cuda_ntt {

using Radix5Word = unsigned long long;

struct Radix5Constants {
    Radix5Word cosine2, cosine_delta, sine1, sine_delta, sine_sum;
};

inline Radix5Constants radix5_constants(Radix5Word root) {
    const auto w2 = gl_mul_host(root, root);
    const auto w3 = gl_mul_host(w2, root);
    const auto w4 = gl_mul_host(w3, root);
    constexpr auto half = (GL_P + 1) / 2;
    const auto a = gl_mul_host(gl_add(root, w4), half);
    const auto b = gl_mul_host(gl_add(w2, w3), half);
    const auto c = gl_mul_host(gl_sub(root, w4), half);
    const auto d = gl_mul_host(gl_sub(w2, w3), half);
    return {b, gl_sub(a, b), c, gl_sub(d, c), gl_add(c, d)};
}

// Symmetric/antisymmetric pairs give all five DFT outputs in six products.
__device__ __forceinline__ void radix5_dft(Radix5Word (&x)[5], Radix5Constants c) {
    const auto s1 = gl_add(x[1], x[4]), s2 = gl_add(x[2], x[3]);
    const auto d1 = gl_sub(x[1], x[4]), d2 = gl_sub(x[2], x[3]);
    const auto sum = gl_add(s1, s2);
    const auto even = gl_add(x[0], gl_mul(c.cosine2, sum));
    const auto u = gl_add(even, gl_mul(c.cosine_delta, s1));
    const auto v = gl_add(even, gl_mul(c.cosine_delta, s2));
    const auto odd = gl_mul(c.sine1, gl_add(d1, d2));
    const auto y = gl_add(odd, gl_mul(c.sine_delta, d2));
    const auto z = gl_sub(gl_mul(c.sine_sum, d1), odd);
    x[0] = gl_add(x[0], sum);
    x[1] = gl_add(u, y);
    x[2] = gl_add(v, z);
    x[3] = gl_sub(v, z);
    x[4] = gl_sub(u, y);
}

// Rotate the five planes without changing the index modulo width. The
// resulting coefficient sets are disjoint between threads, including stores.
__device__ __forceinline__ unsigned radix5_plane(unsigned plane, unsigned rotation) {
    const unsigned sum = plane + rotation;
    return sum < 5 ? sum : sum - 5;
}

} // namespace ecm_cuda_ntt
