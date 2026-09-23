/* Scalar CIOS Montgomery multiplication reference.
 * Algorithm matches OpenCL mont_mul_priv_opt.cl (generic priv_opt).
 * Uses uint64_t for intermediate products; portable C++11.
 *
 * Inputs:  a, b, N  — limb arrays (uint32_t), each of <limbs> elements.
 *          np0       — Montgomery constant (-N^{-1} mod 2^32).
 *          limbs     — number of limbs.
 * Outputs: out       — limb array, result a*b*R^{-1} mod N (R=2^{32*limbs}).
 */

#include "cpu_mont_scalar.h"
#include <cstring>

void cpu_mont_np0_compute(uint32_t *np0_out, const uint32_t *N, uint32_t limbs) {
    // Compute np0 = -N[0]^{-1} mod 2^32 via extended Euclidean algorithm.
    // N[0] is guaranteed odd by caller.
    uint32_t x0 = 1u;
    uint32_t b0 = N[0];
    uint32_t x1 = 0u;
    uint32_t b1 = 0u;  // the modulus is 2^32, represented as 0
    for (int i = 0; i < 32; ++i) {
        if (x0 & 1u) {
            x0 = x0 - b0;
            x1 = x1 - b1;
        }
        x0 >>= 1u;
        x1 = (x1 >> 1u) | ((x0 & 1u) ? 0x80000000u : 0u);
    }
    *np0_out = x1;
}

void cpu_mont_scalar_cios(uint32_t *out, const uint32_t *a, const uint32_t *b,
                          const uint32_t *N, uint32_t np0, uint32_t limbs) {
    if (limbs == 0 || limbs > CPU_MONT_MAX_LIMBS) {
        return;
    }

    uint32_t t[CPU_MONT_MAX_LIMBS + 2];
    uint32_t B[CPU_MONT_MAX_LIMBS];

    std::memset(t, 0, (limbs + 2) * sizeof(uint32_t));
    std::memcpy(B, b, limbs * sizeof(uint32_t));

    for (uint32_t i = 0; i < limbs; ++i) {
        uint32_t ai = a[i];

        // Multiply-accumulate: t += ai * B
        uint64_t carry = 0;
        for (uint32_t j = 0; j < limbs; ++j) {
            uint64_t uv = (uint64_t)t[j] + (uint64_t)ai * (uint64_t)B[j] + carry;
            t[j] = (uint32_t)uv;
            carry = uv >> 32;
        }
        uint64_t top = (uint64_t)t[limbs] + carry;
        t[limbs] = (uint32_t)top;
        t[limbs + 1] = (uint32_t)(top >> 32);

        // Montgomery reduction: m = t[0] * np0 (mod 2^32), then t += m * N
        uint32_t m = (uint32_t)((uint64_t)t[0] * (uint64_t)np0);
        carry = 0;
        for (uint32_t j = 0; j < limbs; ++j) {
            uint64_t uv = (uint64_t)t[j] + (uint64_t)m * (uint64_t)N[j] + carry;
            if (j > 0) {
                t[j - 1] = (uint32_t)uv;
            }
            carry = uv >> 32;
        }
        top = (uint64_t)t[limbs] + carry;
        t[limbs - 1] = (uint32_t)top;
        top = (uint64_t)t[limbs + 1] + (top >> 32);
        t[limbs] = (uint32_t)top;
        t[limbs + 1] = (uint32_t)(top >> 32);
    }

    // Final conditional subtract: if t >= N, t -= N
    uint64_t borrow = 0;
    uint32_t D[CPU_MONT_MAX_LIMBS];
    for (uint32_t i = 0; i < limbs; ++i) {
        uint64_t tv = (uint64_t)t[i];
        uint64_t nv = (uint64_t)N[i];
        uint64_t w = tv - nv - borrow;
        D[i] = (uint32_t)w;
        borrow = (tv < nv + borrow) ? 1 : 0;
    }

    uint32_t need_sub = (t[limbs] != 0 || t[limbs + 1] != 0) ? 1 : 0;
    need_sub = (borrow == 0) ? 1 : need_sub;
    uint32_t mask = 0u - need_sub;

    for (uint32_t i = 0; i < limbs; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
}