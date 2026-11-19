#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ecm_mersenne_ntt::stage1_layout {

enum Slot : unsigned {
    X0, Z0, X1, Z1, A24, Base, A, B, C, D, AA, BB, DA, CB,
    Sum, Diff, E, XA, DiffSq, XD, AE, ZA, ZD
};
constexpr unsigned slots = ZD + 1;
struct Indices { unsigned x[5]; };
struct Sums { Indices left, right, out; unsigned subtract; };
struct Shape { unsigned p, words, curves, length, bpw, digits; };
struct Cursor { const uint32_t* bits; const unsigned* start; unsigned offset, count; };

__device__ inline unsigned scalar_bit(Cursor cursor) {
    const unsigned i = cursor.count - 1 - (*cursor.start + cursor.offset);
    return (cursor.bits[i / 32] >> (i % 32)) & 1u;
}

__device__ inline size_t limb(Shape s, unsigned slot, unsigned curve, unsigned word) {
    return (size_t(slot) * s.curves + curve) * s.words + word;
}
__device__ inline uint32_t limb_mask(Shape s, unsigned i) {
    const unsigned rest = s.p % 32;
    return i + 1 == s.words && rest ? (1u << rest) - 1 : 0xffffffffu;
}

__device__ inline uint32_t linear_product_word(const unsigned long long* digits, Shape s, unsigned bit) {
    const unsigned k = bit / s.bpw, offset = bit % s.bpw;
    auto value = digits[k] >> offset;
    for (unsigned shift = s.bpw - offset, j = k + 1; shift < 32 && j < s.length; shift += s.bpw, ++j)
        value |= digits[j] << shift;
    return uint32_t(value);
}

__device__ inline unsigned long long linear_fold_limb(const unsigned long long* digits, Shape s, unsigned i,
                                                        uint32_t mask) {
    const unsigned bit = 32 * i;
    return static_cast<unsigned long long>(linear_product_word(digits,s,bit) & mask) +
           (linear_product_word(digits,s,s.p + bit) & mask);
}

template<bool Routed>
__device__ inline unsigned output_slot(unsigned out, Cursor cursor) {
    if constexpr (Routed) {
        if (out == XA || out == XD || out == ZA || out == ZD) {
            const unsigned bit = scalar_bit(cursor);
            const bool doubled = out == XD || out == ZD;
            return 2 * (doubled ? bit : 1 - bit) + unsigned(out == ZA || out == ZD);
        }
    }
    return out;
}

__device__ inline unsigned long long add_limb(const uint32_t* state, Shape s, unsigned a, unsigned b,
                                              unsigned curve, unsigned i, uint32_t mask, bool subtract) {
    const auto rhs = state[limb(s,b,curve,i)];
    return static_cast<unsigned long long>(state[limb(s,a,curve,i)]) + (subtract ? (rhs ^ mask) : rhs);
}

} // namespace ecm_mersenne_ntt::stage1_layout
