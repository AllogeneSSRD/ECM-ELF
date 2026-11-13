#pragma once

// Internal IFMA product kernels. Requires AVX512-IFMA compilation.
#include "modular_ifma.h"

#include <immintrin.h>
#include <string.h>

namespace ecm_ifma_karatsuba {

inline constexpr size_t max_limbs = 316;
inline constexpr size_t split_min_limbs = 12;
inline constexpr uint64_t limb_mask = (uint64_t{1} << 52) - 1;

inline constexpr unsigned mul_depth(size_t n)
{
    return n >= 82 && n <= max_limbs ? 2 : 0;
}

inline constexpr unsigned sqr_depth(size_t n)
{
    return n >= 158 && n <= max_limbs ? 2 : 0;
}

inline size_t workspace_columns(size_t n, unsigned depth)
{
    size_t cols = 0;
    while (depth-- && n >= split_min_limbs) {
        n = (n + 1) / 2 + 1;  // The middle product is the widest child.
        cols += 4 * n;         // Two n-limb inputs plus the 2n-limb product.
    }
    return cols;
}

static inline __m512i load(const uint64_t *p) { return _mm512_load_si512(p); }
static inline void store(uint64_t *p, __m512i v) { _mm512_store_si512(p, v); }

/* Schoolbook product of two `len`-limb operands into t[0 .. 2*len-1], which must be
   zeroed.  Two rows per iteration and one b-vector feeding four madds -- the same
   shape as the production kernel, so the base case is not handicapped. */
static void prod_rows(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t len,
                      __m512i zero)
{
    size_t i = 0;
    for (; i + 1 < len; i += 2) {
        const __m512i a0 = load(a + 8 * i), a1 = load(a + 8 * (i + 1));
        const __m512i b0 = load(b);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0);
        store(t + 8 * i, _mm512_madd52lo_epu64(load(t + 8 * i), a0, b0));
        store(t + 8 * (i + 1), _mm512_madd52lo_epu64(load(t + 8 * (i + 1)), a1, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = load(b + 8 * r);
            __m512i c0 = _mm512_add_epi64(load(t + 8 * (i + r)), h0);
            store(t + 8 * (i + r), _mm512_madd52lo_epu64(c0, a0, br));
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c1 = _mm512_add_epi64(load(t + 8 * (i + 1 + r)), h1);
            store(t + 8 * (i + 1 + r), _mm512_madd52lo_epu64(c1, a1, br));
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
        }
        store(t + 8 * (i + len), _mm512_add_epi64(load(t + 8 * (i + len)), h0));
        store(t + 8 * (i + 1 + len), _mm512_add_epi64(load(t + 8 * (i + 1 + len)), h1));
    }
    if (i < len) {
        const __m512i ai = load(a + 8 * i);
        const __m512i b0 = load(b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        store(t + 8 * i, _mm512_madd52lo_epu64(load(t + 8 * i), ai, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = load(b + 8 * r);
            const __m512i acc = _mm512_add_epi64(load(t + 8 * (i + r)), hi);
            store(t + 8 * (i + r), _mm512_madd52lo_epu64(acc, ai, br));
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        store(t + 8 * (i + len), _mm512_add_epi64(load(t + 8 * (i + len)), hi));
    }
}

// The same triangular accumulation as the production square, with the
// double and diagonal left in redundant columns for Karatsuba assembly.
static void square_rows(uint64_t *t, const uint64_t *a, size_t n, __m512i zero)
{
    /* ---- off-diagonal (strict upper triangle) --------------------------- */
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        const __m512i a0 = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i a1 = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
        {   /* row i, j = i+1 -> columns 2i+1 (lo), 2i+2 (hi) */
            const __m512i aj = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
            _mm512_store_si512((__m512i *)(t + 8 * (2 * i + 1)),
                _mm512_madd52lo_epu64(
                    _mm512_load_si512((const __m512i *)(t + 8 * (2 * i + 1))), a0, aj));
            _mm512_store_si512((__m512i *)(t + 8 * (2 * i + 2)),
                _mm512_madd52hi_epu64(
                    _mm512_load_si512((const __m512i *)(t + 8 * (2 * i + 2))), a0, aj));
        }
        __m512i h0 = zero, h1 = zero;
        for (size_t j = i + 2; j < n; j++) {
            const __m512i aj = _mm512_load_si512((const __m512i *)(a + 8 * j));
            __m512i c0 = _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * (i + j))), h0);
            _mm512_store_si512((__m512i *)(t + 8 * (i + j)),
                               _mm512_madd52lo_epu64(c0, a0, aj));
            h0 = _mm512_madd52hi_epu64(zero, a0, aj);
            __m512i c1 = _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + j))), h1);
            _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + j)),
                               _mm512_madd52lo_epu64(c1, a1, aj));
            h1 = _mm512_madd52hi_epu64(zero, a1, aj);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + n))), h0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + n))), h1));
    }
    for (size_t i = 0; i < n; ++i) {
        const __m512i a_i = load(a + 8 * i);
        uint64_t *lo = t + 16 * i;
        uint64_t *hi = lo + 8;
        store(lo, _mm512_madd52lo_epu64(_mm512_slli_epi64(load(lo), 1), a_i, a_i));
        store(hi, _mm512_madd52hi_epu64(_mm512_slli_epi64(load(hi), 1), a_i, a_i));
    }
}

struct Scratch {
    uint64_t *next;

    uint64_t *take(size_t cols)
    {
        uint64_t *p = next;
        next += 8 * cols;
        return p;
    }
};

static void add_halves(uint64_t *out, const uint64_t *a, size_t h, size_t m,
                       __m512i mask, __m512i zero)
{
    __m512i carry = zero;
    for (size_t i = 0; i < h; ++i) {
        const __m512i hi = i < m ? load(a + 8 * (i + h)) : zero;
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(load(a + 8 * i), hi), carry);
        store(out + 8 * i, _mm512_and_si512(s, mask));
        carry = _mm512_srli_epi64(s, 52);
    }
    store(out + 8 * h, carry);
}

// Write 2n redundant product columns. A split writes a0*b0 and a1*b1
// directly into the final low/high regions, preserving their unequal sizes.
// Each recursion reuses the same storage after its children return.
template<bool square>
static void product(uint64_t *out, const uint64_t *a, const uint64_t *b, size_t n, unsigned depth,
                    Scratch scratch, __m512i mask, __m512i zero)
{
    if (depth == 0 || n < split_min_limbs) {
        memset(out, 0, 2 * n * 64);
        if constexpr (square) square_rows(out, a, n, zero);
        else prod_rows(out, a, b, n, zero);
        return;
    }
    const size_t h = (n + 1) / 2;
    const size_t m = n - h;
    uint64_t *low = out;
    uint64_t *high = out + 16 * h;
    product<square>(low, a, b, h, depth - 1, scratch, mask, zero);
    product<square>(high, a + 8 * h, b + 8 * h, m, depth - 1, scratch, mask, zero);

    uint64_t *middle = scratch.take(2 * (h + 1));
    uint64_t *sa = scratch.take(h + 1);
    add_halves(sa, a, h, m, mask, zero);
    uint64_t *sb = sa;
    if constexpr (!square) {
        sb = scratch.take(h + 1);
        add_halves(sb, b, h, m, mask, zero);
    }
    product<square>(middle, sa, sb, h + 1, depth - 1, scratch, mask, zero);

    // Subtract before adding back to out: the shifted middle term overlaps
    // both halves, and they are still needed for the complete cross term.
    for (size_t i = 0; i < 2 * m; ++i) {
        const __m512i parts = _mm512_add_epi64(load(low + 8 * i), load(high + 8 * i));
        store(middle + 8 * i, _mm512_sub_epi64(load(middle + 8 * i), parts));
    }
    for (size_t i = 2 * m; i < 2 * h; ++i)
        store(middle + 8 * i, _mm512_sub_epi64(load(middle + 8 * i), load(low + 8 * i)));
    // n >= 12 makes h + 2(h + 1) <= 2n for both parities.
    for (size_t i = 0; i < 2 * (h + 1); ++i)
        store(out + 8 * (h + i), _mm512_add_epi64(load(out + 8 * (h + i)), load(middle + 8 * i)));
}

template<bool square>
static __m512i normalized_product(uint64_t *t, const uint64_t *a, const uint64_t *b,
                                  size_t n, uint64_t *scratch, unsigned depth)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i mask = _mm512_set1_epi64(static_cast<long long>(limb_mask));
    product<square>(t, a, b, n, depth, Scratch{scratch}, mask, zero);

    // Individual redundant columns can be negative, even though the exact
    // product is nonnegative. Arithmetic shift propagates a signed carry.
    // For n <= 316, leaves at depth 2 have at most 81 limbs. Each column has
    // fewer than 162 radix parts. A split sums two terms and subtracts two
    // terms, so the first level stays within ±324 radix parts and the
    // second within ±1296. The signed carry also fits in a 64-bit lane.
    __m512i carry = zero;
    for (size_t i = 0; i < 2 * n; ++i) {
        const __m512i v = _mm512_add_epi64(load(t + 8 * i), carry);
        store(t + 8 * i, _mm512_and_si512(v, mask));
        carry = _mm512_srai_epi64(v, 52);
    }
    return carry;
}

template<bool square>
inline uint64_t madd_count(size_t n, unsigned depth)
{
    if (!depth || n < split_min_limbs)
        return square ? n * (n + 1) : 2 * n * n;
    const size_t h = (n + 1) / 2;
    return madd_count<square>(h, depth - 1)
        + madd_count<square>(n - h, depth - 1)
        + madd_count<square>(h + 1, depth - 1);
}

} // namespace ecm_ifma_karatsuba
