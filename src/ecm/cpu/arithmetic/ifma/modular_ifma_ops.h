#pragma once

#include "modular_ifma.h"

#include <immintrin.h>

namespace ecm_ifma {

// SoA residues have IFMA_LANES lanes per limb and are canonical in both
// reduction modes. Output may alias an input. Addition uses IFMA_LANES*n
// temporary words that must be disjoint from its operands.
inline constexpr uint64_t limb_mask = (uint64_t{1} << 52) - 1;

inline __m512i ld(const uint64_t* p) { return _mm512_loadu_si512(p); }
inline void st(uint64_t* p, __m512i v) { _mm512_storeu_si512(p, v); }

/* r = a + b mod N, canonical (0 <= r < N).
 *
 * `tmp` (8n words) receives the candidate r - N, formed in the same pass; the
 * second pass selects it where the value was >= N.  Selection rule, with cy the
 * carry out of column n-1 and `borrow` the borrow out of (limbs - N):
 *     value >= N  <=>  cy != 0  OR  borrow == 0
 * With a carry out, the wrapped subtraction gives the same canonical result. */
inline void add(uint64_t *r, const uint64_t *a, const uint64_t *b, uint64_t *tmp,
                const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)limb_mask);
    const __m512i zero = _mm512_setzero_si512();
    __m512i cy = zero, borrow = zero;

    for (size_t i = 0; i < n; i++) {
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(ld(a + 8 * i), ld(b + 8 * i)), cy);
        cy = _mm512_srli_epi64(s, 52);
        const __m512i si = _mm512_and_si512(s, mask);
        st(r + 8 * i, si);
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(si, ld(nb + 8 * i)), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(tmp + 8 * i, _mm512_and_si512(d, mask));
    }

    const __mmask8 need = (__mmask8)(_mm512_cmpneq_epi64_mask(cy, zero) |
                                     _mm512_cmpeq_epi64_mask(borrow, zero));
    for (size_t i = 0; i < n; i++)
        st(r + 8 * i, _mm512_mask_blend_epi64(need, ld(r + 8 * i), ld(tmp + 8 * i)));
}

/* r = a - b mod N, canonical.  One pass subtracts with borrow propagation, the
   second adds N back exactly in the lanes that went negative (there the true value
   is a - b + N, and the discarded carry out of column n-1 cancels the borrow). */
inline void sub(uint64_t *r, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)limb_mask);
    const __m512i zero = _mm512_setzero_si512();
    __m512i borrow = zero;

    for (size_t i = 0; i < n; i++) {
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(ld(a + 8 * i), ld(b + 8 * i)), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(r + 8 * i, _mm512_and_si512(d, mask));
    }

    const __mmask8 neg = _mm512_cmpneq_epi64_mask(borrow, zero);
    __m512i carry = zero;
    for (size_t i = 0; i < n; i++) {
        const __m512i addend = _mm512_mask_blend_epi64(neg, zero, ld(nb + 8 * i));
        const __m512i v = _mm512_add_epi64(_mm512_add_epi64(ld(r + 8 * i), addend), carry);
        carry = _mm512_srli_epi64(v, 52);
        st(r + 8 * i, _mm512_and_si512(v, mask));
    }
}

/* Compute both a + b and a - b with one load of each input limb.
   The outputs must be distinct. Either may exactly alias an input.
   tmp is disjoint scratch with IFMA_LANES*n words, as for add. */
inline void addsub(uint64_t *sum, uint64_t *diff, const uint64_t *a, const uint64_t *b,
                   uint64_t *tmp, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)limb_mask);
    const __m512i zero = _mm512_setzero_si512();
    __m512i carry = zero, sum_borrow = zero, diff_borrow = zero;

    for (size_t i = 0; i < 8 * n; i += 8) {
        const __m512i ai = ld(a + i), bi = ld(b + i);
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(ai, bi), carry);
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(ai, bi), diff_borrow);
        carry = _mm512_srli_epi64(s, 52);
        diff_borrow = _mm512_srli_epi64(d, 63);

        const __m512i low_sum = _mm512_and_si512(s, mask);
        const __m512i reduced = _mm512_sub_epi64(_mm512_sub_epi64(low_sum, ld(nb + i)), sum_borrow);
        sum_borrow = _mm512_srli_epi64(reduced, 63);
        st(sum + i, low_sum);
        st(diff + i, _mm512_and_si512(d, mask));
        st(tmp + i, _mm512_and_si512(reduced, mask));
    }

    const __mmask8 sum_reduce = (__mmask8)(_mm512_cmpneq_epi64_mask(carry, zero) |
                                            _mm512_cmpeq_epi64_mask(sum_borrow, zero));
    const __mmask8 diff_negative = _mm512_cmpneq_epi64_mask(diff_borrow, zero);
    carry = zero;
    for (size_t i = 0; i < 8 * n; i += 8) {
        st(sum + i, _mm512_mask_blend_epi64(sum_reduce, ld(sum + i), ld(tmp + i)));
        const __m512i addend = _mm512_maskz_loadu_epi64(diff_negative, nb + i);
        const __m512i d = _mm512_add_epi64(_mm512_add_epi64(ld(diff + i), addend), carry);
        carry = _mm512_srli_epi64(d, 52);
        st(diff + i, _mm512_and_si512(d, mask));
    }
}

/* r = -a mod N  (0 stays 0, otherwise N - a). */
inline void neg(uint64_t *r, const uint64_t *a, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)limb_mask);
    __m512i borrow = _mm512_setzero_si512();
    /* "a != 0" is the OR over limbs of "limb != 0".  ANDing them would mean
       "every limb is non-zero", which silently zeroes N - a for any sparse a
       (e.g. a = R mod N, or a squared value with a zero limb). */
    __mmask8 nz = 0;
    for (size_t i = 0; i < n; i++) {
        const __m512i ai = ld(a + 8 * i), nk = ld(nb + 8 * i);
        nz = (__mmask8)(nz | ~_mm512_cmpeq_epi64_mask(ai, _mm512_setzero_si512()));
        __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(nk, ai), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(r + 8 * i, _mm512_and_si512(d, mask));
    }
    /* a == 0 gave N, which must be 0 */
    for (size_t i = 0; i < n; i++)
        st(r + 8 * i, _mm512_maskz_mov_epi64(nz, ld(r + 8 * i)));
}

} // namespace ecm_ifma
