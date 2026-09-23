/* ---------------------------------------------------------------------------
 * simd_mont_ifma.cpp ??see simd_mont_ifma.h for the data layout contract.
 *
 * Algorithm: word-level CIOS (Coarsely Integrated Operand Scanning) in radix
 * 2^52, with the 8 lanes being 8 independent curves, so the interleaved
 * multiply/reduce dependency of CIOS is *not* a serialization problem here:
 * every lane walks the same column index i in lockstep and each lane's
 * reduction digit depends only on its own column i.
 *
 * Why CIOS and not SOS: SOS needs 5n^2 madds (a*b, then m = t*Np mod B^n, then
 * m*N) while CIOS needs 4n^2 + 3n, and the reduction digit m_i = t_i * np0
 * mod B costs a single madd per row instead of a whole low product.
 *
 * Per row i the accumulator window is columns [i, i+n], i.e. exactly the
 * columns that row i multiplies into and that the reduction of digit i adds
 * into.  Both are fused into one sweep, so every column is read once and
 * written once per row (~1 load + 0.25 store per madd at 4n madds/row), and
 * every column's new value depends only on pre-row values -> the j loop is
 * fully independent across iterations and pipelines freely.
 *
 * The pending carry out of the previously reduced column lives in the `cy`
 * register (never stored back), which is what keeps the window sliding by one
 * column per row with no shifting of the accumulator.
 *
 * madds per batch mont_mul = n*(4n+3).
 * ------------------------------------------------------------------------- */

#include "simd_mont_ifma.h"

/* MSVC's /arch:AVX512 is a single "AVX512" level and does not define
   __AVX512IFMA__ even though it does accept the vpmadd52 intrinsics, so only
   the GCC/Clang path can assert on the IFMA macro. */
#if !defined(__AVX512F__)
#error "simd_mont_ifma.cpp must be compiled with AVX512 (/arch:AVX512 or -mavx512f)"
#endif
#if !defined(__AVX512IFMA__) && !defined(_MSC_VER)
#error "simd_mont_ifma.cpp must be compiled with AVX512-IFMA (-mavx512ifma)"
#endif

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IFMA_MASK52 0x000FFFFFFFFFFFFFULL

static void *ifma_alloc(size_t bytes)
{
    return _mm_malloc(bytes ? bytes : 64, 64);
}

static void ifma_free(void *p)
{
    if (p) _mm_free(p);
}

/* ---- mpz <-> 52-bit limbs (cold: setup and result extraction) ------------
 * mpz_get_ui() returns `unsigned long`, which is only 32 bits on Windows x64,
 * so it silently truncates a 64-bit limb.  Every limb here is >= 2^32 in
 * practice, so it must go through mpz_export().  (This exact mistake made the
 * modulus, the Montgomery one and every input lose their top 32 bits.)
 * ------------------------------------------------------------------------ */
#define IFMA_EXTRACT_CHUNK 64   /* bits pulled out of the mpz per step */

static uint64_t mpz_low_word(const mpz_t v, unsigned bits)
{
    mpz_t w;
    unsigned char buf[8];
    size_t count = 0;
    uint64_t word = 0;

    mpz_init(w);
    mpz_fdiv_r_2exp(w, v, bits);
    memset(buf, 0, sizeof(buf));
    mpz_export(buf, &count, -1, sizeof(uint64_t), 0, 0, w);
    if (count > 0) memcpy(&word, buf, sizeof(word));
    mpz_clear(w);
    return word;
}

static void mpz_extract_limbs52(uint64_t *dst, size_t n, const mpz_t v)
{
    mpz_t t;
    mpz_init_set(t, v);
    for (size_t i = 0; i < n; i++) {
        dst[i] = mpz_low_word(t, 52) & IFMA_MASK52;
        mpz_fdiv_q_2exp(t, t, 52);
    }
    mpz_clear(t);
}

/* limb (52 bits) -> mpz: mpz_add_ui() also takes unsigned long, so feed it two
   26-bit halves instead of one truncated 52-bit value.  The two 26-bit shifts
   add up to the single 52-bit shift the caller expects. */
static void mpz_add_limb52(mpz_t out, uint64_t limb)
{
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)((limb >> 26) & 0x3FFFFFFu));
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)(limb & 0x3FFFFFFu));
}

int ifma_ctx_init(ifma_ctx_t *c, const mpz_t N)
{
    return ifma_ctx_init_ex(c, N, IFMA_FIELD_AUTO);
}

/* N = 2^k - 1 ?  Returns k (>= 64) on success, 0 otherwise.  Requires N odd */
static size_t mersenne_k(const mpz_t N)
{
    mpz_t t;
    size_t k = 0;
    mpz_init(t);
    mpz_add_ui(t, N, 1);
    if (mpz_popcount(t) == 1) {
        const mp_bitcnt_t b = mpz_scan1(t, 0);
        if (b >= 64) k = (size_t)b;
    }
    mpz_clear(t);
    return k;
}

int ifma_ctx_init_ex(ifma_ctx_t *c, const mpz_t N, int field_mode)
{
    memset(c, 0, sizeof(*c));
    c->ok = 0;
    c->requested = field_mode;
    c->mode = IFMA_FIELD_MONT;
    if (mpz_sgn(N) <= 0 || mpz_even_p(N)) return -1;
    if (mpz_cmp_ui(N, 1) <= 0) return -2;

    const size_t bits = mpz_sizeinbase(N, 2);
    const size_t n = (bits + 51) / 52;

    if (field_mode == IFMA_FIELD_MERS || field_mode == IFMA_FIELD_AUTO) {
        const size_t k = mersenne_k(N);
        if (k) {
            c->mode = IFMA_FIELD_MERS;
            c->k = k;
            c->shift = (unsigned)(52 * n - k);      /* in [0,52) by n = ceil(k/52) */
        } else if (field_mode == IFMA_FIELD_MERS) {
            return -4;                              /* caller demanded it */
        }
    }

    /* c->N is initialized before c->n is published, so ifma_ctx_clear() is
       always safe to call on a half-built context. */
    mpz_init_set(c->N, N);
    c->n = n;

    c->nb = (uint64_t *)ifma_alloc(8 * n * sizeof(uint64_t));
    c->one = (uint64_t *)ifma_alloc(8 * n * sizeof(uint64_t));
    c->scratch = (uint64_t *)ifma_alloc(8 * (2 * n + 2) * sizeof(uint64_t));
    if (!c->nb || !c->one || !c->scratch) {
        ifma_ctx_clear(c);
        return -3;
    }

    /* Broadcast modulus: N is identical in all 8 lanes. */
    {
        uint64_t *limbs = (uint64_t *)calloc(n, sizeof(uint64_t));
        if (!limbs) { ifma_ctx_clear(c); return -3; }
        mpz_extract_limbs52(limbs, n, N);
        for (size_t j = 0; j < n; j++)
            for (unsigned k = 0; k < IFMA_LANES; k++)
                c->nb[8 * j + k] = limbs[j];
        free(limbs);
    }

    if (c->mode == IFMA_FIELD_MERS) {
        /* Plain domain: the identity element is 1, not R mod N, and no
           Montgomery inverse is needed at all. */
        memset(c->one, 0, 8 * n * sizeof(uint64_t));
        for (unsigned lane = 0; lane < IFMA_LANES; lane++) c->one[lane] = 1;
    } else {
        /* np0 = -N^-1 mod 2^52 (Newton iteration doubles the correct bit count
           each step; N is odd so N itself is correct mod 8 to start with). */
        {
            uint64_t inv = c->nb[0];
            for (int k = 0; k < 6; k++) inv *= (uint64_t)2 - inv * c->nb[0];
            c->np0 = (0ULL - inv) & IFMA_MASK52;
        }

        /* one = R mod N, broadcast into all lanes. */
        {
            mpz_t r, v;
            mpz_inits(r, v, NULL);
            mpz_set_ui(r, 1);
            mpz_mul_2exp(r, r, 52 * (unsigned long)n);
            mpz_mod(v, r, N);
            uint64_t *limbs = (uint64_t *)calloc(n, sizeof(uint64_t));
            if (!limbs) { mpz_clears(r, v, NULL); ifma_ctx_clear(c); return -3; }
            mpz_extract_limbs52(limbs, n, v);
            for (size_t j = 0; j < n; j++)
                for (unsigned k = 0; k < IFMA_LANES; k++)
                    c->one[8 * j + k] = limbs[j];
            free(limbs);
            mpz_clears(r, v, NULL);
        }
    }

    memset(c->scratch, 0, 8 * (2 * n + 2) * sizeof(uint64_t));
    c->ok = 1;
    return 0;
}

const char *ifma_field_name(const ifma_ctx_t *c)
{
    static char buf[96];
    if (c->mode == IFMA_FIELD_MERS) {
        snprintf(buf, sizeof(buf), "mersenne (2^k=1, k=%zu, n52=%zu, fold<<%u)",
                 c->k, c->n, c->shift);
        return buf;
    }
    return "montgomery (R=2^52n)";
}

void ifma_ctx_clear(ifma_ctx_t *c)
{
    ifma_free(c->nb);
    ifma_free(c->one);
    ifma_free(c->scratch);
    if (c->n) mpz_clear(c->N);
    memset(c, 0, sizeof(*c));
}

void ifma_set_zero(uint64_t *e, const ifma_ctx_t *c)
{
    memset(e, 0, 8 * c->n * sizeof(uint64_t));
}

void ifma_set_one(uint64_t *e, const ifma_ctx_t *c)
{
    memcpy(e, c->one, 8 * c->n * sizeof(uint64_t));
}

/* ---- mpz <-> SoA helpers (cold: used for setup and result extraction) ---- */

static void ifma_limbs_to_lane(uint64_t *e, unsigned lane, const mpz_t v, const ifma_ctx_t *c)
{
    uint64_t *limbs = (uint64_t *)calloc(c->n, sizeof(uint64_t));
    if (!limbs) return;
    mpz_extract_limbs52(limbs, c->n, v);
    for (size_t i = 0; i < c->n; i++) e[8 * i + lane] = limbs[i];
    free(limbs);
}

static void ifma_lane_to_limbs(mpz_t out, const uint64_t *e, unsigned lane, const ifma_ctx_t *c)
{
    mpz_set_ui(out, 0);
    for (size_t i = c->n; i-- > 0; )
        mpz_add_limb52(out, e[8 * i + lane] & IFMA_MASK52);
}

void ifma_from_mpz_lane(uint64_t *e, unsigned lane, const mpz_t v, const ifma_ctx_t *c)
{
    mpz_t t, r;
    mpz_inits(t, r, NULL);
    mpz_mod(t, v, c->N);
    if (c->mode == IFMA_FIELD_MERS) {   /* plain domain: no R */
        ifma_limbs_to_lane(e, lane, t, c);
        mpz_clears(t, r, NULL);
        return;
    }
    mpz_set_ui(r, 1);
    mpz_mul_2exp(r, r, 52 * (unsigned long)c->n);
    mpz_mod(r, r, c->N);            /* R mod N */
    mpz_mul(t, t, r);
    mpz_mod(t, t, c->N);            /* v*R mod N */
    ifma_limbs_to_lane(e, lane, t, c);
    mpz_clears(t, r, NULL);
}

void ifma_from_u64_lane(uint64_t *e, unsigned lane, uint64_t v, const ifma_ctx_t *c)
{
    mpz_t t;
    mpz_init_set_ui(t, v);
    ifma_from_mpz_lane(e, lane, t, c);
    mpz_clear(t);
}

void ifma_to_mpz_lane(mpz_t out, const uint64_t *e, unsigned lane, const ifma_ctx_t *c)
{
    if (c->mode == IFMA_FIELD_MERS) {   /* plain domain: already an integer */
        ifma_lane_to_limbs(out, e, lane, c);
        mpz_mod(out, out, c->N);
        return;
    }
    mpz_t v, r, rinv;
    mpz_inits(v, r, rinv, NULL);
    ifma_lane_to_limbs(v, e, lane, c);
    mpz_set_ui(r, 1);
    mpz_mul_2exp(r, r, 52 * (unsigned long)c->n);
    mpz_invert(rinv, r, c->N);      /* R^-1 mod N (GCD is 1: N is odd) */
    mpz_mul(v, v, rinv);
    mpz_mod(out, v, c->N);
    mpz_clears(v, r, rinv, NULL);
}

uint64_t ifma_madd_count(const ifma_ctx_t *c)
{
    if (c->mode == IFMA_FIELD_MERS) return 2 * (uint64_t)c->n * (uint64_t)c->n;
    return (uint64_t)c->n * (4 * (uint64_t)c->n + 3);
}

/* ---------------------------------------------------------------------------
 * The kernel.
 * ------------------------------------------------------------------------- */
static void ifma_cios(uint64_t *out, const uint64_t *a, const uint64_t *b,
                      const ifma_ctx_t *c)
{
    const size_t n = c->n;
    uint64_t *t = c->scratch;
    const uint64_t *nb = c->nb;

    const __m512i mask = _mm512_set1_epi64((long long)IFMA_MASK52);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i np0v = _mm512_set1_epi64((long long)c->np0);

    /* Column i (and the extra top column) must start at zero; only the columns
       touched below are cleared, i.e. all of them. */
    for (size_t k = 0; k < 2 * n + 2; k++)
        _mm512_store_si512((__m512i *)(t + 8 * k), zero);

    __m512i cy = zero;  /* pending carry out of the last reduced column */

    for (size_t i = 0; i < n; i++) {
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)(b));
        const __m512i nb0 = _mm512_load_si512((const __m512i *)(nb));

        /* --- column i: exact low digit, then the reduction digit m --- */
        __m512i v = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * i)), cy);
        v = _mm512_madd52lo_epu64(v, ai, b0);              /* + lo(a_i*b_0)  */
        __m512i digit = _mm512_and_si512(v, mask);
        __m512i m = _mm512_madd52lo_epu64(zero, digit, np0v);  /* m_i (low 52) */
        __m512i lo0 = _mm512_madd52lo_epu64(zero, m, nb0);     /* lo(m_i*N_0)  */
        /* carry out of column i = (v + lo0 + hi0*B) >> 52, and v+lo0 is an
           exact multiple of B, so its own >> 52 is a single bit. */
        __m512i c1 = _mm512_srli_epi64(v, 52);
        __m512i c2 = _mm512_srli_epi64(_mm512_add_epi64(digit, lo0), 52);
        cy = _mm512_add_epi64(c1, c2);

        /* --- column i+1: high parts of j = 0 --- */
        __m512i acc = _mm512_load_si512((const __m512i *)(t + 8 * (i + 1)));
        acc = _mm512_madd52hi_epu64(acc, ai, b0);
        acc = _mm512_madd52hi_epu64(acc, m, nb0);

        /* --- columns i+1 .. i+n --- */
        for (size_t j = 1; j < n; j++) {
            const __m512i bj = _mm512_load_si512((const __m512i *)(b + 8 * j));
            const __m512i nbj = _mm512_load_si512((const __m512i *)(nb + 8 * j));
            /* low parts -> column i+j */
            acc = _mm512_madd52lo_epu64(acc, ai, bj);
            acc = _mm512_madd52lo_epu64(acc, m, nbj);
            _mm512_store_si512((__m512i *)(t + 8 * (i + j)), acc);
            /* high parts -> column i+j+1 */
            acc = _mm512_load_si512((const __m512i *)(t + 8 * (i + j + 1)));
            acc = _mm512_madd52hi_epu64(acc, ai, bj);
            acc = _mm512_madd52hi_epu64(acc, m, nbj);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)), acc);
    }

    /* --- result = columns n .. 2n-1 (plus the pending carry), normalized --- */
    __m512i cb = cy;                 /* the carry out of column n-1 */
    for (size_t k = n; k < 2 * n; k++) {
        __m512i x = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * k)), cb);
        cb = _mm512_srli_epi64(x, 52);
        _mm512_store_si512((__m512i *)(t + 8 * k), _mm512_and_si512(x, mask));
    }
    const __m512i top = cb;          /* 0 or 1: r may be >= 2^(52n) */

    /* --- conditional subtraction of N over n+1 limbs --- */
    __m512i borrow = zero;
    for (size_t k = 0; k < n; k++) {
        __m512i rk = _mm512_load_si512((const __m512i *)(t + 8 * (n + k)));
        __m512i nk = _mm512_load_si512((const __m512i *)(nb + 8 * k));
        __mmask8 b1 = _mm512_cmplt_epu64_mask(rk, nk);
        __m512i d = _mm512_sub_epi64(rk, nk);
        __mmask8 b2 = _mm512_cmplt_epu64_mask(d, borrow);
        d = _mm512_sub_epi64(d, borrow);
        borrow = _mm512_maskz_set1_epi64((__mmask8)(b1 | b2), 1);
        _mm512_store_si512((__m512i *)(out + 8 * k), d);
    }
    /* r >= N  <=>  top >= borrow (N has no limb n) */
    const __mmask8 need_sub = _mm512_cmpge_epu64_mask(top, borrow);

    /* keep the original where no subtraction was needed */
    for (size_t k = 0; k < n; k++) {
        __m512i orig = _mm512_load_si512((const __m512i *)(t + 8 * (n + k)));
        __m512i sub = _mm512_load_si512((const __m512i *)(out + 8 * k));
        _mm512_store_si512((__m512i *)(out + 8 * k),
                           _mm512_mask_blend_epi64(need_sub, orig, sub));
    }
}

/* ---------------------------------------------------------------------------
 * Mersenne kernel: N = 2^k - 1, so 2^k = 1 (mod N).
 *
 * Radix 2^52, n = ceil(k/52) limbs, sh = 52n - k in [0,52).  Then
 *
 *     B^n = 2^(52n) = 2^(k+sh) = 2^sh   (mod N)
 *     2^(52c) = 2^(52(c-n) + sh)        for every c >= n
 *
 * so a digit sitting at column c >= n folds back to columns c-n and c-n+1
 * with a left shift of sh bits: a shift and an add instead of the whole
 * m_i*N reduction row of CIOS.  A mul therefore costs 2n^2 madds instead of
 * n(4n+3) (1.9-2x fewer), and no Montgomery constant is needed at all ??the
 * field representation is the plain one.
 *
 * Three passes, all O(n) vector ops except the product:
 *   1. product    : n rows; row i accumulates columns [i, i+n] with the same
 *                   register-carried sliding window as ifma_cios, 2n madds.
 *                   Columns are 64-bit lazy: column c receives at most n
 *                   products of 52 bits, so it cannot overflow.
 *   2. normalize  : carry chain over the 2n product columns.
 *   3. fold       : columns n..2n-1 (and the carry) move down by n columns and
 *                   sh bits, then the bits >= k of limb n-1 and any carry out
 *                   of column n-1 keep folding until nothing is left above
 *                   position k; finally N is subtracted if the value reached
 *                   the non-canonical 2^k - 1.
 *
 * The folding shift is exact only for digits < 2^52 (a 2-limb split of
 * d << sh needs d*2^sh < 2^104), which is why the high columns are
 * normalized before they are folded and why column n is normalized again
 * after the last fold lands on it.
 * ------------------------------------------------------------------------- */

/* t[tcol .. tcol+1] += (d * 2^sh) split into two 52-bit limbs.  d < 2^52. */
static inline void ifma_fold_digit(uint64_t *t, size_t tcol, __m512i d,
                                   unsigned sh, __m512i mask)
{
    const __m512i lo = _mm512_and_si512(_mm512_slli_epi64(d, sh), mask);
    const __m512i hi = (sh == 0) ? _mm512_setzero_si512()
                                 : _mm512_srli_epi64(d, 52 - sh);
    _mm512_store_si512((__m512i *)(t + 8 * tcol),
                       _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * tcol)), lo));
    _mm512_store_si512((__m512i *)(t + 8 * (tcol + 1)),
                       _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (tcol + 1))), hi));
}

/* 52-bit normalization of columns [from,to): returns the carry out of to-1. */
static inline __m512i ifma_carry_pass(uint64_t *t, size_t from, size_t to,
                                      __m512i carry, __m512i mask)
{
    for (size_t k = from; k < to; k++) {
        const __m512i v = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * k)), carry);
        carry = _mm512_srli_epi64(v, 52);
        _mm512_store_si512((__m512i *)(t + 8 * k), _mm512_and_si512(v, mask));
    }
    return carry;
}

static void ifma_mersenne_mul(uint64_t *out, const uint64_t *a, const uint64_t *b,
                              const ifma_ctx_t *c)
{
    const size_t n = c->n;
    const unsigned sh = c->shift;
    const __m512i mask = _mm512_set1_epi64((long long)IFMA_MASK52);
    const __m512i zero = _mm512_setzero_si512();
    uint64_t *t = c->scratch;             /* 8*(2n+2) words = 2n+2 columns */

    for (size_t k = 0; k < 2 * n + 2; k++)
        _mm512_store_si512((__m512i *)(t + 8 * k), zero);

    /* ---- 1. product: columns [i, i+n] per row, 2n madds per row ----------
       Two rows are processed per iteration.  A single row reaches only
       ~0.67 madd/cycle at n52=58: each column then costs one load, one store
       and one add for its two madds, i.e. as much memory traffic as madd work
       (measured with tools/bench/mers_loop_bench.cpp).  Fusing two rows keeps
       one b-vector for four madds and both rows' columns are independent, so
       the loop runs at 1.0 madd/cycle = the hardware roof. */
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        const __m512i a0 = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i a1 = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), a0, b0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1)),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1))), a1, b0));
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            __m512i c0 = _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * (i + r))), h0);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)),
                               _mm512_madd52lo_epu64(c0, a0, br));
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c1 = _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + r))), h1);
            _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + r)),
                               _mm512_madd52lo_epu64(c1, a1, br));
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + n))), h0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + n))), h1));
    }
    if (i < n) {                                    /* odd n: one row left */
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), ai, b0));
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            const __m512i acc = _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * (i + r))), hi);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)),
                               _mm512_madd52lo_epu64(acc, ai, br));
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + n))), hi));
    }

    if (getenv("IFMA_NOTail")) { for (size_t q = 0; q < 8 * n; q++) out[q] = t[q]; return; }
    /* ---- 2. normalize the 2n product columns ---------------------------- */
    __m512i top = ifma_carry_pass(t, 0, 2 * n, zero, mask);

    /* ---- 3. fold the high columns down, highest first ------------------- */
    /* The carry out of column 2n-1 is a digit at column 2n -> columns n, n+1. */
    ifma_fold_digit(t, n, top, sh, mask);
    /* Columns 2n-1 .. n+1 are 52-bit digits now; each lands at c-n, c-n+1.
       Column n is folded last because column 2n-1 also lands on it. */
    for (size_t cc = 2 * n - 1; cc > n; cc--) {
        const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * cc));
        _mm512_store_si512((__m512i *)(t + 8 * cc), zero);
        ifma_fold_digit(t, cc - n, d, sh, mask);
    }
    /* Column n holds p_n plus the fold of column 2n-1: normalize it first so
       the folding shift stays exact, and push any overflow to column n+1. */
    {
        const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * n));
        _mm512_store_si512((__m512i *)(t + 8 * n), zero);
        ifma_fold_digit(t, 0, _mm512_and_si512(d, mask), sh, mask);
        const __m512i extra = _mm512_srli_epi64(d, 52);
        if (_mm512_test_epi64_mask(extra, extra) != 0) ifma_fold_digit(t, 1, extra, sh, mask);
    }

    /* ---- 4. drain whatever is left at or above position k ----------------
       The top sh bits of limb n-1 fold back by 2^k = 1, a carry out of limb n-1
       by 2^(52n) = 2^sh.  The masking has to be re-checked AFTER the carry
       chain: the carry can push limb n-1 back up to exactly 2^(52-sh), i.e. one
       bit above k, and the previous version broke out of the loop right there
       and returned a *non-canonical* representative (>= 2^k).  mpz_mod() in
       ifma_to_mpz_lane() hides that from any test that reads values back as
       mpz, but the next field op then sees an operand >= N and produces
       garbage ??that was the "sparse operand" corruption seen in the point
       selftest (k = 677 / 991 at sh = 51 / 49). */
    if (sh != 0) {
        const __m512i lowmask = _mm512_set1_epi64((long long)((1ULL << (52 - sh)) - 1));
        for (int it = 0; it < 6; it++) {
            /* 1. fold the bits >= k of limb n-1 down to limb 0 */
            const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * (n - 1)));
            _mm512_store_si512((__m512i *)(t + 8 * (n - 1)), _mm512_and_si512(d, lowmask));
            _mm512_store_si512((__m512i *)(t + 8 * 0), _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * 0)),
                _mm512_srli_epi64(d, 52 - sh)));
            /* 2. carry chain over the low n limbs */
            const __m512i carry = ifma_carry_pass(t, 0, n, zero, mask);
            /* 3. anything still at/above k?  (re-check the top limb: the carry
                  chain may have pushed it back over 2^k) */
            const __m512i d2 = _mm512_load_si512((const __m512i *)(t + 8 * (n - 1)));
            const __m512i above = _mm512_srli_epi64(d2, 52 - sh);
            if (_mm512_test_epi64_mask(above, above) == 0 &&
                _mm512_test_epi64_mask(carry, carry) == 0) break;
            /* 4. fold the carry (position 52n = 2^sh) and go round again */
            if (_mm512_test_epi64_mask(carry, carry) != 0) ifma_fold_digit(t, 0, carry, sh, mask);
        }
    } else {
        for (int it = 0; it < 6; it++) {
            const __m512i carry = ifma_carry_pass(t, 0, n, zero, mask);
            if (_mm512_test_epi64_mask(carry, carry) == 0) break;
            ifma_fold_digit(t, 0, carry, sh, mask);
        }
    }

    /* ---- 5. canonical: the only value >= N below 2^k is 2^k-1 itself ---- */
    {
        __m512i borrow = zero;
        for (size_t k = 0; k < n; k++) {
            const __m512i rk = _mm512_load_si512((const __m512i *)(t + 8 * k));
            const __m512i nk = _mm512_load_si512((const __m512i *)(c->nb + 8 * k));
            const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(rk, nk), borrow);
            borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
            _mm512_store_si512((__m512i *)(out + 8 * k), d);
        }
        /* borrow is 0/1 per lane (not a mask): 1 means r < N, so N is subtracted
           exactly where the final borrow is 0.  mask_blend takes b where set. */
        const __mmask8 need_sub = _mm512_cmpeq_epi64_mask(borrow, zero);
        for (size_t k = 0; k < n; k++) {
            const __m512i orig = _mm512_load_si512((const __m512i *)(t + 8 * k));
            const __m512i sub = _mm512_load_si512((const __m512i *)(out + 8 * k));
            _mm512_store_si512((__m512i *)(out + 8 * k),
                               _mm512_mask_blend_epi64(need_sub, orig, sub));
        }
    }
}

void ifma_mont_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *c)
{
    if (c->mode == IFMA_FIELD_MERS) { ifma_mersenne_mul(out, a, b, c); return; }
    ifma_cios(out, a, b, c);
}

void ifma_mont_sqr(uint64_t *out, const uint64_t *a, const ifma_ctx_t *c)
{
    if (c->mode == IFMA_FIELD_MERS) { ifma_mersenne_mul(out, a, a, c); return; }
    ifma_cios(out, a, a, c);
}
