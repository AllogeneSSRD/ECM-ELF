/* ---------------------------------------------------------------------------
 * mers_karatsuba.cpp -- is a sub-quadratic product worth it in the Mersenne fold
 * domain?
 *
 * Context: our fold-domain multiply is schoolbook, 2n^2 madd instructions for n
 * 52-bit limbs, and it runs at 84-96% of this machine's vpmadd52 roof
 * (docs/ECM_Montgomery_STAGE1.md section 11).  n = bits/52, so n is 25 (M1277),
 * 58 (M3001) or 77 (M4001) -- small, which is exactly where sub-quadratic methods
 * historically struggle to pay off.
 *
 * The repo already contains a GPU Karatsuba (kernels/opencl/mont_mul/
 * mont_mul_karatsuba_2048b.cl) and docs/DEV_COOP_KARATSUBA_2048.md explains why it
 * deliberately does NOT use the 3-multiply identity: "the carry corrections need
 * per-limb propagation, complexity and correctness risk far exceed the gain".  That
 * reasoning is about 32-bit-limb CIOS on a GPU.  Here the situation differs:
 *
 *   * the fold domain accumulates each column in a 64-bit lane, so the Karatsuba
 *     cross terms are just extra add/sub passes over 2h columns -- there is no
 *     per-limb carry propagation until the single final normalize pass;
 *   * a column's intermediate value may go negative, which is harmless because the
 *     accumulated sum is the exact product (>= 0) and every partial stays well
 *     inside 2^63.
 *
 * So the 3-multiply identity should be usable here.  This tool measures whether it
 * actually wins, and by how much, at the limb counts we care about:
 *
 *   a = a0 + a1*B^h, b = b0 + b1*B^h        (h = ceil(n/2))
 *   a*b = m0 + (m1 - m0 - m2)*B^h + m2*B^2h,  m0 = a0b0, m2 = a1b1,
 *                                             m1 = (a0+a1)(b0+b1)
 *
 * Cost model (instructions per batch, before measurement):
 *   schoolbook : 2n^2 madds
 *   Karatsuba-1: 2*(h^2 + h^2 + (h+1)^2) madds + ~5 combine passes over ~2h columns
 *   Karatsuba-2: 9 base products of ~n/4 + 2x the combine work
 *
 * Correctness is checked against the production ifma_mont_mul on random canonical
 * operands (byte-compare of the whole 8n-limb element), so the prototype cannot
 * silently "win" by being wrong.
 *
 * usage: mers_karatsuba [k] [levels] [reps]
 * ------------------------------------------------------------------------- */
#include "simd_mont_ifma.h"
#include "../../src/cpu/simd_mont_ifma.cpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

static double now_ns(void)
{
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static inline __m512i K_ld(const uint64_t *p) { return _mm512_load_si512((const __m512i *)p); }
static inline void    K_st(uint64_t *p, __m512i v) { _mm512_store_si512((__m512i *)p, v); }

/* Schoolbook product of two `len`-limb operands into t[0 .. 2*len+1], which must be
   zeroed.  Two rows per iteration and one b-vector feeding four madds -- the same
   shape as the production kernel, so the base case is not handicapped. */
static void prod_rows(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t len,
                      __m512i mask, __m512i zero)
{
    size_t i = 0;
    for (; i + 1 < len; i += 2) {
        const __m512i a0 = K_ld(a + 8 * i), a1 = K_ld(a + 8 * (i + 1));
        const __m512i b0 = K_ld(b);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0);
        K_st(t + 8 * i, _mm512_madd52lo_epu64(K_ld(t + 8 * i), a0, b0));
        K_st(t + 8 * (i + 1), _mm512_madd52lo_epu64(K_ld(t + 8 * (i + 1)), a1, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = K_ld(b + 8 * r);
            __m512i c0 = _mm512_add_epi64(K_ld(t + 8 * (i + r)), h0);
            K_st(t + 8 * (i + r), _mm512_madd52lo_epu64(c0, a0, br));
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c1 = _mm512_add_epi64(K_ld(t + 8 * (i + 1 + r)), h1);
            K_st(t + 8 * (i + 1 + r), _mm512_madd52lo_epu64(c1, a1, br));
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
        }
        K_st(t + 8 * (i + len), _mm512_add_epi64(K_ld(t + 8 * (i + len)), h0));
        K_st(t + 8 * (i + 1 + len), _mm512_add_epi64(K_ld(t + 8 * (i + 1 + len)), h1));
    }
    if (i < len) {
        const __m512i ai = K_ld(a + 8 * i);
        const __m512i b0 = K_ld(b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        K_st(t + 8 * i, _mm512_madd52lo_epu64(K_ld(t + 8 * i), ai, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = K_ld(b + 8 * r);
            const __m512i acc = _mm512_add_epi64(K_ld(t + 8 * (i + r)), hi);
            K_st(t + 8 * (i + r), _mm512_madd52lo_epu64(acc, ai, br));
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        K_st(t + 8 * (i + len), _mm512_add_epi64(K_ld(t + 8 * (i + len)), hi));
    }
    (void)mask;
}

/* ---------------------------------------------------------------------------
 * Scratch discipline: a bump allocator hands every sub-product its own columns, so
 * recursion can never overwrite a live intermediate (the first version of this tool
 * shared one buffer and produced garbage).  Only freshly handed-out columns are
 * zeroed; prod_rows() accumulates into them.
 * ------------------------------------------------------------------------ */
typedef struct { uint64_t *base; size_t used, cap; } kbump_t;

static uint64_t *kget(kbump_t *s, size_t cols)
{
    if (s->used + cols > s->cap) { printf("  !! scratch overflow\n"); exit(3); }
    uint64_t *p = s->base + 8 * s->used;
    s->used += cols;
    for (size_t k = 0; k < cols; k++) _mm512_store_si512((__m512i *)(p + 8 * k), _mm512_setzero_si512());
    return p;
}

static void kadd(uint64_t *dst, const uint64_t *src, size_t cols, int sub)
{
    for (size_t k = 0; k < cols; k++) {
        __m512i v = _mm512_load_si512((const __m512i *)(dst + 8 * k));
        const __m512i w = _mm512_load_si512((const __m512i *)(src + 8 * k));
        v = sub ? _mm512_sub_epi64(v, w) : _mm512_add_epi64(v, w);
        _mm512_store_si512((__m512i *)(dst + 8 * k), v);
    }
}

/* dst[0..nout) = x (lx limbs, zero-extended) + y (ly limbs, zero-extended).
   The two operands can have different lengths (a0 has h limbs, a1 has len-h), which is
   exactly what an earlier version got wrong. */
static void array_add_mixed(uint64_t *dst, const uint64_t *x, size_t lx,
                            const uint64_t *y, size_t ly, size_t nout,
                            __m512i mask, __m512i zero)
{
    __m512i cy = zero;
    for (size_t k = 0; k < nout; k++) {
        const __m512i xv = (k < lx) ? K_ld(x + 8 * k) : zero;
        const __m512i yv = (k < ly) ? K_ld(y + 8 * k) : zero;
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(xv, yv), cy);
        cy = _mm512_srli_epi64(s, 52);
        K_st(dst + 8 * k, _mm512_and_si512(s, mask));
    }
    K_st(dst + 8 * (nout - 1), cy);
}

/* Borrow-propagating subtract, radix 2^52: dst[] -= src[] with the borrow carried
 * forward, so every stored column stays a NON-NEGATIVE 52-bit digit.
 *
 * This is the crux of the whole exercise.  A plain column-wise subtract leaves
 * "negative columns" (borrows) inside the accumulator, and the fold tail's carry
 * pass -- `carry = v >> 52` -- then reads a 2^64-sized carry out of them and
 * produces garbage.  That is the same trap docs/DEV_COOP_KARATSUBA_2048.md describes
 * for the GPU version ("the carry corrections need per-limb propagation"). */
static void ksub(uint64_t *dst, const uint64_t *src, size_t cols, __m512i mask)
{
    __m512i borrow = _mm512_setzero_si512();
    for (size_t k = 0; k < cols; k++) {
        __m512i d = _mm512_sub_epi64(_mm512_load_si512((const __m512i *)(dst + 8 * k)),
                                     _mm512_load_si512((const __m512i *)(src + 8 * k)));
        d = _mm512_sub_epi64(d, borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        _mm512_store_si512((__m512i *)(dst + 8 * k), _mm512_and_si512(d, mask));
    }
}

/* Returns a fresh 2*len+2 column array holding a*b (zeroed by the allocator). */
static uint64_t *mul_out(kbump_t *s, const uint64_t *a, const uint64_t *b, size_t len,
                         int levels, __m512i mask, __m512i zero)
{
    /* Width: the m2 term lands at offset 2h with width 2h+2, so the accumulator must
       hold 4h+2 columns -- that is 2*len+2 for even len but 2*len+4 for odd len, and
       the first version of this tool allocated 2*len+2 and silently wrote two columns
       into the neighbouring scratch region (which then corrupted m0 before the
       subtraction read it).  The tail only reads columns < 2n, so extra columns are
       harmless as long as they are zero. */
    const size_t cols = 2 * len + 4;
    uint64_t *r = kget(s, cols);
    if (levels <= 0 || len < 12) {               /* base case: schoolbook */
        prod_rows(r, a, b, len, mask, zero);
        ifma_carry_pass(r, 0, cols, zero, mask);  /* hand back CANONICAL digits */
        return r;
    }
    /* a = a0 + a1*X with X = 2^(52h): a0 has h limbs, but a1 = floor(a/X) has len-h
       limbs -- WHICH IS h-1 FOR ODD len, not h.  Treating a1 as h limbs reads one
       column past the operand and adds the wrong value into a0+a1; nothing else
       explains why every odd-limb case (n52 = 25 and 77) mismatched while the even
       ones (58, 116) passed. */
    const size_t h = (len + 1) / 2;
    const size_t m = len - h;                    /* limbs in a1 */
    uint64_t *m0 = mul_out(s, a, b, h, levels - 1, mask, zero);
    uint64_t *m2 = mul_out(s, a + 8 * h, b + 8 * h, m, levels - 1, mask, zero);
    uint64_t *sa = kget(s, h + 1);
    uint64_t *sb = kget(s, h + 1);
    array_add_mixed(sa, a, h, a + 8 * h, m, h + 1, mask, zero);
    array_add_mixed(sb, b, h, b + 8 * h, m, h + 1, mask, zero);
    uint64_t *m1 = mul_out(s, sa, sb, h + 1, levels - 1, mask, zero);

    /* Widths must be each region's real width (an earlier version passed cols and read
       straight into the neighbouring scratch region), and every positive term goes in
       first so the partial value can never dip below the subtrahend (m1 >= m0+m2 gives
       m0 + (m1<<h) + (m2<<2h) >= (m0+m2)<<h, hence a final borrow of 0). */
    const size_t w0 = 2 * h + 4, w1 = 2 * (h + 1) + 4, w2 = 2 * m + 4;
    kadd(r,          m0, w0, 0);                 /* + m0                */
    kadd(r + 8 * h,  m1, w1, 0);                 /* + m1 << h           */
    kadd(r + 16 * h, m2, w2, 0);                 /* + m2 << 2h          */

    /* The subtraction needs CANONICAL digits on both sides: a column of an
       unnormalised accumulator accumulates up to ~n products (n*2^52), so the
       "borrow" out of such a column is not 0/1 and ksub()'s mask would silently drop
       the high bits.  One carry pass here makes the minuend canonical, and the
       sub-products are canonical by construction (each mul_out normalises before
       returning).  This pass is exactly the "carry correction" cost the GPU version
       in docs/DEV_COOP_KARATSUBA_2048.md refused to pay -- here it is one pass over
       2n+4 columns, and it is what the timing below has to cover. */
    ifma_carry_pass(r, 0, cols, zero, mask);

    ksub(r + 8 * h,  m0, w0, mask);              /* - m0 << h           */
    ksub(r + 8 * h,  m2, w2, mask);              /* - m2 << h           */
    return r;
}

int main(int argc, char **argv)
{
    const int k = (argc > 1) ? atoi(argv[1]) : 4001;
    const int levels = (argc > 2) ? atoi(argv[2]) : 2;
    const int reps = (argc > 3) ? atoi(argv[3]) : 2000;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1);
    mpz_mul_2exp(N, N, (mp_bitcnt_t)k);
    mpz_sub_ui(N, N, 1);

    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, IFMA_FIELD_MERS) != 0) { printf("ctx failed\n"); return 2; }
    const size_t n = c.n;
    const __m512i mask = _mm512_set1_epi64((long long)IFMA_MASK52);
    const __m512i zero = _mm512_setzero_si512();

    uint64_t *a = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *b = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *ref = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *got = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    const size_t wcols = 2 * n + 2;
    uint64_t *t = (uint64_t *)_mm_malloc(8 * wcols * sizeof(uint64_t), 64);
    const size_t scap = 40 * n + 128;                 /* generous peak for 3 levels */
    uint64_t *scratch = (uint64_t *)_mm_malloc(8 * scap * sizeof(uint64_t), 64);

    uint64_t s = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < 8 * n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; a[i] = s & 0xFFFFFFFFFFFFFULL; }
    for (size_t i = 0; i < 8 * n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b[i] = s & 0xFFFFFFFFFFFFFULL; }
    { const size_t top = (size_t)(k / 52); const unsigned tb = (unsigned)(k % 52);
      for (unsigned lane = 0; lane < 8; lane++)
        for (size_t j = top; j < n; j++)
          a[8*j+lane] = b[8*j+lane] = (j == top && tb) ? (a[8*j+lane] & ((1ULL << tb) - 1) & ~(1ULL << (tb-1))) : 0; }

    ifma_mont_mul(ref, a, b, &c);
    for (int lv = 0; lv <= 4; lv++) {
        kbump_t sc = { scratch, 0, scap };
        uint64_t *r = mul_out(&sc, a, b, n, lv, mask, zero);
        ifma_mersenne_finish(got, r, &c, zero);
        int bad = memcmp(ref, got, 8 * n * sizeof(uint64_t)) != 0;
        int at = -1;
        for (size_t i = 0; i < 8 * n && at < 0; i++) if (ref[i] != got[i]) at = (int)i;
        printf("  levels=%d: %s", lv, bad ? "MISMATCH" : "identical");
        if (bad) printf(" first diff at word %d (col %d lane %d): ref=%016llx got=%016llx",
                        at, at / 8, at % 8,
                        (unsigned long long)ref[at], (unsigned long long)got[at]);
        printf("   [scratch %zu]\n", sc.used);
    }
    printf("\n  k=%d n52=%zu  production mul = ", k, n);

    for (int r = 0; r < 50; r++) ifma_mont_mul(got, a, b, &c);
    double best_prod = 1e30;
    for (int r = 0; r < 7; r++) {
        const double t0 = now_ns();
        for (int q = 0; q < reps; q++) ifma_mont_mul(got, a, b, &c);
        const double d = (now_ns() - t0) / reps; if (d < best_prod) best_prod = d;
    }
    printf("%.1f ns/batch (2n^2 = %llu madds)\n", best_prod, (unsigned long long)(2 * n * n));
    for (int lv = 1; lv <= levels; lv++) {
        double best = 1e30;
        for (int r = 0; r < 7; r++) {
            const double t0 = now_ns();
            for (int q = 0; q < reps; q++) {
                kbump_t sc = { scratch, 0, scap };
                uint64_t *rr = mul_out(&sc, a, b, n, lv, mask, zero);
                ifma_mersenne_finish(got, rr, &c, zero);
            }
            const double d = (now_ns() - t0) / reps; if (d < best) best = d;
        }
        printf("  levels=%d  karatsuba = %.1f ns   -> %+.1f%% vs production\n",
               lv, best, 100.0 * (best - best_prod) / best_prod);
    }

    _mm_free(a); _mm_free(b); _mm_free(ref); _mm_free(got); _mm_free(t); _mm_free(scratch);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    return 0;
}
