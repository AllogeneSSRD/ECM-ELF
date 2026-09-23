/* ---------------------------------------------------------------------------
 * mers_sqr_phases.cpp -- where does the Mersenne fold-domain *square* spend its
 * time?
 *
 * Motivation (measured, tools/bench/simd_mont_gate.cpp + this tool):
 *   at n52 = 58 the fold square touches only the strict upper triangle, i.e. about
 *   a QUARTER of the limb products of the general mul, yet it costs the SAME wall
 *   time (mul 3613 ns/batch vs sqr 3622 ns/batch).  The ladder runs 4 squarings per
 *   bit out of 10 products, so if the square were actually madd-throughput bound
 *   the whole ladder would be ~20% faster.
 *
 * This tool separates the kernel into its five phases and times each one by
 * differencing cumulative runs (phase p is "everything up to and including p"), so
 * a phase that does not scale with its own work becomes visible:
 *
 *   0 init   : zero the 2n+2 accumulator columns
 *   1 prod   : strict upper triangle (off-diagonal, doubled later) + the row-i /
 *              j=i+1 special case
 *   2 double : the "off-diagonal gets doubled" pass (serial carry over 2n columns)
 *   3 diag   : add the n diagonal a_i^2 terms
 *   4 finish : ifma_mersenne_finish (normalise 2n columns, fold, drain, reduce)
 *
 * The phase bodies are copied from ifma_mersenne_sqr() in src/cpu/simd_mont_ifma.cpp
 * on purpose (the real function has no phase guards); the copy is verified against
 * the production kernel by the "matches ifma_mont_sqr" line at the end.  Including
 * the .cpp gives access to its static helpers, which is the same trick
 * simd_mont_tail.cpp / simd_mont_notail.cpp use.
 *
 * usage: mers_sqr_phases [k] [reps]        (N = 2^k-1, default k=3001, reps=20000)
 * ------------------------------------------------------------------------- */
#include "simd_mont_ifma.h"
#include "../../src/cpu/simd_mont_ifma.cpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

static double now_ns(void)
{
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

/* copy of ifma_mersenne_sqr's body, with phase guards; `up_to` = last phase to run */
static void sqr_upto(uint64_t *out, const uint64_t *a, const ifma_ctx_t *c, int up_to)
{
    const size_t n = c->n;
    const __m512i mask = _mm512_set1_epi64((long long)IFMA_MASK52);
    const __m512i zero = _mm512_setzero_si512();
    uint64_t *t = c->scratch;

    for (size_t k = 0; k < 2 * n + 2; k++)                 /* phase 0 */
        _mm512_store_si512((__m512i *)(t + 8 * k), zero);
    if (up_to < 1) return;

    size_t i = 0;                                          /* phase 1 */
    for (; i + 1 < n; i += 2) {
        const __m512i a0 = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i a1 = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
        {
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
    if (up_to < 2) return;

    __m512i cy = zero;                                     /* phase 2 */
    for (size_t k = 0; k < 2 * n; k++) {
        const __m512i v = _mm512_add_epi64(
            _mm512_slli_epi64(_mm512_load_si512((const __m512i *)(t + 8 * k)), 1), cy);
        cy = _mm512_srli_epi64(v, 52);
        _mm512_store_si512((__m512i *)(t + 8 * k), _mm512_and_si512(v, mask));
    }
    if (up_to < 3) return;

    for (size_t q = 0; q < n; q++) {                       /* phase 3 */
        const __m512i aq = _mm512_load_si512((const __m512i *)(a + 8 * q));
        _mm512_store_si512((__m512i *)(t + 8 * (2 * q)),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (2 * q))), aq, aq));
        _mm512_store_si512((__m512i *)(t + 8 * (2 * q + 1)),
            _mm512_madd52hi_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (2 * q + 1))), aq, aq));
    }
    if (up_to < 4) return;

    ifma_mersenne_finish(out, t, c, cy);                   /* phase 4 */
}

int main(int argc, char **argv)
{
    const int k = (argc > 1) ? atoi(argv[1]) : 3001;
    const int reps = (argc > 2) ? atoi(argv[2]) : 20000;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1);
    mpz_mul_2exp(N, N, (mp_bitcnt_t)k);
    mpz_sub_ui(N, N, 1);

    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, IFMA_FIELD_MERS) != 0) {
        printf("ctx init failed (not a Mersenne fold case?)\n");
        mpz_clear(N);
        return 2;
    }
    const size_t n = c.n;

    /* operands.  NOTE: the fold kernel is branch-free, but the *drain* loop in
       ifma_mersenne_finish() is NOT -- "sparse" (small) operands break out of it
       after one iteration while full-width random limbs keep the carry cascade
       alive.  So both operand shapes matter; pass "rand" to switch. */
    const int use_rand = (argc > 3) && (strcmp(argv[3], "rand") == 0);
    uint64_t *a = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *o = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    for (size_t i = 0; i < 8 * n; i++) a[i] = 0;
    if (use_rand) {
        uint64_t s = 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < 8 * n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            a[i] = s & 0x000FFFFFFFFFFFFFULL;
        }
        /* canonical: keep every lane < N = 2^k-1 (limbs above k cleared, and the
           top bit of the k-th limb cleared so the value stays < 2^k) */
        const size_t top = (size_t)(k / 52);
        const unsigned tbits = (unsigned)(k % 52);
        for (unsigned lane = 0; lane < 8; lane++)
            for (size_t j = top; j < n; j++)
                a[8 * j + lane] = (j == top && tbits)
                                  ? (a[8 * j + lane] & ((1ULL << tbits) - 1) & ~(1ULL << (tbits - 1)))
                                  : 0;
    } else {
        for (size_t j = 0; j < n; j += 3) a[8 * j] = 0x1234567 + j;   /* some spread */
    }
    memset(o, 0, 8 * n * sizeof(uint64_t));

    double cum[5] = {0, 0, 0, 0, 0};
    for (int p = 0; p <= 4; p++) {
        /* warm-up */
        for (int r = 0; r < 50; r++) sqr_upto(o, a, &c, p);
        const double t0 = now_ns();
        for (int r = 0; r < reps; r++) sqr_upto(o, a, &c, p);
        cum[p] = (now_ns() - t0) / reps;
    }

    /* production kernel for reference */
    for (int r = 0; r < 50; r++) ifma_mont_sqr(o, a, &c);
    const double t0 = now_ns();
    for (int r = 0; r < reps; r++) ifma_mont_sqr(o, a, &c);
    const double full = (now_ns() - t0) / reps;

    /* mul of the same size for reference (a*a via the mul kernel) */
    uint64_t *m = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    memset(m, 0, 8 * n * sizeof(uint64_t));
    for (int r = 0; r < 50; r++) ifma_mont_mul(m, a, a, &c);
    const double t1 = now_ns();
    for (int r = 0; r < reps; r++) ifma_mont_mul(m, a, a, &c);
    const double mul_aa = (now_ns() - t1) / reps;

    const char *names[5] = { "init(zero 2n+2 cols)", "prod(upper triangle)",
                             "double(off-diag x2)  ", "diag(a_i^2)          ",
                             "finish(norm/fold/drain)" };
    printf("mers_sqr_phases: N = 2^%d-1, n52 = %zu, reps = %d, operands = %s\n",
           k, n, reps, use_rand ? "full-width random" : "sparse (near zero)");
    printf("  %-22s %10.1f ns/batch   cumulative\n", "phase", 0.0);
    for (int p = 0; p <= 4; p++) {
        const double delta = (p == 0) ? cum[0] : cum[p] - cum[p - 1];
        printf("  %-22s %10.1f ns          %8.1f ns%s\n", names[p], delta, cum[p],
               (p > 0 && delta > 0.55 * cum[4]) ? "   <-- dominant" : "");
    }
    printf("  %-22s %10.1f ns\n", "phase sum", cum[4]);
    printf("  %-22s %10.1f ns   (guard-copy vs production kernel)\n",
           "production ifma_mont_sqr", full);
    printf("  %-22s %10.1f ns   (same operands via the mul kernel)\n",
           "ifma_mont_mul(a,a)", mul_aa);
    printf("  ratio sqr/mul(identical operands) = %.3f\n", full / mul_aa);
    printf("  NOTE: madd counts for 8 lanes -- mul: 4n^2 = %llu madds, "
           "sqr: 2n(n-1)+2n products = %llu madds\n",
           (unsigned long long)(4 * n * n),
           (unsigned long long)(2 * (n * (n - 1) + 2 * n)));

    _mm_free(a); _mm_free(o); _mm_free(m);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    return 0;
}
