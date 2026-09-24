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
#include "simd_mont_curve.h"      /* mont_soa_* : the real ladder, timed below */
#include "ecm_mont_cpu.h"         /* mont_build_s / mont_expand_bits */
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

    /* ---- finish, split into its own sub-phases -------------------------
       4 norm    : 52-bit normalize of the 2n product columns (serial carry chain)
       5 +fold   : fold columns 2n-1 .. n+1 down by 2^(52n) = 2^sh
       6 +drain  : fold everything left at/above position k, carry again
       7 +canon  : the final "value == N means value is 2^k-1" reduction
       Copied from ifma_mersenne_finish(); keep in sync with src/cpu/simd_mont_ifma.cpp. */
    const unsigned sh = c->shift;
    __m512i top = ifma_carry_pass(t, 0, 2 * n, cy, mask);
    if (up_to < 5) return;

    ifma_fold_digit(t, n, top, sh, mask);
    for (size_t cc = 2 * n - 1; cc > n; cc--) {
        const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * cc));
        _mm512_store_si512((__m512i *)(t + 8 * cc), zero);
        ifma_fold_digit(t, cc - n, d, sh, mask);
    }
    {
        const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * n));
        _mm512_store_si512((__m512i *)(t + 8 * n), zero);
        ifma_fold_digit(t, 0, _mm512_and_si512(d, mask), sh, mask);
        const __m512i extra = _mm512_srli_epi64(d, 52);
        if (_mm512_test_epi64_mask(extra, extra) != 0) ifma_fold_digit(t, 1, extra, sh, mask);
    }
    if (up_to < 6) return;

    if (sh != 0) {
        const __m512i lowmask = _mm512_set1_epi64((long long)((1ULL << (52 - sh)) - 1));
        for (int it = 0; it < 6; it++) {
            const __m512i d = _mm512_load_si512((const __m512i *)(t + 8 * (n - 1)));
            _mm512_store_si512((__m512i *)(t + 8 * (n - 1)), _mm512_and_si512(d, lowmask));
            _mm512_store_si512((__m512i *)(t + 8 * 0), _mm512_add_epi64(
                _mm512_load_si512((const __m512i *)(t + 8 * 0)),
                _mm512_srli_epi64(d, 52 - sh)));
            const __m512i carry = ifma_carry_pass(t, 0, n, zero, mask);
            const __m512i d2 = _mm512_load_si512((const __m512i *)(t + 8 * (n - 1)));
            const __m512i above = _mm512_srli_epi64(d2, 52 - sh);
            if (_mm512_test_epi64_mask(above, above) == 0 &&
                _mm512_test_epi64_mask(carry, carry) == 0) break;
            if (_mm512_test_epi64_mask(carry, carry) != 0) ifma_fold_digit(t, 0, carry, sh, mask);
        }
    } else {
        for (int it = 0; it < 6; it++) {
            const __m512i carry = ifma_carry_pass(t, 0, n, zero, mask);
            if (_mm512_test_epi64_mask(carry, carry) == 0) break;
            ifma_fold_digit(t, 0, carry, sh, mask);
        }
    }
    if (up_to < 7) return;

    {
        __m512i borrow = zero;
        for (size_t k = 0; k < n; k++) {
            const __m512i rk = _mm512_load_si512((const __m512i *)(t + 8 * k));
            const __m512i nk = _mm512_load_si512((const __m512i *)(c->nb + 8 * k));
            const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(rk, nk), borrow);
            borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
            _mm512_store_si512((__m512i *)(out + 8 * k), d);
        }
        const __mmask8 need_sub = _mm512_cmpeq_epi64_mask(borrow, zero);
        for (size_t k = 0; k < n; k++) {
            const __m512i orig = _mm512_load_si512((const __m512i *)(t + 8 * k));
            const __m512i sub = _mm512_load_si512((const __m512i *)(out + 8 * k));
            _mm512_store_si512((__m512i *)(out + 8 * k),
                               _mm512_mask_blend_epi64(need_sub, orig, sub));
        }
    }
}

/* Independent vpmadd52 chains: the machine's own madd roof, measured in the same
 * process as the kernels so every comparison below is clock independent (this
 * laptop is a Strix Point HX 370: 256-bit FPU datapath, so 512-bit AVX-512 runs
 * as 2x256-bit and the roof is about half a desktop Zen 5's). */
static double peak_gmadds(int iters)
{
    const __m512i x = _mm512_set1_epi64(0x000FFFFFFFFFFFFFULL);
    __m512i a0 = _mm512_set1_epi64(1), a1 = _mm512_set1_epi64(2);
    __m512i a2 = _mm512_set1_epi64(3), a3 = _mm512_set1_epi64(4);
    __m512i a4 = _mm512_set1_epi64(5), a5 = _mm512_set1_epi64(6);
    __m512i a6 = _mm512_set1_epi64(7), a7 = _mm512_set1_epi64(8);
    for (int r = 0; r < 1000; r++) {           /* warm-up */
        a0 = _mm512_madd52lo_epu64(a0, x, x); a1 = _mm512_madd52lo_epu64(a1, x, x);
        a2 = _mm512_madd52lo_epu64(a2, x, x); a3 = _mm512_madd52lo_epu64(a3, x, x);
        a4 = _mm512_madd52lo_epu64(a4, x, x); a5 = _mm512_madd52lo_epu64(a5, x, x);
        a6 = _mm512_madd52lo_epu64(a6, x, x); a7 = _mm512_madd52lo_epu64(a7, x, x);
    }
    const double t0 = now_ns();
    for (int r = 0; r < iters; r++) {
        a0 = _mm512_madd52lo_epu64(a0, x, x); a1 = _mm512_madd52lo_epu64(a1, x, x);
        a2 = _mm512_madd52lo_epu64(a2, x, x); a3 = _mm512_madd52lo_epu64(a3, x, x);
        a4 = _mm512_madd52lo_epu64(a4, x, x); a5 = _mm512_madd52lo_epu64(a5, x, x);
        a6 = _mm512_madd52lo_epu64(a6, x, x); a7 = _mm512_madd52lo_epu64(a7, x, x);
    }
    const double ns = now_ns() - t0;
    /* 8 chains x 8 lanes per iteration; the volatile sink stops the compiler from
       deleting the loop (an unused pure chain disappears at /O2 -- that is how this
       function first came back as "inf Gmadd/s"). */
    static volatile uint64_t sink;
    sink = (uint64_t)_mm512_reduce_add_epi64(
        _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3)))
         + (uint64_t)_mm512_reduce_add_epi64(
        _mm512_add_epi64(_mm512_add_epi64(a4, a5), _mm512_add_epi64(a6, a7)));
    return 8.0 * 8.0 * iters / ns;             /* Gmadd/s (per-lane madds) */
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

    double cum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int p = 0; p <= 7; p++) {
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

    /* ---- interleaved A/B: the guarded copy above IS the pre-2026-09-24 kernel
       (separate double / diagonal / normalize passes, subtract-and-blend
       canonicalization), the production call is the fused one.  Alternating the
       two inside one process is the only reliable comparison on a laptop that
       throttles; report the best of `rounds` blocks. */
    const int rounds = 7;
    double best_old = 1e30, best_new = 1e30;
    for (int r = 0; r < rounds; r++) {
        double t = now_ns();
        for (int q = 0; q < 4000; q++) sqr_upto(o, a, &c, 7);
        t = (now_ns() - t) / 4000; if (t < best_old) best_old = t;
        t = now_ns();
        for (int q = 0; q < 4000; q++) ifma_mont_sqr(o, a, &c);
        t = (now_ns() - t) / 4000; if (t < best_new) best_new = t;
    }

    /* mul of the same size for reference (a*a via the mul kernel) */
    uint64_t *m = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    memset(m, 0, 8 * n * sizeof(uint64_t));
    for (int r = 0; r < 50; r++) ifma_mont_mul(m, a, a, &c);
    const double t1 = now_ns();
    for (int r = 0; r < reps; r++) ifma_mont_mul(m, a, a, &c);
    const double mul_aa = (now_ns() - t1) / reps;

    const char *names[8] = { "init(zero 2n+2 cols)", "prod(upper triangle)",
                             "double(off-diag x2)  ", "diag(a_i^2)          ",
                             "finish: normalize    ", "finish: fold hi cols ",
                             "finish: drain >= k   ", "finish: canonical    " };
    printf("mers_sqr_phases: N = 2^%d-1, n52 = %zu, reps = %d, operands = %s\n",
           k, n, reps, use_rand ? "full-width random" : "sparse (near zero)");
    printf("  %-22s %10s            cumulative\n", "phase", "delta");
    for (int p = 0; p <= 7; p++) {
        const double delta = (p == 0) ? cum[0] : cum[p] - cum[p - 1];
        printf("  %-22s %10.1f ns          %8.1f ns%s\n", names[p], delta, cum[p],
               (p > 0 && delta > 0.55 * cum[7]) ? "   <-- dominant" : "");
    }
    printf("  %-22s %10.1f ns\n", "phase sum", cum[7]);
    printf("  %-22s %10.1f ns   (guard-copy vs production kernel)\n",
           "production ifma_mont_sqr", full);
    printf("  %-22s %10.1f ns   (same operands via the mul kernel)\n",
           "ifma_mont_mul(a,a)", mul_aa);
    printf("  ratio sqr/mul(identical operands) = %.3f\n", full / mul_aa);
    printf("\n  --- interleaved A/B (best of %d blocks) ----------------------------\n", rounds);
    printf("  pre-2026-09-24 sqr (this file's copy) : %8.1f ns\n", best_old);
    printf("  production sqr (fused)                : %8.1f ns   %+.1f%%\n",
           best_new, 100.0 * (best_new - best_old) / best_old);
    printf("  NOTE: madd counts for 8 lanes -- mul: 4n^2 = %llu madds, "
           "sqr: 2n(n-1)+2n products = %llu madds\n",
           (unsigned long long)(4 * n * n),
           (unsigned long long)(2 * (n * (n - 1) + 2 * n)));

    /* ---- roof and working-set sensitivity ---------------------------------
       Everything above is ns per call, which mixes in whatever clock the CPU felt
       like using.  Gmadd/s (madd instructions x 8 lanes / second) is clock
       independent, and dividing by the machine's own roof (independent chains in
       this same process) says how much room a kernel rewrite could possibly find. */
    const double peak = peak_gmadds(2000000);
    const uint64_t mul_madds = 2 * (uint64_t)n * (uint64_t)n;              /* instr/batch */
    const uint64_t sqr_madds = (uint64_t)n * (uint64_t)(n - 1) + 2 * (uint64_t)n;
    printf("\n  machine roof (8 independent chains) : %6.2f Gmadd/s\n", peak);

    /* Working-set sweep: the ladder's pool is ~22 field elements (22 x 8n words),
       while these loops otherwise run out of a handful of L1-resident buffers.  If
       the kernel is memory bound, inflating the operand set must slow it down. */
    printf("\n  working-set sweep (operand buffers) -- mul / sqr, Gmadd/s and %% of roof\n");
    double best_mul = 0, best_sqr = 0;      /* single-set (L1-resident) best times */
    for (int sets = 1; sets <= 32; sets *= 4) {
        std::vector<uint64_t *> buf(sets);
        for (int s = 0; s < sets; s++) {
            buf[s] = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
            for (size_t i = 0; i < 8 * n; i++) buf[s][i] = a[i % (8 * n)];
        }
        uint64_t *res = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
        double bm = 1e30, bs = 1e30;
        for (int r = 0; r < 5; r++) {
            double t = now_ns();
            for (int q = 0; q < 2000; q++) {
                const uint64_t *u = buf[q % sets], *v = buf[(q + 1) % sets];
                ifma_mont_mul(res, u, v, &c);
            }
            t = (now_ns() - t) / 2000; if (t < bm) bm = t;
            t = now_ns();
            for (int q = 0; q < 2000; q++) ifma_mont_sqr(res, buf[q % sets], &c);
            t = (now_ns() - t) / 2000; if (t < bs) bs = t;
        }
        const double gm = 8.0 * mul_madds / bm, gs = 8.0 * sqr_madds / bs;
        if (sets == 1) { best_mul = bm; best_sqr = bs; }
        printf("    %2d set(s): %5.2f Gmadd/s (%3.0f%%) / %5.2f Gmadd/s (%3.0f%%)"
               "   [%6.1f ns / %6.1f ns]\n",
               sets, gm, 100.0 * gm / peak, gs, 100.0 * gs / peak, bm, bs);
        for (int s = 0; s < sets; s++) _mm_free(buf[s]);
        _mm_free(res);
    }

    /* ---- the real ladder, in this same process ---------------------------
       The per-bit model (6 mul + 4 sqr) and the ladder's measured cost per bit are
       measured under identical conditions here, so the difference is the auxiliary
       SoA work (soa_add/soa_sub/soa_neg/soa_cond_sub passes and the copy in xz_add)
       plus any per-op overhead the isolated kernel calls do not have. */
    {
        /* Two exponent sizes in the SAME process: a ~4k-bit exponent finishes in
           ~0.1 s (burst), a B1=1e5 exponent takes seconds (sustained).  If the
           per-bit cost jumps in the second case, the difference between this
           harness and ecm.exe is a power/clock effect, not kernel code. */
        const uint64_t ladder_b1s[2] = { 3000, 100000 };
        for (int li = 0; li < 2; li++) {
        mpz_t Nm, s;
        mpz_inits(Nm, s, NULL);
        mpz_set_ui(Nm, 1);
        mpz_mul_2exp(Nm, Nm, (mp_bitcnt_t)k);
        mpz_sub_ui(Nm, Nm, 1);
        const size_t s_bits = mont_build_s(s, ladder_b1s[li], 1);

        mont_soa_ctx_t sc;
        if (mont_soa_init(&sc, Nm, IFMA_FIELD_MERS) == 0) {
            size_t nb_bits = 0;
            uint8_t *bits = mont_expand_bits(s, &nb_bits);
            uint64_t sg[IFMA_LANES];
            for (unsigned i = 0; i < IFMA_LANES; i++) sg[i] = 12345 + i;
            std::vector<mpz_t> bx(IFMA_LANES), bg(IFMA_LANES);
            for (unsigned i = 0; i < IFMA_LANES; i++) mpz_inits(bx[i], bg[i], NULL);

            const int nrun = (ladder_b1s[li] >= 100000) ? 4 : 6;
            for (int r = 0; r < 2; r++) mont_soa_stage1_bits(&sc, bits, nb_bits, sg, bx.data(), bg.data());
            double best = 1e30;
            for (int r = 0; r < nrun; r++) {
                const double t = now_ns();
                mont_soa_stage1_bits(&sc, bits, nb_bits, sg, bx.data(), bg.data());
                const double d = now_ns() - t;
                if (d < best) best = d;
            }
            const double per_bit = best / (double)nb_bits;
            const double model = 6.0 * best_mul + 4.0 * best_sqr;
            const double ladder_madds = 6.0 * mul_madds + 4.0 * sqr_madds;
            printf("\n  ladder (same process): B1=%llu s_bits=%zu (call %.2f s), %.1f us/bit,"
                   " %.2f Gmadd/s per-lane (%.0f%% of roof)\n",
                   (unsigned long long)ladder_b1s[li], nb_bits, best / 1e9,
                   per_bit / 1000.0,
                   8.0 * ladder_madds / per_bit,
                   100.0 * (8.0 * ladder_madds / per_bit) / peak);
            printf("  model 6 mul + 4 sqr (best-of, same process) = %.1f us/bit"
                   "   -> auxiliary SoA work = %.1f us/bit (%.0f%%)\n",
                   model / 1000.0, (per_bit - model) / 1000.0,
                   100.0 * (per_bit - model) / per_bit);

            for (unsigned i = 0; i < IFMA_LANES; i++) mpz_clears(bx[i], bg[i], NULL);
            free(bits);
            mont_soa_clear(&sc);
        }
        mpz_clears(Nm, s, NULL);
        }
    }

    _mm_free(a); _mm_free(o); _mm_free(m);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    return 0;
}
