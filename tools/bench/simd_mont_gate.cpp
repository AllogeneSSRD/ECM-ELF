/* ---------------------------------------------------------------------------
 * simd_mont_gate.cpp — SIMD (AVX512-IFMA, 8 lanes = 8 curves) Montgomery
 * multiplication vs the production scalar GMP stack.
 *
 * Baseline = the real production primitives, i.e. this file includes
 * ecm_edwards_mont.h and calls mont_mul()/mont_sqr(), which are exactly
 *   mpn_mul_n()/mpn_sqr() + the landed REDC dispatch (mpn_redc_1 / mpn_redc_n).
 * Not mpz_mod, not a toy reimplementation.
 *
 * Gate (docs/ECM_EDWARDS_STAGE1.md §13): >= 2.0x at n52 = 77 (~4000 bit) and
 * n52 = 125 (~6500 bit); n52 = 154 (8000 bit) is measured but not gated.
 * Ratios are measured alternating A/B, minimum of `rounds` blocks, because this
 * laptop throttles under sustained load (see §9's retracted w-sweep).
 *
 * Build: cmake --build build_vs18 --config Release --target simd_mont_gate
 * ------------------------------------------------------------------------- */

#include "simd_mont_ifma.h"
#include "ecm_edwards_mont.h"

#include <gmp.h>
#include <emmintrin.h>
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

/* ---- timing ---------------------------------------------------------- */

static double calibrate_ghz_n(long long N)
{
    /* Dependent chain of SSE2 64-bit adds: 1 cycle latency each on Zen. */
    __m128i x = _mm_set_epi64x(1, 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (long long i = 0; i < N; i++) x = _mm_add_epi64(x, x);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    volatile long long sink = _mm_cvtsi128_si64(x);
    (void)sink;
    return (double)N / sec / 1e9;
}

static double calibrate_ghz(void)
{
    return calibrate_ghz_n(300000000LL);
}

static double now_sec(void)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* ---- random modulus -------------------------------------------------- */

static void random_modulus(mpz_t N, size_t bits, gmp_randstate_t rs)
{
    mpz_urandomb(N, rs, (mp_bitcnt_t)bits);
    mpz_setbit(N, bits - 1);      /* exact bit length */
    if (mpz_even_p(N)) mpz_add_ui(N, N, 1);
}

/* ---- correctness: SIMD Montgomery output vs an independent mpz check --- */

/* limb (52 bits) -> mpz.  mpz_add_ui() takes `unsigned long` (32 bits on
   Windows x64), so feeding it a whole 52-bit limb truncates it: feed two
   26-bit halves instead.  This is the same trap that broke the kernel's own
   limb extraction, so the test must not repeat it. */
static void mpz_add_limb52(mpz_t out, uint64_t limb)
{
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)((limb >> 26) & 0x3FFFFFFu));
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)(limb & 0x3FFFFFFu));
}

static uint64_t mpz_low52(const mpz_t v)
{
    mpz_t w;
    unsigned char buf[8];
    size_t count = 0;
    uint64_t word = 0;
    mpz_init(w);
    mpz_fdiv_r_2exp(w, v, 52);
    memset(buf, 0, sizeof(buf));
    mpz_export(buf, &count, -1, sizeof(uint64_t), 0, 0, w);
    if (count) memcpy(&word, buf, sizeof(word));
    mpz_clear(w);
    return word;
}

/* Independent limb extraction (does not use the library's own helper, so a
   wrong radix convention cannot make the test agree with itself). */
static void lanes_to_mpz(mpz_t out, const uint64_t *e, size_t n, unsigned lane)
{
    mpz_set_ui(out, 0);
    for (size_t i = n; i-- > 0; )
        mpz_add_limb52(out, e[8 * i + lane] & 0xFFFFFFFFFFFFFULL);
}

struct CheckResult { int trials; int fails; };

static CheckResult check_correctness(const mpz_t N, ifma_ctx_t *ctx,
                                     const mont_ctx_t *mctx, size_t n52,
                                     gmp_randstate_t rs, int trials)
{
    CheckResult res = { 0, 0 };
    const size_t n = ctx->n;
    if (n != n52) { printf("    !! limb count mismatch\n"); res.fails++; return res; }

    std::vector<uint64_t> a(8 * n), b(8 * n), r(8 * n), t(8 * n);
    mpz_t R, Rinv, av, bv, want, got, one, tmp;
    mpz_inits(R, Rinv, av, bv, want, got, one, tmp, NULL);
    mpz_set_ui(R, 1);
    mpz_mul_2exp(R, R, 52 * (unsigned long)n);
    mpz_invert(Rinv, R, N);                 /* R^-1 mod N */

    for (int trial = 0; trial < trials; trial++) {
        const int edge = trial % 3;
        for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
            if (edge == 0) {
                mpz_urandomm(av, rs, N);
                mpz_urandomm(bv, rs, N);
            } else if (edge == 1) {         /* boundary: 0, 1, N-1, N-2 */
                static const int pick[4] = { 0, 1, -1, -2 };
                const int pa = pick[(lane + trial) & 3], pb = pick[(lane + 2 + trial) & 3];
                if (pa >= 0) mpz_set_ui(av, (unsigned long)pa); else mpz_add_ui(av, N, (unsigned long)pa);
                if (pb >= 0) mpz_set_ui(bv, (unsigned long)pb); else mpz_add_ui(bv, N, (unsigned long)pb);
            } else {                        /* 2^k boundaries inside the top limb */
                const size_t sh = 52 * (n - 1) + (size_t)(lane % 3);
                mpz_urandomm(av, rs, N);
                mpz_setbit(av, sh);         /* may exceed N, from_mpz reduces */
                mpz_urandomm(bv, rs, N);
            }
            ifma_from_mpz_lane(a.data(), lane, av, ctx);
            ifma_from_mpz_lane(b.data(), lane, bv, ctx);
        }
        ifma_mont_mul(r.data(), a.data(), b.data(), ctx);

        for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
            /* expected Montgomery product in the SAME domain: a*b*R^-1 mod N */
            lanes_to_mpz(av, a.data(), n, lane);
            lanes_to_mpz(bv, b.data(), n, lane);
            mpz_mul(want, av, bv);
            mpz_mod(want, want, N);
            mpz_mul(want, want, Rinv);
            mpz_mod(want, want, N);
            lanes_to_mpz(got, r.data(), n, lane);
            res.trials++;
            if (mpz_cmp(want, got) != 0) {
                if (res.fails < 3)
                    gmp_printf("    !! FAIL trial=%d lane=%u want=%Zx got=%Zx\n", trial, lane, want, got);
                res.fails++;
            }
            if (mpz_cmp(got, N) >= 0) {      /* results must stay reduced */
                if (res.fails < 3) gmp_printf("    !! FAIL not reduced lane=%u\n", lane);
                res.fails++;
            }
        }

        /* x * 1_mont == x, and squaring agrees with mul(x,x) */
        {
            std::vector<uint64_t> s(8 * n);
            ifma_set_one(t.data(), ctx);
            ifma_mont_mul(r.data(), a.data(), t.data(), ctx);
            ifma_mont_sqr(s.data(), a.data(), ctx);
            for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
                lanes_to_mpz(one, r.data(), n, lane);
                lanes_to_mpz(got, a.data(), n, lane);
                res.trials++;
                if (mpz_cmp(one, got) != 0) { res.fails++; if (res.fails < 4) printf("    !! FAIL x*1 != x\n"); }
                /* square vs mul */
                std::vector<uint64_t> aa(8 * n);
                memcpy(aa.data(), a.data(), 8 * n * sizeof(uint64_t));
                ifma_mont_mul(aa.data(), a.data(), a.data(), ctx);
                if (memcmp(aa.data(), s.data(), 8 * n * sizeof(uint64_t)) != 0) {
                    res.trials++; res.fails++;
                    if (res.fails < 5) printf("    !! FAIL sqr != mul(a,a)\n");
                }
            }
        }

        /* Cross-domain check against the production scalar path: the ordinary
           integer a*b mod N must agree with mont_to/mont_mul/mont_from. */
        {
            mont_t ma, mb, mr;
            /* r was clobbered by the block above (x*1); recompute it. */
            ifma_mont_mul(r.data(), a.data(), b.data(), ctx);
            for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
                lanes_to_mpz(av, a.data(), n, lane);   /* Montgomery value */
                lanes_to_mpz(bv, b.data(), n, lane);
                ifma_to_mpz_lane(want, r.data(), lane, ctx);   /* -> ordinary */
                /* ordinary a,b that produced av,bv were reduced; recover them */
                mpz_mul(tmp, av, Rinv); mpz_mod(av, tmp, N);
                mpz_mul(tmp, bv, Rinv); mpz_mod(bv, tmp, N);
                mont_to(&ma, av, mctx);
                mont_to(&mb, bv, mctx);
                mont_mul(&mr, &ma, &mb, mctx);
                mont_from(got, &mr, mctx);
                res.trials++;
                if (mpz_cmp(want, got) != 0) {
                    if (res.fails < 2) {
                        mpz_t c1, c2;
                        mpz_inits(c1, c2, NULL);
                        lanes_to_mpz(c1, r.data(), n, lane);
                        mpz_mul(c2, c1, Rinv); mpz_mod(c2, c2, N);
                        gmp_printf("    !! FAIL vs production lane=%u a=%Zx b=%Zx\n", lane, av, bv);
                        gmp_printf("       simd_ord=%Zx gmp=%Zx bench_ord=%Zx\n", want, got, c2);
                        mpz_clears(c1, c2, NULL);
                    }
                    res.fails++;
                }
            }
        }
    }
    mpz_clears(R, Rinv, av, bv, want, got, one, tmp, NULL);
    return res;
}

/* ---- timing ---------------------------------------------------------- */

struct Timing { double ns_per_op; };

static Timing time_simd(ifma_ctx_t *ctx, int kind, long long iters, int rounds)
{
    const size_t n = ctx->n;
    std::vector<std::vector<uint64_t> > acc(4, std::vector<uint64_t>(8 * n));
    std::vector<uint64_t> b(8 * n);
    for (int k = 0; k < 4; k++) ifma_set_one(acc[k].data(), ctx);
    memcpy(b.data(), ctx->one, 8 * n * sizeof(uint64_t));
    b[0] ^= 1;                                  /* not 1, not zero */

    double best = 1e30;
    for (int rd = 0; rd < rounds; rd++) {
        const double t0 = now_sec();
        long long i = 0;
        for (; i + 4 <= iters; i += 4) {
            for (int k = 0; k < 4; k++) {
                if (kind == 0) ifma_mont_mul(acc[k].data(), acc[k].data(), b.data(), ctx);
                else           ifma_mont_sqr(acc[k].data(), acc[k].data(), ctx);
            }
        }
        for (; i < iters; i++) ifma_mont_mul(acc[0].data(), acc[0].data(), b.data(), ctx);
        const double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    volatile uint64_t sink = acc[0][0] ^ acc[1][3] ^ acc[2][7] ^ acc[3][11 % (8 * n)];
    (void)sink;
    return Timing{ best / (double)iters * 1e9 };
}

static Timing time_gmp(const mont_ctx_t *mctx, int kind, long long iters, int rounds)
{
    static mont_t acc[4], b;
    memset(acc, 0, sizeof(acc));
    memset(&b, 0, sizeof(b));
    for (int k = 0; k < 4; k++) acc[k] = mctx->one;
    b = mctx->one;
    b.l[0] ^= 1;

    double best = 1e30;
    for (int rd = 0; rd < rounds; rd++) {
        const double t0 = now_sec();
        long long i = 0;
        for (; i + 4 <= iters; i += 4) {
            for (int k = 0; k < 4; k++) {
                if (kind == 0) mont_mul(&acc[k], &acc[k], &b, mctx);
                else           mont_sqr(&acc[k], &acc[k], mctx);
            }
        }
        for (; i < iters; i++) mont_mul(&acc[0], &acc[0], &b, mctx);
        const double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    volatile mp_limb_t sink = acc[0].l[0] ^ acc[1].l[3] ^ acc[2].l[7] ^ acc[3].l[11];
    (void)sink;
    return Timing{ best / (double)iters * 1e9 };
}

/* Edwards stage-1 op mix at w-NAF w=12 (docs §13):
   doubling = 4 sqr + 4 mul, addition = 9 mul, add density 1/(w+1) = 1/13
   -> per bit 4 sqr and 4.69 mul  -> sqr share 0.46, mul share 0.54. */
#define MIX_SQR 0.46
#define MIX_MUL 0.54

/* ---- debug: dump raw limbs for a tiny modulus ------------------------- */

static void print_limbs(const char *tag, const uint64_t *e, size_t n, unsigned lane)
{
    printf("  %-10s:", tag);
    for (size_t i = 0; i < n; i++) printf(" %013llx", (unsigned long long)(e[8 * i + lane] & 0xFFFFFFFFFFFFFULL));
    printf("\n");
}

static void print_mpz_limbs(const char *tag, const mpz_t v, size_t n)
{
    mpz_t t;
    mpz_init_set(t, v);
    printf("  %-10s:", tag);
    for (size_t i = 0; i < n; i++) {
        printf(" %013llx", (unsigned long long)mpz_low52(t));
        mpz_fdiv_q_2exp(t, t, 52);
    }
    printf("\n");
    mpz_clear(t);
}

static int dbg_mode(void)
{
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 4242u);

    const size_t bits = 156;                      /* n52 = 3 */
    mpz_t N;
    mpz_init(N);
    random_modulus(N, bits, rs);

    ifma_ctx_t ctx;
    if (ifma_ctx_init(&ctx, N) != 0) { printf("init failed\n"); return 2; }
    const size_t n = ctx.n;
    printf("dbg: %zu bits -> n52=%zu, np0=%013llx\n", bits, n, (unsigned long long)ctx.np0);
    gmp_printf("  N        : %Zx\n", N);
    print_limbs("N limbs", ctx.nb, n, 0);
    print_limbs("one", ctx.one, n, 0);

    std::vector<uint64_t> a(8 * n, 0), b(8 * n, 0), r(8 * n, 0), t(8 * n, 0);
    mpz_t R, Rinv, av, bv, want;
    mpz_inits(R, Rinv, av, bv, want, NULL);
    mpz_set_ui(R, 1);
    mpz_mul_2exp(R, R, 52 * (unsigned long)n);
    mpz_invert(Rinv, R, N);

    for (int trial = 0; trial < 3; trial++) {
        printf("trial %d\n", trial);
        for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
            mpz_urandomm(av, rs, N);
            mpz_urandomm(bv, rs, N);
            ifma_from_mpz_lane(a.data(), lane, av, &ctx);
            ifma_from_mpz_lane(b.data(), lane, bv, &ctx);
        }
        ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);
        print_limbs("a(mont)", a.data(), n, 0);
        print_limbs("b(mont)", b.data(), n, 0);
        print_limbs("out", r.data(), n, 0);

        /* the kernel's own Montgomery-domain exit vs the bench's */
        {
            mpz_t k1, k2, o1, o2;
            mpz_inits(k1, k2, o1, o2, NULL);
            ifma_to_mpz_lane(k1, r.data(), 0, &ctx);
            lanes_to_mpz(k2, r.data(), n, 0);
            mpz_mul(k2, k2, Rinv); mpz_mod(k2, k2, N);
            gmp_printf("  conv  : kernel=%Zx bench=%Zx %s\n", k1, k2, mpz_cmp(k1, k2) ? "DIFF" : "ok");
            /* and both against the pure-mpz ordinary product */
            {
                mont_ctx_t mc;
                mont_t ma, mb, mr;
                if (mont_init(&mc, N) == 0) {
                    lanes_to_mpz(o1, a.data(), n, 0);
                    lanes_to_mpz(o2, b.data(), n, 0);
                    mpz_mul(o1, o1, Rinv); mpz_mod(o1, o1, N);   /* ordinary a */
                    mpz_mul(o2, o2, Rinv); mpz_mod(o2, o2, N);   /* ordinary b */
                    mpz_mul(k2, o1, o2); mpz_mod(k2, k2, N);     /* ground truth */
                    mont_to(&ma, o1, &mc);
                    mont_to(&mb, o2, &mc);
                    mont_mul(&mr, &ma, &mb, &mc);
                    mont_from(o1, &mr, &mc);
                    gmp_printf("  prod  : truth=%Zx gmp=%Zx simd=%Zx  %s\n", k2, o1, k1,
                               (mpz_cmp(k2, o1) || mpz_cmp(k2, k1)) ? "MISMATCH" : "ok");
                    mont_clear(&mc);
                }
            }
            mpz_clears(k1, k2, o1, o2, NULL);
        }

        /* expected = a*b*R^-1 mod N in the same Montgomery domain */
        {
            mpz_t A, B;
            mpz_inits(A, B, NULL);
            mpz_set_ui(A, 0);
            for (size_t i = n; i-- > 0; ) mpz_add_limb52(A, a[8 * i + 0]);
            mpz_set_ui(B, 0);
            for (size_t i = n; i-- > 0; ) mpz_add_limb52(B, b[8 * i + 0]);
            mpz_mul(want, A, B);
            mpz_mod(want, want, N);
            mpz_mul(want, want, Rinv);
            mpz_mod(want, want, N);
            print_mpz_limbs("want", want, n);
            mpz_clears(A, B, NULL);
        }
    }

    /* x * 1 */
    printf("x*1 check\n");
    ifma_set_one(t.data(), &ctx);
    for (unsigned lane = 0; lane < IFMA_LANES; lane++) ifma_from_u64_lane(a.data(), lane, 12345u + lane, &ctx);
    ifma_mont_mul(r.data(), a.data(), t.data(), &ctx);
    print_limbs("x", a.data(), n, 0);
    print_limbs("x*1", r.data(), n, 0);

    mpz_clears(R, Rinv, av, bv, want, NULL);
    ifma_ctx_clear(&ctx);
    mpz_clear(N);
    gmp_randclear(rs);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "dbg") == 0) return dbg_mode();

    size_t bits_list[8];
    int nb = 0;
    for (int i = 1; i < argc && nb < 8; i++) bits_list[nb++] = (size_t)atoi(argv[i]);
    if (nb == 0) {
        bits_list[nb++] = 1000;   /* control: n52 = 20 */
        bits_list[nb++] = 4003;   /* gate:    n52 = 77  */
        bits_list[nb++] = 6500;   /* gate:    n52 = 125 */
        bits_list[nb++] = 8000;   /* measure: n52 = 154 */
    }

    const double ghz = calibrate_ghz();
    printf("simd_mont_gate: %d AVX512-IFMA lanes, measured core clock %.3f GHz\n\n", IFMA_LANES, ghz);

    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 20250924u);

    int gate_fail = 0;
    for (int s = 0; s < nb; s++) {
        const size_t bits = bits_list[s];
        const size_t n52 = (bits + 51) / 52;
        const size_t n64 = (bits + 63) / 64;

        mpz_t N;
        mpz_init(N);
        random_modulus(N, bits, rs);

        ifma_ctx_t ctx;
        if (ifma_ctx_init(&ctx, N) != 0) { printf("ctx init failed\n"); mpz_clear(N); return 2; }
        mont_ctx_t mctx;
        if (mont_init(&mctx, N) != 0) { printf("mont_init failed\n"); mpz_clear(N); return 2; }

        printf("== %zu bit  (n52=%zu, n64=%zu, redc=%s, madds/batch-mul=%llu)\n",
               bits, n52, n64, mctx.use_redc_n ? "redc_n" : "redc_1",
               (unsigned long long)ifma_madd_count(&ctx));

        const CheckResult cr = check_correctness(N, &ctx, &mctx, n52, rs, 9);
        printf("   correctness: %d checks, %d failures  -> %s\n", cr.trials, cr.fails,
               cr.fails ? "BAD" : "ok");

        const long long iters = (bits <= 1200) ? 30000 : (bits <= 4500 ? 8000 : (bits <= 7000 ? 4000 : 2500));
        const int rounds = 7;

        double best_simd_mul = 1e30, best_gmp_mul = 1e30, best_simd_sqr = 1e30, best_gmp_sqr = 1e30;
        /* clock sampled immediately AFTER each block, i.e. while the machine is
           still in the thermal/power state that block ran in (this laptop
           throttles, and the idle-clock calibration above is only a baseline). */
        double ghz_after_simd = 1e30, ghz_after_gmp = 1e30;
        for (int rd = 0; rd < 3; rd++) {          /* alternate A/B blocks */
            Timing a = time_simd(&ctx, 0, iters, rounds);
            const double g1 = calibrate_ghz_n(30000000LL);
            Timing b = time_gmp(&mctx, 0, iters, rounds);
            const double g2 = calibrate_ghz_n(30000000LL);
            Timing c = time_simd(&ctx, 1, iters, rounds);
            const double g3 = calibrate_ghz_n(30000000LL);
            Timing d = time_gmp(&mctx, 1, iters, rounds);
            const double g4 = calibrate_ghz_n(30000000LL);
            (void)g1; (void)g2; (void)g3; (void)g4;
            ghz_after_simd = std::min(ghz_after_simd, std::min(g1, g3));
            ghz_after_gmp = std::min(ghz_after_gmp, std::min(g2, g4));
            best_simd_mul = std::min(best_simd_mul, a.ns_per_op);
            best_gmp_mul  = std::min(best_gmp_mul,  b.ns_per_op);
            best_simd_sqr = std::min(best_simd_sqr, c.ns_per_op);
            best_gmp_sqr  = std::min(best_gmp_sqr,  d.ns_per_op);
        }

        const double simd_mul_per_curve = best_simd_mul / IFMA_LANES;
        const double simd_sqr_per_curve = best_simd_sqr / IFMA_LANES;
        const double r_mul = best_gmp_mul / simd_mul_per_curve;
        const double r_sqr = best_gmp_sqr / simd_sqr_per_curve;
        /* exact weighted ratio, per curve, at the documented ladder op mix */
        const double simd_blend = MIX_MUL * simd_mul_per_curve + MIX_SQR * simd_sqr_per_curve;
        const double gmp_blend  = MIX_MUL * best_gmp_mul + MIX_SQR * best_gmp_sqr;
        const double blend = gmp_blend / simd_blend;

        const double cyc_batch = best_simd_mul * ghz_after_simd;
        const double u = (double)ifma_madd_count(&ctx) / cyc_batch;

        printf("   mul: simd %8.1f ns/batch (%7.1f ns/curve)  gmp %9.1f ns  ratio %5.2fx\n",
               best_simd_mul, simd_mul_per_curve, best_gmp_mul, r_mul);
        printf("   sqr: simd %8.1f ns/batch (%7.1f ns/curve)  gmp %9.1f ns  ratio %5.2fx\n",
               best_simd_sqr, simd_sqr_per_curve, best_gmp_sqr, r_sqr);
        printf("   blended (0.54 mul + 0.46 sqr, w=12 ladder): %5.2fx\n", blend);
        printf("   kernel: %.2f madd/cycle (u), using %.2f GHz sampled right after a\n", u, ghz_after_simd);
        printf("           SIMD block and %.2f GHz after a GMP block; %llu madds in %.0f cycles/batch\n\n",
               ghz_after_gmp, (unsigned long long)ifma_madd_count(&ctx), cyc_batch);

        if (n52 == 77 || n52 == 125) {
            if (blend < 2.0) { printf("   GATE FAIL (n52=%zu needs >= 2.0x)\n\n", n52); gate_fail = 1; }
            else             printf("   GATE PASS (n52=%zu >= 2.0x)\n\n", n52);
        }

        ifma_ctx_clear(&ctx);
        mont_clear(&mctx);
        mpz_clear(N);
    }

    gmp_randclear(rs);
    printf("gate summary: %s\n", gate_fail ? "FAIL" : "PASS");
    return gate_fail ? 1 : 0;
}
