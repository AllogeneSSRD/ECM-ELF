/* mers_test.cpp — standalone correctness + timing for the Mersenne fold kernel.
 *
 *   cl /nologo /O2 /arch:AVX512 /I third_party\gmp-zen3\dist\include \
 *      mers_test.cpp src\cpu\simd_mont_ifma.cpp /Fe:mers_test.exe \
 *      /link /LIBPATH:third_party\gmp-zen3\dist\lib gmp.lib
 *
 * Checks ifma_mont_mul/sqr against mpz in BOTH field modes, with random,
 * adversarial and structured (zero-limb, 2^52j, top-boundary) inputs.
 */
#include "simd_mont_ifma.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

static double now_s(void)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void mpz_add_limb52(mpz_t out, uint64_t limb)
{
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)((limb >> 26) & 0x3FFFFFFu));
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)(limb & 0x3FFFFFFu));
}

static void lane_to_mpz(mpz_t out, const uint64_t *e, size_t n, unsigned lane)
{
    mpz_set_ui(out, 0);
    for (size_t i = n; i-- > 0; ) mpz_add_limb52(out, e[8 * i + lane] & 0xFFFFFFFFFFFFFULL);
}

struct Res { int trials, fails; };

/* one modulus: correctness in both modes */
static Res test_modulus(const mpz_t N, const char *tag, int trials, gmp_randstate_t rs)
{
    Res res = { 0, 0 };
    for (int mode = 0; mode < 2; mode++) {
        ifma_ctx_t ctx;
        const int want_mode = mode ? IFMA_FIELD_MERS : IFMA_FIELD_MONT;
        const int rc = ifma_ctx_init_ex(&ctx, N, want_mode);
        if (rc != 0) {
            if (mode == 1) { printf("  %s: mersenne mode rejected (rc=%d) -- skipped\n", tag, rc); continue; }
            printf("  %s: mont init failed rc=%d\n", tag, rc);
            res.fails++;
            continue;
        }
        if (ctx.mode != want_mode) {
            printf("  %s: mode %d not honoured (got %d)\n", tag, want_mode, ctx.mode);
            res.fails++;
            ifma_ctx_clear(&ctx);
            continue;
        }
        const size_t n = ctx.n;
        std::vector<uint64_t> a(8 * n), b(8 * n), r(8 * n), s(8 * n), one(8 * n);
        mpz_t av, bv, want, got, tmp;
        mpz_inits(av, bv, want, got, tmp, NULL);

        for (int trial = 0; trial < trials; trial++) {
            const int kind = trial % 5;
            for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
                mpz_urandomm(av, rs, N);
                mpz_urandomm(bv, rs, N);
                switch (kind) {
                case 1: switch ((lane + trial) & 3) {          /* 0 / 1 / N-1 / N-2 */
                        case 0: mpz_set_ui(av, 0); break;
                        case 1: mpz_set_ui(av, 1); break;
                        case 2: mpz_sub_ui(av, N, 1); break;
                        default: mpz_sub_ui(av, N, 2); break; }
                        switch ((lane + 1 + trial) & 3) {
                        case 0: mpz_set_ui(bv, 0); break;
                        case 1: mpz_set_ui(bv, 1); break;
                        case 2: mpz_sub_ui(bv, N, 1); break;
                        default: mpz_sub_ui(bv, N, 2); break; }
                        break;
                case 2: mpz_set_ui(av, 0); mpz_setbit(av, 52 * (size_t)(1 + lane % 3)); break;   /* sparse limbs */
                case 3: mpz_sub_ui(av, N, 1); mpz_sub_ui(bv, N, 1); break;                       /* (N-1)^2 */
                case 4: {                                                                        /* near 2^k */
                        const size_t k = mpz_sizeinbase(N, 2);
                        mpz_set_ui(av, 1); mpz_mul_2exp(av, av, k);       /* 2^k, may be >= N */
                        mpz_sub_ui(av, av, (unsigned long)(lane % 2));
                        mpz_urandomm(bv, rs, N);
                        break; }
                default: break;
                }
                ifma_from_mpz_lane(a.data(), lane, av, &ctx);
                ifma_from_mpz_lane(b.data(), lane, bv, &ctx);
            }
            ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);
            ifma_mont_sqr(s.data(), a.data(), &ctx);
            ifma_set_one(one.data(), &ctx);

            for (unsigned lane = 0; lane < IFMA_LANES; lane++) {
                /* ordinary values that the kernel actually received */
                ifma_to_mpz_lane(av, a.data(), lane, &ctx);
                ifma_to_mpz_lane(bv, b.data(), lane, &ctx);
                mpz_mul(want, av, bv);
                mpz_mod(want, want, N);
                ifma_to_mpz_lane(got, r.data(), lane, &ctx);
                /* in mersenne mode the domain IS the plain one, so the raw limbs
                   must agree with the converted value (independent check) */
                if (ctx.mode == IFMA_FIELD_MERS) {
                    lane_to_mpz(tmp, r.data(), n, lane);
                    if (mpz_cmp(tmp, got) != 0) {
                        if (res.fails < 12) {
                            gmp_printf("  !! %s non-canonical raw=%Zx  (N=%Zx)  want=%Zx\n",
                                       tag, tmp, N, want);
                        }
                        res.fails++;
                    }
                }
                res.trials++;
                if (mpz_cmp(want, got) != 0) {
                    if (res.fails < 4)
                        gmp_printf("  !! %s mode=%d trial=%d lane=%u mul MISMATCH\n"
                                   "     a=%Zx\n     b=%Zx\n     want=%Zx\n     got =%Zx\n",
                                   tag, mode, trial, lane, av, bv, want, got);
                    res.fails++;
                }
                if (mpz_cmp(got, N) >= 0) {
                    if (res.fails < 6) printf("  !! %s mode=%d not reduced\n", tag, mode);
                    res.fails++;
                }
                /* square */
                mpz_mul(want, av, av); mpz_mod(want, want, N);
                ifma_to_mpz_lane(got, s.data(), lane, &ctx);
                res.trials++;
                if (mpz_cmp(want, got) != 0) {
                    if (res.fails < 8) gmp_printf("  !! %s mode=%d sqr MISMATCH a=%Zx\n", tag, mode, av);
                    res.fails++;
                }
                /* x*1 == x */
                {
                    std::vector<uint64_t> t1(8 * n);
                    ifma_mont_mul(t1.data(), a.data(), one.data(), &ctx);
                    ifma_to_mpz_lane(got, t1.data(), lane, &ctx);
                    res.trials++;
                    if (mpz_cmp(got, av) != 0) {
                        if (res.fails < 10) gmp_printf("  !! %s mode=%d x*1 != x\n", tag, mode);
                        res.fails++;
                    }
                }
            }
        }
        mpz_clears(av, bv, want, got, tmp, NULL);
        ifma_ctx_clear(&ctx);
    }
    return res;
}

int main(int argc, char **argv)
{
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 12345);

    const int trials = (argc > 1) ? atoi(argv[1]) : 40;
    Res total = { 0, 0 };
    mpz_t N;
    mpz_init(N);

    /* the acceptance modulus plus a spread of k that exercises every sh */
    static const size_t ks[] = { 3001, 991, 677, 2203, 4003, 6000, 8011, 9850,
                                 2964, 3016, 2965, 3002, 3000, 3003, 12323, 1024, 100 };
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) {
        mpz_set_ui(N, 1);
        mpz_mul_2exp(N, N, (mp_bitcnt_t)ks[i]);
        mpz_sub_ui(N, N, 1);
        char tag[64];
        snprintf(tag, sizeof(tag), "2^%zu-1", ks[i]);
        const Res r = test_modulus(N, tag, trials, rs);
        printf("%-12s trials=%5d fails=%d%s\n", tag, r.trials, r.fails, r.fails ? "   <== FAIL" : "");
        fflush(stdout);
        total.trials += r.trials;
        total.fails += r.fails;
    }

    /* a non-Mersenne modulus: AUTO must fall back to Montgomery */
    {
        mpz_set_ui(N, 1);
        mpz_mul_2exp(N, N, 3001);
        mpz_sub_ui(N, N, 3);
        if (mpz_even_p(N)) mpz_add_ui(N, N, 1);
        ifma_ctx_t ctx;
        const int rc = ifma_ctx_init_ex(&ctx, N, IFMA_FIELD_AUTO);
        printf("auto on 2^3001-3: rc=%d mode=%s\n", rc, rc == 0 ? ifma_field_name(&ctx) : "-");
        if (rc == 0) ifma_ctx_clear(&ctx);
        const int rc2 = ifma_ctx_init_ex(&ctx, N, IFMA_FIELD_MERS);
        printf("force mersenne on 2^3001-3: rc=%d (expect non-zero)\n", rc2);
        if (rc2 == 0) ifma_ctx_clear(&ctx);
    }

    /* timing on the acceptance modulus */
    {
        mpz_set_ui(N, 1);
        mpz_mul_2exp(N, N, 3001);
        mpz_sub_ui(N, N, 1);
        for (int mode = 0; mode < 2; mode++) {
            ifma_ctx_t ctx;
            if (ifma_ctx_init_ex(&ctx, N, mode ? IFMA_FIELD_MERS : IFMA_FIELD_MONT) != 0) continue;
            const size_t n = ctx.n;
            std::vector<uint64_t> a(8 * n), b(8 * n), r(8 * n);
            for (size_t i = 0; i < 8 * n; i++) { a[i] = (uint64_t)(i * 2654435761u + 1) & 0xFFFFFFFFFFFFFULL; b[i] = (uint64_t)(i * 40503u + 7) & 0xFFFFFFFFFFFFFULL; }
            const int iters = 20000;
            ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);          /* warm */
            double t0 = now_s();
            for (int i = 0; i < iters; i++) ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);
            const double dt = now_s() - t0;
            ifma_mont_sqr(r.data(), a.data(), &ctx);
            double t1 = now_s();
            for (int i = 0; i < iters; i++) ifma_mont_sqr(r.data(), a.data(), &ctx);
            const double dt2 = now_s() - t1;
            const double ns = dt / iters * 1e9;
            const double nsq = dt2 / iters * 1e9;
            const uint64_t madds = ifma_madd_count(&ctx);
            const uint64_t smadds = ifma_sqr_madd_count(&ctx);
            const double mixed = 0.5 * ns + 0.5 * nsq;   /* a doubling is 4S+4M */
            printf("3001 bit %-24s mul %8.1f ns (%llu madds, %.2f Gmadd/s)  sqr %8.1f ns (%llu madds, %.2f Gmadd/s)  4S+4M=%7.1f ns/curve\n",
                   ifma_field_name(&ctx), ns, (unsigned long long)madds, madds / ns,
                   nsq, (unsigned long long)smadds, smadds / nsq, mixed * 4 / 8.0);
            ifma_ctx_clear(&ctx);
        }
    }

    /* timing sweep on Mersenne moduli: alternating A/B, min of 3 rounds */
    {
        static const size_t tks[] = { 1000, 2203, 3001, 4003, 6000, 8011, 9850 };
        printf("\n  k    n52 | mont ns  mers ns | ratio | mont madd/c  mers madd/c\n");
        mpz_t Nm;
        mpz_init(Nm);
        for (size_t i = 0; i < sizeof(tks) / sizeof(tks[0]); i++) {
            mpz_set_ui(Nm, 1);
            mpz_mul_2exp(Nm, Nm, (mp_bitcnt_t)tks[i]);
            mpz_sub_ui(Nm, Nm, 1);
            double best[2] = { 1e30, 1e30 };
            uint64_t madds[2] = { 0, 0 };
            size_t nn = 0;
            for (int round = 0; round < 3; round++) {
                for (int mode = 0; mode < 2; mode++) {
                    ifma_ctx_t ctx;
                    if (ifma_ctx_init_ex(&ctx, Nm, mode ? IFMA_FIELD_MERS : IFMA_FIELD_MONT) != 0) continue;
                    const size_t n = ctx.n;
                    nn = n;
                    std::vector<uint64_t> a(8 * n), b(8 * n), r(8 * n);
                    for (size_t j = 0; j < 8 * n; j++) {
                        a[j] = (uint64_t)(j * 2654435761u + 1) & 0xFFFFFFFFFFFFFULL;
                        b[j] = (uint64_t)(j * 40503u + 7) & 0xFFFFFFFFFFFFFULL;
                    }
                    const int iters = 20000;
                    ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);
                    const double t0 = now_s();
                    for (int it = 0; it < iters; it++) ifma_mont_mul(r.data(), a.data(), b.data(), &ctx);
                    const double ns = (now_s() - t0) / iters * 1e9;
                    if (ns < best[mode]) best[mode] = ns;
                    madds[mode] = ifma_madd_count(&ctx);
                    ifma_ctx_clear(&ctx);
                }
            }
            printf("  %-5zu %3zu | %7.1f  %7.1f | %5.2fx | %6.2f       %6.2f\n",
                   tks[i], nn, best[0], best[1], best[0] / best[1],
                   madds[0] / (best[0] * 4.0), madds[1] / (best[1] * 4.0));
            fflush(stdout);
        }
        mpz_clear(Nm);
        printf("\n");
    }

    mpz_clear(N);
    gmp_randclear(rs);
    printf("TOTAL trials=%d fails=%d -> %s\n", total.trials, total.fails, total.fails ? "FAIL" : "PASS");
    return total.fails ? 1 : 0;
}
