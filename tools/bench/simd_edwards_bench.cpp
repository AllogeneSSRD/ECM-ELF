/* ---------------------------------------------------------------------------
 * simd_edwards_bench.cpp — M2 verification + timing for the batched Edwards
 * stage-1 layer (simd_edwards.{h,cpp}).
 *
 * Two modes:
 *   verify <Ndec> <B1> <sigma> [sigma...]   compare against the production
 *                                           scalar ladder on the same curves
 *   bench  <Ndec> <B1> <rounds>             interleaved A/B/C timing:
 *                                           scalar w=12, scalar w=8, SIMD w=8
 *
 * What "correct" means here: the scalar and SIMD ladders evaluate the same
 * signed-digit expansion of the same exponent s over the same curve, using
 * field ops that both return canonical residues < N, so Qx/Qz must match
 * BIT FOR BIT (not just "same factor").  The dictionary window may differ
 * (w=8 vs w=12) because (2j+1)P does not depend on w.  The factor test
 * gcd(Qz, N) is the externally visible result that must agree with p95.
 *
 * Build: cmake --build build_vs18 --config Release --target simd_edwards_bench
 * ------------------------------------------------------------------------- */

#include "simd_edwards.h"
#include "ecm_edwards_cpu.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

static double now_sec(void)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* s = 48 * lcm(1..B1) — same construction as the scalar standalone main. */
static void build_s(mpz_t s, uint64_t B1)
{
    mpz_set_ui(s, 48);
    for (uint64_t p = 2; p <= B1; ++p) {
        bool prime = true;
        for (uint64_t q = 2; q * q <= p; ++q) if (p % q == 0) { prime = false; break; }
        if (!prime) continue;
        uint64_t v = p;
        while (v <= B1 / p) v *= p;
        mpz_mul_ui(s, s, (unsigned long)v);
    }
}

static void random_modulus(mpz_t N, size_t bits, gmp_randstate_t rs)
{
    mpz_urandomb(N, rs, (mp_bitcnt_t)bits);
    mpz_setbit(N, bits - 1);
    if (mpz_even_p(N)) mpz_add_ui(N, N, 1);
}

/* Holds mpz_t members, so it is deliberately non-copyable: copying an mpz_t
   array shares limb pointers and double-clears.  Fill it by reference. */
struct ScalarResult {
    std::vector<mpz_t> Qx, Qz, factor;
    double sec;
    explicit ScalarResult(int k) : Qx(k), Qz(k), factor(k), sec(0)
    {
        for (int i = 0; i < k; i++) { mpz_init(Qx[i]); mpz_init(Qz[i]); mpz_init(factor[i]); }
    }
    ~ScalarResult()
    {
        for (size_t i = 0; i < Qx.size(); i++) { mpz_clear(Qx[i]); mpz_clear(Qz[i]); mpz_clear(factor[i]); }
    }
    ScalarResult(const ScalarResult &) = delete;
    ScalarResult &operator=(const ScalarResult &) = delete;
};

static void run_scalar(ScalarResult &r, const mpz_t N, const mpz_t s,
                       const std::vector<uint64_t> &sigma, int w)
{
    edwards_set_naf_w(w);
    const double t0 = now_sec();
    for (size_t i = 0; i < sigma.size(); i++) {
        mpz_set_ui(r.factor[i], 1);
        edwards_stage1_curve(r.factor[i], r.Qx[i], r.Qz[i], N, sigma[i], s);
    }
    r.sec = now_sec() - t0;
}

/* ---- checkpoint/resume 往返自测 ---- */
static int      g_cap_lanes;
static size_t   g_cap_digit;
static mpz_t   *g_cap_x, *g_cap_y, *g_cap_z;

static int cap_cb(void *p, size_t done, size_t total, const uint64_t *Rx,
                  const uint64_t *Ry, const uint64_t *Rz)
{
    ed_soa_ctx_t *c = (ed_soa_ctx_t *)p;
    (void)total;
    if (done == 0) return 0;       /* i=0 是空 tick, 继续跑到下一个 16384 边界再捕获 */
    g_cap_digit = done;
    for (int k = 0; k < g_cap_lanes; k++) {
        ifma_to_mpz_lane(g_cap_x[k], Rx, (unsigned)k, &c->mc);
        ifma_to_mpz_lane(g_cap_y[k], Ry, (unsigned)k, &c->mc);
        ifma_to_mpz_lane(g_cap_z[k], Rz, (unsigned)k, &c->mc);
    }
    return 1;                      /* 在第一个回调点中止, 模拟 checkpoint */
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s verify <Ndec> <B1> <sigma> [sigma...]   (up to 8)\n"
                "       %s bench  <Ndec> <B1> <rounds>\n", argv[0], argv[0]);
        return 1;
    }
    const bool bench = strcmp(argv[1], "bench") == 0;
    mpz_t N, s;
    mpz_inits(N, s, NULL);

    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 987654321u);

    if (strncmp(argv[2], "bits:", 5) == 0) random_modulus(N, (size_t)atoi(argv[2] + 5), rs);
    else mpz_set_str(N, argv[2], 10);
    const uint64_t B1 = strtoull(argv[3], NULL, 10);
    build_s(s, B1);
    /* debugging aid: force a small exponent so the ladder can be traced */
    if (const char *ov = getenv("ED_SOA_S_OVERRIDE")) mpz_set_str(s, ov, 10);

    std::vector<uint64_t> sigma;
    int w_simd = 8;
    int rounds = 1;
    if (bench) {
        for (int i = 0; i < IFMA_LANES; i++) sigma.push_back(1000u + 7u * (uint64_t)i);
        rounds = (argc > 4) ? atoi(argv[4]) : 1;
        if (argc > 5) w_simd = atoi(argv[5]);          /* dictionary window */
        if (w_simd < 3 || w_simd > 12) { fprintf(stderr, "w must be 3..12\n"); return 1; }
    } else {
        for (int i = 4; i < argc && (int)sigma.size() < IFMA_LANES; i++) sigma.push_back(strtoull(argv[i], NULL, 10));
        while (sigma.size() < IFMA_LANES) sigma.push_back(sigma.back());   /* pad: only lanes < given are compared */
    }
    const int lanes = (int)sigma.size();

    printf("N       : %zu bits\n", mpz_sizeinbase(N, 2));
    printf("B1      : %llu, s_bits = %zu, sigmas = %d, SIMD w = %d (m = %d)\n",
           (unsigned long long)B1, mpz_sizeinbase(s, 2), lanes, w_simd, 1 << (w_simd - 2));
    fflush(stdout);

    ed_soa_ctx_t ctx;
    if (ed_soa_init(&ctx, N, w_simd) != 0) { fprintf(stderr, "ed_soa_init failed\n"); return 2; }

    {
        const int sf = ed_soa_field_selftest(&ctx, 3);
        printf("field ops: add/sub/neg vs mpz: %d failures -> %s\n", sf, sf ? "BAD" : "ok");
        const int sp = ed_soa_point_selftest(&ctx, 2);
        printf("point ops: dbl/add/add_affine vs mpz: %d failures -> %s\n", sp, sp ? "BAD" : "ok");
        fflush(stdout);
    }

    const double t_set0 = now_sec();
    if (ed_soa_set_curves(&ctx, sigma.data(), lanes) != 0) { fprintf(stderr, "set_curves failed\n"); return 2; }
    const double t_set = now_sec() - t_set0;
    printf("dict    : %zu entries, %.2f MB, build %.3f s\n",
           ctx.m, (double)ed_soa_dict_bytes(&ctx) / (1024.0 * 1024.0), t_set);
    fflush(stdout);

    if (!bench) {
        /* Compare the setup the SIMD context actually holds against a fresh
           Atkin-Morain evaluation, so a setup bug is visible before the ladder. */
        if (getenv("ED_SOA_DEBUG")) {
            mpz_t d0, px0, py0, got;
            mpz_inits(d0, px0, py0, got, NULL);
            edwards_atkin_morain(d0, px0, py0, sigma[0], N);
            const size_t lw = 8 * ctx.n;
            ifma_to_mpz_lane(got, ctx.d, 0, &ctx.mc);
            gmp_printf("  setup d   : %s\n", mpz_cmp(got, d0) == 0 ? "ok" : "DIFF");
            ifma_to_mpz_lane(got, ctx.dict + 0 * lw, 0, &ctx.mc);
            gmp_printf("  setup P.x : %s\n", mpz_cmp(got, px0) == 0 ? "ok" : "DIFF");
            ifma_to_mpz_lane(got, ctx.dict + 1 * lw, 0, &ctx.mc);
            gmp_printf("  setup P.y : %s\n", mpz_cmp(got, py0) == 0 ? "ok" : "DIFF");
            /* dict[1] must be 3P: check against 2P+P in the scalar mpz domain */
            mpz_clears(d0, px0, py0, got, NULL);
            fflush(stdout);
        }
        ScalarResult sc(lanes);
        run_scalar(sc, N, s, sigma, w_simd);
        std::vector<mpz_t> sx(lanes), sz(lanes), sf(lanes);
        for (int i = 0; i < lanes; i++) { mpz_inits(sx[i], sz[i], sf[i], NULL); }
        const double t0 = now_sec();
        ed_soa_stage1(&ctx, s, lanes, sx.data(), sz.data(), sf.data());
        const double simd_sec = now_sec() - t0;

        int fails = 0;
        for (int i = 0; i < lanes; i++) {
            const bool qx_ok = mpz_cmp(sx[i], sc.Qx[i]) == 0;
            const bool qz_ok = mpz_cmp(sz[i], sc.Qz[i]) == 0;
            const bool f_ok = mpz_cmp(sf[i], sc.factor[i]) == 0;
            if (getenv("ED_SOA_DEBUG") && i == 0) {
                const size_t lw = 8 * ctx.n;
                mpz_t dv, dx, dxy;
                mpz_inits(dv, dx, dxy, NULL);
                edwards_atkin_morain(dv, dx, dxy, sigma[0], N);   /* dv=d, dx=Px */
                gmp_printf("  trace lane0: d=%Zd\n", dv);
                gmp_printf("  trace lane0: Px=%Zd\n", dx);
                gmp_printf("  trace lane0: Py=%Zd\n", dxy);
                { mpz_t t2; mpz_init(t2); mpz_mul(t2, dx, dxy); mpz_mul(t2, t2, dv);
                  mpz_mod(t2, t2, N); gmp_printf("  trace lane0: dict0 dxy=%Zd\n", t2); mpz_clear(t2); }
                gmp_printf("  trace lane0: simd Qx=%Zd\n  trace lane0: simd Qz=%Zd\n", sx[0], sz[0]);
                gmp_printf("  trace lane0: gmp  Qx=%Zd\n  trace lane0: gmp  Qz=%Zd\n", sc.Qx[0], sc.Qz[0]);
                mpz_clears(dv, dx, dxy, NULL);
                (void)lw;
                fflush(stdout);
            }
            if (!qx_ok || !qz_ok || !f_ok) {
                fails++;
                printf("  lane %d MISMATCH: Qx %s Qz %s factor %s\n", i,
                       qx_ok ? "ok" : "DIFF", qz_ok ? "ok" : "DIFF", f_ok ? "ok" : "DIFF");
                if (!qz_ok) gmp_printf("    simd Qz = %Zx\n    gmp  Qz = %Zx\n", sz[i], sc.Qz[i]);
            } else {
                gmp_printf("  lane %d ok  factor = %Zd\n", i, sf[i]);
            }
        }
        printf("scalar  : %d curves in %.3f s (%.3f s/curve, w=%d)\n", lanes, sc.sec, sc.sec / lanes, w_simd);

        /* ---- resume 往返自测: 跑到第一个回调点中止 -> set_resume -> 跑完 ---- */
        if (mpz_sizeinbase(s, 2) > (size_t)ED_SOA_PROGRESS_BITS) {
            std::vector<mpz_t> cx(lanes), cy(lanes), cz(lanes), rx2(lanes), rz2(lanes), rf2(lanes);
            for (int i = 0; i < lanes; i++) {
                mpz_inits(cx[i], cy[i], cz[i], rx2[i], rz2[i], rf2[i], NULL);
            }
            g_cap_lanes = lanes; g_cap_digit = 0;
            g_cap_x = cx.data(); g_cap_y = cy.data(); g_cap_z = cz.data();

            ed_soa_set_progress(&ctx, cap_cb, &ctx);
            const int ab = ed_soa_stage1(&ctx, s, lanes, rx2.data(), rz2.data(), rf2.data());
            ed_soa_set_progress(&ctx, NULL, NULL);

            int sres = -99, rfail = -1;
            if (ab == 1 && g_cap_digit > 0) {
                sres = ed_soa_set_resume(&ctx, g_cap_digit, lanes, cx.data(), cy.data(), cz.data());
                if (sres == 0) {
                    ed_soa_stage1(&ctx, s, lanes, rx2.data(), rz2.data(), rf2.data());
                    rfail = 0;
                    for (int i = 0; i < lanes; i++) {
                        if (mpz_cmp(rx2[i], sx[i]) != 0 || mpz_cmp(rz2[i], sz[i]) != 0 ||
                            mpz_cmp(rf2[i], sf[i]) != 0) rfail++;
                    }
                }
            }
            printf("resume  : aborted@digit=%zu of s_bits=%zu, set_resume=%d, "
                   "lanes differing after resume = %d -> %s\n",
                   g_cap_digit, (size_t)mpz_sizeinbase(s, 2), sres, rfail,
                   (ab == 1 && sres == 0 && rfail == 0) ? "OK" : "BAD");
            for (int i = 0; i < lanes; i++) {
                mpz_clears(cx[i], cy[i], cz[i], rx2[i], rz2[i], rf2[i], NULL);
            }
        }        printf("simd    : %d curves in %.3f s (%.3f s/curve, %dx batch)\n",
               lanes, simd_sec + t_set, (simd_sec + t_set) / lanes, IFMA_LANES);
        printf("stage-1 ratio (incl. dictionary, fair vs scalar): %.2fx\n", sc.sec / (simd_sec + t_set));
        printf("ladder-only ratio (dict excluded both sides):     %.2fx\n",
               (sc.sec - 0.0) / simd_sec);
        printf("result: %s\n", fails ? "FAIL" : "PASS");
        for (int i = 0; i < lanes; i++) { mpz_clears(sx[i], sz[i], sf[i], NULL); }
        ed_soa_clear(&ctx);
        mpz_clears(N, s, NULL);
        gmp_randclear(rs);
        return fails ? 1 : 0;
    }

    /* ---- bench: interleaved A/B/C, min of `rounds` ----
       NOTE: ScalarResult owns mpz_t's, so it must never be copied (mpz_t is an
       array type whose copy would share limb pointers and double-clear); only
       the times are retained across rounds. */
    double best12 = 1e30, best8 = 1e30, best_simd = 1e30;
    std::vector<mpz_t> bx(lanes), bz(lanes), bf(lanes);
    for (int i = 0; i < lanes; i++) mpz_inits(bx[i], bz[i], bf[i], NULL);

    double last_simd = 0;
    for (int rd = 0; rd < rounds; rd++) {
        ScalarResult r12(lanes), r8(lanes);
        run_scalar(r12, N, s, sigma, 12);
        run_scalar(r8, N, s, sigma, w_simd);
        const double t0 = now_sec();
        ed_soa_stage1(&ctx, s, lanes, bx.data(), bz.data(), bf.data());
        const double ladder = now_sec() - t0;
        const double simd_sec = ladder + t_set;    /* dictionary is part of stage-1 */

        if (r12.sec < best12) best12 = r12.sec;
        if (r8.sec < best8) best8 = r8.sec;
        if (simd_sec < best_simd) { best_simd = simd_sec; last_simd = ladder; }

        /* agreement check every round: the ladder result must be identical */
        int fails = 0;
        for (int i = 0; i < lanes; i++)
            if (mpz_cmp(bx[i], r8.Qx[i]) != 0 || mpz_cmp(bz[i], r8.Qz[i]) != 0 ||
                mpz_cmp(bf[i], r8.factor[i]) != 0) fails++;
        printf("round %d: scalar w=12 %.3f s, scalar w=8 %.3f s, simd %.3f s | %s\n",
               rd, r12.sec, r8.sec, simd_sec, fails ? "RESULT MISMATCH" : "results identical");
        fflush(stdout);
    }

    const double per12 = best12 / lanes, per8 = best8 / lanes, per_simd = best_simd / lanes;
    printf("\nper curve: scalar w=12 %.3f s | scalar w=8 %.3f s | simd w=8 %.3f s (ladder %.3f s + dict/8 %.3f s)\n",
           per12, per8, per_simd, last_simd / lanes, t_set / lanes);
    printf("speedup  : vs scalar w=12 %.2fx | vs scalar w=8 %.2fx\n", per12 / per_simd, per8 / per_simd);

    for (int i = 0; i < lanes; i++) mpz_clears(bx[i], bz[i], bf[i], NULL);
    ed_soa_clear(&ctx);
    mpz_clears(N, s, NULL);
    gmp_randclear(rs);
    return 0;
}
