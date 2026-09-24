/* ---------------------------------------------------------------------------
 * mont_simd_verify.cpp -- M2 acceptance: the 8-lane SIMD Suyama-Montgomery stage 1
 * must agree, lane by lane, with the scalar MPN path.
 *
 * For a batch of sigmas and a given (N, B1) it checks per lane
 *   * gcd(Z,N) identical (stage-1 verdict), and
 *   * the normalised Montgomery x byte-identical whenever neither side hit.
 * It also reports the first timing comparison (batch wall time vs 8 scalar runs).
 *
 * usage: mont_simd_verify <N> <B1> <sigma0> [count] [torsion]
 * ------------------------------------------------------------------------- */
#include "simd_mont_curve.h"
#include "ecm_mont_cpu.h"
#include "ecm_save.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <N> <B1> <sigma0> [count] [torsion]\n", argv[0]);
        return 1;
    }
    mpz_t N, s;
    mpz_inits(N, s, NULL);
    if (mpz_set_str(N, argv[1], 10) != 0) { fprintf(stderr, "bad N\n"); return 1; }
    const uint64_t B1 = strtoull(argv[2], nullptr, 10);
    const uint64_t sigma0 = strtoull(argv[3], nullptr, 10);
    int count = (argc > 4) ? atoi(argv[4]) : 8;
    if (count < 1) count = 1;
    if (count > IFMA_LANES) count = IFMA_LANES;
    const uint64_t torsion = (argc > 5) ? strtoull(argv[5], nullptr, 10) : 1;

    const size_t s_bits = mont_build_s(s, B1, torsion);
    uint64_t sigmas[IFMA_LANES];
    for (int k = 0; k < IFMA_LANES; k++) sigmas[k] = sigma0 + (uint64_t)k;

    mont_soa_ctx_t ctx;
    if (mont_soa_init(&ctx, N, IFMA_FIELD_AUTO) != 0) { fprintf(stderr, "init failed\n"); return 2; }
    printf("N bits=%zu  s_bits=%zu  field=%s  B1=%llu  torsion=%llu  curves=%d\n",
           mpz_sizeinbase(N, 2), s_bits, ifma_field_name(&ctx.mc),
           (unsigned long long)B1, (unsigned long long)torsion, count);

    mpz_t sx[IFMA_LANES], sg[IFMA_LANES];
    for (int k = 0; k < IFMA_LANES; k++) { mpz_inits(sx[k], sg[k], NULL); }

    const auto t0 = std::chrono::steady_clock::now();
    mont_soa_stage1(&ctx, s, sigmas, sx, sg);
    const auto t1 = std::chrono::steady_clock::now();
    const double simd_s = std::chrono::duration<double>(t1 - t0).count();

    /* scalar reference, one curve at a time (same s, same convention) */
    int fails = 0;
    double scalar_s = 0.0;
    for (int k = 0; k < count; k++) {
        mpz_t xx, gg;
        mpz_inits(xx, gg, NULL);
        const auto a0 = std::chrono::steady_clock::now();
        mont_stage1_curve_x(gg, xx, N, sigmas[k], s);
        const auto a1 = std::chrono::steady_clock::now();
        scalar_s += std::chrono::duration<double>(a1 - a0).count();

        const int hit_simd = (mpz_cmp_ui(sg[k], 1) > 0);
        const int hit_scalar = (mpz_cmp_ui(gg, 1) > 0);
        int ok = (hit_simd == hit_scalar);
        if (ok && !hit_simd) ok = (mpz_cmp(sx[k], xx) == 0);
        if (!ok) {
            fails++;
            char *a = mpz_get_str(nullptr, 16, sx[k]);
            char *b = mpz_get_str(nullptr, 16, xx);
            printf("  lane %d FAIL: gcd simd=%s scalar=%s | x simd=%.32s scalar=%.32s\n",
                   k, hit_simd ? ">1" : "1", hit_scalar ? ">1" : "1", a, b);
            free(a); free(b);
        } else {
            char *g = mpz_get_str(nullptr, 10, sg[k]);
            printf("  lane %d ok  %s  gcd=%s\n", k, hit_simd ? "HIT " : "miss", g);
            free(g);
        }
        mpz_clears(xx, gg, NULL);
    }
    printf("\nTIMING: simd batch(8) = %.4f s   scalar x%d = %.4f s   speedup = %.2fx\n",
           simd_s, count, scalar_s, scalar_s > 0 ? scalar_s / simd_s : 0.0);

    /* Optional: one SHARED save file for the whole task (same N and B1, many curves).
       Mirrors the reference layout (e.g. gmp-ecm's 3001_B1e5.save = 10 lines in one
       file): one self-contained line per curve, because the reference reader parses
       METHOD/B1/N from every line. */
    if (argc > 6) {
        mpz_t xs[IFMA_LANES];
        int hit[IFMA_LANES];
        for (int k = 0; k < IFMA_LANES; k++) {
            mpz_init(xs[k]);
            hit[k] = (mpz_cmp_ui(sg[k], 1) > 0) ? 1 : 0;
            if (hit[k]) mpz_set(xs[k], sg[k]);      /* hit: store the factor */
            else        mpz_set(xs[k], sx[k]);      /* miss: the normalised x */
        }
        /* the save writer takes the FULL per-curve sigma array (a run with random
           sigmas has no "base + i" relationship; see ecm_save.h) */
        uint64_t sigma_arr[IFMA_LANES];
        for (int k = 0; k < IFMA_LANES; k++) sigma_arr[k] = sigmas[k];
        const bool ok = ecm_append_save_lines_mont(argv[6], N, (double)B1, sigma_arr,
                                                   (uint32_t)IFMA_LANES, xs, hit, argv[1]);
        printf("shared save file: %s -> %s (%u curve lines)\n",
               ok ? "written" : "FAILED", argv[6], (unsigned)IFMA_LANES);
        for (int k = 0; k < IFMA_LANES; k++) mpz_clear(xs[k]);
    }

    for (int k = 0; k < IFMA_LANES; k++) mpz_clears(sx[k], sg[k], NULL);
    mont_soa_clear(&ctx);
    mpz_clears(N, s, NULL);
    printf("RESULT: %s (%d/%d lanes disagree)\n", fails ? "FAIL" : "PASS", fails, count);
    return fails ? 1 : 0;
}
