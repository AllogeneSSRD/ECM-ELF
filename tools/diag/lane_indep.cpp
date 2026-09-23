/* lane_indep.cpp — is lane 0's result independent of the other lanes' contents?
 * Computes the same lane-0 product three times with different garbage in lanes
 * 1..7 and compares lane 0 only.  A SIMD-8 kernel must be lane independent.
 */
#include "simd_mont_ifma.h"
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 3001;
    const int mode = (argc > 2) ? atoi(argv[2]) : IFMA_FIELD_MERS;
    const int trials = (argc > 3) ? atoi(argv[3]) : 50;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);
    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, mode) != 0) { printf("init failed\n"); return 2; }
    const size_t n = c.n, lw = 8 * n;
    printf("%s n=%zu\n", ifma_field_name(&c), n);

    std::vector<uint64_t> A(lw), B(lw), R0(lw), R1(lw), R2(lw);
    unsigned long long seed = 12345;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return (uint64_t)(seed >> 13) & 0xFFFFFFFFFFFFFULL; };
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 77u);

    int lane_dep_mul = 0, lane_dep_sqr = 0, lane_dep_add = 0;
    for (int t = 0; t < trials; t++) {
        mpz_t av, bv; mpz_inits(av, bv, NULL);
        mpz_urandomm(av, rs, N); mpz_urandomm(bv, rs, N);
        for (unsigned lane = 0; lane < 8; lane++) {
            ifma_from_mpz_lane(A.data(), lane, av, &c);
            ifma_from_mpz_lane(B.data(), lane, bv, &c);
        }
        ifma_mont_mul(R0.data(), A.data(), B.data(), &c);          /* all lanes equal */
        /* variant 1: lanes 1..7 = 0 */
        for (unsigned lane = 1; lane < 8; lane++)
            for (size_t i = 0; i < n; i++) { A[8 * i + lane] = 0; B[8 * i + lane] = 0; }
        ifma_mont_mul(R1.data(), A.data(), B.data(), &c);
        /* variant 2: lanes 1..7 = garbage */
        for (unsigned lane = 1; lane < 8; lane++)
            for (size_t i = 0; i < n; i++) { A[8 * i + lane] = rnd(); B[8 * i + lane] = rnd(); }
        ifma_mont_mul(R2.data(), A.data(), B.data(), &c);
        for (size_t i = 0; i < n; i++) {
            if (R0[8 * i] != R1[8 * i] || R0[8 * i] != R2[8 * i]) { lane_dep_mul++; break; }
        }
        /* same for sqr */
        for (unsigned lane = 0; lane < 8; lane++) ifma_from_mpz_lane(A.data(), lane, av, &c);
        ifma_mont_sqr(R0.data(), A.data(), &c);
        for (unsigned lane = 1; lane < 8; lane++)
            for (size_t i = 0; i < n; i++) A[8 * i + lane] = rnd();
        ifma_mont_sqr(R1.data(), A.data(), &c);
        for (size_t i = 0; i < n; i++) {
            if (R0[8 * i] != R1[8 * i]) { lane_dep_sqr++; break; }
        }
        mpz_clears(av, bv, NULL);
    }
    printf("lane-dependent: mul=%d/%d sqr=%d/%d (trials)\n", lane_dep_mul, trials, lane_dep_sqr, trials);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    gmp_randclear(rs);
    return 0;
}
