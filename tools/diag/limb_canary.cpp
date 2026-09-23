/* dirt.cpp -- do the kernels ever leave bits >= 2^52 in a result limb?
   Every mpz-level check masks limbs with IFMA_MASK52 (and ifma_to_mpz_lane()
   additionally applies mpz_mod), so a result whose limbs are "correct mod 2^52
   but dirty above" passes all of them while breaking any consumer that assumes
   canonical 52-bit limbs (soa_add/soa_sub/soa_cond_sub).

   usage: dirt <k> <mode 0=mont 1=mers> <trials> */
#include "simd_mont_ifma.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static const uint64_t MASK52 = 0xFFFFFFFFFFFFFULL;

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 4003;
    const int mode = (argc > 2) ? atoi(argv[2]) : IFMA_FIELD_MONT;
    const int trials = (argc > 3) ? atoi(argv[3]) : 20;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);

    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, mode) != 0) { printf("init failed\n"); return 2; }
    const size_t n = c.n, lw = 8 * n;

    std::vector<uint64_t> A(lw), B(lw), M(lw), S(lw);
    mpz_t av, bv;
    mpz_inits(av, bv, NULL);
    gmp_randstate_t rs;
    gmp_randinit_mt(rs); gmp_randseed_ui(rs, 4242u);

    int dirty_mul = 0, dirty_sqr = 0, dirty_in = 0, shown = 0;
    for (int t = 0; t < trials; t++) {
        for (unsigned lane = 0; lane < 8; lane++) {
            mpz_urandomm(av, rs, N);
            mpz_urandomm(bv, rs, N);
            ifma_from_mpz_lane(A.data(), lane, av, &c);
            ifma_from_mpz_lane(B.data(), lane, bv, &c);
        }
        ifma_mont_mul(M.data(), A.data(), B.data(), &c);
        ifma_mont_sqr(S.data(), A.data(), &c);
        for (unsigned lane = 0; lane < 8; lane++) {
            int d = 0, di = 0;
            for (size_t i = 0; i < n; i++) {
                if (M[8 * i + lane] > MASK52) d++;
                if (S[8 * i + lane] > MASK52) d++;
            }
            for (size_t i = 0; i < n; i++) {
                if (A[8 * i + lane] > MASK52) di++;
                if (B[8 * i + lane] > MASK52) di++;
            }
            if (di) dirty_in++;
            if (d) {
                if (dirty_mul == 0 && dirty_sqr == 0 && shown < 3) {
                    shown++;
                    printf("  trial %d lane %u: dirty mul/sqr limbs=%d\n", t, lane, d);
                    for (size_t i = 0; i < n && i < 4; i++) {
                        printf("    in  limb %2zu = %016llx\n", i,
                               (unsigned long long)A[8 * i + lane]);
                        printf("    mul limb %2zu = %016llx\n", i,
                               (unsigned long long)M[8 * i + lane]);
                        printf("    sqr limb %2zu = %016llx\n", i,
                               (unsigned long long)S[8 * i + lane]);
                    }
                }
            }
            /* count separately */
            for (size_t i = 0; i < n; i++) {
                if (M[8 * i + lane] > MASK52) { dirty_mul++; break; }
            }
            for (size_t i = 0; i < n; i++) {
                if (S[8 * i + lane] > MASK52) { dirty_sqr++; break; }
            }
        }
    }
    printf("k=%zu mode=%s n=%zu trials=%d -> dirty inputs=%d, mul results dirty=%d, sqr results dirty=%d\n",
           k, mode ? "mersenne" : "montgomery", n, trials, dirty_in, dirty_mul, dirty_sqr);
    mpz_clears(av, bv, NULL);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    gmp_randclear(rs);
    return 0;
}
