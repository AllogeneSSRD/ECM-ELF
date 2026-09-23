/* canary.cpp — is the kernel's *raw* result canonical (< N)?  mpz_mod() in
   ifma_to_mpz_lane() hides non-canonical representatives, so this reads the raw
   limbs and compares against a*b (mersenne) / a*b*R^-1 (montgomery). */
#include "simd_mont_ifma.h"
#include <gmp.h>
#include <stdio.h>
#include <vector>

static void add52(mpz_t out, uint64_t limb)
{
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)((limb >> 26) & 0x3FFFFFFu));
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)(limb & 0x3FFFFFFu));
}

static void raw_to_mpz(mpz_t out, const uint64_t *e, size_t n, unsigned lane)
{
    mpz_set_ui(out, 0);
    for (size_t i = n; i-- > 0; ) add52(out, e[8 * i + lane] & 0xFFFFFFFFFFFFFULL);
}

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 4003;
    const int mode = (argc > 2) ? atoi(argv[2]) : IFMA_FIELD_MONT;
    const int trials = (argc > 3) ? atoi(argv[3]) : 200;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);

    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, mode) != 0) { printf("init failed\n"); return 2; }
    const size_t n = c.n, lw = 8 * n;
    std::vector<uint64_t> A(lw), B(lw), R(lw);
    mpz_t Rm, Rinv, av, bv, want, raw, m;
    mpz_inits(Rm, Rinv, av, bv, want, raw, m, NULL);
    mpz_set_ui(Rm, 1); mpz_mul_2exp(Rm, Rm, 52 * (unsigned long)n);
    mpz_mod(Rm, Rm, N);
    mpz_invert(Rinv, Rm, N);
    gmp_randstate_t rs;
    gmp_randinit_mt(rs); gmp_randseed_ui(rs, 31337u);

    int noncanon = 0, wrong = 0;
    for (int t = 0; t < trials; t++) {
        for (unsigned lane = 0; lane < 8; lane++) {
            mpz_urandomm(av, rs, N);
            mpz_urandomm(bv, rs, N);
            ifma_from_mpz_lane(A.data(), lane, av, &c);
            ifma_from_mpz_lane(B.data(), lane, bv, &c);
        }
        ifma_mont_mul(R.data(), A.data(), B.data(), &c);
        for (unsigned lane = 0; lane < 8; lane++) {
            raw_to_mpz(av, A.data(), n, lane);
            raw_to_mpz(bv, B.data(), n, lane);
            raw_to_mpz(raw, R.data(), n, lane);
            if (mode == IFMA_FIELD_MERS) {
                mpz_mul(want, av, bv); mpz_mod(want, want, N);
            } else {
                mpz_mul(want, av, bv); mpz_mod(want, want, N);
                mpz_mul(want, want, Rinv); mpz_mod(want, want, N);
            }
            if (mpz_cmp(raw, N) >= 0) {
                noncanon++;
                if (noncanon <= 2) gmp_printf("  NON-CANONICAL lane=%u raw=%Zx (N=%Zx)\n", lane, raw, N);
            }
            if (mpz_cmp(raw, want) != 0) {
                wrong++;
                if (wrong <= 2) gmp_printf("  WRONG lane=%u\n    want=%Zx\n    raw =%Zx\n", lane, want, raw);
            }
        }
    }
    printf("k=%zu mode=%s trials=%d -> non-canonical=%d wrong=%d\n", k,
           mode ? "mersenne" : "montgomery", trials, noncanon, wrong);
    mpz_clears(Rm, Rinv, av, bv, want, raw, m, NULL);
    ifma_ctx_clear(&c);
    mpz_clear(N);
    gmp_randclear(rs);
    return 0;
}
