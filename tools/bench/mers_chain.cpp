/* mers_chain.cpp — chained field-op walk in Mersenne mode vs mpz, step by step.
 *
 * The isolated-op tests (mers_test.cpp) pass, but the *ladder* diverges for some
 * exponents, i.e. the bug is value/state dependent.  This walks a long chain of
 * mul/sqr (like a ladder does) and reports the first step where the kernel's
 * value stops being a*b mod N.
 *
 * build: cl /O2 /arch:AVX512 /I src\cpu mers_chain.cpp src\cpu\simd_mont_ifma.cpp
 */
#include "simd_mont_ifma.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 3001;
    const long steps = (argc > 2) ? atol(argv[2]) : 200000;
    const int mode = (argc > 3) ? atoi(argv[3]) : IFMA_FIELD_MERS;

    mpz_t N, x, want, got, a;
    mpz_inits(N, x, want, got, a, NULL);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);

    ifma_ctx_t c;
    if (ifma_ctx_init_ex(&c, N, mode) != 0) { printf("init failed\n"); return 2; }
    const size_t n = c.n;
    std::vector<uint64_t> X(8 * n), R(8 * n);
    printf("k=%zu n=%zu mode=%s steps=%ld\n", k, n, ifma_field_name(&c), steps);

    gmp_randstate_t rs;
    gmp_randinit_mt(rs); gmp_randseed_ui(rs, 777);
    mpz_urandomm(x, rs, N);
    ifma_from_mpz_lane(X.data(), 0, x, &c);

    long first_bad = -1;
    for (long i = 0; i < steps; i++) {
        /* mirror an Edwards doubling chain: squares and products of the point
           coordinates, exactly the mix the ladder uses */
        if ((i % 3) == 0) {
            ifma_mont_sqr(R.data(), X.data(), &c);
            mpz_mul(want, x, x); mpz_mod(want, want, N);
        } else {
            mpz_urandomm(a, rs, N);
            std::vector<uint64_t> A(8 * n, 0);
            ifma_from_mpz_lane(A.data(), 0, a, &c);
            ifma_mont_mul(R.data(), X.data(), A.data(), &c);
            mpz_mul(want, x, a); mpz_mod(want, want, N);
        }
        lane_to_mpz(got, R.data(), n, 0);
        if (mpz_cmp(got, want) != 0) {
            first_bad = i;
            gmp_printf("  step %ld MISMATCH (op=%s)\n    x    =%Zx\n    want =%Zx\n    got  =%Zx\n",
                       i, (i % 3) == 0 ? "sqr" : "mul", x, want, got);
            break;
        }
        if (mpz_cmp(got, N) >= 0) {
            first_bad = i;
            gmp_printf("  step %ld NOT CANONICAL: got >= N\n    got =%Zx\n    N   =%Zx\n", i, got, N);
            break;
        }
        /* feed the value back in (this is what makes it a chain, not a test) */
        memcpy(X.data(), R.data(), 8 * n * sizeof(uint64_t));
        mpz_set(x, want);
    }

    printf("chain: %s at step %ld of %ld\n", first_bad < 0 ? "OK" : "FAILED", first_bad, steps);
    ifma_ctx_clear(&c);
    mpz_clears(N, x, want, got, a, NULL);
    gmp_randclear(rs);
    return first_bad < 0 ? 0 : 1;
}
