/* ---------------------------------------------------------------------------
 * ecm_edwards_standalone.cpp -- one-curve Edwards stage-1 dump.
 *
 * Prints everything needed to cross-check a stage-1 run against Prime95 or an
 * independent reference ladder (d, base point, exponent size, Qx/Qz, the
 * projective invariant u = (z+y)/(z-y), y_affine and gcd(Qz,N)).
 *
 * This used to live inside src/cpu/ecm_edwards_cpu.cpp behind
 * BUILD_ECM_EDWARDS_STANDALONE, which put a CLI main() and its private helpers in
 * the production translation unit.  It is now an ordinary bench tool built only
 * from the public API (src/cpu/ecm_edwards_cpu.h).
 *
 * usage: ecm_edwards_standalone <N-decimal> <sigma> <B1> [naf_w]
 * ------------------------------------------------------------------------- */
#include "ecm_edwards_cpu.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* s = 48 * lcm(1..B1): the stage-1 exponent, built the same way the driver does. */
static void build_s(mpz_t s, uint64_t B1)
{
    std::vector<char> composite(B1 + 1, 0);
    mpz_set_ui(s, 48);
    for (uint64_t p = 2; p <= B1; ++p) {
        if (composite[p]) continue;
        for (uint64_t q = p * 2; q <= B1; q += p) composite[q] = 1;
        uint64_t v = p;
        while (v <= B1 / p) v *= p;               /* highest power of p <= B1 */
        mpz_mul_ui(s, s, (unsigned long)v);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <N-decimal> <sigma> <B1> [naf_w]\n", argv[0]);
        return 1;
    }
    mpz_t N, d, s, Qx, Qz, factor, t, u, yaff;
    mpz_inits(N, d, s, Qx, Qz, factor, t, u, yaff, NULL);

    if (mpz_set_str(N, argv[1], 10) != 0) { fprintf(stderr, "bad N\n"); return 1; }
    const uint64_t sigma = strtoull(argv[2], nullptr, 10);
    const uint64_t B1 = strtoull(argv[3], nullptr, 10);
    if (argc > 4) edwards_set_naf_w(atoi(argv[4]));

    mpz_t Px, Py;
    mpz_inits(Px, Py, NULL);
    edwards_atkin_morain(d, Px, Py, sigma, N);

    build_s(s, B1);
    const int rc = edwards_stage1_curve(factor, Qx, Qz, N, sigma, s);

    /* u = Qx/Qz = (z+y)/(z-y) = (1+y_aff)/(1-y_aff)  (projective invariant) */
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, Qz, N);
    if (mpz_invert(t, Qz, N)) {
        mpz_mul(u, Qx, t);
        mpz_mod(u, u, N);
        /* y_aff = (u-1)/(u+1) */
        mpz_add_ui(t, u, 1);
        if (mpz_invert(t, t, N)) {
            mpz_sub_ui(yaff, u, 1);
            mpz_mul(yaff, yaff, t);
            mpz_mod(yaff, yaff, N);
        } else {
            mpz_set_ui(yaff, 0);
        }
    } else {
        mpz_set_ui(u, 0);
        mpz_set_ui(yaff, 0);
    }

    gmp_printf("N       = %Zd\n", N);
    gmp_printf("sigma   = %llu\n", (unsigned long long)sigma);
    gmp_printf("B1      = %llu\n", (unsigned long long)B1);
    gmp_printf("w       = %d\n", edwards_get_naf_w());
    gmp_printf("d       = %Zd\n", d);
    gmp_printf("P.x     = %Zd\n", Px);
    gmp_printf("P.y     = %Zd\n", Py);
    gmp_printf("s_bits  = %zu\n", mpz_sizeinbase(s, 2));
    gmp_printf("Qx      = %Zd\n", Qx);
    gmp_printf("Qz      = %Zd\n", Qz);
    gmp_printf("y_affine= %Zd\n", yaff);
    gmp_printf("u       = %Zd\n", u);
    gmp_printf("gcd(Qz,N)= %Zd\n", g);
    gmp_printf("stage1_rc= %d  (1=factor, 0=no factor, -1=error)\n", rc);

    mpz_clears(N, d, s, Qx, Qz, factor, t, u, yaff, Px, Py, NULL);
    mpz_clear(g);
    return 0;
}
