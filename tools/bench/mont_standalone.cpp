/* ---------------------------------------------------------------------------
 * mont_standalone.cpp -- one-curve Suyama-sigma Montgomery stage-1 dump.
 *
 * Purpose: compare against gmp-ecm param 0 / Prime95-style references.  Prints the
 * curve A, the exponent size, the saved quantity (normalised Montgomery x), the
 * raw (X:Z) and gcd(Z,N) -- everything needed to diff against a reference save line.
 *
 * usage: mont_standalone <N-decimal|expr> <sigma> <B1> [torsion]
 *        torsion default 1 = gmp-ecm convention (lcm(1..B1)); 12 = Prime95 choose12
 * ------------------------------------------------------------------------- */
#include "ecm_mont_cpu.h"
#include "ecm_save.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <N> <sigma> <B1> [torsion=1] [save-file]\n", argv[0]);
        return 1;
    }
    mpz_t N, s, A, X0, Z0, x, factor;
    mpz_inits(N, s, A, X0, Z0, x, factor, NULL);

    if (mpz_set_str(N, argv[1], 10) != 0) { fprintf(stderr, "bad N\n"); return 1; }
    const uint64_t sigma = strtoull(argv[2], nullptr, 10);
    const uint64_t B1 = strtoull(argv[3], nullptr, 10);
    const uint64_t torsion = (argc > 4) ? strtoull(argv[4], nullptr, 10) : 1;

    const size_t s_bits = mont_build_s(s, B1, torsion);
    mont_suyama_curve(A, X0, Z0, sigma, N);
    mpz_t Qx, Qz, g;
    mpz_inits(Qx, Qz, g, NULL);
    int rc = mont_stage1_curve(nullptr, Qx, Qz, N, sigma, s);
    mpz_gcd(g, Qz, N);
    if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) rc = 1;
    /* the normalised x = X/Z exists only when Z is a unit mod N (no hit) */
    const int x_valid = (mpz_sgn(Qz) != 0) ? (mpz_invert(x, Qz, N) != 0) : 0;
    if (x_valid) {
        mpz_mul(x, Qx, x);
        mpz_mod(x, x, N);
    } else {
        mpz_set_ui(x, 0);
    }
    mpz_clear(Qx);

    char *sx = mpz_get_str(nullptr, 16, x);
    char *sa = mpz_get_str(nullptr, 10, A);
    char *sz = mpz_get_str(nullptr, 16, Qz);
    gmp_printf("sigma    = %llu\n", (unsigned long long)sigma);
    gmp_printf("B1       = %llu  torsion = %llu\n", (unsigned long long)B1,
               (unsigned long long)torsion);
    gmp_printf("s_bits   = %zu\n", s_bits);
    gmp_printf("A        = %s\n", sa);
    gmp_printf("X0       = %Zx\nZ0      = %Zx\n", X0, Z0);
    gmp_printf("x        = 0x%s\n", sx);
    gmp_printf("Z        = 0x%s\n", sz);
    gmp_printf("x_valid  = %d\n", x_valid);
    gmp_printf("gcd(Z,N) = %Zd\n", g);
    gmp_printf("rc       = %d\n", rc);

    /* optional: append a reference-family text save line (M3 writer) */
    if (argc > 5) {
        mpz_t xv;
        mpz_init(xv);
        int hitf = 0;
        if (rc == 1) { mpz_set(xv, g); hitf = 1; }   /* hit curves store the factor */
        else         { mpz_set(xv, x); }
        const int h = hitf;
        const bool ok = ecm_append_save_lines_mont(argv[5], N, (double)B1, sigma, 1, &xv, &h,
                                                   argv[1]);
        printf("save-line: %s (%s)\n", ok ? "written" : "FAILED", argv[5]);
        mpz_clear(xv);
    }

    free(sx); free(sa); free(sz);
    mpz_clears(N, s, A, X0, Z0, x, g, NULL);
    mpz_clear(Qz);
    return 0;
}
