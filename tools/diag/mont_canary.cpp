/* mont_canary.cpp -- is the *scalar* Edwards Montgomery layer correct?
 *
 * Uses exactly the header the scalar stage-1 uses (ecm_edwards_mont.h) and checks,
 * for many random and adversarial inputs:
 *   * mul/sqr against mpz          (value)
 *   * every limb < N / result canonical (representation)
 *   * add/sub/neg against mpz
 * The SIMD twin of this check (canary.cpp / dirt.cpp) found two real bugs, so this
 * is the same instrument aimed at the mpn path.
 *
 *   usage: mont_canary <bits> [trials]
 */
#include "ecm_edwards_mont.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const long bits = (argc > 1) ? atol(argv[1]) : 3001;
    const int trials = (argc > 2) ? atoi(argv[2]) : 200;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (unsigned long)bits); mpz_sub_ui(N, N, 1);

    mont_ctx_t ctx;
    if (mont_init(&ctx, N) != 0) { printf("mont_init failed (too many limbs)\n"); return 2; }
    printf("bits=%ld nlimbs=%zu use_redc_n=%d\n", bits, ctx.nlimbs,
           (int)ctx.use_redc_n);

    gmp_randstate_t rs;
    gmp_randinit_mt(rs); gmp_randseed_ui(rs, 20260923u);

    mpz_t a, b, want, got;
    mpz_inits(a, b, want, got, NULL);
    mont_t A, B, R;
    int wrong = 0, noncanon = 0, shown = 0;

    for (int t = 0; t < trials; t++) {
        mpz_urandomm(a, rs, N);
        mpz_urandomm(b, rs, N);
        mont_to(&A, a, &ctx);
        mont_to(&B, b, &ctx);

        for (int op = 0; op < 5; op++) {
            switch (op) {
            case 0: mont_mul(&R, &A, &B, &ctx); mpz_mul(want, a, b); break;
            case 1: mont_sqr(&R, &A, &ctx);     mpz_mul(want, a, a); break;
            case 2: mont_add(&R, &A, &B, &ctx); mpz_add(want, a, b); break;
            case 3: mont_sub(&R, &A, &B, &ctx); mpz_sub(want, a, b); break;
            default: mont_neg(&R, &A, &ctx);    mpz_neg(want, a);     break;
            }
            mpz_mod(want, want, N);
            /* canonical?  compare the raw limb array, not a reduction of it */
            {
                mpz_t raw;
                mpz_init(raw);
                mpz_import(raw, ctx.nlimbs, -1, sizeof(mp_limb_t), 0, 0, R.l);
                if (mpz_cmp(raw, N) >= 0) {
                    noncanon++;
                    if (shown < 5) { gmp_printf("  NON-CANONICAL op=%d t=%d raw=%Zx\n", op, t, raw); shown++; }
                }
                mpz_clear(raw);
            }
            mont_from(got, &R, &ctx);
            if (mpz_cmp(got, want) != 0) {
                wrong++;
                if (shown < 5) {
                    gmp_printf("  WRONG op=%d t=%d\n    want=%Zx\n    got =%Zx\n", op, t, want, got);
                    shown++;
                }
            }
        }
    }
    printf("bits=%ld trials=%d -> wrong=%d non-canonical=%d\n", bits, trials, wrong, noncanon);
    mpz_clears(a, b, want, got, NULL);
    mont_clear(&ctx);
    mpz_clear(N);
    gmp_randclear(rs);
    return (wrong || noncanon) ? 1 : 0;
}
