#include <cstdio>
#include <gmp.h>
#include "../src/cpu/ecm_edwards_mont.h"

int main() {
    mpz_t N, a, b, c, expect;
    mpz_inits(N, a, b, c, expect, NULL);
    mpz_set_str(N, "286687326998758938951352611912760867599570623646035140467198604923365359511060601008752319138765710819327", 10);
    mpz_set_ui(a, 12345);
    mpz_set_ui(b, 67890);

    mont_ctx_t ctx;
    mont_init(&ctx, N);

    mont_t A, B, C;
    mont_to(&A, a, &ctx);
    mont_to(&B, b, &ctx);

    mont_mul(&C, &A, &B, &ctx);
    mont_from(c, &C, &ctx);
    mpz_mul(expect, a, b); mpz_mod(expect, expect, N);
    printf("mul: %s\n", mpz_cmp(c, expect) == 0 ? "OK" : "FAIL");
    if (mpz_cmp(c, expect) != 0) { gmp_printf("  c=%Zd\n  e=%Zd\n", c, expect); }

    mont_sqr(&C, &A, &ctx);
    mont_from(c, &C, &ctx);
    mpz_mul(expect, a, a); mpz_mod(expect, expect, N);
    printf("sqr: %s\n", mpz_cmp(c, expect) == 0 ? "OK" : "FAIL");

    mont_add(&C, &A, &B, &ctx);
    mont_from(c, &C, &ctx);
    mpz_add(expect, a, b); mpz_mod(expect, expect, N);
    printf("add: %s\n", mpz_cmp(c, expect) == 0 ? "OK" : "FAIL");

    mont_sub(&C, &A, &B, &ctx);
    mont_from(c, &C, &ctx);
    mpz_sub(expect, a, b); mpz_mod(expect, expect, N);
    printf("sub: %s\n", mpz_cmp(c, expect) == 0 ? "OK" : "FAIL");

    // mont_set_ui: 1 -> R mod N; from_mont should give 1
    mont_t ONE;
    mont_set_ui(&ONE, 1, &ctx);
    mont_from(c, &ONE, &ctx);
    printf("one: %s (got %lu)\n", mpz_cmp_ui(c, 1) == 0 ? "OK" : "FAIL", mpz_get_ui(c));

    // mont_to / mont_from roundtrip
    mont_to(&A, a, &ctx);
    mont_from(c, &A, &ctx);
    printf("roundtrip a: %s\n", mpz_cmp(c, a) == 0 ? "OK" : "FAIL");

    mont_clear(&ctx);
    return 0;
}
