/* selftest_many.cpp — run the SoA field/point selftests with many trials. */
#include "simd_edwards.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 3001;
    const int ftrials = (argc > 2) ? atoi(argv[2]) : 50;
    const int ptrials = (argc > 3) ? atoi(argv[3]) : 200;
    const int w = (argc > 4) ? atoi(argv[4]) : 8;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);

    for (int mode = 1; mode >= 0; mode--) {          /* mersenne first */
        ed_soa_ctx_t ctx;
        if (ed_soa_init_ex(&ctx, N, w, mode ? IFMA_FIELD_MERS : IFMA_FIELD_MONT) != 0) {
            printf("init failed\n"); continue;
        }
        printf("mode=%-24s w=%d\n", ed_soa_field_name(&ctx), w);
        /* the point-op selftest needs c->d, i.e. set_curves must have run */
        uint64_t sg[8];
        for (int i = 0; i < 8; i++) sg[i] = 20260922u + (uint64_t)i;
        if (ed_soa_set_curves(&ctx, sg, 8) != 0) { printf("  set_curves failed\n"); }
        const int f = ed_soa_field_selftest(&ctx, ftrials);
        printf("  field selftest (%d trials): %d failures\n", ftrials, f);
        fflush(stdout);
        const int p = ed_soa_point_selftest(&ctx, ptrials);
        printf("  point selftest (%d trials): %d failures\n", ptrials, p);
        fflush(stdout);
        ed_soa_clear(&ctx);
    }
    mpz_clear(N);
    return 0;
}
