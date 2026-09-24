/* ---------------------------------------------------------------------------
 * mont_ckpt_ab.cpp -- what does the checkpoint/progress hook cost the ladder?
 *
 * The ladder gained a "reached the next checkpoint offset?" test per bit plus a
 * callback every `chunk` bits.  This tool times three variants of the SAME
 * workload in one process, interleaved, so CPU frequency drift cannot fake a
 * difference:
 *
 *   old      : the ladder as it was before checkpoints (git HEAD, renamed symbols)
 *   new-none : today's ladder with no callback at all   (cb == NULL)
 *   new-prog : today's ladder with the progress/pause callback that the driver
 *              always installs (checkpoint autosave + progress bar)
 *
 * Reported as ns per ladder bit for the whole 8-lane batch.
 *
 * build: tools\build_tool.bat tools\bench\mont_ckpt_ab.cpp ^
 *            src\cpu\simd_mont_curve.cpp src\cpu\simd_mont_ifma.cpp ^
 *            src\cpu\ecm_mont_cpu.cpp src\core\ecm_stage1_exp.cpp ^
 *            .bench_tmp\ab_old\old_ladder.cpp
 * run:   build_vs18\tools\mont_ckpt_ab.exe [k_bits] [B1] [reps]
 * ------------------------------------------------------------------------- */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <vector>

#include "ecm_mont_cpu.h"
#include "simd_mont_curve.h"

/* the pre-checkpoint ladder, renamed by the build command (see the header note) */
int old_mont_soa_init(mont_soa_ctx_t *c, const mpz_t N, int field_mode);
void old_mont_soa_clear(mont_soa_ctx_t *c);
int old_mont_soa_stage1_bits(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                             const uint64_t sigmas[IFMA_LANES], mpz_t *out_x,
                             mpz_t *out_gcd);

static int cb_progress(void *p, size_t bitnum)
{
    volatile size_t *sink = (volatile size_t *)p;
    *sink = bitnum;                     /* mimic a progress update: no pause */
    return 0;
}

int main(int argc, char **argv)
{
    const unsigned k = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 10) : 3001;
    const double B1 = (argc > 2) ? atof(argv[2]) : 1e5;
    const int reps = (argc > 3) ? atoi(argv[3]) : 5;

    mpz_t N, s;
    mpz_inits(N, s, NULL);
    mpz_set_ui(N, 1);
    mpz_mul_2exp(N, N, k);
    mpz_sub_ui(N, N, 1);
    mont_build_s(s, (uint64_t)B1, 1);
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);

    uint64_t sg[IFMA_LANES];
    for (unsigned i = 0; i < IFMA_LANES; i++) sg[i] = 1000003ull * (i + 1) + 12345ull;

    std::vector<mpz_t> x(IFMA_LANES), g(IFMA_LANES);
    for (unsigned i = 0; i < IFMA_LANES; i++) mpz_inits(x[i], g[i], NULL);

    mont_soa_ctx_t cnew;
    mont_soa_ctx_t cold;
    if (mont_soa_init(&cnew, N, IFMA_FIELD_AUTO) != 0 ||
        old_mont_soa_init(&cold, N, IFMA_FIELD_AUTO) != 0) {
        printf("init failed (no AVX512-IFMA?)\n");
        return 1;
    }
    const size_t chunk = (nbits / 512 < 4096) ? 4096 : (nbits / 512);
    std::vector<uint64_t> state(mont_soa_state_words(&cnew));
    volatile size_t sink = 0;

    printf("mont_ckpt_ab: N = 2^%u-1, B1 = %.0f, s_bits = %zu, %d rep(s), chunk = %zu\n",
           k, B1, nbits, reps, chunk);

    double t_old = 0, t_none = 0, t_prog = 0;
    int rc_old = 0, rc_none = 0, rc_prog = 0;
    for (int r = 0; r < reps; r++) {
        {   /* old */
            const auto t0 = std::chrono::steady_clock::now();
            rc_old = old_mont_soa_stage1_bits(&cold, bits, nbits, sg, x.data(), g.data());
            t_old += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        {   /* new, no callback */
            const auto t0 = std::chrono::steady_clock::now();
            rc_none = mont_soa_stage1_bits(&cnew, bits, nbits, sg, x.data(), g.data());
            t_none += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        {   /* new, with the driver's progress callback (never pauses here) */
            size_t bn = 0;
            const auto t0 = std::chrono::steady_clock::now();
            rc_prog = mont_soa_stage1_bits_ex(&cnew, bits, nbits, sg, 0, state.data(), &bn,
                                              x.data(), g.data(), cb_progress, (void *)&sink,
                                              chunk);
            t_prog += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }

    const double per_bit_old = t_old / reps / (double)nbits * 1e9;
    const double per_bit_none = t_none / reps / (double)nbits * 1e9;
    const double per_bit_prog = t_prog / reps / (double)nbits * 1e9;

    printf("  old ladder          : %8.2f ns/bit  (%.3f s/batch)\n", per_bit_old, t_old / reps);
    printf("  new, cb = NULL      : %8.2f ns/bit  (%+.2f%%)\n", per_bit_none,
           100.0 * (per_bit_none - per_bit_old) / per_bit_old);
    printf("  new, cb = progress  : %8.2f ns/bit  (%+.2f%%)  <- what the driver installs\n",
           per_bit_prog, 100.0 * (per_bit_prog - per_bit_old) / per_bit_old);
    printf("  rc: old=%d none=%d prog=%d\n", rc_old, rc_none, rc_prog);

    mont_soa_clear(&cnew);
    old_mont_soa_clear(&cold);
    for (unsigned i = 0; i < IFMA_LANES; i++) mpz_clears(x[i], g[i], NULL);
    free(bits);
    mpz_clears(N, s, NULL);
    return 0;
}
