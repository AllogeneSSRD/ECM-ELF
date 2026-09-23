// gmp_mpn_microbench.c — measure GMP mpn kernel throughput with a calibrated clock.
//
// 1. Calibrates the effective core clock (GHz) with a dependent-add chain
//    (each `add` has 1-cycle latency, so N adds take exactly N cycles).
// 2. Measures mpn_addmul_1 / mpn_mul_n / mpn_sqr at several sizes and reports
//    ns per limb and cycles per limb.
//
// Build (mingw64 gcc):
//   gcc -O2 -o gmp_mpn_microbench.exe gmp_mpn_microbench.c -I. -L. -lgmp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <gmp.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Dependent add chain: 8 adds per iteration, each 1-cycle latency on all
// modern x86-64 => exactly 8 cycles per iteration.
static double calibrate_ghz(void) {
    const uint64_t iters = 200000000ULL;   // 1.6e9 cycles
    uint64_t n = iters;
    volatile uint64_t sink;
    double t0 = now_s();
    __asm__ __volatile__(
        "xor %%rax, %%rax\n\t"
        "1:\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "add $1, %%rax\n\t"
        "dec %1\n\t"
        "jnz 1b\n\t"
        : "=a"(sink), "+r"(n)
        :
        : "cc");
    double t1 = now_s();
    (void)sink;
    return (double)iters * 8.0 / (t1 - t0) / 1e9;
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 5;
    const double ghz = calibrate_ghz();
    printf("calibrated core clock: %.3f GHz\n", ghz);

    const size_t sizes[] = { 8, 16, 32, 63, 128 };
    const int nsizes = 5;

    printf("\n%-6s %10s %12s %12s %12s %12s\n",
           "n", "iters", "addmul1_ns", "addmul1_c/l", "mul_n_c/l2", "sqr_c/l2");

    for (int si = 0; si < nsizes; si++) {
        size_t n = sizes[si];
        mp_limb_t *rp = malloc((2 * n + 8) * sizeof(mp_limb_t));
        mp_limb_t *up = malloc((n + 1) * sizeof(mp_limb_t));
        mp_limb_t *vp = malloc((n + 1) * sizeof(mp_limb_t));
        mp_limb_t *tp = malloc((2 * n + 8) * sizeof(mp_limb_t));
        for (size_t i = 0; i < n; i++) {
            up[i] = (mp_limb_t)(0x9E3779B97F4A7C15ULL * (i + 1));
            vp[i] = (mp_limb_t)(0xC2B2AE3D27D4EB4FULL * (i + 3));
            rp[i] = (mp_limb_t)(i * 1234567 + 89);
            tp[i] = 0;
        }
        const mp_limb_t k = 0xF1357AEA2E62A9C5ULL;

        long iters = (long)(2e7 / (double)n);
        if (iters < 200) iters = 200;

        // mpn_addmul_1
        double t0 = now_s();
        mp_limb_t carry = 0;
        for (long it = 0; it < iters; it++)
            carry = mpn_addmul_1(rp, up, n, k + (mp_limb_t)it);
        double t1 = now_s();
        double am_ns = (t1 - t0) / (double)iters / (double)n * 1e9;
        double am_cpl = am_ns * ghz;

        // mpn_mul_n
        long miters = iters / 8;
        if (miters < 50) miters = 50;
        t0 = now_s();
        for (long it = 0; it < miters; it++)
            mpn_mul_n(tp, up, vp, n);
        t1 = now_s();
        double mn_cpl2 = (t1 - t0) / (double)miters / ((double)n * (double)n) * 1e9 * ghz;

        // mpn_sqr
        t0 = now_s();
        for (long it = 0; it < miters; it++)
            mpn_sqr(tp, up, n);
        t1 = now_s();
        double sq_cpl2 = (t1 - t0) / (double)miters / ((double)n * (double)n) * 1e9 * ghz;

        (void)carry;
        printf("%-6zu %10ld %12.3f %12.3f %12.3f %12.3f\n",
               n, iters, am_ns, am_cpl, mn_cpl2, sq_cpl2);
        fflush(stdout);
        free(rp); free(up); free(vp); free(tp);
    }
    (void)reps;
    return 0;
}
