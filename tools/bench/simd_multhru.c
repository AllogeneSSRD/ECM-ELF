// simd_multhru.c — measure AVX-512 multiply throughput (independent-op chains).
//
// Calibrates the core clock with a dependent add chain, then reports ops/cycle for
//   vpmadd52luq / vpmadd52huq  (IFMA: 8 lanes x 52x52 -> 104-bit fused MAC)
//   vpmuludq                   (16 lanes x 32x32 -> 64-bit product)
//   vpaddq                     (baseline, 8 lanes of 64-bit add)
// using 8 independent accumulators so latency does not limit the measurement.
//
// Build: gcc -O2 -mavx512f -mavx512ifma -mavx512dq -mavx512bw simd_multhru.c -o simd_multhru
#include <stdio.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double calibrate_ghz(void) {
    const uint64_t iters = 100000000ULL;   // 8 adds/iter = 8 cycles/iter
    uint64_t n = iters;
    volatile uint64_t sink;
    double t0 = now_s();
    __asm__ __volatile__(
        "xor %%rax, %%rax\n\t"
        "1:\n\t"
        "add $1, %%rax\n\tadd $1, %%rax\n\tadd $1, %%rax\n\tadd $1, %%rax\n\t"
        "add $1, %%rax\n\tadd $1, %%rax\n\tadd $1, %%rax\n\tadd $1, %%rax\n\t"
        "dec %1\n\tjnz 1b\n\t"
        : "=a"(sink), "+r"(n) : : "cc");
    double t1 = now_s();
    (void)sink;
    return (double)iters * 8.0 / (t1 - t0) / 1e9;
}

#define DEFINE_BENCH(NAME, INSN)                                            \
static double NAME(double ghz, long iters) {                                \
    __asm__ __volatile__(                                                   \
        "vpxorq %%zmm0, %%zmm0, %%zmm0\n\t"                                 \
        "vpxorq %%zmm1, %%zmm1, %%zmm1\n\t"                                 \
        "vpxorq %%zmm2, %%zmm2, %%zmm2\n\t"                                 \
        "vpxorq %%zmm3, %%zmm3, %%zmm3\n\t"                                 \
        "vpxorq %%zmm4, %%zmm4, %%zmm4\n\t"                                 \
        "vpxorq %%zmm5, %%zmm5, %%zmm5\n\t"                                 \
        "vpxorq %%zmm6, %%zmm6, %%zmm6\n\t"                                 \
        "vpxorq %%zmm7, %%zmm7, %%zmm7\n\t"                                 \
        "vmovdqa64 %%zmm0, %%zmm8\n\t"                                      \
        "vmovdqa64 %%zmm0, %%zmm9\n\t"                                      \
        "1:\n\t"                                                            \
        REP10(INSN)                                                         \
        "dec %[n]\n\tjnz 1b\n\t"                                            \
        : : [n] "r"(iters)                                                  \
        : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",          \
          "zmm8","zmm9","cc");                                              \
    (void)ghz;                                                              \
    return 0.0;                                                             \
}

// 10 independent instructions per iteration, rotating over 8 accumulators.
#define REP10(I) I " %%zmm8, %%zmm0, %%zmm0\n\t" I " %%zmm8, %%zmm1, %%zmm1\n\t" \
                 I " %%zmm8, %%zmm2, %%zmm2\n\t" I " %%zmm8, %%zmm3, %%zmm3\n\t" \
                 I " %%zmm8, %%zmm4, %%zmm4\n\t" I " %%zmm8, %%zmm5, %%zmm5\n\t" \
                 I " %%zmm8, %%zmm6, %%zmm6\n\t" I " %%zmm8, %%zmm7, %%zmm7\n\t" \
                 I " %%zmm8, %%zmm0, %%zmm0\n\t" I " %%zmm8, %%zmm1, %%zmm1\n\t"

DEFINE_BENCH(bench_ifma_lo, "vpmadd52luq")
DEFINE_BENCH(bench_ifma_hi, "vpmadd52huq")
DEFINE_BENCH(bench_pmuludq, "vpmuludq")
DEFINE_BENCH(bench_paddq,   "vpaddq")

static double time_it(void (*unused)(void)) { (void)unused; return 0.0; }

int main(void) {
    const long iters = 200000000L;   // x10 ops per iteration
    const double ghz = calibrate_ghz();
    printf("calibrated core clock: %.3f GHz\n\n", ghz);
    printf("%-14s %12s %12s\n", "op", "ops/cyc", "ops/sec");
    struct { const char *name; double (*fn)(double, long); } tbl[] = {
        {"vpmadd52luq", bench_ifma_lo},
        {"vpmadd52huq", bench_ifma_hi},
        {"vpmuludq",    bench_pmuludq},
        {"vpaddq",      bench_paddq},
    };
    for (unsigned i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++) {
        double t0 = now_s();
        tbl[i].fn(ghz, iters);
        double t1 = now_s();
        double ops = (double)iters * 10.0;
        double secs = t1 - t0;
        printf("%-14s %12.2f %12.3e\n", tbl[i].name, ops / secs / (ghz * 1e9), ops / secs);
    }
    (void)time_it;
    return 0;
}
