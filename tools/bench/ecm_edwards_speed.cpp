// ecm_edwards_speed.cpp — focused Edwards stage-1 speed benchmark.
//
// Usage: ecm_edwards_speed <bits> <B1> <w> <reps>
//   times edwards_stage1_curve on N = 2^bits - 1, fixed sigma, `reps` runs.
//   reports min / median / avg wall-clock seconds.
//
// Compile (MSVC):
//   cl /O2 /EHsc /utf-8 -I<gmp>/include -Isrc/cpu \
//      tools/ecm_edwards_speed.cpp src/cpu/ecm_edwards_cpu.cpp gmp.lib
// Compile (gcc):
//   g++ -O3 -march=native -I<gmp>/include -Isrc/cpu \
//       tools/ecm_edwards_speed.cpp src/cpu/ecm_edwards_cpu.cpp -lgmp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <gmp.h>
#include "ecm_edwards_cpu.h"

static void compute_s(mpz_t s, unsigned long B1) {
    mpz_set_ui(s, 48);
    std::vector<char> sieve(B1 + 1, 1);
    sieve[0] = sieve[1] = 0;
    for (unsigned long p = 2; p * p <= B1; p++) {
        if (!sieve[p]) continue;
        for (unsigned long q = p * p; q <= B1; q += p) sieve[q] = 0;
    }
    for (unsigned long p = 2; p <= B1; p++) {
        if (!sieve[p]) continue;
        unsigned long v = p;
        while (v <= B1 / p) v *= p;
        mpz_mul_ui(s, s, v);
    }
}

int main(int argc, char **argv) {
    unsigned long bits = (argc > 1) ? strtoul(argv[1], nullptr, 10) : 991;
    unsigned long B1   = (argc > 2) ? strtoul(argv[2], nullptr, 10) : 1000000;
    int w              = (argc > 3) ? atoi(argv[3]) : 8;
    int reps           = (argc > 4) ? atoi(argv[4]) : 3;

    edwards_set_naf_w(w);

    mpz_t s, N, f;
    mpz_inits(s, N, f, NULL);
    compute_s(s, B1);

    mpz_ui_pow_ui(N, 2, bits);
    mpz_sub_ui(N, N, 1);

    // Fixed sigma: verified factors for M347/M677/M991/M4003 respectively.
    const uint64_t sigma =
        (bits == 347)  ? 20260922ULL :
        (bits == 677)  ? 6581585141005897ULL :
        (bits == 991)  ? 105413044550089ULL :
        (bits == 4003) ? 2027329164697536ULL : 105413044550089ULL;

    std::vector<double> times;
    times.reserve(reps);
    int rc = 0;

    printf("bits=%lu B1=%lu w=%d reps=%d sigma=%llu s_bits=%zu\n",
           bits, B1, w, reps, (unsigned long long)sigma, mpz_sizeinbase(s, 2));
    fflush(stdout);

    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        rc = edwards_stage1_curve(f, nullptr, nullptr, N, sigma, s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(secs);
        printf("  run %2d: %.3f s  rc=%d\n", r + 1, secs, rc);
        fflush(stdout);
    }

    std::sort(times.begin(), times.end());
    double sum = 0;
    for (double t : times) sum += t;
    printf("min=%.3f median=%.3f avg=%.3f s  (%d runs, rc=%d)\n",
           times.front(), times[times.size() / 2], sum / times.size(), reps, rc);

    mpz_clears(s, N, f, NULL);
    return 0;
}
