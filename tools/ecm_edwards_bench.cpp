// ecm_edwards_bench.cpp — Edwards stage-1 benchmark: sweep N and w-NAF window.
//
// Compile (link the Edwards library, no standalone main):
//   cl /O2 /EHsc /utf-8 -I<gmp>/include -Isrc/cpu \
//      tools/ecm_edwards_bench.cpp src/cpu/ecm_edwards_cpu.cpp gmp.lib
//
// For each (N, w): times edwards_stage1_curve and prints the NAF dictionary memory.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <gmp.h>
#include "ecm_edwards_cpu.h"
#include "ecm_edwards_mont.h"

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

struct Case { const char *name; const char *n_str; unsigned long long sigma; };

int main(int argc, char **argv) {
    unsigned long B1 = 1000000;
    if (argc >= 2) B1 = strtoul(argv[1], nullptr, 10);

    const Case cases[] = {
        {"M347",  nullptr, 20260922ULL},          // set below
        {"M677",  nullptr, 6581585141005897ULL},
        {"M991",  nullptr, 105413044550089ULL},
        {"M4003", nullptr, 2027329164697536ULL},
    };
    const int ns[] = {347, 677, 991, 4003};
    const int n_cases = 4;

    mpz_t s, N, f;
    mpz_inits(s, N, f, NULL);
    compute_s(s, B1);
    printf("B1=%lu, s_bits=%zu, ED_MONT_MAX_LIMBS=%d, mp_bits_per_limb=%d\n",
           B1, mpz_sizeinbase(s, 2), (int)ED_MONT_MAX_LIMBS, (int)GMP_NUMB_BITS);
    printf("point bytes = %d, affine bytes = %d\n",
           (int)(4 * ED_MONT_MAX_LIMBS * sizeof(mp_limb_t)),
           (int)(3 * ED_MONT_MAX_LIMBS * sizeof(mp_limb_t)));
    printf("\n%-6s %4s | %10s %9s | %8s | %s\n",
           "N", "w", "dict_entries", "dict_kB", "time_s", "result");

    for (int ci = 0; ci < n_cases; ci++) {
        mpz_ui_pow_ui(N, 2, ns[ci]);
        mpz_sub_ui(N, N, 1);

        // w sweep (fewer values for the slow 4003-bit case)
        static const int w_all[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14};
        static const int w_small[] = {2, 4, 8, 12, 14};
        const int *ws = (ns[ci] >= 4000) ? w_small : w_all;
        const int nw = (ns[ci] >= 4000) ? 5 : 10;

        for (int wi = 0; wi < nw; wi++) {
            const int w = ws[wi];
            edwards_set_naf_w(w);
            const size_t entries = (size_t)1 << (w - 2);
            const double dict_kB = (double)entries *
                (4 + 3) * ED_MONT_MAX_LIMBS * sizeof(mp_limb_t) / 1024.0;

            const clock_t t0 = clock();
            const int rc = edwards_stage1_curve(f, nullptr, nullptr, N, cases[ci].sigma, s);
            const clock_t t1 = clock();
            const double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;

            char res[64] = "-";
            if (rc > 0) gmp_snprintf(res, sizeof(res), "factor=%Zd", f);

            printf("%-6s %4d | %10zu %9.0f | %8.2f | %s\n",
                   cases[ci].name, w, entries, dict_kB, secs, res);
            fflush(stdout);
        }
    }

    mpz_clears(s, N, f, NULL);
    return 0;
}
