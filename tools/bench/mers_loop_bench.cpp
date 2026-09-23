/* mers_loop_bench.cpp — which loop shape reaches 1 madd52/cycle for the
 * Mersenne product (2 madds per column instead of CIOS's 4)?
 *
 * Each variant does n rows x (n-1) inner iterations on the SoA scratch, i.e.
 * the same 2n^2 madds a fold mul needs.  Cycles are derived from a calibrated
 * 4.0 GHz (see probe2/probe3: dependent-LCG chain says 4.01 GHz, and 8
 * independent vpmadd52 chains saturate at 4.00 Gmadd/s = 1 madd/cycle).
 *
 * build: cl /O2 /arch:AVX512 mers_loop_bench.cpp
 */
#include <immintrin.h>
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>

#define MASK52 0x000FFFFFFFFFFFFFULL

static double now_s(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static double g_ghz = 4.0;

/* --- L0: the shipped shape: 2 madds, load b, load t, store t, add -------- */
static void L0(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t n, int reps)
{
    const __m512i zero = _mm512_setzero_si512();
    for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i < n; i++) {
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        __m512i acc = _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), ai, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i), acc);
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            acc = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + r))), hi);
            acc = _mm512_madd52lo_epu64(acc, ai, br);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)), acc);
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        acc = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + n))), hi);
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)), acc);
    }
}

/* --- L1: same, but no t store (accumulate into a register and drop it) --- */
static void L1(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t n, int reps)
{
    const __m512i zero = _mm512_setzero_si512();
    for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i < n; i++) {
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        __m512i acc = _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), ai, b0);
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            acc = _mm512_add_epi64(acc, hi);
            acc = _mm512_madd52lo_epu64(acc, ai, br);
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * i), acc);
    }
}

/* --- L2: 2-row unroll (4 madds, 1 b load, 2 t loads, 2 stores, 2 adds) --- */
static void L2(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t n, int reps)
{
    const __m512i zero = _mm512_setzero_si512();
    for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i + 1 < n; i += 2) {
        const __m512i a0 = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i a1 = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), a0, b0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1)),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1))), a1, b0));
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            __m512i c0 = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + r))), h0);
            c0 = _mm512_madd52lo_epu64(c0, a0, br);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)), c0);
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c1 = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + r))), h1);
            c1 = _mm512_madd52lo_epu64(c1, a1, br);
            _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + r)), c1);
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + n))), h0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + n)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + n))), h1));
    }
}

/* --- L3: 4-row unroll (8 madds, 1 b load, 4 t loads, 4 stores) ---------- */
static void L3(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t n, int reps)
{
    const __m512i zero = _mm512_setzero_si512();
    for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i + 3 < n; i += 4) {
        __m512i av[4], hv[4];
        for (int q = 0; q < 4; q++) {
            av[q] = _mm512_load_si512((const __m512i *)(a + 8 * (i + q)));
            hv[q] = _mm512_madd52hi_epu64(zero, av[q], _mm512_load_si512((const __m512i *)b));
            _mm512_store_si512((__m512i *)(t + 8 * (i + q)),
                _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (i + q))),
                                      av[q], _mm512_load_si512((const __m512i *)b)));
        }
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            for (int q = 0; q < 4; q++) {
                __m512i c = _mm512_add_epi64(
                    _mm512_load_si512((const __m512i *)(t + 8 * (i + q + r))), hv[q]);
                c = _mm512_madd52lo_epu64(c, av[q], br);
                _mm512_store_si512((__m512i *)(t + 8 * (i + q + r)), c);
                hv[q] = _mm512_madd52hi_epu64(zero, av[q], br);
            }
        }
        for (int q = 0; q < 4; q++)
            _mm512_store_si512((__m512i *)(t + 8 * (i + q + n)),
                _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + q + n))), hv[q]));
    }
}

/* --- L4: CIOS-shaped reference (4 madds, 3 loads, 1 store) -------------- */
static void L4(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t n, int reps)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i m = _mm512_set1_epi64(0x0FEDCBA9876ull);
    for (int rep = 0; rep < reps; rep++)
    for (size_t i = 0; i < n; i++) {
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i acc = _mm512_madd52hi_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1))), ai, b0);
        acc = _mm512_madd52hi_epu64(acc, m, b0);
        for (size_t r = 1; r < n; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            const __m512i nr = _mm512_load_si512((const __m512i *)(b + 8 * ((r + 7) % n)));
            acc = _mm512_madd52lo_epu64(acc, ai, br);
            acc = _mm512_madd52lo_epu64(acc, m, nr);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)), acc);
            acc = _mm512_load_si512((const __m512i *)(t + 8 * (i + r + 1)));
            acc = _mm512_madd52hi_epu64(acc, ai, br);
            acc = _mm512_madd52hi_epu64(acc, m, nr);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + n)), acc);
    }
}

typedef void (*fn_t)(uint64_t *, const uint64_t *, const uint64_t *, size_t, int);

static void bench(const char *name, fn_t f, std::vector<uint64_t> &t,
                  const std::vector<uint64_t> &a, const std::vector<uint64_t> &b,
                  size_t n, int madds_per_iter, int iters_per_round)
{
    f(t.data(), a.data(), b.data(), n, 1);              /* warm */
    double best = 1e30;
    for (int round = 0; round < 3; round++) {
        const double t0 = now_s();
        f(t.data(), a.data(), b.data(), n, iters_per_round);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    const double iters = (double)iters_per_round * (double)n * (double)(n - 1);
    const double madds = iters * madds_per_iter;
    const double cyc = best * g_ghz * 1e9;
    printf("  %-28s %7.0f cyc/iter  %5.2f madd/cyc  %6.2f Gmadd/s\n",
           name, cyc / iters, madds / cyc, madds / best / 1e9);
}

int main(void)
{
    const size_t n = 58;
    std::vector<uint64_t> t(8 * (2 * n + 4)), a(8 * n), b(8 * n);
    for (size_t j = 0; j < t.size(); j++) t[j] = (uint64_t)(j * 2654435761u) & MASK52;
    for (size_t j = 0; j < 8 * n; j++) {
        a[j] = (uint64_t)(j * 40503u + 1) & MASK52;
        b[j] = (uint64_t)(j * 22695477u + 3) & MASK52;
    }
    const int rounds = 300;
    printf("n=%zu, %d rounds of n*(n-1) iterations, clock assumed %.2f GHz\n", n, rounds, g_ghz);
    bench("L0 shipped (2M 2L 1S)", L0, t, a, b, n, 2, rounds);
    bench("L1 no store (2M 1L ~S)", L1, t, a, b, n, 2, rounds);
    bench("L2 2-row unroll (4M 3L 2S)", L2, t, a, b, n, 4, rounds);
    bench("L3 4-row unroll (8M 5L 4S)", L3, t, a, b, n, 8, rounds);
    bench("L4 CIOS shape (4M 4L 1S)", L4, t, a, b, n, 4, rounds);
    /* re-run L0 to expose drift */
    bench("L0 again", L0, t, a, b, n, 2, rounds);
    return 0;
}
