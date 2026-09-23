/* probe2.c - peak vpmadd52 throughput (Gmadd/s, clock-independent) + clock estimate.
 * build: cl /nologo /O2 /arch:AVX512 probe2.c /Fe:probe2.exe
 */
#include <immintrin.h>
#include <windows.h>
#include <stdio.h>
#include <stdint.h>

volatile uint64_t g_sink;

static double now_s(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* LCG: latency = imul(3) + add(1) = 4 cycles on Zen, cannot be folded */
__declspec(noinline) static uint64_t lcg(uint64_t n)
{
    uint64_t x = 12345;
    for (uint64_t i = 0; i < n; i++) x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    return x;
}

/* 8 independent vpmadd52luq chains -> peak madd/cycle (zmm) */
__declspec(noinline) static double madd_peak(uint64_t n)
{
    const __m512i a = _mm512_set1_epi64(0x000FFFFFFFFFFFFFull);
    const __m512i b = _mm512_set1_epi64(3);
    __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512(), c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512(), c7 = _mm512_setzero_si512();
    const double t0 = now_s();
    for (uint64_t i = 0; i < n; i++) {
        c0 = _mm512_madd52lo_epu64(c0, a, b);
        c1 = _mm512_madd52lo_epu64(c1, a, b);
        c2 = _mm512_madd52lo_epu64(c2, a, b);
        c3 = _mm512_madd52lo_epu64(c3, a, b);
        c4 = _mm512_madd52lo_epu64(c4, a, b);
        c5 = _mm512_madd52lo_epu64(c5, a, b);
        c6 = _mm512_madd52lo_epu64(c6, a, b);
        c7 = _mm512_madd52lo_epu64(c7, a, b);
    }
    const double dt = now_s() - t0;
    __m512i s = _mm512_add_epi64(
        _mm512_add_epi64(_mm512_add_epi64(c0, c1), _mm512_add_epi64(c2, c3)),
        _mm512_add_epi64(_mm512_add_epi64(c4, c5), _mm512_add_epi64(c6, c7)));
    g_sink += (uint64_t)_mm512_reduce_add_epi64(s);
    return (double)n * 8.0 / dt;      /* madds per second */
}

/* mixed lo+hi on the same accumulator pair, like the CIOS inner loop */
__declspec(noinline) static double madd_mixed(uint64_t n)
{
    const __m512i a = _mm512_set1_epi64(0x000FFFFFFFFFFFFFull);
    const __m512i b = _mm512_set1_epi64(0x000FEDCBA9876543ull);
    __m512i c0 = _mm512_setzero_si512(), d0 = _mm512_setzero_si512();
    __m512i c1 = _mm512_setzero_si512(), d1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512(), d2 = _mm512_setzero_si512();
    __m512i c3 = _mm512_setzero_si512(), d3 = _mm512_setzero_si512();
    const double t0 = now_s();
    for (uint64_t i = 0; i < n; i++) {
        c0 = _mm512_madd52lo_epu64(c0, a, b);  d0 = _mm512_madd52hi_epu64(d0, a, b);
        c1 = _mm512_madd52lo_epu64(c1, a, b);  d1 = _mm512_madd52hi_epu64(d1, a, b);
        c2 = _mm512_madd52lo_epu64(c2, a, b);  d2 = _mm512_madd52hi_epu64(d2, a, b);
        c3 = _mm512_madd52lo_epu64(c3, a, b);  d3 = _mm512_madd52hi_epu64(d3, a, b);
    }
    const double dt = now_s() - t0;
    __m512i s = _mm512_add_epi64(_mm512_add_epi64(c0, c1), _mm512_add_epi64(c2, c3));
    s = _mm512_add_epi64(s, _mm512_add_epi64(_mm512_add_epi64(d0, d1), _mm512_add_epi64(d2, d3)));
    g_sink += (uint64_t)_mm512_reduce_add_epi64(s);
    return (double)n * 8.0 / dt;
}

int main(void)
{
    double t = now_s();
    g_sink += lcg(50000000ull);
    t = now_s() - t;
    printf("lcg 5e7 (latency 4): %.3f s -> clock ~ %.2f GHz\n", t, 4.0 * 5e7 / t / 1e9);
    fflush(stdout);

    double m = madd_peak(50000000ull);
    printf("vpmadd52luq peak  : %.2f Gmadd/s (zmm, 8 indep chains)\n", m / 1e9);
    fflush(stdout);
    m = madd_mixed(50000000ull);
    printf("vpmadd52 lo+hi mix: %.2f Gmadd/s\n", m / 1e9);
    fflush(stdout);

    /* longer run: sustained (thermal) */
    m = madd_peak(400000000ull);
    printf("vpmadd52luq 40x   : %.2f Gmadd/s (sustained ~1s)\n", m / 1e9);
    fflush(stdout);
    t = now_s();
    g_sink += lcg(50000000ull);
    t = now_s() - t;
    printf("lcg again         : clock ~ %.2f GHz\n", 4.0 * 5e7 / t / 1e9);
    printf("sink=%llu\n", (unsigned long long)g_sink);
    fflush(stdout);
    return 0;
}
