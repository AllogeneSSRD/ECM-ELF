/* probe3.c - is vpmadd52 zmm port-limited at 1/cycle, or was probe2 latency-limited?
 * 4 / 8 / 12 / 16 independent chains (+ hi variant).
 * build: cl /nologo /O2 /arch:AVX512 probe3.c /Fe:probe3.exe
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

#define MK(NAME, N, OP)                                                 \
__declspec(noinline) static double NAME(uint64_t it)                    \
{                                                                       \
    const __m512i a = _mm512_set1_epi64(0x000FFFFFFFFFFFFFull);         \
    const __m512i b = _mm512_set1_epi64(3);                             \
    __m512i c[16];                                                      \
    for (int k = 0; k < (N); k++) c[k] = _mm512_setzero_si512();        \
    const double t0 = now_s();                                          \
    for (uint64_t i = 0; i < it; i++) {                                 \
        for (int k = 0; k < (N); k++) c[k] = OP(c[k], a, b);            \
    }                                                                   \
    const double dt = now_s() - t0;                                     \
    __m512i s = _mm512_setzero_si512();                                 \
    for (int k = 0; k < (N); k++) s = _mm512_add_epi64(s, c[k]);        \
    g_sink += (uint64_t)_mm512_reduce_add_epi64(s);                     \
    return (double)it * (double)(N) / dt;                               \
}

MK(thru4,  4, _mm512_madd52lo_epu64)
MK(thru8,  8, _mm512_madd52lo_epu64)
MK(thru12, 12, _mm512_madd52lo_epu64)
MK(thru16, 16, _mm512_madd52lo_epu64)
MK(thruh8, 8, _mm512_madd52hi_epu64)

int main(void)
{
    double m;
    m = thru4(20000000ull);  printf("  4 chains: %.2f Gmadd/s\n", m / 1e9); fflush(stdout);
    m = thru8(20000000ull);  printf("  8 chains: %.2f Gmadd/s\n", m / 1e9); fflush(stdout);
    m = thru12(20000000ull); printf(" 12 chains: %.2f Gmadd/s\n", m / 1e9); fflush(stdout);
    m = thru16(20000000ull); printf(" 16 chains: %.2f Gmadd/s\n", m / 1e9); fflush(stdout);
    m = thruh8(20000000ull); printf("  8 hi    : %.2f Gmadd/s\n", m / 1e9); fflush(stdout);
    printf("sink=%llu\n", (unsigned long long)g_sink); fflush(stdout);
    return 0;
}
