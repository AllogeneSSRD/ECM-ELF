/* ---------------------------------------------------------------------------
 * lane_width.cpp -- does widening the SoA batch from 8 to 16 lanes (two zmm per
 * field-element column) actually speed up the product loop?
 *
 * The reasoning to check: our fold-domain product loop spends, per iteration,
 *   4 madds + 1 b-load + 2 t-loads + 2 t-stores + 2 adds        (8 lanes)
 * so ~7 non-madd instructions per 4 madds.  Widening to 16 lanes doubles the madds
 * per iteration (8) but ALSO doubles the loads and stores (the data volume doubles),
 * so the ratio stays 4:7 unless something is genuinely fixed per call.  This tool
 * measures both loop shapes back to back in one process, plus the pure-madd roof, and
 * reports per-lane throughput.  If variant B is not faster, widening buys nothing and
 * the field layer really is finished (docs/ECM_Montgomery_STAGE1.md section 15.2).
 *
 * Both variants run the same double loop (2 rows fused, one b-vector feeding four
 * madds) over n columns of 52-bit limbs; variant B keeps each column as two zmm
 * (low 8 lanes, high 8 lanes).
 *
 * usage: lane_width [n] [reps]
 * ------------------------------------------------------------------------- */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define M52 0x000FFFFFFFFFFFFFULL

static double now_ns(void)
{
    using namespace std::chrono;
    return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

/* ---- variant A: 8 lanes per column (the production shape) ---- */
static void prod8(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t len)
{
    const __m512i mask = _mm512_set1_epi64((long long)M52);
    const __m512i zero = _mm512_setzero_si512();
    for (size_t k = 0; k < 2 * len + 2; k++)
        _mm512_store_si512((__m512i *)(t + 8 * k), zero);
    size_t i = 0;
    for (; i + 1 < len; i += 2) {
        const __m512i a0 = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i a1 = _mm512_load_si512((const __m512i *)(a + 8 * (i + 1)));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), a0, b0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1)),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1))), a1, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            __m512i c0 = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + r))), h0);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)), _mm512_madd52lo_epu64(c0, a0, br));
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c1 = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + r))), h1);
            _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + r)), _mm512_madd52lo_epu64(c1, a1, br));
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + len)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + len))), h0));
        _mm512_store_si512((__m512i *)(t + 8 * (i + 1 + len)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + 1 + len))), h1));
    }
    if (i < len) {
        const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
        const __m512i b0 = _mm512_load_si512((const __m512i *)b);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0);
        _mm512_store_si512((__m512i *)(t + 8 * i),
            _mm512_madd52lo_epu64(_mm512_load_si512((const __m512i *)(t + 8 * i)), ai, b0));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = _mm512_load_si512((const __m512i *)(b + 8 * r));
            const __m512i acc = _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + r))), hi);
            _mm512_store_si512((__m512i *)(t + 8 * (i + r)), _mm512_madd52lo_epu64(acc, ai, br));
            hi = _mm512_madd52hi_epu64(zero, ai, br);
        }
        _mm512_store_si512((__m512i *)(t + 8 * (i + len)),
            _mm512_add_epi64(_mm512_load_si512((const __m512i *)(t + 8 * (i + len))), hi));
    }
}

/* ---- variant B: 16 lanes per column (each column is two zmm) ----
   Column c lives at t[16c .. 16c+15]: lanes 0..7 in the first zmm, 8..15 in the second. */
static void prod16(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t len)
{
    const __m512i mask = _mm512_set1_epi64((long long)M52);
    const __m512i zero = _mm512_setzero_si512();
    for (size_t k = 0; k < 2 * len + 2; k++) {
        _mm512_store_si512((__m512i *)(t + 16 * k), zero);
        _mm512_store_si512((__m512i *)(t + 16 * k + 8), zero);
    }
#define LD16(p, c) _mm512_load_si512((const __m512i *)((p) + 16 * (c)))
#define LD16H(p, c) _mm512_load_si512((const __m512i *)((p) + 16 * (c) + 8))
#define ST16(p, c, v) _mm512_store_si512((__m512i *)((p) + 16 * (c)), (v))
#define ST16H(p, c, v) _mm512_store_si512((__m512i *)((p) + 16 * (c) + 8), (v))
    size_t i = 0;
    for (; i + 1 < len; i += 2) {
        const __m512i a0 = LD16(a, i), a0h = LD16H(a, i);
        const __m512i a1 = LD16(a, i + 1), a1h = LD16H(a, i + 1);
        const __m512i b0 = LD16(b, 0), b0h = LD16H(b, 0);
        __m512i h0 = _mm512_madd52hi_epu64(zero, a0, b0), h0h = _mm512_madd52hi_epu64(zero, a0h, b0h);
        __m512i h1 = _mm512_madd52hi_epu64(zero, a1, b0), h1h = _mm512_madd52hi_epu64(zero, a1h, b0h);
        ST16(t, i, _mm512_madd52lo_epu64(LD16(t, i), a0, b0));
        ST16H(t, i, _mm512_madd52lo_epu64(LD16H(t, i), a0h, b0h));
        ST16(t, i + 1, _mm512_madd52lo_epu64(LD16(t, i + 1), a1, b0));
        ST16H(t, i + 1, _mm512_madd52lo_epu64(LD16H(t, i + 1), a1h, b0h));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = LD16(b, r), brh = LD16H(b, r);
            __m512i c0 = _mm512_add_epi64(LD16(t, i + r), h0);
            ST16(t, i + r, _mm512_madd52lo_epu64(c0, a0, br));
            h0 = _mm512_madd52hi_epu64(zero, a0, br);
            __m512i c0h = _mm512_add_epi64(LD16H(t, i + r), h0h);
            ST16H(t, i + r, _mm512_madd52lo_epu64(c0h, a0h, brh));
            h0h = _mm512_madd52hi_epu64(zero, a0h, brh);
            __m512i c1 = _mm512_add_epi64(LD16(t, i + 1 + r), h1);
            ST16(t, i + 1 + r, _mm512_madd52lo_epu64(c1, a1, br));
            h1 = _mm512_madd52hi_epu64(zero, a1, br);
            __m512i c1h = _mm512_add_epi64(LD16H(t, i + 1 + r), h1h);
            ST16H(t, i + 1 + r, _mm512_madd52lo_epu64(c1h, a1h, brh));
            h1h = _mm512_madd52hi_epu64(zero, a1h, brh);
        }
        ST16(t, i + len, _mm512_add_epi64(LD16(t, i + len), h0));
        ST16H(t, i + len, _mm512_add_epi64(LD16H(t, i + len), h0h));
        ST16(t, i + 1 + len, _mm512_add_epi64(LD16(t, i + 1 + len), h1));
        ST16H(t, i + 1 + len, _mm512_add_epi64(LD16H(t, i + 1 + len), h1h));
    }
    if (i < len) {
        const __m512i ai = LD16(a, i), aih = LD16H(a, i);
        const __m512i b0 = LD16(b, 0), b0h = LD16H(b, 0);
        __m512i hi = _mm512_madd52hi_epu64(zero, ai, b0), hih = _mm512_madd52hi_epu64(zero, aih, b0h);
        ST16(t, i, _mm512_madd52lo_epu64(LD16(t, i), ai, b0));
        ST16H(t, i, _mm512_madd52lo_epu64(LD16H(t, i), aih, b0h));
        for (size_t r = 1; r < len; r++) {
            const __m512i br = LD16(b, r), brh = LD16H(b, r);
            ST16(t, i + r, _mm512_madd52lo_epu64(_mm512_add_epi64(LD16(t, i + r), hi), ai, br));
            hi = _mm512_madd52hi_epu64(zero, ai, br);
            ST16H(t, i + r, _mm512_madd52lo_epu64(_mm512_add_epi64(LD16H(t, i + r), hih), aih, brh));
            hih = _mm512_madd52hi_epu64(zero, aih, brh);
        }
        ST16(t, i + len, _mm512_add_epi64(LD16(t, i + len), hi));
        ST16H(t, i + len, _mm512_add_epi64(LD16H(t, i + len), hih));
    }
#undef LD16
#undef LD16H
#undef ST16
#undef ST16H
}

/* ---- variant C: 8 lanes/column, but REGISTER-BLOCKED ----
   Variant A touches memory 5 times per 4 madds (t-column load+store per row, plus the
   b-vector load), i.e. about as many memory instructions as madds.  Here the output
   columns are accumulated in registers in blocks of 8, so a whole block of products
   costs 8 loads + 8 stores instead of ~1.25 memory ops per madd.  This is the real
   question the 8-vs-16-lane experiment was a proxy for: is the product loop limited by
   the load/store ports rather than by the madd ports?  */
#define BLK 8
static void prod_regblk(uint64_t *t, const uint64_t *a, const uint64_t *b, size_t len)
{
    const __m512i mask = _mm512_set1_epi64((long long)M52);
    const __m512i zero = _mm512_setzero_si512();
    const size_t ncol = 2 * len + 2;
    for (size_t k = 0; k < ncol + 2; k++)
        _mm512_store_si512((__m512i *)(t + 8 * k), zero);

    for (size_t c0 = 0; c0 < ncol; c0 += BLK) {
        __m512i acc[BLK];
        for (int q = 0; q < BLK; q++) acc[q] = zero;
        for (size_t i = 0; i < len; i++) {
            /* products (i,j) with c0 <= i+j < c0+BLK  =>  j in [c0-i, c0+BLK-i) */
            long js = (long)c0 - (long)i;      if (js < 0) js = 0;
            long je = (long)c0 + BLK - (long)i; if (je > (long)len) je = (long)len;
            if (js >= je) continue;
            const __m512i ai = _mm512_load_si512((const __m512i *)(a + 8 * i));
            for (long j = js; j < je; j++) {
                const __m512i bj = _mm512_load_si512((const __m512i *)(b + 8 * (size_t)j));
                const size_t c = (size_t)((long)i + j - (long)c0);
                acc[c] = _mm512_madd52lo_epu64(acc[c], ai, bj);
                if (c + 1 < BLK) {
                    acc[c + 1] = _mm512_madd52hi_epu64(acc[c + 1], ai, bj);
                } else if (c0 + BLK < ncol + 2) {   /* high half spills to the next block */
                    uint64_t *dst = t + 8 * (c0 + BLK);
                    _mm512_store_si512((__m512i *)dst,
                        _mm512_add_epi64(_mm512_load_si512((const __m512i *)dst),
                                         _mm512_madd52hi_epu64(zero, ai, bj)));
                }
            }
        }
        for (int q = 0; q < BLK; q++)
            if (c0 + (size_t)q < ncol)
                _mm512_store_si512((__m512i *)(t + 8 * (c0 + q)),
                                   _mm512_and_si512(acc[q], mask));
    }
}

/* pure-madd roof: 8 independent chains, no memory traffic */
static double roof_gmadds(int iters)
{
    const __m512i x = _mm512_set1_epi64((long long)M52);
    __m512i a0 = _mm512_set1_epi64(1), a1 = _mm512_set1_epi64(2);
    __m512i a2 = _mm512_set1_epi64(3), a3 = _mm512_set1_epi64(4);
    __m512i a4 = _mm512_set1_epi64(5), a5 = _mm512_set1_epi64(6);
    __m512i a6 = _mm512_set1_epi64(7), a7 = _mm512_set1_epi64(8);
    for (int r = 0; r < 2000; r++) {
        a0 = _mm512_madd52lo_epu64(a0, x, x); a1 = _mm512_madd52lo_epu64(a1, x, x);
        a2 = _mm512_madd52lo_epu64(a2, x, x); a3 = _mm512_madd52lo_epu64(a3, x, x);
        a4 = _mm512_madd52lo_epu64(a4, x, x); a5 = _mm512_madd52lo_epu64(a5, x, x);
        a6 = _mm512_madd52lo_epu64(a6, x, x); a7 = _mm512_madd52lo_epu64(a7, x, x);
    }
    const double t0 = now_ns();
    for (int r = 0; r < iters; r++) {
        a0 = _mm512_madd52lo_epu64(a0, x, x); a1 = _mm512_madd52lo_epu64(a1, x, x);
        a2 = _mm512_madd52lo_epu64(a2, x, x); a3 = _mm512_madd52lo_epu64(a3, x, x);
        a4 = _mm512_madd52lo_epu64(a4, x, x); a5 = _mm512_madd52lo_epu64(a5, x, x);
        a6 = _mm512_madd52lo_epu64(a6, x, x); a7 = _mm512_madd52lo_epu64(a7, x, x);
    }
    const double ns = now_ns() - t0;
    static volatile uint64_t sink;
    sink = (uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a0, a1),
                                                             _mm512_add_epi64(a2, a3)));
    return 8.0 * 8.0 * iters / ns;            /* per-lane madds per ns */
}

int main(int argc, char **argv)
{
    const size_t n = (argc > 1) ? (size_t)atoi(argv[1]) : 58;
    const int reps = (argc > 2) ? atoi(argv[2]) : 3000;

    uint64_t *a8 = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *b8 = (uint64_t *)_mm_malloc(8 * n * sizeof(uint64_t), 64);
    uint64_t *t8 = (uint64_t *)_mm_malloc(8 * (2 * n + 4) * sizeof(uint64_t), 64);
    uint64_t *a16 = (uint64_t *)_mm_malloc(16 * n * sizeof(uint64_t), 64);
    uint64_t *b16 = (uint64_t *)_mm_malloc(16 * n * sizeof(uint64_t), 64);
    uint64_t *t16 = (uint64_t *)_mm_malloc(16 * (2 * n + 2) * sizeof(uint64_t), 64);
    for (size_t i = 0; i < 8 * n; i++)  { a8[i]  = (i * 2654435761u) & M52; b8[i]  = (i * 40503u + 7) & M52; }
    for (size_t i = 0; i < 16 * n; i++) { a16[i] = (i * 2654435761u) & M52; b16[i] = (i * 40503u + 7) & M52; }

    /* madd instructions per call: the loop is (len/2) row-pairs x (len-1) iterations
       x 4 madds, plus boundaries -- count them the same way for both variants. */
    const size_t iters = (n / 2) * (n - 1);
    const double madds8 = 4.0 * iters + 2.0 * (n / 2) * 2;      /* + boundary madds */
    const double madds16 = 2.0 * madds8;                        /* two zmm per column */

    double best8 = 1e30, best16 = 1e30, bestrb = 1e30;
    for (int r = 0; r < 7; r++) {
        double t0 = now_ns();
        for (int q = 0; q < reps; q++) prod8(t8, a8, b8, n);
        double d = (now_ns() - t0) / reps; if (d < best8) best8 = d;
        t0 = now_ns();
        for (int q = 0; q < reps; q++) prod16(t16, a16, b16, n);
        d = (now_ns() - t0) / reps; if (d < best16) best16 = d;
        t0 = now_ns();
        for (int q = 0; q < reps; q++) prod_regblk(t8, a8, b8, n);
        d = (now_ns() - t0) / reps; if (d < bestrb) bestrb = d;
    }

    const double roof = roof_gmadds(1000000);
    const double r8 = madds8 * 8.0 / best8;         /* per-lane madds/ns */
    const double r16 = madds16 * 8.0 / best16;
    const double rrb = madds8 * 8.0 / bestrb;
    printf("lane_width: n52=%zu, reps=%d\n", n, reps);
    printf("  8 lanes/column (production)  %8.1f ns/call -> %6.2f per-lane madd/ns\n",
           best8, r8);
    printf("  16 lanes/column (2 zmm)      %8.1f ns/call -> %6.2f  (%+.1f%% vs A)\n",
           best16, r16, 100.0 * (r16 - r8) / r8);
    printf("  8 lanes, register-blocked    %8.1f ns/call -> %6.2f  (%+.1f%% vs A)\n",
           bestrb, rrb, 100.0 * (rrb - r8) / r8);
    printf("  (reference roof loop = %.2f per-lane madd/ns; it may be compiler-transformed,\n"
           "   so treat it as indicative only -- the A/B/C comparison is the measurement)\n",
           roof * 8.0);

    _mm_free(a8); _mm_free(b8); _mm_free(t8);
    _mm_free(a16); _mm_free(b16); _mm_free(t16);
    return 0;
}
