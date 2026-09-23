/* ──────────────────────────────────────────────────────────────────────────
 * AVX512 / AVX2  batched CIOS Montgomery multiplication
 *
 * Data layout: Structure of Arrays (SoA).
 *   Input arrays a, b, N, out are [kInstances × max_limbs] uint32_t.
 *   Instance k limb j is at arr[k * max_limbs + j].
 *
 * Internal working buffers (t, B_local) use SoA [limbs+2][16] uint64_t
 * so that each column of 16 64-bit lanes can be loaded/stored with a single
 * 512-bit move.
 *
 * Each 64-bit lane holds a 32-bit limb zero-extended; vpmuludq computes
 * the 32×32→64 product, and the accumulator fits in 64 bits + carry.
 *
 * AVX512 path: 16 curves in two half-register passes (lanes 0-7, 8-15).
 * AVX2 path:   8 curves, one pass.
 *
 * ──────────────────────────────────────────────────────────────────────── */
#include "cpu_mont_scalar.h"   /* for CPU_MONT_MAX_LIMBS */
#include "cpu_mont_avx.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cstring>
#include <immintrin.h>

/* ══════════════════════════════════════════════════════════════════════════
 * CPUID  helpers
 * ════════════════════════════════════════════════════════════════════════ */

bool cpu_has_avx512f()
{
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 16)) != 0;   // EBX bit16
#else
    return __builtin_cpu_supports("avx512f");
#endif
}

bool cpu_has_avx2()
{
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;    // EBX bit5
#else
    return __builtin_cpu_supports("avx2");
#endif
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Low-level AVX512 helpers  (lanes = 8)
 * ──────────────────────────────────────────────────────────────────────── */

/* broadcast 32-bit scalar → 8×64-bit (zero-extended) */
static inline __m512i m512_bcst32_64(uint32_t x)
{
    return _mm512_set1_epi64((uint64_t)x);
}

/* load 8 contiguous uint64_t from SoA column */
static inline __m512i m512_ld64(const void *p)
{
    return _mm512_load_epi64(p);
}

/* store 8 contiguous uint64_t to SoA column */
static inline void m512_st64(void *p, __m512i v)
{
    _mm512_store_epi64(p, v);
}

/* 32×32→64 unsigned multiply (vpmuludq) – low halves of 64-bit lanes */
static inline __m512i m512_mul32_64(__m512i a, __m512i b)
{
    return _mm512_mul_epu32(a, b);
}

/* ── MAC step:  t += ai * Bj + carry,  return new t (lower 32b) & new carry (upper 32b) ── */
static inline void m512_mac_carry(__m512i t_in, __m512i ai, __m512i bj, __m512i carry_in,
                                  __m512i *t_out, __m512i *carry_out)
{
    __m512i prod = m512_mul32_64(ai, bj);
    __m512i s1   = _mm512_add_epi64(t_in, prod);
    __m512i s2   = _mm512_add_epi64(s1, carry_in);

    *t_out      = _mm512_and_epi64(s2, _mm512_set1_epi64(0xFFFFFFFFULL));
    *carry_out  = _mm512_srli_epi64(s2, 32);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Low-level AVX2 helpers  (lanes = 4)
 * ──────────────────────────────────────────────────────────────────────── */

static inline __m256i m256_bcst32_64(uint32_t x)
{
    return _mm256_set1_epi64x((uint64_t)x);
}
static inline __m256i m256_ld64(const void *p)
{
    return _mm256_load_si256((const __m256i *)p);
}
static inline void m256_st64(void *p, __m256i v)
{
    _mm256_store_si256((__m256i *)p, v);
}
static inline __m256i m256_mul32_64(__m256i a, __m256i b)
{
    return _mm256_mul_epu32(a, b);
}
static inline void m256_mac_carry(__m256i t_in, __m256i ai, __m256i bj, __m256i carry_in,
                                  __m256i *t_out, __m256i *carry_out)
{
    __m256i prod = m256_mul32_64(ai, bj);
    __m256i s1   = _mm256_add_epi64(t_in, prod);
    __m256i s2   = _mm256_add_epi64(s1, carry_in);
    *t_out       = _mm256_and_si256(s2, _mm256_set1_epi64x(0xFFFFFFFFULL));
    *carry_out   = _mm256_srli_epi64(s2, 32);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Core batched CIOS  (template by SIMD width)
 * ════════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────────
 *  Unified AVX512 batch  (16 curves)
 * ──────────────────────────────────────────────────────────────────────── */
void avx512_mont_cios_batch(uint32_t *out, const uint32_t *a, const uint32_t *b,
                            const uint32_t *N, uint32_t np0, uint32_t limbs,
                            uint32_t max_limbs, uint32_t kInstances)
{
    if (limbs == 0 || limbs > max_limbs || kInstances != 16 || limbs > CPU_MONT_MAX_LIMBS)
        return;

    const uint32_t STRIDE = max_limbs;   /* elements between instances in user arrays */
    const uint32_t LIM = limbs;
    const uint32_t K   = 16;

    /* ── SoA conversion buffers ──
     * t[LIM+2][16]  — accumulator, 64-bit per lane
     * B[LIM][16]    — cached operand b, 64-bit
     * a64[LIM][16]  — operand a, 64-bit (pre-extracted)
     * On MSVC: __declspec(align(64))
     */
#if defined(_MSC_VER)
    __declspec(align(64))
#endif
    uint64_t t_soa[CPU_MONT_MAX_LIMBS + 2][16];
#if defined(_MSC_VER)
    __declspec(align(64))
#endif
    uint64_t B_soa[CPU_MONT_MAX_LIMBS][16];
#if defined(_MSC_VER)
    __declspec(align(64))
#endif
    uint64_t a64_soa[CPU_MONT_MAX_LIMBS][16];

    /* Zero all */
    std::memset(t_soa,   0, sizeof(t_soa));
    std::memset(B_soa,   0, sizeof(B_soa));
    std::memset(a64_soa, 0, sizeof(a64_soa));

    /* Pack a, b from [K×STRIDE] to SoA [LIM][16] 64-bit */
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t j = 0; j < LIM; ++j) {
            B_soa  [j][k] = (uint64_t)b[k * STRIDE + j];
            a64_soa[j][k] = (uint64_t)a[k * STRIDE + j];
        }
    }

    /* ── CIOS outer loop ── */
    for (uint32_t i = 0; i < LIM; ++i) {

        /* ─── MAC:  t += a[i] * B ─── */
        {
            __m512i carry_lo = _mm512_setzero_si512();
            __m512i carry_hi = _mm512_setzero_si512();

            __m512i ai_lo = m512_ld64(&a64_soa[i][0]);   /* curves 0-7 */
            __m512i ai_hi = m512_ld64(&a64_soa[i][8]);   /* curves 8-15 */

            for (uint32_t j = 0; j < LIM; ++j) {
                __m512i tj_lo = m512_ld64(&t_soa[j][0]);
                __m512i tj_hi = m512_ld64(&t_soa[j][8]);
                __m512i bj_lo = m512_ld64(&B_soa[j][0]);
                __m512i bj_hi = m512_ld64(&B_soa[j][8]);

                __m512i nlo, ncarry_lo;
                __m512i nhi, ncarry_hi;
                m512_mac_carry(tj_lo, ai_lo, bj_lo, carry_lo, &nlo, &ncarry_lo);
                m512_mac_carry(tj_hi, ai_hi, bj_hi, carry_hi, &nhi, &ncarry_hi);

                m512_st64(&t_soa[j][0], nlo);
                m512_st64(&t_soa[j][8], nhi);
                carry_lo = ncarry_lo;
                carry_hi = ncarry_hi;
            }
            /* handle t[LIM] (extra limb) */
            {
                __m512i tl_lo = m512_ld64(&t_soa[LIM][0]);
                __m512i tl_hi = m512_ld64(&t_soa[LIM][8]);

                tl_lo = _mm512_add_epi64(tl_lo, carry_lo);
                tl_hi = _mm512_add_epi64(tl_hi, carry_hi);

                /* carry for LIM+1 */
                __m512i co_lo = _mm512_srli_epi64(tl_lo, 32);
                __m512i co_hi = _mm512_srli_epi64(tl_hi, 32);
                tl_lo = _mm512_and_epi64(tl_lo, _mm512_set1_epi64(0xFFFFFFFFULL));
                tl_hi = _mm512_and_epi64(tl_hi, _mm512_set1_epi64(0xFFFFFFFFULL));

                __m512i tn_lo = m512_ld64(&t_soa[LIM + 1][0]);
                __m512i tn_hi = m512_ld64(&t_soa[LIM + 1][8]);

                tn_lo = _mm512_add_epi64(tn_lo, co_lo);
                tn_hi = _mm512_add_epi64(tn_hi, co_hi);

                m512_st64(&t_soa[LIM][0],        tl_lo);
                m512_st64(&t_soa[LIM][8],        tl_hi);
                m512_st64(&t_soa[LIM + 1][0],    tn_lo);
                m512_st64(&t_soa[LIM + 1][8],    tn_hi);
            }
        }

        /* ─── Reduction: m = t[0] * np0 (mod 2^32) ─── */
        __m512i t0_lo = m512_ld64(&t_soa[0][0]);
        __m512i t0_hi = m512_ld64(&t_soa[0][8]);
        __m512i m_lo  = m512_mul32_64(t0_lo, m512_bcst32_64(np0));
        __m512i m_hi  = m512_mul32_64(t0_hi, m512_bcst32_64(np0));
        /* m is already low-32-bit due to vpmuludq, mask just in case */
        m_lo = _mm512_and_epi64(m_lo, _mm512_set1_epi64(0xFFFFFFFFULL));
        m_hi = _mm512_and_epi64(m_hi, _mm512_set1_epi64(0xFFFFFFFFULL));

        /* ─── Reduction MAC:  t += m * N, shift down ─── */
        {
            __m512i carry_lo = _mm512_setzero_si512();
            __m512i carry_hi = _mm512_setzero_si512();

            for (uint32_t j = 0; j < LIM; ++j) {
                __m512i tj_lo = m512_ld64(&t_soa[j][0]);
                __m512i tj_hi = m512_ld64(&t_soa[j][8]);
                __m512i nj     = m512_bcst32_64(N[j]);   /* N[j] same for all curves */

                __m512i nlo, ncarry_lo;
                __m512i nhi, ncarry_hi;
                m512_mac_carry(tj_lo, m_lo, nj, carry_lo, &nlo, &ncarry_lo);
                m512_mac_carry(tj_hi, m_hi, nj, carry_hi, &nhi, &ncarry_hi);

                if (j > 0) {
                    m512_st64(&t_soa[j - 1][0], nlo);
                    m512_st64(&t_soa[j - 1][8], nhi);
                }
                /* j == 0 result is discarded (t[-1] doesn't exist) */
                carry_lo = ncarry_lo;
                carry_hi = ncarry_hi;
            }
            /* shift-down tail (matches scalar 2-step carry propagation) */
            {
                /* Step 1: top = t[LIM] + carry; t[LIM-1] = lo32(top); carry = hi32(top) */
                __m512i tl_lo = m512_ld64(&t_soa[LIM][0]);
                __m512i tl_hi = m512_ld64(&t_soa[LIM][8]);

                tl_lo = _mm512_add_epi64(tl_lo, carry_lo);
                tl_hi = _mm512_add_epi64(tl_hi, carry_hi);

                __m512i co_lo = _mm512_srli_epi64(tl_lo, 32);
                __m512i co_hi = _mm512_srli_epi64(tl_hi, 32);
                tl_lo = _mm512_and_epi64(tl_lo, _mm512_set1_epi64(0xFFFFFFFFULL));
                tl_hi = _mm512_and_epi64(tl_hi, _mm512_set1_epi64(0xFFFFFFFFULL));

                m512_st64(&t_soa[LIM - 1][0], tl_lo);
                m512_st64(&t_soa[LIM - 1][8], tl_hi);

                /* Step 2: top = t[LIM+1] + carry; t[LIM] = lo32(top); t[LIM+1] = hi32(top) */
                __m512i tn_lo = m512_ld64(&t_soa[LIM + 1][0]);
                __m512i tn_hi = m512_ld64(&t_soa[LIM + 1][8]);

                tn_lo = _mm512_add_epi64(tn_lo, co_lo);
                tn_hi = _mm512_add_epi64(tn_hi, co_hi);

                m512_st64(&t_soa[LIM][0],
                    _mm512_and_epi64(tn_lo, _mm512_set1_epi64(0xFFFFFFFFULL)));
                m512_st64(&t_soa[LIM][8],
                    _mm512_and_epi64(tn_hi, _mm512_set1_epi64(0xFFFFFFFFULL)));
                m512_st64(&t_soa[LIM + 1][0], _mm512_srli_epi64(tn_lo, 32));
                m512_st64(&t_soa[LIM + 1][8], _mm512_srli_epi64(tn_hi, 32));
            }
        }
    }

    /* ── Final conditional subtract ── */
    for (uint32_t k = 0; k < K; ++k) {
        /* extract t for curve k */
        uint32_t t_local[CPU_MONT_MAX_LIMBS + 2];
        for (uint32_t j = 0; j < LIM + 2; ++j) {
            t_local[j] = (uint32_t)t_soa[j][k];
        }

        uint64_t borrow = 0;
        uint32_t D[CPU_MONT_MAX_LIMBS];
        for (uint32_t j = 0; j < LIM; ++j) {
            uint64_t tv = (uint64_t)t_local[j];
            uint64_t nv = (uint64_t)N[j];
            uint64_t w  = tv - nv - borrow;
            D[j]        = (uint32_t)w;
            borrow      = (tv < nv + borrow) ? 1 : 0;
        }

        uint32_t need_sub = (t_local[LIM] != 0 || t_local[LIM + 1] != 0) ? 1 : 0;
        need_sub          = (borrow == 0) ? 1 : need_sub;
        uint32_t mask     = 0u - need_sub;

        for (uint32_t j = 0; j < LIM; ++j) {
            out[k * STRIDE + j] = (D[j] & mask) | (t_local[j] & ~mask);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AVX2 batch  (8 curves)
 * ════════════════════════════════════════════════════════════════════════ */

void avx2_mont_cios_batch(uint32_t *out, const uint32_t *a, const uint32_t *b,
                          const uint32_t *N, uint32_t np0, uint32_t limbs,
                          uint32_t max_limbs, uint32_t kInstances)
{
    if (limbs == 0 || limbs > max_limbs || kInstances != 8 || limbs > CPU_MONT_MAX_LIMBS)
        return;

    const uint32_t STRIDE = max_limbs;
    const uint32_t LIM    = limbs;
    const uint32_t K      = 8;

#if defined(_MSC_VER)
    __declspec(align(32))
#endif
    uint64_t t_soa[CPU_MONT_MAX_LIMBS + 2][8];
#if defined(_MSC_VER)
    __declspec(align(32))
#endif
    uint64_t B_soa[CPU_MONT_MAX_LIMBS][8];
#if defined(_MSC_VER)
    __declspec(align(32))
#endif
    uint64_t a64_soa[CPU_MONT_MAX_LIMBS][8];

    std::memset(t_soa,   0, sizeof(t_soa));
    std::memset(B_soa,   0, sizeof(B_soa));
    std::memset(a64_soa, 0, sizeof(a64_soa));

    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t j = 0; j < LIM; ++j) {
            B_soa  [j][k] = (uint64_t)b[k * STRIDE + j];
            a64_soa[j][k] = (uint64_t)a[k * STRIDE + j];
        }
    }

    for (uint32_t i = 0; i < LIM; ++i) {
        /* ─── MAC ─── */
        {
            __m256i carry  = _mm256_setzero_si256();
            __m256i ai     = m256_ld64(&a64_soa[i][0]);

            for (uint32_t j = 0; j < LIM; ++j) {
                __m256i tj = m256_ld64(&t_soa[j][0]);
                __m256i bj = m256_ld64(&B_soa[j][0]);
                __m256i nt, nc;
                m256_mac_carry(tj, ai, bj, carry, &nt, &nc);
                m256_st64(&t_soa[j][0], nt);
                carry = nc;
            }
            /* t[LIM] */
            {
                __m256i tl = m256_ld64(&t_soa[LIM][0]);
                tl = _mm256_add_epi64(tl, carry);
                __m256i co = _mm256_srli_epi64(tl, 32);
                tl = _mm256_and_si256(tl, _mm256_set1_epi64x(0xFFFFFFFFULL));

                __m256i tn = m256_ld64(&t_soa[LIM + 1][0]);
                tn = _mm256_add_epi64(tn, co);
                m256_st64(&t_soa[LIM][0],     tl);
                m256_st64(&t_soa[LIM + 1][0], tn);
            }
        }

        /* ─── Reduction ─── */
        {
            __m256i t0  = m256_ld64(&t_soa[0][0]);
            __m256i m   = m256_mul32_64(t0, m256_bcst32_64(np0));
            m = _mm256_and_si256(m, _mm256_set1_epi64x(0xFFFFFFFFULL));

            __m256i carry = _mm256_setzero_si256();
            for (uint32_t j = 0; j < LIM; ++j) {
                __m256i tj = m256_ld64(&t_soa[j][0]);
                __m256i nj = m256_bcst32_64(N[j]);
                __m256i nt, nc;
                m256_mac_carry(tj, m, nj, carry, &nt, &nc);
                if (j > 0) {
                    m256_st64(&t_soa[j - 1][0], nt);
                }
                carry = nc;
            }
            /* shift-down tail */
            {
                __m256i tl = m256_ld64(&t_soa[LIM][0]);
                tl = _mm256_add_epi64(tl, carry);
                __m256i co = _mm256_srli_epi64(tl, 32);
                tl = _mm256_and_si256(tl, _mm256_set1_epi64x(0xFFFFFFFFULL));
                m256_st64(&t_soa[LIM - 1][0], tl);

                __m256i tn = m256_ld64(&t_soa[LIM + 1][0]);
                tn = _mm256_add_epi64(tn, co);
                m256_st64(&t_soa[LIM][0],     _mm256_and_si256(tn, _mm256_set1_epi64x(0xFFFFFFFFULL)));
                m256_st64(&t_soa[LIM + 1][0], _mm256_srli_epi64(tn, 32));
            }
        }
    }

    /* ── Final conditional subtract ── */
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t t_local[CPU_MONT_MAX_LIMBS + 2];
        for (uint32_t j = 0; j < LIM + 2; ++j) {
            t_local[j] = (uint32_t)t_soa[j][k];
        }

        uint64_t borrow = 0;
        uint32_t D[CPU_MONT_MAX_LIMBS];
        for (uint32_t j = 0; j < LIM; ++j) {
            uint64_t tv = (uint64_t)t_local[j];
            uint64_t nv = (uint64_t)N[j];
            uint64_t w  = tv - nv - borrow;
            D[j]        = (uint32_t)w;
            borrow      = (tv < nv + borrow) ? 1 : 0;
        }

        uint32_t need_sub = (t_local[LIM] != 0 || t_local[LIM + 1] != 0) ? 1 : 0;
        need_sub          = (borrow == 0) ? 1 : need_sub;
        uint32_t mask     = 0u - need_sub;

        for (uint32_t j = 0; j < LIM; ++j) {
            out[k * STRIDE + j] = (D[j] & mask) | (t_local[j] & ~mask);
        }
    }
}