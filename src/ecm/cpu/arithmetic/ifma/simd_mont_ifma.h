/* ---------------------------------------------------------------------------
 * simd_mont_ifma.h — batched Montgomery multiplication, 8 curves in one zmm.
 *
 * Layout (SoA, "lane = curve"):
 *   an element is uint64_t e[8*n]; column i is the 64-byte vector e + 8*i;
 *   its lane k (k in [0,8)) is 52-bit limb i of curve k.
 *   So one full field element is n zmm registers, and a batch of 8 curves is
 *   ONE array — no interleaving at the call site, no per-limb lane gather.
 *
 * Radix: 2^52 (IFMA / vpmadd52luq|huq).  A limb is 52 bits, so n = ceil(bits/52)
 *   limbs; a lazy accumulator column can sum ~2^12 products of 52x52 bits
 *   before it would overflow 64 bits, which is what makes the single-pass
 *   word-level CIOS below safe with 64-bit accumulators.
 *
 * !!! This translation unit requires AVX512-IFMA: compile it with
 *     /arch:AVX512 (MSVC) or -mavx512f -mavx512ifma (GCC/Clang), and only
 *     call it after runtime CPUID detection of AVX512F+AVX512IFMA.
 *
 * Threading: a context owns its scratch accumulator, so it is NOT thread
 *   safe — give every worker thread its own ifma_ctx_t (they are cheap:
 *   ~(10n+2) zmm-sized buffers).
 *
 * Aliasing: ifma_mont_mul() reads all of a and b before it writes out, so
 *   out may alias a or b (in-place squaring is fine).  out must not alias
 *   the context scratch.
 * ------------------------------------------------------------------------- */
#ifndef SIMD_MONT_IFMA_H
#define SIMD_MONT_IFMA_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IFMA_LANES 8

/* Field mode: how a multiplication reduces modulo N.
 *
 *   IFMA_FIELD_MONT  Montgomery, R = 2^(52n).  Works for every odd N, but the
 *                    reduction is a full extra n-limb multiply per row, so a
 *                    mul costs n(4n+3) madds.
 *   IFMA_FIELD_MERS  Mersenne fold for N = 2^k - 1: 2^k = 1 (mod N) turns the
 *                    reduction into a shift and an add, so a mul costs 2n^2
 *                    madds (~2x fewer).  Values live in the *plain* domain
 *                    (one = 1, no R); everything above the kernel (point ops,
 *                    dictionary, checkpoint, gcd) is domain agnostic.
 *   IFMA_FIELD_AUTO  MERS when N = 2^k - 1 (k >= 64), else MONT.
 */
#define IFMA_FIELD_MONT 0
#define IFMA_FIELD_MERS 1
#define IFMA_FIELD_AUTO 2

typedef struct {
    size_t    n;        /* 52-bit limbs */
    uint64_t  np0;      /* -N^-1 mod 2^52 (Montgomery mode only) */
    uint64_t *nb;       /* 8*n: broadcast modulus, nb[8*j+k] = N[j] */
    uint64_t *one;      /* 8*n: R mod N (mont) or 1 (mersenne), every lane */
    uint64_t *scratch;  /* 8*(2*n+2): CIOS / fold accumulator, 64-byte aligned */
    mpz_t     N;        /* modulus, kept for the (rare) mpz conversions */
    int       mode;     /* effective: IFMA_FIELD_MONT | IFMA_FIELD_MERS */
    int       requested;/* what the caller asked for (an IFMA_FIELD_* value) */
    size_t    k;        /* mersenne: bit length of N, N = 2^k - 1 */
    unsigned  shift;    /* mersenne: 52*n - k, in [0,52) */
    int       ok;
} ifma_ctx_t;

/* N must be odd and > 1.  Returns 0 on success, non-zero on rejection. */
int  ifma_ctx_init(ifma_ctx_t *c, const mpz_t N);                /* = AUTO */
int  ifma_ctx_init_ex(ifma_ctx_t *c, const mpz_t N, int field_mode);
/* "montgomery" / "mersenne(2^k=1, n52=..)" for the startup log. */
const char *ifma_field_name(const ifma_ctx_t *c);
void ifma_ctx_clear(ifma_ctx_t *c);

void ifma_set_zero(uint64_t *e, const ifma_ctx_t *c);
void ifma_set_one (uint64_t *e, const ifma_ctx_t *c);   /* = R mod N (Mont 1) */

/* Montgomery representation of v mod N written into one lane only.
   In mersenne mode this is just v mod N (no R). */
void ifma_from_mpz_lane(uint64_t *e, unsigned lane, const mpz_t v, const ifma_ctx_t *c);
void ifma_from_u64_lane(uint64_t *e, unsigned lane, uint64_t v, const ifma_ctx_t *c);
/* lane -> ordinary integer (Montgomery domain exit; identity in mersenne mode). */
void ifma_to_mpz_lane(mpz_t out, const uint64_t *e, unsigned lane, const ifma_ctx_t *c);

/* out = a*b mod N (mont: a*b*R^-1).  Requires a,b < N (every caller below
   preserves that, and both modes return a canonical result < N). */
void ifma_mont_mul(uint64_t *out, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *c);
void ifma_mont_sqr(uint64_t *out, const uint64_t *a, const ifma_ctx_t *c);

/* Exposed for the gate benchmark: cycles attributed to the madds themselves. */
uint64_t ifma_madd_count(const ifma_ctx_t *c);   /* madds per batch field mul */
uint64_t ifma_sqr_madd_count(const ifma_ctx_t *c); /* madds per batch square */

#ifdef __cplusplus
}
#endif

#endif /* SIMD_MONT_IFMA_H */
