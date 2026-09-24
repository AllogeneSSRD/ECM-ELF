/* ---------------------------------------------------------------------------
 * simd_mont_curve.cpp -- 8-lane SoA Suyama-sigma Montgomery stage 1.
 *
 * Point ops are the same formulas as the scalar reference (ecm_mont_cpu.cpp), just
 * vectorised over 8 curves:
 *
 *   xdbl : A=(X+Z)^2, B=(X-Z)^2, E=A-B=4XZ, X2=A*B, Z2=E*(B+a24*E)      [3M+2S]
 *   xadd : t4=(Xp+Zp)(Xq-Zq), t5=(Xp-Zp)(Xq+Zq),
 *          X3=(t4+t5)^2, Z3=(t4-t5)^2*xdiff                             [3M+2S]
 *
 * where xdiff is the AFFINE x of the fixed difference point (the start point).
 * Additions/subtractions are the same SoA helpers the Edwards path uses (limbs are
 * canonical, per-lane carry, conditional subtract of N).
 *
 * All 8 lanes share the exponent bits, so the ladder has no divergence at all.
 * ------------------------------------------------------------------------- */
#include "simd_mont_curve.h"
#include "ecm_mont_cpu.h"        /* mont_expand_bits() */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

/* 52-bit limb mask (the kernel header keeps this private, like simd_edwards.cpp does) */
#ifndef IFMA_M52
#define IFMA_M52 0x000FFFFFFFFFFFFFULL
#endif

/* --------------------------------------------------------------------------
 * SoA field helpers (add / sub with per-lane canonical reduction)
 *
 * Pass budget per helper -- this is the ladder's non-multiply cost, and it was
 * measured at 18-26% of a ladder bit (docs/ECM_Montgomery_STAGE1.md §11.1), so the
 * number of passes over the n columns matters:
 *
 *   soa_add : 2 passes (add + candidate in one pass, then one select pass)
 *   soa_sub : 2 passes (subtract with borrow, then add N where the borrow says so)
 *
 * The previous versions cost 3 and 5 passes: add used a separate two-pass
 * conditional subtract, and sub was implemented as neg + add (2 + 3).  The ladder
 * runs 4 additions and 4 subtractions per bit, so this removes ~12 of ~30 passes.
 * ------------------------------------------------------------------------ */
namespace {

static inline __m512i ld(const uint64_t *p) { return _mm512_load_si512((const __m512i *)p); }
static inline void    st(uint64_t *p, __m512i v) { _mm512_store_si512((__m512i *)p, v); }

/* r = a + b mod N, canonical (0 <= r < N).
 *
 * `tmp` (8n words) receives the candidate r - N, formed in the same pass; the
 * second pass selects it where the value was >= N.  Selection rule, with cy the
 * carry out of column n-1 and `borrow` the borrow out of (limbs - N):
 *     value >= N  <=>  cy != 0  OR  borrow == 0
 * (cy != 0 forces the masked limbs to be < N, so the limb-wise difference is
 * exactly value - N in that case; see the derivation in the dev doc §13.7.) */
static void soa_add(uint64_t *r, const uint64_t *a, const uint64_t *b, uint64_t *tmp,
                    const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)IFMA_M52);
    const __m512i zero = _mm512_setzero_si512();
    __m512i cy = zero, borrow = zero;

    for (size_t i = 0; i < n; i++) {
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(ld(a + 8 * i), ld(b + 8 * i)), cy);
        cy = _mm512_srli_epi64(s, 52);
        const __m512i si = _mm512_and_si512(s, mask);
        st(r + 8 * i, si);
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(si, ld(nb + 8 * i)), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(tmp + 8 * i, _mm512_and_si512(d, mask));
    }

    const __mmask8 need = (__mmask8)(_mm512_cmpneq_epi64_mask(cy, zero) |
                                     _mm512_cmpeq_epi64_mask(borrow, zero));
    for (size_t i = 0; i < n; i++)
        st(r + 8 * i, _mm512_mask_blend_epi64(need, ld(r + 8 * i), ld(tmp + 8 * i)));
}

/* r = a - b mod N, canonical.  One pass subtracts with borrow propagation, the
   second adds N back exactly in the lanes that went negative (there the true value
   is a - b + N, and the discarded carry out of column n-1 cancels the borrow). */
static void soa_sub(uint64_t *r, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    const __m512i mask = _mm512_set1_epi64((long long)IFMA_M52);
    const __m512i zero = _mm512_setzero_si512();
    __m512i borrow = zero;

    for (size_t i = 0; i < n; i++) {
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(ld(a + 8 * i), ld(b + 8 * i)), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(r + 8 * i, _mm512_and_si512(d, mask));
    }

    const __mmask8 neg = _mm512_cmpneq_epi64_mask(borrow, zero);
    __m512i carry = zero;
    for (size_t i = 0; i < n; i++) {
        const __m512i addend = _mm512_mask_blend_epi64(neg, zero, ld(nb + 8 * i));
        const __m512i v = _mm512_add_epi64(_mm512_add_epi64(ld(r + 8 * i), addend), carry);
        carry = _mm512_srli_epi64(v, 52);
        st(r + 8 * i, _mm512_and_si512(v, mask));
    }
}

/* --------------------------------------------------------------------------
 * SoA point ops
 * ------------------------------------------------------------------------ */
struct soa_xz { uint64_t *X, *Z; };

/* X2 = (X+Z)^2 (X-Z)^2 ,  Z2 = 4XZ * ((X-Z)^2 + a24*4XZ) */
static void xz_dbl(const soa_xz &r, const soa_xz &p, const uint64_t *a24,
                   uint64_t *A_, uint64_t *B_, uint64_t *E_, uint64_t *t, uint64_t *tmp,
                   const ifma_ctx_t *mc)
{
    soa_add(A_, p.X, p.Z, tmp, mc);
    ifma_mont_sqr(A_, A_, mc);
    soa_sub(B_, p.X, p.Z, mc);
    ifma_mont_sqr(B_, B_, mc);
    soa_sub(E_, A_, B_, mc);                       /* E = 4XZ */
    ifma_mont_mul(r.X, A_, B_, mc);
    ifma_mont_mul(t, a24, E_, mc);
    soa_add(t, t, B_, tmp, mc);                    /* B + a24*E */
    ifma_mont_mul(r.Z, E_, t, mc);
}

/* X3 = (t4+t5)^2 ,  Z3 = (t4-t5)^2 * xdiff.
   r must not alias p or q's *live* values: the result lands directly in r.X (no
   copy pass), which is safe here because every read of p and q happens first. */
static void xz_add(const soa_xz &r, const soa_xz &p, const soa_xz &q, const uint64_t *xdiff,
                   uint64_t *t0, uint64_t *t1, uint64_t *t2, uint64_t *t3,
                   uint64_t *t4, uint64_t *t5, uint64_t *tmp, const ifma_ctx_t *mc)
{
    soa_add(t0, p.X, p.Z, tmp, mc);
    soa_sub(t1, p.X, p.Z, mc);
    soa_add(t2, q.X, q.Z, tmp, mc);
    soa_sub(t3, q.X, q.Z, mc);
    ifma_mont_mul(t4, t0, t3, mc);
    ifma_mont_mul(t5, t1, t2, mc);
    soa_add(t0, t4, t5, tmp, mc);
    soa_sub(t1, t4, t5, mc);
    ifma_mont_sqr(r.X, t0, mc);                    /* was: sqr into t4 + memcpy */
    ifma_mont_sqr(t5, t1, mc);
    ifma_mont_mul(r.Z, t5, xdiff, mc);
}

} /* anonymous namespace */

/* --------------------------------------------------------------------------
 * context
 * ------------------------------------------------------------------------ */
#define MONT_SOA_POOL 22

int mont_soa_init(mont_soa_ctx_t *c, const mpz_t N, int field_mode)
{
    memset(c, 0, sizeof(*c));
    if (ifma_ctx_init_ex(&c->mc, N, field_mode) != 0) return -1;
    c->n = c->mc.n;
    c->pool_elems = MONT_SOA_POOL;
    c->pool = (uint64_t *)_mm_malloc(MONT_SOA_POOL * 8 * c->n * sizeof(uint64_t), 64);
    if (!c->pool) { ifma_ctx_clear(&c->mc); return -1; }
    return 0;
}

void mont_soa_clear(mont_soa_ctx_t *c)
{
    if (c->pool) _mm_free(c->pool);
    c->pool = nullptr;
    ifma_ctx_clear(&c->mc);
    memset(c, 0, sizeof(*c));
}

/* --------------------------------------------------------------------------
 * one batch: [s]P for 8 curves
 * ------------------------------------------------------------------------ */
int mont_soa_stage1_bits(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                         const uint64_t sigmas[IFMA_LANES],
                         mpz_t *out_x, mpz_t *out_gcd)
{
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n, lw = 8 * n;
    uint64_t *pool = c->pool;

    soa_xz R0 = { pool + 0 * lw, pool + 1 * lw };
    soa_xz R1 = { pool + 2 * lw, pool + 3 * lw };
    soa_xz T  = { pool + 4 * lw, pool + 5 * lw };
    uint64_t *a24 = pool + 6 * lw, *xdiff = pool + 7 * lw;
    uint64_t *A_ = pool + 8 * lw, *B_ = pool + 9 * lw, *E_ = pool + 10 * lw, *tt = pool + 11 * lw;
    uint64_t *t0 = pool + 12 * lw, *t1 = pool + 13 * lw, *t2 = pool + 14 * lw;
    uint64_t *t3 = pool + 15 * lw, *t4 = pool + 16 * lw, *t5 = pool + 17 * lw;
    uint64_t *tmp = pool + 18 * lw;      /* scratch for soa_add's candidate r-N */

    /* per-lane curve setup: sigma -> A, a24, start (X0:Z0), affine xdiff */
    mpz_t u, v, t, num, den, X0, Z0, Amz, inv;
    mpz_inits(u, v, t, num, den, X0, Z0, Amz, inv, NULL);
    for (unsigned k = 0; k < IFMA_LANES; k++) {
        mpz_set_ui(u, (unsigned long)sigmas[k]);
        mpz_mul(u, u, u);
        mpz_sub_ui(u, u, 5);                       /* u = sigma^2 - 5 */
        mpz_set_ui(v, (unsigned long)sigmas[k]);
        mpz_mul_ui(v, v, 4);                       /* v = 4*sigma */
        mpz_sub(t, v, u);
        mpz_powm_ui(num, t, 3, mc->N);
        mpz_mul_ui(t, u, 3);
        mpz_add(t, t, v);                          /* 3u+v */
        mpz_mul(num, num, t);
        mpz_mod(num, num, mc->N);
        mpz_powm_ui(den, u, 3, mc->N);
        mpz_mul_ui(den, den, 4);
        mpz_mul(den, den, v);
        mpz_mod(den, den, mc->N);
        if (mpz_invert(inv, den, mc->N) == 0) mpz_set_ui(inv, 0);
        mpz_mul(num, num, inv);
        mpz_sub_ui(num, num, 2);
        mpz_mod(Amz, num, mc->N);                  /* A */
        mpz_powm_ui(X0, u, 3, mc->N);
        mpz_powm_ui(Z0, v, 3, mc->N);

        ifma_from_mpz_lane(R0.X, k, X0, mc);
        ifma_from_mpz_lane(R0.Z, k, Z0, mc);

        mpz_add_ui(num, Amz, 2);
        mpz_set_ui(inv, 4);
        mpz_invert(inv, inv, mc->N);
        mpz_mul(num, num, inv);
        mpz_mod(num, num, mc->N);                  /* a24 = (A+2)/4 */
        ifma_from_mpz_lane(a24, k, num, mc);

        mpz_invert(inv, Z0, mc->N);
        mpz_mul(num, X0, inv);
        mpz_mod(num, num, mc->N);                  /* affine x of the start point */
        ifma_from_mpz_lane(xdiff, k, num, mc);
    }
    /* NOTE: the mpz temporaries are freed only AFTER the results loop below --
       that loop still uses `inv` (and an earlier version cleared it here, which is a
       use-after-free that shows up as STATUS_HEAP_CORRUPTION at exit). */

    /* R1 = 2*R0, then the branch-free ladder over the shared exponent bits */
    xz_dbl(R1, R0, a24, A_, B_, E_, tt, tmp, mc);

    soa_xz *p0 = &R0, *p1 = &R1, *pt = &T;
    for (size_t i = 1; i < nbits; i++) {           /* bits[0] is the implicit top bit */
        if (bits[i]) {
            xz_add(*pt, *p1, *p0, xdiff, t0, t1, t2, t3, t4, t5, tmp, mc);
            xz_dbl(*p1, *p1, a24, A_, B_, E_, tt, tmp, mc);
            soa_xz *sw = p0; p0 = pt; pt = sw;
        } else {
            xz_add(*pt, *p0, *p1, xdiff, t0, t1, t2, t3, t4, t5, tmp, mc);
            xz_dbl(*p0, *p0, a24, A_, B_, E_, tt, tmp, mc);
            soa_xz *sw = p1; p1 = pt; pt = sw;
        }
    }

    /* results: per lane, normalise x and take gcd(Z, N) */
    mpz_t Z;
    mpz_init(Z);
    for (unsigned k = 0; k < IFMA_LANES; k++) {
        ifma_to_mpz_lane(Z, p0->Z, k, mc);
        mpz_gcd(out_gcd[k], Z, mc->N);
        if (mpz_sgn(Z) != 0 && mpz_invert(inv, Z, mc->N)) {
            mpz_t X;
            mpz_init(X);
            ifma_to_mpz_lane(X, p0->X, k, mc);
            mpz_mul(out_x[k], X, inv);
            mpz_mod(out_x[k], out_x[k], mc->N);
            mpz_clear(X);
        } else {
            mpz_set_ui(out_x[k], 0);
        }
    }
    mpz_clear(Z);
    mpz_clears(u, v, t, num, den, X0, Z0, Amz, inv, NULL);
    return 0;
}

int mont_soa_stage1(mont_soa_ctx_t *c, const mpz_t s, const uint64_t sigmas[IFMA_LANES],
                    mpz_t *out_x, mpz_t *out_gcd)
{
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    const int rc = mont_soa_stage1_bits(c, bits, nbits, sigmas, out_x, out_gcd);
    free(bits);
    return rc;
}

void mont_soa_op_counts(uint64_t *out)
{
    /* per ladder bit: one xdbl (3M+2S) + one xadd (3M+2S) = 6 M + 4 S, in permille */
    out[0] = 6000;
    out[1] = 4000;
}
