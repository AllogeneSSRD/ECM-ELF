/* ---------------------------------------------------------------------------
 * simd_edwards.cpp — see simd_edwards.h for the contract.
 *
 * Field ops are vectorised on top of ifma_mont_mul (batched, lane = curve).
 * Everything here is "the same formula, 8 lanes wide": the scalar reference is
 * src/cpu/ecm_edwards_cpu.cpp ed_dbl_mont/ed_add_mont/ed_add_affine_mont, and
 * a modular op in this file must return the same canonical residue < N as
 * mont_add/mont_sub/mont_neg do there (both are exact modular arithmetic, so
 * the ladder's final Qx/Qz are bit-identical to the scalar path's).
 * ------------------------------------------------------------------------- */

#include "simd_edwards.h"

#if !defined(__AVX512F__)
#error "simd_edwards.cpp must be compiled with AVX512 (/arch:AVX512)"
#endif

#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

/* The production curve construction and the scalar reference path. */
#include "ecm_edwards_cpu.h"

#define IFMA_M52 0x000FFFFFFFFFFFFFULL

static const __m512i kMask = _mm512_set1_epi64((long long)IFMA_M52);

static inline __m512i ld(const uint64_t *p) { return _mm512_load_si512((const __m512i *)p); }
static inline void    st(uint64_t *p, __m512i v) { _mm512_store_si512((__m512i *)p, v); }

/* ---------------------------------------------------------------------------
 * SoA modular field ops (all inputs/outputs < N, canonical)
 * ------------------------------------------------------------------------- */

static void soa_zero(uint64_t *r, const ifma_ctx_t *mc)
{
    memset(r, 0, 8 * mc->n * sizeof(uint64_t));
}

static void soa_copy(uint64_t *r, const uint64_t *a, const ifma_ctx_t *mc)
{
    memcpy(r, a, 8 * mc->n * sizeof(uint64_t));
}

/* r -= N where the (n+1)-limb value (r, top) is >= N; two passes so that no
   separate copy of the originals is needed. */
static void soa_cond_sub(uint64_t *r, __m512i top, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    __m512i borrow = _mm512_setzero_si512();

    for (size_t i = 0; i < n; i++) {
        const __m512i rk = ld(r + 8 * i), nk = ld(nb + 8 * i);
        const __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(rk, nk), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
    }
    const __mmask8 need = _mm512_cmpge_epu64_mask(top, borrow);

    borrow = _mm512_setzero_si512();
    for (size_t i = 0; i < n; i++) {
        const __m512i rk = ld(r + 8 * i), nk = ld(nb + 8 * i);
        __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(rk, nk), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(r + 8 * i, _mm512_mask_blend_epi64(need, rk, _mm512_and_si512(d, kMask)));
    }
}

/* r = a + b mod N.  Limbs are 52 bits, so a limb-wise sum with a running carry
   register cannot overflow 64 bits. */
static void soa_add(uint64_t *r, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    __m512i cy = _mm512_setzero_si512();
    for (size_t i = 0; i < n; i++) {
        const __m512i s = _mm512_add_epi64(_mm512_add_epi64(ld(a + 8 * i), ld(b + 8 * i)), cy);
        cy = _mm512_srli_epi64(s, 52);
        st(r + 8 * i, _mm512_and_si512(s, kMask));
    }
    soa_cond_sub(r, cy, mc);
}

/* r = -a mod N  (0 stays 0, otherwise N - a). */
static void soa_neg(uint64_t *r, const uint64_t *a, const ifma_ctx_t *mc)
{
    const size_t n = mc->n;
    const uint64_t *nb = mc->nb;
    __m512i borrow = _mm512_setzero_si512();
    /* "a != 0" is the OR over limbs of "limb != 0".  ANDing them would mean
       "every limb is non-zero", which silently zeroes N - a for any sparse a
       (e.g. a = R mod N, or a squared value with a zero limb). */
    __mmask8 nz = 0;
    for (size_t i = 0; i < n; i++) {
        const __m512i ai = ld(a + 8 * i), nk = ld(nb + 8 * i);
        nz = (__mmask8)(nz | ~_mm512_cmpeq_epi64_mask(ai, _mm512_setzero_si512()));
        __m512i d = _mm512_sub_epi64(_mm512_sub_epi64(nk, ai), borrow);
        borrow = _mm512_maskz_set1_epi64(_mm512_movepi64_mask(d), 1);
        st(r + 8 * i, _mm512_and_si512(d, kMask));
    }
    /* a == 0 gave N, which must be 0 */
    for (size_t i = 0; i < n; i++)
        st(r + 8 * i, _mm512_maskz_mov_epi64(nz, ld(r + 8 * i)));
}

/* r = a - b mod N = a + (N - b); `t` is 8n scratch. */
static void soa_sub(uint64_t *r, const uint64_t *a, const uint64_t *b,
                    uint64_t *t, const ifma_ctx_t *mc)
{
    soa_neg(t, b, mc);
    soa_add(r, a, t, mc);
}

/* ---------------------------------------------------------------------------
 * Point ops (mirror ecm_edwards_cpu.cpp)
 * ------------------------------------------------------------------------- */

typedef struct { uint64_t *x, *y, *z, *t; } soa_pt;

static uint64_t *arena_slot(ed_soa_ctx_t *c, int idx)
{
    return c->arena + (size_t)idx * 8 * c->n;
}

/* r = 2p : 4 sqr + 4 mul (a = 1 case). */
static void ed_soa_dbl(ed_soa_ctx_t *c, soa_pt r, const soa_pt p)
{
    const ifma_ctx_t *mc = &c->mc;
    uint64_t *A = arena_slot(c, 0), *B = arena_slot(c, 1), *C = arena_slot(c, 2);
    uint64_t *E = arena_slot(c, 3), *F = arena_slot(c, 4), *G = arena_slot(c, 5);
    uint64_t *H = arena_slot(c, 6), *t = arena_slot(c, 7), *o = arena_slot(c, 8);

    ifma_mont_sqr(A, p.x, mc);
    ifma_mont_sqr(B, p.y, mc);
    ifma_mont_sqr(C, p.z, mc);
    soa_add(C, C, C, mc);                     /* C = 2Z^2 */
    soa_add(t, p.x, p.y, mc);
    ifma_mont_sqr(E, t, mc);
    soa_sub(E, E, A, o, mc);
    soa_sub(E, E, B, o, mc);                  /* E = (X+Y)^2 - A - B */
    soa_add(G, A, B, mc);                     /* G = A + B */
    soa_sub(F, G, C, o, mc);                  /* F = G - C */
    soa_sub(H, A, B, o, mc);                  /* H = A - B */
    ifma_mont_mul(r.x, E, F, mc);
    ifma_mont_mul(r.y, G, H, mc);
    ifma_mont_mul(r.t, E, H, mc);
    ifma_mont_mul(r.z, F, G, mc);
}

/* r = p + q : 9 mul (extended coordinates, both non-affine). */
static void ed_soa_add(ed_soa_ctx_t *c, soa_pt r, const soa_pt p, const soa_pt q)
{
    const ifma_ctx_t *mc = &c->mc;
    uint64_t *A = arena_slot(c, 0), *B = arena_slot(c, 1), *C = arena_slot(c, 2);
    uint64_t *D = arena_slot(c, 3), *E = arena_slot(c, 4), *F = arena_slot(c, 5);
    uint64_t *G = arena_slot(c, 6), *H = arena_slot(c, 7), *t = arena_slot(c, 8), *o = arena_slot(c, 9);

    ifma_mont_mul(A, p.x, q.x, mc);
    ifma_mont_mul(B, p.y, q.y, mc);
    ifma_mont_mul(C, p.t, q.t, mc);
    ifma_mont_mul(C, C, c->d, mc);            /* C = d*T1*T2 */
    ifma_mont_mul(D, p.z, q.z, mc);
    soa_add(t, p.x, p.y, mc);
    soa_add(E, q.x, q.y, mc);
    ifma_mont_mul(E, t, E, mc);
    soa_sub(E, E, A, o, mc);
    soa_sub(E, E, B, o, mc);
    soa_sub(F, D, C, o, mc);
    soa_add(G, D, C, mc);
    soa_sub(H, B, A, o, mc);
    ifma_mont_mul(r.x, E, F, mc);
    ifma_mont_mul(r.y, G, H, mc);
    ifma_mont_mul(r.t, E, H, mc);
    ifma_mont_mul(r.z, F, G, mc);
}

/* r = p + q where q = (x, y, dxy = d*x*y) is affine: 7 mul. */
static void ed_soa_add_affine(ed_soa_ctx_t *c, soa_pt r, const soa_pt p,
                              const uint64_t *qx, const uint64_t *qy, const uint64_t *qdxy)
{
    const ifma_ctx_t *mc = &c->mc;
    uint64_t *A = arena_slot(c, 0), *B = arena_slot(c, 1), *C = arena_slot(c, 2);
    uint64_t *E = arena_slot(c, 3), *F = arena_slot(c, 4), *G = arena_slot(c, 5);
    uint64_t *H = arena_slot(c, 6), *t = arena_slot(c, 7), *o = arena_slot(c, 8);

    ifma_mont_mul(A, p.x, qx, mc);
    ifma_mont_mul(B, p.y, qy, mc);
    ifma_mont_mul(C, p.t, qdxy, mc);
    soa_add(t, p.x, p.y, mc);
    soa_add(E, qx, qy, mc);
    ifma_mont_mul(E, t, E, mc);
    soa_sub(E, E, A, o, mc);
    soa_sub(E, E, B, o, mc);
    soa_sub(F, p.z, C, o, mc);
    soa_add(G, p.z, C, mc);
    soa_sub(H, B, A, o, mc);
    ifma_mont_mul(r.x, E, F, mc);
    ifma_mont_mul(r.y, G, H, mc);
    ifma_mont_mul(r.t, E, H, mc);
    ifma_mont_mul(r.z, F, G, mc);
}

/* ---------------------------------------------------------------------------
 * w-NAF digits — copied from ecm_edwards_cpu.cpp naf_digits (which is static
 * there).  Both produce a valid signed representation of the same exponent, so
 * the ladder's result is identical regardless; kept in sync deliberately.
 * ------------------------------------------------------------------------- */
static void soa_naf_digits(const mpz_t k, int w, std::vector<int> &digits)
{
    digits.clear();
    const size_t nbits = mpz_sizeinbase(k, 2);
    if (nbits == 0) return;
    const int max_val = (1 << (w - 1)) - 1;

    std::vector<int> out(nbits + 1, 0);
    int value = 0, addin = 1, carry = 0;
    size_t start = 0;

    for (size_t bitnum = 0; bitnum < nbits; bitnum++) {
        int this_bit = carry + (mpz_tstbit(k, bitnum) ? 1 : 0);
        carry = this_bit >> 1;
        this_bit &= 1;
        if (this_bit) {
            if (value == 0) start = bitnum;
            value += addin;
        }
        if (value == 0) continue;
        addin <<= 1;

        bool complete;
        if (bitnum == nbits - 1) complete = true;
        else if (addin < max_val) complete = false;
        else if (value <= max_val && addin - value <= max_val) complete = false;
        else complete = true;
        if (!complete) continue;

        if (value <= max_val) out[start] = value;
        else { out[start] = value - addin; carry = 1; }
        value = 0;
        addin = 1;
    }
    if (carry) out[nbits] = 1;
    size_t sz = out.size();
    while (sz > 1 && out[sz - 1] == 0) sz--;
    out.resize(sz);
    digits = std::move(out);
}

/* ---------------------------------------------------------------------------
 * Field-op self test (mirrors what the ladder relies on)
 * ------------------------------------------------------------------------- */
int ed_soa_field_selftest(ed_soa_ctx_t *c, int trials)
{
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n, lw = 8 * n;
    const uint64_t *N = NULL;
    (void)N;
    std::vector<uint64_t> A(8 * n), B(8 * n), R(8 * n), T(8 * n);
    mpz_t av, bv, want, got, tmp;
    mpz_inits(av, bv, want, got, tmp, NULL);
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 13579u);
    int fails = 0;

    for (int t = 0; t < trials; t++) {
        for (unsigned k = 0; k < IFMA_LANES; k++) {
            mpz_urandomm(av, rs, mc->N);
            mpz_urandomm(bv, rs, mc->N);
            ifma_from_mpz_lane(A.data(), k, av, mc);
            ifma_from_mpz_lane(B.data(), k, bv, mc);
        }
        for (int op = 0; op < 3; op++) {
            if (op == 0) soa_add(R.data(), A.data(), B.data(), mc);
            else if (op == 1) soa_sub(R.data(), A.data(), B.data(), T.data(), mc);
            else soa_neg(R.data(), A.data(), mc);            for (unsigned k = 0; k < IFMA_LANES; k++) {
                ifma_to_mpz_lane(got, R.data(), k, mc);
                ifma_to_mpz_lane(av, A.data(), k, mc);
                ifma_to_mpz_lane(bv, B.data(), k, mc);
                if (op == 0) { mpz_add(want, av, bv); }
                else if (op == 1) { mpz_sub(want, av, bv); }
                else { mpz_neg(want, av); }
                mpz_mod(want, want, mc->N);
                if (mpz_cmp(want, got) != 0) {
                    if (fails < 5)
                        gmp_printf("  [selftest] op=%d lane=%u want=%Zx got=%Zx\n", op, k, want, got);
                    fails++;
                }
            }
        }
        (void)lw;
        /* aliased variants: the ladder does r = r - A and r = r + A in place,
           which the distinct-buffer cases above do not exercise. */
        for (int op = 0; op < 3; op++) {
            std::vector<uint64_t> Aa(A), Ba(B);
            mpz_t av0, bv0, want0;
            mpz_inits(av0, bv0, want0, NULL);
            ifma_to_mpz_lane(av0, Aa.data(), 0, mc);
            ifma_to_mpz_lane(bv0, Ba.data(), 0, mc);
            if (op == 0) { soa_add(Aa.data(), Aa.data(), Ba.data(), mc);       mpz_add(want0, av0, bv0); }
            else if (op == 1) { soa_sub(Aa.data(), Aa.data(), Ba.data(), T.data(), mc); mpz_sub(want0, av0, bv0); }
            else { soa_neg(Aa.data(), Aa.data(), mc);                          mpz_neg(want0, av0); }
            mpz_mod(want0, want0, mc->N);
            ifma_to_mpz_lane(got, Aa.data(), 0, mc);
            if (mpz_cmp(want0, got) != 0) {
                if (fails < 12) gmp_printf("  [selftest] ALIASED op=%d lane=0 want=%Zx got=%Zx\n", op, want0, got);
                fails++;
            }
            mpz_clears(av0, bv0, want0, NULL);
        }
        /* doubling form: r = r + r (all three the same buffer) */
        {
            std::vector<uint64_t> Aa(A);
            mpz_t av0, want0;
            mpz_inits(av0, want0, NULL);
            ifma_to_mpz_lane(av0, Aa.data(), 0, mc);
            soa_add(Aa.data(), Aa.data(), Aa.data(), mc);
            mpz_add(want0, av0, av0); mpz_mod(want0, want0, mc->N);
            ifma_to_mpz_lane(got, Aa.data(), 0, mc);
            if (mpz_cmp(want0, got) != 0) {
                if (fails < 14) gmp_printf("  [selftest] SELF-ADD lane=0 want=%Zx got=%Zx\n", want0, got);
                fails++;
            }
            mpz_clears(av0, want0, NULL);
        }
        /* adversarial patterns + chained sub: the ladder subtracts twice in a
           row through the same scratch, which random cases did not cover. */
        for (int pat = 0; pat < 4; pat++) {
            std::vector<uint64_t> Aa(8 * n), Ba(8 * n), Tt(8 * n);
            mpz_t av0, bv0, want0;
            mpz_inits(av0, bv0, want0, NULL);
            if (pat == 0) { mpz_set_ui(av0, 0); mpz_set_ui(bv0, 0); }
            else if (pat == 1) { mpz_set_ui(av0, 1); mpz_set_ui(bv0, 1); }
            else if (pat == 2) { mpz_sub_ui(av0, mc->N, 1); mpz_sub_ui(bv0, mc->N, 1); }
            else { mpz_sub_ui(av0, mc->N, 1); mpz_set_ui(bv0, 1); }
            ifma_from_mpz_lane(Aa.data(), 0, av0, mc);
            ifma_from_mpz_lane(Ba.data(), 0, bv0, mc);
            if (getenv("ED_SOA_DEBUG")) {
                /* decompose soa_sub into neg + add and check each half against
                   the Montgomery-domain expectation */
                mpz_t Rm, ea, eb, exp, gotm;
                mpz_inits(Rm, ea, eb, exp, gotm, NULL);
                mpz_set_ui(Rm, 1);
                mpz_mul_2exp(Rm, Rm, 52 * (unsigned long)n);
                mpz_mod(ea, av0, mc->N); mpz_mul(ea, ea, Rm); mpz_mod(ea, ea, mc->N);
                mpz_mod(eb, bv0, mc->N); mpz_mul(eb, eb, Rm); mpz_mod(eb, eb, mc->N);
                soa_neg(Tt.data(), Ba.data(), mc);
                ifma_to_mpz_lane(gotm, Tt.data(), 0, mc);
                mpz_sub(exp, mc->N, eb); mpz_mod(exp, exp, mc->N);
                gmp_printf("  [dbg] pat=%d neg: exp=%Zx got=%Zx %s\n", pat, exp, gotm,
                           mpz_cmp(exp, gotm) ? "DIFF" : "ok");
                soa_add(Aa.data(), Aa.data(), Tt.data(), mc);
                ifma_to_mpz_lane(gotm, Aa.data(), 0, mc);
                mpz_add(exp, ea, mc->N); mpz_sub(exp, exp, eb); mpz_mod(exp, exp, mc->N);
                gmp_printf("  [dbg] pat=%d add: exp=%Zx got=%Zx %s\n", pat, exp, gotm,
                           mpz_cmp(exp, gotm) ? "DIFF" : "ok");
                mpz_clears(Rm, ea, eb, exp, gotm, NULL);
            }
            soa_sub(Aa.data(), Aa.data(), Ba.data(), Tt.data(), mc);
            soa_sub(Aa.data(), Aa.data(), Ba.data(), Tt.data(), mc);   /* chained */
            mpz_sub(want0, av0, bv0); mpz_sub(want0, want0, bv0); mpz_mod(want0, want0, mc->N);
            ifma_to_mpz_lane(got, Aa.data(), 0, mc);
            if (mpz_cmp(want0, got) != 0) {
                if (fails < 16) gmp_printf("  [selftest] CHAINED sub pat=%d want=%Zx got=%Zx\n", pat, want0, got);
                fails++;
            }
            /* and add then sub back, which must return the original */
            soa_sub(Aa.data(), Aa.data(), Ba.data(), Tt.data(), mc);
            mpz_sub(want0, av0, bv0); mpz_sub(want0, want0, bv0); mpz_mod(want0, want0, mc->N);
            soa_add(Aa.data(), Aa.data(), Ba.data(), mc);
            ifma_to_mpz_lane(got, Aa.data(), 0, mc);
            if (mpz_cmp(want0, got) != 0) {
                if (fails < 18) gmp_printf("  [selftest] ADD-BACK pat=%d want=%Zx got=%Zx\n", pat, want0, got);
                fails++;
            }
            mpz_clears(av0, bv0, want0, NULL);
        }
    }
    mpz_clears(av, bv, want, got, tmp, NULL);
    gmp_randclear(rs);
    return fails;
}

/* ---------------------------------------------------------------------------
 * Point-op self test: each op separately against the same formulas evaluated in
 * mpz.  Field helpers and the kernel are already verified, so a mismatch here
 * means the op wired its slots or its result buffers wrongly.
 * ------------------------------------------------------------------------- */
typedef struct { mpz_t X, Y, Z, T; } mpz_pt;

static void mpz_pt_init(mpz_pt *p) { mpz_inits(p->X, p->Y, p->Z, p->T, NULL); }
static void mpz_pt_clear(mpz_pt *p) { mpz_clears(p->X, p->Y, p->Z, p->T, NULL); }
static void mpz_pt_set(mpz_pt *d, const mpz_pt *s)
{
    mpz_set(d->X, s->X); mpz_set(d->Y, s->Y); mpz_set(d->Z, s->Z); mpz_set(d->T, s->T);
}

/* r = 2p (mirrors ed_dbl_mont) */
static void mpz_ref_dbl(mpz_pt *r, const mpz_pt *p, const mpz_t N)
{
    mpz_t A, B, C, E, F, G, H, t;
    mpz_inits(A, B, C, E, F, G, H, t, NULL);
    mpz_mul(A, p->X, p->X); mpz_mod(A, A, N);
    mpz_mul(B, p->Y, p->Y); mpz_mod(B, B, N);
    mpz_mul(C, p->Z, p->Z); mpz_mod(C, C, N);
    mpz_add(C, C, C); mpz_mod(C, C, N);
    mpz_add(t, p->X, p->Y); mpz_mod(t, t, N);
    mpz_mul(E, t, t); mpz_mod(E, E, N);
    mpz_sub(E, E, A); mpz_mod(E, E, N);
    mpz_sub(E, E, B); mpz_mod(E, E, N);
    mpz_add(G, A, B); mpz_mod(G, G, N);
    mpz_sub(F, G, C); mpz_mod(F, F, N);
    mpz_sub(H, A, B); mpz_mod(H, H, N);
    mpz_mul(r->X, E, F); mpz_mod(r->X, r->X, N);
    mpz_mul(r->Y, G, H); mpz_mod(r->Y, r->Y, N);
    mpz_mul(r->T, E, H); mpz_mod(r->T, r->T, N);
    mpz_mul(r->Z, F, G); mpz_mod(r->Z, r->Z, N);
    mpz_clears(A, B, C, E, F, G, H, t, NULL);
}

/* r = p + q (mirrors ed_add_mont, q non-affine) */
static void mpz_ref_add(mpz_pt *r, const mpz_pt *p, const mpz_pt *q, const mpz_t d, const mpz_t N)
{
    mpz_t A, B, C, D, E, F, G, H, t;
    mpz_inits(A, B, C, D, E, F, G, H, t, NULL);
    mpz_mul(A, p->X, q->X); mpz_mod(A, A, N);
    mpz_mul(B, p->Y, q->Y); mpz_mod(B, B, N);
    mpz_mul(C, p->T, q->T); mpz_mod(C, C, N);
    mpz_mul(C, C, d); mpz_mod(C, C, N);
    mpz_mul(D, p->Z, q->Z); mpz_mod(D, D, N);
    mpz_add(t, p->X, p->Y); mpz_mod(t, t, N);
    mpz_add(E, q->X, q->Y); mpz_mod(E, E, N);
    mpz_mul(E, t, E); mpz_mod(E, E, N);
    mpz_sub(E, E, A); mpz_mod(E, E, N);
    mpz_sub(E, E, B); mpz_mod(E, E, N);
    mpz_sub(F, D, C); mpz_mod(F, F, N);
    mpz_add(G, D, C); mpz_mod(G, G, N);
    mpz_sub(H, B, A); mpz_mod(H, H, N);
    mpz_mul(r->X, E, F); mpz_mod(r->X, r->X, N);
    mpz_mul(r->Y, G, H); mpz_mod(r->Y, r->Y, N);
    mpz_mul(r->T, E, H); mpz_mod(r->T, r->T, N);
    mpz_mul(r->Z, F, G); mpz_mod(r->Z, r->Z, N);
    mpz_clears(A, B, C, D, E, F, G, H, t, NULL);
}

/* r = p + (x2, y2, dxy2) affine (mirrors ed_add_affine_mont) */
static void mpz_ref_add_affine(mpz_pt *r, const mpz_pt *p,
                               const mpz_t x2, const mpz_t y2, const mpz_t dxy2, const mpz_t N)
{
    mpz_t A, B, C, E, F, G, H, t;
    mpz_inits(A, B, C, E, F, G, H, t, NULL);
    mpz_mul(A, p->X, x2); mpz_mod(A, A, N);
    mpz_mul(B, p->Y, y2); mpz_mod(B, B, N);
    mpz_mul(C, p->T, dxy2); mpz_mod(C, C, N);
    mpz_add(t, p->X, p->Y); mpz_mod(t, t, N);
    mpz_add(E, x2, y2); mpz_mod(E, E, N);
    mpz_mul(E, t, E); mpz_mod(E, E, N);
    mpz_sub(E, E, A); mpz_mod(E, E, N);
    mpz_sub(E, E, B); mpz_mod(E, E, N);
    mpz_sub(F, p->Z, C); mpz_mod(F, F, N);
    mpz_add(G, p->Z, C); mpz_mod(G, G, N);
    mpz_sub(H, B, A); mpz_mod(H, H, N);
    mpz_mul(r->X, E, F); mpz_mod(r->X, r->X, N);
    mpz_mul(r->Y, G, H); mpz_mod(r->Y, r->Y, N);
    mpz_mul(r->T, E, H); mpz_mod(r->T, r->T, N);
    mpz_mul(r->Z, F, G); mpz_mod(r->Z, r->Z, N);
    mpz_clears(A, B, C, E, F, G, H, t, NULL);
}

int ed_soa_point_selftest(ed_soa_ctx_t *c, int trials)
{
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n, lw = 8 * n;
    std::vector<uint64_t> px(8 * n), py(8 * n), pz(8 * n), pt(8 * n);
    std::vector<uint64_t> qx(8 * n), qy(8 * n), qt(8 * n);
    std::vector<uint64_t> ox(8 * n), oy(8 * n), oz(8 * n), ot(8 * n);
    mpz_t v, d_mpz;
    mpz_inits(v, d_mpz, NULL);
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 24680u);
    int fails = 0;

    for (int t = 0; t < trials; t++) {
        /* lane 0: random point in Montgomery form + a random affine point */
        mpz_pt P, Q, Rref, Rref2, Gref, Aref;
        mpz_pt_init(&P); mpz_pt_init(&Q); mpz_pt_init(&Rref);
        mpz_pt_init(&Rref2); mpz_pt_init(&Gref); mpz_pt_init(&Aref);

        mpz_urandomm(P.X, rs, mc->N); mpz_urandomm(P.Y, rs, mc->N);
        mpz_urandomm(P.Z, rs, mc->N); mpz_urandomm(P.T, rs, mc->N);
        ifma_from_mpz_lane(px.data(), 0, P.X, mc);
        ifma_from_mpz_lane(py.data(), 0, P.Y, mc);
        ifma_from_mpz_lane(pz.data(), 0, P.Z, mc);
        ifma_from_mpz_lane(pt.data(), 0, P.T, mc);

        /* dictionaries' affine entry: x2, y2, dxy = d*x2*y2 */
        mpz_t x2, y2, dxy;
        mpz_inits(x2, y2, dxy, NULL);
        mpz_urandomm(x2, rs, mc->N); mpz_urandomm(y2, rs, mc->N);
        ifma_from_mpz_lane(qx.data(), 0, x2, mc);
        ifma_from_mpz_lane(qy.data(), 0, y2, mc);
        ifma_to_mpz_lane(v, c->d, 0, mc);
        mpz_set(d_mpz, v);
        mpz_mul(dxy, x2, y2); mpz_mod(dxy, dxy, mc->N);
        mpz_mul(dxy, dxy, d_mpz); mpz_mod(dxy, dxy, mc->N);
        ifma_from_mpz_lane(qt.data(), 0, dxy, mc);

        /* --- dbl --- */
        soa_pt Ppt = { px.data(), py.data(), pz.data(), pt.data() };
        soa_pt Opt = { ox.data(), oy.data(), oz.data(), ot.data() };
        ed_soa_dbl(c, Opt, Ppt);
        mpz_ref_dbl(&Rref, &P, mc->N);
        struct { const char *name; const std::vector<uint64_t> *e; mpz_srcptr ref; } chk[4] = {
            { "dbl.X", &ox, Rref.X }, { "dbl.Y", &oy, Rref.Y },
            { "dbl.Z", &oz, Rref.Z }, { "dbl.T", &ot, Rref.T },
        };
        for (int i = 0; i < 4; i++) {
            ifma_to_mpz_lane(v, chk[i].e->data(), 0, mc);
            if (mpz_cmp(v, chk[i].ref) != 0) {
                if (fails < 6) gmp_printf("  [ptsel] %s want=%Zx got=%Zx\n", chk[i].name, chk[i].ref, v);
                fails++;
            }
        }

        /* --- add_affine --- */
        memset(ox.data(), 0, 8 * n * sizeof(uint64_t));
        ed_soa_add_affine(c, Opt, Ppt, qx.data(), qy.data(), qt.data());
        mpz_ref_add_affine(&Rref2, &P, x2, y2, dxy, mc->N);
        chk[0] = { "addaf.X", &ox, Rref2.X }; chk[1] = { "addaf.Y", &oy, Rref2.Y };
        chk[2] = { "addaf.Z", &oz, Rref2.Z }; chk[3] = { "addaf.T", &ot, Rref2.T };
        for (int i = 0; i < 4; i++) {
            ifma_to_mpz_lane(v, chk[i].e->data(), 0, mc);
            if (mpz_cmp(v, chk[i].ref) != 0) {
                if (fails < 6) gmp_printf("  [ptsel] %s want=%Zx got=%Zx\n", chk[i].name, chk[i].ref, v);
                fails++;
            }
        }

        /* --- add (non-affine) --- */
        mpz_urandomm(Q.X, rs, mc->N); mpz_urandomm(Q.Y, rs, mc->N);
        mpz_urandomm(Q.Z, rs, mc->N); mpz_urandomm(Q.T, rs, mc->N);
        ifma_from_mpz_lane(qx.data(), 0, Q.X, mc);
        ifma_from_mpz_lane(qy.data(), 0, Q.Y, mc);
        ifma_from_mpz_lane(oz.data(), 0, Q.Z, mc);
        ifma_from_mpz_lane(ot.data(), 0, Q.T, mc);
        soa_pt Qpt = { qx.data(), qy.data(), oz.data(), ot.data() };
        std::vector<uint64_t> tx(8 * n), ty(8 * n), tz(8 * n), tt(8 * n);
        soa_pt Tpt = { tx.data(), ty.data(), tz.data(), tt.data() };
        ed_soa_add(c, Tpt, Ppt, Qpt);
        mpz_ref_add(&Gref, &P, &Q, d_mpz, mc->N);
        chk[0] = { "add.X", &tx, Gref.X }; chk[1] = { "add.Y", &ty, Gref.Y };
        chk[2] = { "add.Z", &tz, Gref.Z }; chk[3] = { "add.T", &tt, Gref.T };
        for (int i = 0; i < 4; i++) {
            ifma_to_mpz_lane(v, chk[i].e->data(), 0, mc);
            if (mpz_cmp(v, chk[i].ref) != 0) {
                if (fails < 6) gmp_printf("  [ptsel] %s want=%Zx got=%Zx\n", chk[i].name, chk[i].ref, v);
                fails++;
            }
        }

        /* --- step-by-step manual replay of dbl, explicit buffers, vs mpz ---
           Bisects ed_soa_dbl: if these steps are all correct the op's own slot
           wiring is at fault; if a step fails, the primitive is. */
        {
            std::vector<std::vector<uint64_t> > sA(9, std::vector<uint64_t>(8 * n));
            uint64_t *mA = sA[0].data(), *mB = sA[1].data(), *mC = sA[2].data();
            uint64_t *mE = sA[3].data(), *mF = sA[4].data(), *mG = sA[5].data();
            uint64_t *mH = sA[6].data(), *mt = sA[7].data(), *mo = sA[8].data();
            mpz_t wA, wB, wC, wE, wF, wG, wH;
            mpz_inits(wA, wB, wC, wE, wF, wG, wH, NULL);
            const mpz_t *NX = (const mpz_t *)mc->N;

            ifma_mont_sqr(mA, Ppt.x, mc);
            ifma_to_mpz_lane(v, mA, 0, mc);
            mpz_mul(wA, P.X, P.X); mpz_mod(wA, wA, (mpz_ptr)NX);
            if (mpz_cmp(v, wA) != 0) { if (fails < 8) gmp_printf("  [step] A want=%Zx got=%Zx\n", wA, v); fails++; }

            ifma_mont_sqr(mB, Ppt.y, mc);
            mpz_mul(wB, P.Y, P.Y); mpz_mod(wB, wB, (mpz_ptr)NX);
            ifma_to_mpz_lane(v, mB, 0, mc);
            if (mpz_cmp(v, wB) != 0) { if (fails < 8) gmp_printf("  [step] B want=%Zx got=%Zx\n", wB, v); fails++; }

            ifma_mont_sqr(mC, Ppt.z, mc);
            soa_add(mC, mC, mC, mc);
            mpz_mul(wC, P.Z, P.Z); mpz_mod(wC, wC, (mpz_ptr)NX);
            mpz_add(wC, wC, wC); mpz_mod(wC, wC, (mpz_ptr)NX);
            ifma_to_mpz_lane(v, mC, 0, mc);
            if (mpz_cmp(v, wC) != 0) { if (fails < 8) gmp_printf("  [step] C want=%Zx got=%Zx\n", wC, v); fails++; }

            soa_add(mt, Ppt.x, Ppt.y, mc);
            {
                mpz_t w1;
                mpz_init(w1);
                mpz_add(w1, P.X, P.Y); mpz_mod(w1, w1, (mpz_ptr)NX);
                ifma_to_mpz_lane(v, mt, 0, mc);
                if (mpz_cmp(v, w1) != 0) { if (fails < 12) gmp_printf("  [step] t=x+y want=%Zx got=%Zx\n", w1, v); fails++; }
                mpz_clear(w1);
            }
            ifma_mont_sqr(mE, mt, mc);
            {
                mpz_t w2;
                mpz_init(w2);
                mpz_add(w2, P.X, P.Y); mpz_mod(w2, w2, (mpz_ptr)NX);
                mpz_mul(w2, w2, w2); mpz_mod(w2, w2, (mpz_ptr)NX);
                ifma_to_mpz_lane(v, mE, 0, mc);
                if (mpz_cmp(v, w2) != 0) { if (fails < 12) gmp_printf("  [step] t^2 want=%Zx got=%Zx\n", w2, v); fails++; }
                mpz_clear(w2);
            }
            mpz_add(wE, P.X, P.Y); mpz_mod(wE, wE, (mpz_ptr)NX);
            mpz_mul(wE, wE, wE); mpz_mod(wE, wE, (mpz_ptr)NX);
            mpz_sub(wE, wE, wA); mpz_mod(wE, wE, (mpz_ptr)NX);
            mpz_sub(wE, wE, wB); mpz_mod(wE, wE, (mpz_ptr)NX);
            soa_sub(mE, mE, mA, mo, mc);
            soa_sub(mE, mE, mB, mo, mc);
            ifma_to_mpz_lane(v, mE, 0, mc);
            if (mpz_cmp(v, wE) != 0) {
                if (fails < 12) {
                    /* Is E consistent with the SIMD's OWN A and B?
                       yes -> the subtraction is fine and the mismatch is in the
                       inputs/reference; no -> soa_sub misbehaves here. */
                    mpz_t vA, vB, self, t2;
                    mpz_inits(vA, vB, self, t2, NULL);
                    ifma_to_mpz_lane(vA, mA, 0, mc);
                    ifma_to_mpz_lane(vB, mB, 0, mc);
                    /* t^2 as the SIMD holds it (mt is still x+y there) */
                    mpz_add(t2, P.X, P.Y); mpz_mod(t2, t2, (mpz_ptr)NX);
                    mpz_mul(t2, t2, t2); mpz_mod(t2, t2, (mpz_ptr)NX);
                    mpz_sub(self, t2, vA); mpz_mod(self, self, (mpz_ptr)NX);
                    mpz_sub(self, self, vB); mpz_mod(self, self, (mpz_ptr)NX);
                    gmp_printf("  [step] E self-consistent? %d   A==refA? %d   B==refB? %d\n",
                               mpz_cmp(v, self) == 0, mpz_cmp(vA, wA) == 0, mpz_cmp(vB, wB) == 0);
                    mpz_clears(vA, vB, self, t2, NULL);
                }
                fails++;
            }

            soa_add(mG, mA, mB, mc);
            mpz_add(wG, wA, wB); mpz_mod(wG, wG, (mpz_ptr)NX);
            ifma_to_mpz_lane(v, mG, 0, mc);
            if (mpz_cmp(v, wG) != 0) { if (fails < 8) gmp_printf("  [step] G want=%Zx got=%Zx\n", wG, v); fails++; }

            soa_sub(mF, mG, mC, mo, mc);
            mpz_sub(wF, wG, wC); mpz_mod(wF, wF, (mpz_ptr)NX);
            ifma_to_mpz_lane(v, mF, 0, mc);
            if (mpz_cmp(v, wF) != 0) { if (fails < 8) gmp_printf("  [step] F want=%Zx got=%Zx\n", wF, v); fails++; }

            soa_sub(mH, mA, mB, mo, mc);
            mpz_sub(wH, wA, wB); mpz_mod(wH, wH, (mpz_ptr)NX);
            ifma_to_mpz_lane(v, mH, 0, mc);
            if (mpz_cmp(v, wH) != 0) { if (fails < 8) gmp_printf("  [step] H want=%Zx got=%Zx\n", wH, v); fails++; }

            /* final products of the manual replay, compared with ed_soa_dbl */
            ifma_mont_mul(mH, mE, mF, mc);
            ifma_to_mpz_lane(v, mH, 0, mc);
            if (mpz_cmp(v, Rref.X) != 0) { if (fails < 8) gmp_printf("  [step] manual X want=%Zx got=%Zx\n", Rref.X, v); fails++; }
            mpz_clears(wA, wB, wC, wE, wF, wG, wH, NULL);
        }

        mpz_clears(x2, y2, dxy, NULL);
        mpz_pt_clear(&P); mpz_pt_clear(&Q); mpz_pt_clear(&Rref);
        mpz_pt_clear(&Rref2); mpz_pt_clear(&Gref); mpz_pt_clear(&Aref);
        (void)lw;
    }
    mpz_clears(v, d_mpz, NULL);
    gmp_randclear(rs);
    return fails;
}

/* ---------------------------------------------------------------------------
 * Context
 * ------------------------------------------------------------------------- */

size_t ed_soa_dict_words(const ed_soa_ctx_t *c) { return c->set ? 3 * c->m * 8 * c->n : 0; }
size_t ed_soa_dict_bytes(const ed_soa_ctx_t *c) { return ed_soa_dict_words(c) * sizeof(uint64_t); }

void ed_soa_set_progress(ed_soa_ctx_t *c, ed_soa_progress_fn fn, void *ctx)
{
    c->progress = fn;
    c->progress_ctx = ctx;
}

int ed_soa_init(ed_soa_ctx_t *c, const mpz_t N, int w)
{
    memset(c, 0, sizeof(*c));
    if (w < 3 || w > 12) return -1;
    if (ifma_ctx_init(&c->mc, N) != 0) return -2;
    c->n = c->mc.n;
    c->w = w;
    c->m = (size_t)1 << (w - 2);

    const size_t lane_words = 8 * c->n;
    c->dict = (uint64_t *)_mm_malloc(3 * c->m * lane_words * sizeof(uint64_t), 64);
    c->arena = (uint64_t *)_mm_malloc((size_t)ED_SOA_ARENA * lane_words * sizeof(uint64_t), 64);
    c->d = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    if (!c->dict || !c->arena || !c->d) { ed_soa_clear(c); return -3; }
    memset(c->arena, 0, (size_t)ED_SOA_ARENA * lane_words * sizeof(uint64_t));
    return 0;
}

void ed_soa_clear(ed_soa_ctx_t *c)
{
    if (c->dict) _mm_free(c->dict);
    if (c->arena) _mm_free(c->arena);
    if (c->d) _mm_free(c->d);
    ifma_ctx_clear(&c->mc);
    memset(c, 0, sizeof(*c));
}

/* per-lane batch inversion of m values, mirroring mpz_batch_invert.
   mpz_t is an array type: MSVC rejects std::vector<mpz_t>(n) (C3074), so the
   scratch here is a malloc'd mpz_t array with explicit init/clear. */
static void soa_batch_invert(mpz_t *iz, mpz_t *z, size_t m, const mpz_t N)
{
    if (m == 0) return;
    mpz_t *prefix = (mpz_t *)malloc(m * sizeof(mpz_t));
    if (!prefix) return;
    for (size_t i = 0; i < m; i++) mpz_init(prefix[i]);
    mpz_set(prefix[0], z[0]);
    for (size_t i = 1; i < m; i++) {
        mpz_mul(prefix[i], prefix[i - 1], z[i]);
        mpz_mod(prefix[i], prefix[i], N);
    }
    mpz_t inv;
    mpz_init(inv);
    /* 前缀积做法要求所有 z 都可逆。合数 N 下某个 z 可能与 N 共享因子(甚至为 0),
       此时 mpz_invert 失败 -> 整条 lane 的字典报废。以前没查返回值, 所以 m 一大
       (w 大)就静默出错: 这正是 w=8 通过、w=12 结果不一致的根因。 */
    const int inv_ok = (mpz_invert(inv, prefix[m - 1], N) != 0);
    if (!inv_ok && getenv("ED_SOA_DEBUG")) {
        size_t bad = 0;
        mpz_t g;
        mpz_init(g);
        for (size_t i = 0; i < m; i++) {
            mpz_gcd(g, z[i], N);
            if (mpz_cmp_ui(g, 1) != 0) bad++;
        }
        fprintf(stderr, "[soa] batch_invert FAILED: m=%zu, %zu value(s) not invertible mod N\n",
                m, bad);
        mpz_clear(g);
    }
    for (size_t i = m; i-- > 0;) {
        if (i > 0) {
            mpz_mul(iz[i], inv, prefix[i - 1]); mpz_mod(iz[i], iz[i], N);
            mpz_mul(inv, inv, z[i]); mpz_mod(inv, inv, N);
        } else {
            mpz_set(iz[i], inv);
        }
    }
    mpz_clear(inv);
    for (size_t i = 0; i < m; i++) mpz_clear(prefix[i]);
    free(prefix);
}

int ed_soa_set_curves(ed_soa_ctx_t *c, const uint64_t *sigma, int lanes)
{
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n;
    const size_t lane_words = 8 * n;
    if (!c->dict) return -1;
    if (lanes > IFMA_LANES) return -1;
    c->set = 0;

    /* --- per lane: Atkin-Morain -> d and P in Montgomery form --- */
    std::vector<uint64_t> Px(8 * n, 0), Py(8 * n, 0), Pt(8 * n, 0), Pz(8 * n, 0);
    mpz_t d, px, py, t;
    mpz_inits(d, px, py, t, NULL);
    for (int k = 0; k < lanes; k++) {
        edwards_atkin_morain(d, px, py, sigma[k], mc->N);
        ifma_from_mpz_lane(c->d, (unsigned)k, d, mc);
        ifma_from_mpz_lane(Px.data(), (unsigned)k, px, mc);
        ifma_from_mpz_lane(Py.data(), (unsigned)k, py, mc);
        /* z = 1 (Montgomery), t = x*y */
        for (size_t i = 0; i < n; i++) Pz[8 * i + k] = mc->one[8 * i + k];
    }
    ifma_mont_mul(Pt.data(), Px.data(), Py.data(), mc);

    /* --- dictionary: dict[0] = P, then cur += 2P repeatedly --- */
    /* entry j occupies 3*lane_words at offset j*3*lane_words */
    soa_pt P;
    P.x = Px.data(); P.y = Py.data(); P.z = Pz.data(); P.t = Pt.data();

    uint64_t *e0 = c->dict + 0 * 3 * lane_words;   /* x of entry 0 */
    uint64_t *e1 = e0 + 1 * lane_words;            /* y of entry 0 */
    uint64_t *e2 = e0 + 2 * lane_words;            /* z of entry 0 */
    soa_copy(e0, P.x, mc); soa_copy(e1, P.y, mc); soa_copy(e2, P.z, mc);

    /* running accumulator in arena slots the ops never touch (they use 0..9) */
    uint64_t *cx = arena_slot(c, 10), *cy = arena_slot(c, 11);
    uint64_t *cz = arena_slot(c, 12), *ct = arena_slot(c, 13);
    /* 2P must NOT live in the arena: the ops it is passed to use the arena. */
    uint64_t *bx = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *by = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *bz = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *bt = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    if (!bx || !by || !bz || !bt) {
        if (bx) _mm_free(bx); if (by) _mm_free(by); if (bz) _mm_free(bz); if (bt) _mm_free(bt);
        mpz_clears(d, px, py, t, NULL);
        return -3;
    }

    soa_copy(cx, P.x, mc); soa_copy(cy, P.y, mc); soa_copy(cz, P.z, mc); soa_copy(ct, P.t, mc);
    soa_pt cc = { cx, cy, cz, ct };
    soa_pt bb = { bx, by, bz, bt };
    ed_soa_dbl(c, bb, P);                    /* 2P */

    for (size_t j = 1; j < c->m; j++) {
        ed_soa_add(c, cc, cc, bb);           /* cur += 2P */
        uint64_t *base = c->dict + j * 3 * lane_words;
        soa_copy(base + 0 * lane_words, cx, mc);
        soa_copy(base + 1 * lane_words, cy, mc);
        soa_copy(base + 2 * lane_words, cz, mc);
    }

    /* --- normalise to affine: Z -> 1, one mpz batch inversion per lane --- */
    mpz_t *zval = (mpz_t *)malloc((size_t)lanes * c->m * sizeof(mpz_t));
    mpz_t *zinv = (mpz_t *)malloc((size_t)lanes * c->m * sizeof(mpz_t));
    if (!zval || !zinv) {
        free(zval); free(zinv);
        _mm_free(bx); _mm_free(by); _mm_free(bz); _mm_free(bt);
        mpz_clears(d, px, py, t, NULL);
        return -3;
    }
    for (int k = 0; k < lanes; k++)
        for (size_t j = 0; j < c->m; j++) { mpz_init(zval[k * c->m + j]); mpz_init(zinv[k * c->m + j]); }

    for (size_t j = 0; j < c->m; j++) {
        uint64_t *base = c->dict + j * 3 * lane_words;
        const uint64_t *zv = base + 2 * lane_words;
        for (int k = 0; k < lanes; k++) ifma_to_mpz_lane(zval[k * c->m + j], zv, (unsigned)k, mc);
    }
    for (int k = 0; k < lanes; k++)
        soa_batch_invert(zinv + k * c->m, zval + k * c->m, c->m, mc->N);

    /* 诊断: 逐 lane 统计 Z 与 N 不互素的字典项 (这些项会让该 lane 的字典报废) */
    c->bad_inv = 0;
    if (getenv("ED_SOA_DEBUG")) {
        mpz_t g;
        mpz_init(g);
        for (int k = 0; k < lanes; k++) {
            size_t bad = 0;
            for (size_t j = 0; j < c->m; j++) {
                mpz_gcd(g, zval[k * c->m + j], mc->N);
                if (mpz_cmp_ui(g, 1) != 0) bad++;
            }
            if (bad) {
                fprintf(stderr, "[soa] lane %d: %zu/%zu dictionary Z values share a factor with N\n",
                        k, bad, c->m);
                c->bad_inv += (int)bad;
            }
        }
        mpz_clear(g);
    }

    std::vector<uint64_t> iz(8 * n, 0), xy(8 * n, 0);
    for (size_t j = 0; j < c->m; j++) {
        uint64_t *base = c->dict + j * 3 * lane_words;
        for (int k = 0; k < lanes; k++) ifma_from_mpz_lane(iz.data(), (unsigned)k, zinv[k * c->m + j], mc);
        ifma_mont_mul(base + 0 * lane_words, base + 0 * lane_words, iz.data(), mc);
        ifma_mont_mul(base + 1 * lane_words, base + 1 * lane_words, iz.data(), mc);
        ifma_mont_mul(xy.data(), base + 0 * lane_words, base + 1 * lane_words, mc);
        ifma_mont_mul(base + 2 * lane_words, xy.data(), c->d, mc);   /* dxy = d*x*y */
    }
    for (int k = 0; k < lanes; k++)
        for (size_t j = 0; j < c->m; j++) { mpz_clear(zval[k * c->m + j]); mpz_clear(zinv[k * c->m + j]); }
    free(zval); free(zinv);

    _mm_free(bx); _mm_free(by); _mm_free(bz); _mm_free(bt);
    mpz_clears(d, px, py, t, NULL);
    if (getenv("ED_SOA_DEBUG")) {
        for (size_t j = 0; j < 3; j++) {
            const uint64_t *base = c->dict + j * 3 * lane_words;
            fprintf(stderr, "[dict] j=%zu x=%llx y=%llx dxy=%llx (lane0 limb0)\n", j,
                    (unsigned long long)(base[0] & IFMA_M52),
                    (unsigned long long)(base[lane_words] & IFMA_M52),
                    (unsigned long long)(base[2 * lane_words] & IFMA_M52));
        }
    }
    c->set = 1;
    return 0;
}

int ed_soa_stage1(ed_soa_ctx_t *c, const mpz_t s, int lanes,
                  mpz_t *Qx, mpz_t *Qz, mpz_t *factor)
{
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n;
    const size_t lane_words = 8 * n;
    if (!c->set) return -1;
    if (lanes > IFMA_LANES) return -1;

    /* R = (0, 1, 1, 0) in Montgomery form */
    uint64_t *rx = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *ry = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *rz = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *rt = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *nx = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *ny = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    uint64_t *nd = (uint64_t *)_mm_malloc(lane_words * sizeof(uint64_t), 64);
    if (!rx || !ry || !rz || !rt || !nx || !ny || !nd) {
        if (rx) _mm_free(rx); if (ry) _mm_free(ry); if (rz) _mm_free(rz); if (rt) _mm_free(rt);
        if (nx) _mm_free(nx); if (ny) _mm_free(ny); if (nd) _mm_free(nd);
        return -3;
    }
    soa_pt R = { rx, ry, rz, rt };
    soa_zero(R.x, mc);
    soa_copy(R.y, mc->one, mc);
    soa_copy(R.z, mc->one, mc);
    soa_zero(R.t, mc);

    std::vector<int> digits;
    soa_naf_digits(s, c->w, digits);
    const size_t total = digits.size();

    const int dbg = getenv("ED_SOA_DEBUG") != NULL;
    if (dbg) {
        size_t nz = 0;
        for (size_t i = 0; i < total; i++) if (digits[i] != 0) nz++;
        fprintf(stderr, "[soa] w=%d m=%zu total=%zu nonzero=%zu | R.z limb0 lane0=%llx R.y limb0 lane0=%llx\n",
                c->w, c->m, total, nz,
                (unsigned long long)(R.z[0] & IFMA_M52), (unsigned long long)(R.y[0] & IFMA_M52));
    }

    for (size_t i = 0; i < total; i++) {
        /* 批内进度: 每 ED_SOA_PROGRESS_BITS 个 digit 回调一次, 让上层进度条有更新、
           也让 SIGINT 在批内就有响应 (返回非 0 = 中止, 此时不写任何输出)。 */
        if (c->progress && total > ED_SOA_PROGRESS_BITS &&
            (i % ED_SOA_PROGRESS_BITS) == 0) {
            if (c->progress(c->progress_ctx, i, total) != 0) {
                _mm_free(rx); _mm_free(ry); _mm_free(rz); _mm_free(rt);
                _mm_free(nx); _mm_free(ny); _mm_free(nd);
                return 1;
            }
        }
        ed_soa_dbl(c, R, R);
        const int dgt = digits[total - 1 - i];
        if (dgt != 0) {
            const size_t idx = (size_t)(dgt > 0 ? dgt : -dgt) - 1;
            const size_t j = idx / 2;
            const uint64_t *base = c->dict + j * 3 * lane_words;
            if (dbg && i < 5)
                fprintf(stderr, "[soa] step %zu dgt=%d j=%zu dict.x=%llx dict.z... (affine)\n",
                        i, dgt, j, (unsigned long long)(base[0] & IFMA_M52));
            if (dgt > 0) {
                ed_soa_add_affine(c, R, R, base + 0 * lane_words, base + 1 * lane_words,
                                  base + 2 * lane_words);
            } else {
                /* (-x, y, -dxy) */
                soa_neg(nx, base + 0 * lane_words, mc);
                soa_copy(ny, base + 1 * lane_words, mc);
                soa_neg(nd, base + 2 * lane_words, mc);
                ed_soa_add_affine(c, R, R, nx, ny, nd);
            }
        }
        if (dbg && (i < 5 || (i % 500) == 0))
            fprintf(stderr, "[soa] after step %zu: R.z limb0 lane0=%llx R.y limb0 lane0=%llx\n",
                    i, (unsigned long long)(R.z[0] & IFMA_M52), (unsigned long long)(R.y[0] & IFMA_M52));
        if (dbg && total <= 8) {
            mpz_t vx, vy, vz, vt;
            mpz_inits(vx, vy, vz, vt, NULL);
            ifma_to_mpz_lane(vx, R.x, 0, mc);
            ifma_to_mpz_lane(vy, R.y, 0, mc);
            ifma_to_mpz_lane(vz, R.z, 0, mc);
            ifma_to_mpz_lane(vt, R.t, 0, mc);
            gmp_fprintf(stderr, "[pt] after step %zu: X=%Zd\n[pt] after step %zu: Y=%Zd\n"
                                "[pt] after step %zu: Z=%Zd\n[pt] after step %zu: T=%Zd\n",
                        i, vx, i, vy, i, vz, i, vt);
            mpz_clears(vx, vy, vz, vt, NULL);
        }
    }

    /* Qx = z + y, Qz = z - y (ordinary), factor = gcd(Qz, N) */
    mpz_t zo, yo, tmp, qz;
    mpz_inits(zo, yo, tmp, qz, NULL);
    for (int k = 0; k < lanes; k++) {
        ifma_to_mpz_lane(zo, R.z, (unsigned)k, mc);
        ifma_to_mpz_lane(yo, R.y, (unsigned)k, mc);
        mpz_add(tmp, zo, yo); mpz_mod(tmp, tmp, mc->N);
        if (Qx) mpz_set(Qx[k], tmp);
        mpz_sub(tmp, zo, yo); mpz_mod(qz, tmp, mc->N);
        if (Qz) mpz_set(Qz[k], qz);
        if (factor) {
            mpz_gcd(tmp, qz, mc->N);
            if (mpz_cmp_ui(tmp, 1) > 0 && mpz_cmp(tmp, mc->N) < 0) mpz_set(factor[k], tmp);
            else mpz_set_ui(factor[k], 1);
        }
    }
    mpz_clears(zo, yo, tmp, qz, NULL);
    _mm_free(rx); _mm_free(ry); _mm_free(rz); _mm_free(rt);
    _mm_free(nx); _mm_free(ny); _mm_free(nd);
    return 0;
}
