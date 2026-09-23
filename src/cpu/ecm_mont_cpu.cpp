/* ---------------------------------------------------------------------------
 * ecm_mont_cpu.cpp -- scalar (MPN) Suyama-sigma Montgomery ECM stage 1.
 *
 * See ecm_mont_cpu.h for the conventions.  Point arithmetic is the classic x-only
 * Montgomery ladder:
 *
 *   invariant (R0,R1) = (kP,(k+1)P), difference R1-R0 = P  (constant!)
 *   per bit:  bit=1 -> R0 += R1 (diff P), R1 = 2*R1
 *             bit=0 -> R1 += R0 (diff P), R0 = 2*R0
 *
 * with xDBL / xADD at 3M+2S each, i.e. 6M+4S per bit.  Everything runs in the
 * Montgomery domain (mont_t), so additions are plain modular add/sub and the
 * multiplications go through mont_mul/mont_sqr (MPN + GMP REDC).
 *
 * The single most delicate detail (documented because getting it wrong silently
 * produced ~200 wrong verdicts during design): xADD's third argument must be the
 * *affine* x of the difference point, Xdiff/Zdiff -- never the raw projective X of
 * a point whose Z != 1.
 *
 * The ladder keeps three point buffers and swaps pointers instead of copying
 * (a mont_t copy is ~1.3 KB; copying twice per bit would dominate the cost).
 * ------------------------------------------------------------------------- */
#include "ecm_mont_cpu.h"
#include "ecm_edwards_mont.h"          /* generic MPN Montgomery layer */

#include <stdlib.h>
#include <vector>

/* --------------------------------------------------------------------------
 * s = torsion * lcm(1..B1)
 * ------------------------------------------------------------------------ */
size_t mont_build_s(mpz_t s, uint64_t B1, uint64_t torsion)
{
    std::vector<char> composite(B1 + 1, 0);
    mpz_set_ui(s, torsion ? torsion : 1);
    for (uint64_t p = 2; p <= B1; ++p) {
        if (composite[p]) continue;
        for (uint64_t q = p * 2; q <= B1; q += p) composite[q] = 1;
        uint64_t v = p;
        while (v <= B1 / p) v *= p;               /* highest power of p <= B1 */
        mpz_mul_ui(s, s, (unsigned long)v);
    }
    return (size_t)mpz_sizeinbase(s, 2);
}

/* --------------------------------------------------------------------------
 * sigma -> (A, X0, Z0)   [plain domain, mod N]
 * ------------------------------------------------------------------------ */
void mont_suyama_curve(mpz_t A, mpz_t X0, mpz_t Z0, uint64_t sigma, const mpz_t N)
{
    mpz_t u, v, t, num, den, inv, three_u_plus_v;
    mpz_inits(u, v, t, num, den, inv, three_u_plus_v, NULL);

    mpz_set_ui(u, (unsigned long)sigma);
    mpz_mul(u, u, u);
    mpz_sub_ui(u, u, 5);                          /* u = sigma^2 - 5 */
    mpz_set_ui(v, (unsigned long)sigma);
    mpz_mul_ui(v, v, 4);                          /* v = 4*sigma */

    mpz_sub(t, v, u);                             /* v-u */
    mpz_powm_ui(num, t, 3, N);                    /* (v-u)^3 */
    mpz_mul_ui(three_u_plus_v, u, 3);
    mpz_add(three_u_plus_v, three_u_plus_v, v);   /* 3u+v */
    mpz_mul(num, num, three_u_plus_v);            /* An = (v-u)^3 (3u+v) */
    mpz_mod(num, num, N);

    mpz_powm_ui(den, u, 3, N);                    /* u^3 */
    mpz_mul_ui(den, den, 4);
    mpz_mul(den, den, v);
    mpz_mod(den, den, N);                         /* Ad = 4 u^3 v */

    if (mpz_invert(inv, den, N) == 0) {           /* gcd(den,N) > 1 => a factor */
        mpz_set_ui(inv, 0);
    }
    mpz_mul(num, num, inv);
    mpz_sub_ui(num, num, 2);
    mpz_mod(A, num, N);                           /* A = An/Ad - 2 */

    mpz_powm_ui(X0, u, 3, N);                     /* start (X:Z) = (u^3 : v^3) */
    mpz_powm_ui(Z0, v, 3, N);

    mpz_clears(u, v, t, num, den, inv, three_u_plus_v, NULL);
}

/* --------------------------------------------------------------------------
 * x-only point ops, Montgomery domain
 * ------------------------------------------------------------------------ */
namespace {

struct xz { mont_t X, Z; };

struct ladder_ctx {
    const mont_ctx_t *mc;
    mont_t a24;        /* (A+2)/4 */
    mont_t xdiff;      /* affine x of the start point (the fixed difference) */
};

/* X2 = (X+Z)^2 (X-Z)^2 ,  Z2 = 4XZ * ((X-Z)^2 + a24*4XZ)          [3M+2S] */
static void xdbl(xz &r, const xz &p, const ladder_ctx &c)
{
    mont_t A_, B_, E_, t;
    mont_add(&A_, &p.X, &p.Z, c.mc);
    mont_sqr(&A_, &A_, c.mc);                     /* A = (X+Z)^2 */
    mont_sub(&B_, &p.X, &p.Z, c.mc);
    mont_sqr(&B_, &B_, c.mc);                     /* B = (X-Z)^2 */
    mont_sub(&E_, &A_, &B_, c.mc);                /* E = 4XZ */
    mont_mul(&r.X, &A_, &B_, c.mc);
    mont_mul(&t, &c.a24, &E_, c.mc);
    mont_add(&t, &t, &B_, c.mc);                  /* B + a24*E   (== A + ((A-2)/4)E) */
    mont_mul(&r.Z, &E_, &t, c.mc);
}

/* X3 = (t4+t5)^2 ,  Z3 = (t4-t5)^2 * xdiff  with t4=(Xp+Zp)(Xq-Zq), t5=(Xp-Zp)(Xq+Zq) */
static void xadd(xz &r, const xz &p, const xz &q, const ladder_ctx &c)
{
    mont_t t0, t1, t2, t3, t4, t5;
    mont_add(&t0, &p.X, &p.Z, c.mc);
    mont_sub(&t1, &p.X, &p.Z, c.mc);
    mont_add(&t2, &q.X, &q.Z, c.mc);
    mont_sub(&t3, &q.X, &q.Z, c.mc);
    mont_mul(&t4, &t0, &t3, c.mc);
    mont_mul(&t5, &t1, &t2, c.mc);
    mont_add(&t0, &t4, &t5, c.mc);
    mont_sub(&t1, &t4, &t5, c.mc);
    mont_sqr(&t4, &t0, c.mc);                     /* X3 */
    mont_sqr(&t5, &t1, c.mc);                     /* (t4-t5)^2 */
    mont_mul(&t2, &t5, &c.xdiff, c.mc);           /* Z3 */
    mont_set(&r.X, &t4, c.mc);
    mont_set(&r.Z, &t2, c.mc);
}

} /* anonymous namespace */

/* --------------------------------------------------------------------------
 * [s]P and the stage-1 verdict
 * ------------------------------------------------------------------------ */
uint8_t *mont_expand_bits(const mpz_t s, size_t *out_nbits)
{
    *out_nbits = 0;
    if (mpz_sgn(s) == 0) return NULL;

    size_t nbytes = 0;
    uint8_t *be = (uint8_t *)mpz_export(NULL, &nbytes, 1 /* most significant first */,
                                        1 /* one byte per word */, 0, 0, s);
    if (be == NULL || nbytes == 0) { free(be); return NULL; }

    unsigned lead = 0;                              /* leading zero bits of byte 0 */
    while (lead < 8 && ((be[0] >> (7 - lead)) & 1u) == 0) lead++;

    const size_t nbits = nbytes * 8 - lead;
    uint8_t *bits = (uint8_t *)malloc(nbits);
    if (bits == NULL) { free(be); return NULL; }
    for (size_t i = 0; i < nbits; i++) {
        const size_t j = i + lead;                  /* bit index in the byte string */
        bits[i] = (uint8_t)((be[j >> 3] >> (7 - (j & 7))) & 1u);
    }
    free(be);
    *out_nbits = nbits;
    return bits;
}

int mont_stage1_curve_bits(mpz_t factor, mpz_t Qx, mpz_t Qz, const mpz_t N,
                           uint64_t sigma, const uint8_t *bits, size_t nbits)
{
    mont_ctx_t mc;
    if (mont_init(&mc, N) != 0) return -1;        /* beyond ED_MONT_MAX_LIMBS */

    mpz_t A, X0, Z0, a24, xdiff, inv;
    mpz_inits(A, X0, Z0, a24, xdiff, inv, NULL);
    mont_suyama_curve(A, X0, Z0, sigma, N);

    mpz_add_ui(a24, A, 2);
    mpz_set_ui(inv, 4);
    mpz_invert(inv, inv, N);
    mpz_mul(a24, a24, inv);
    mpz_mod(a24, a24, N);                         /* a24 = (A+2)/4 */

    mpz_invert(inv, Z0, N);
    mpz_mul(xdiff, X0, inv);
    mpz_mod(xdiff, xdiff, N);                      /* affine x of the start point */

    ladder_ctx c;
    c.mc = &mc;
    mont_to(&c.a24, a24, &mc);
    mont_to(&c.xdiff, xdiff, &mc);

    xz R0, R1, T;
    mont_to(&R0.X, X0, &mc);
    mont_to(&R0.Z, Z0, &mc);
    xdbl(R1, R0, c);                               /* R1 = 2P */

    xz *p0 = &R0, *p1 = &R1, *pt = &T;
    for (size_t i = 1; i < nbits; i++) {           /* bits[0] is the implicit top bit */
        if (bits[i]) {
            xadd(*pt, *p1, *p0, c);                /* pt = R0 + R1 (diff P) */
            xdbl(*p1, *p1, c);                     /* R1 = 2*R1 */
            xz *sw = p0; p0 = pt; pt = sw;
        } else {
            xadd(*pt, *p0, *p1, c);                /* pt = R0 + R1 (diff P) */
            xdbl(*p0, *p0, c);                     /* R0 = 2*R0 */
            xz *sw = p1; p1 = pt; pt = sw;
        }
    }

    mont_from(Qx, &p0->X, &mc);
    mont_from(Qz, &p0->Z, &mc);
    mpz_mod(Qx, Qx, N);
    mpz_mod(Qz, Qz, N);

    int rc = 0;
    if (factor) {
        mpz_gcd(factor, Qz, N);
        if (mpz_cmp_ui(factor, 1) > 0 && mpz_cmp(factor, N) < 0) rc = 1;
    }
    mpz_clears(A, X0, Z0, a24, xdiff, inv, NULL);
    mont_clear(&mc);
    return rc;
}

int mont_stage1_curve(mpz_t factor, mpz_t Qx, mpz_t Qz,
                      const mpz_t N, uint64_t sigma, const mpz_t s)
{
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    const int rc = mont_stage1_curve_bits(factor, Qx, Qz, N, sigma, bits, nbits);
    free(bits);
    return rc;
}

int mont_stage1_curve_bits_x(mpz_t factor, mpz_t x, const mpz_t N,
                             uint64_t sigma, const uint8_t *bits, size_t nbits)
{
    mpz_t Qx, Qz, inv;
    mpz_inits(Qx, Qz, inv, NULL);
    const int rc = mont_stage1_curve_bits(factor, Qx, Qz, N, sigma, bits, nbits);
    if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
        mpz_mul(x, Qx, inv);
        mpz_mod(x, x, N);
    } else {
        mpz_set(x, Qx);
    }
    mpz_clears(Qx, Qz, inv, NULL);
    return rc;
}

int mont_stage1_curve_x(mpz_t factor, mpz_t x, const mpz_t N, uint64_t sigma, const mpz_t s)
{
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    const int rc = mont_stage1_curve_bits_x(factor, x, N, sigma, bits, nbits);
    free(bits);
    return rc;
}
