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
#include "../prac/descriptor.h"

#include "ecm/cpu/arithmetic/ifma/modular_ifma_ops.h"
#include <stdlib.h>
#include <string.h>
#include <utility>

namespace {
using ecm_ifma::add;
using ecm_ifma::addsub;
using ecm_ifma::sub;

/* --------------------------------------------------------------------------
 * SoA point ops
 * ------------------------------------------------------------------------ */
struct soa_xz { uint64_t *X, *Z; };

/* X2 = (X+Z)^2 (X-Z)^2 ,  Z2 = 4XZ * ((X-Z)^2 + a24*4XZ) */
static inline void xz_dbl_from_sums(const soa_xz &r, const uint64_t *a24,
                                    uint64_t *A_, uint64_t *B_, uint64_t *E_,
                                    uint64_t *t, uint64_t *tmp, const ifma_ctx_t *mc)
{
    ifma_mont_sqr(A_, A_, mc);
    ifma_mont_sqr(B_, B_, mc);
    sub(E_, A_, B_, mc);                       /* E = 4XZ */
    ifma_mont_mul(r.X, A_, B_, mc);
    ifma_mont_mul(t, a24, E_, mc);
    add(t, t, B_, tmp, mc);                    /* B + a24*E */
    ifma_mont_mul(r.Z, E_, t, mc);
}

static void xz_dbl(const soa_xz &r, const soa_xz &p, const uint64_t *a24,
                   uint64_t *A_, uint64_t *B_, uint64_t *E_, uint64_t *t, uint64_t *tmp,
                   const ifma_ctx_t *mc)
{
    add(A_, p.X, p.Z, tmp, mc);
    sub(B_, p.X, p.Z, mc);
    xz_dbl_from_sums(r, a24, A_, B_, E_, t, tmp, mc);
}

/* X3 = (t4+t5)^2 * zdiff ,  Z3 = (t4-t5)^2 * xdiff.
   PRAC uses a projective difference and a result disjoint from all inputs. */
static void xz_add(const soa_xz &r, const soa_xz &p, const soa_xz &q,
                   const uint64_t *xdiff, const uint64_t *zdiff,
                   uint64_t *t0, uint64_t *t1, uint64_t *t2, uint64_t *t3,
                   uint64_t *t4, uint64_t *t5, uint64_t *tmp, const ifma_ctx_t *mc)
{
    add(t0, p.X, p.Z, tmp, mc);
    sub(t1, p.X, p.Z, mc);
    add(t2, q.X, q.Z, tmp, mc);
    sub(t3, q.X, q.Z, mc);
    ifma_mont_mul(t4, t0, t3, mc);
    ifma_mont_mul(t5, t1, t2, mc);
    add(t0, t4, t5, tmp, mc);
    sub(t1, t4, t5, mc);
    ifma_mont_sqr(t0, t0, mc);
    ifma_mont_mul(r.X, t0, zdiff, mc);
    ifma_mont_sqr(t5, t1, mc);
    ifma_mont_mul(r.Z, t5, xdiff, mc);
}

/* One binary ladder bit reuses the doubled point's X±Z in the fixed-affine
   differential addition. The sum has its own storage. Doubling updates P. */
static void xz_dbl_add(const soa_xz &p, const soa_xz &q, const soa_xz &sum,
                       const uint64_t *a24, const uint64_t *xdiff,
                       uint64_t *plus, uint64_t *minus, uint64_t *q_plus,
                       uint64_t *q_minus, uint64_t *v, uint64_t *u,
                       uint64_t *tmp, const ifma_ctx_t *mc)
{
    addsub(plus, minus, p.X, p.Z, tmp, mc);
    addsub(q_plus, q_minus, q.X, q.Z, tmp, mc);
    ifma_mont_mul(v, plus, q_minus, mc);
    ifma_mont_mul(u, minus, q_plus, mc);
    addsub(q_plus, q_minus, v, u, tmp, mc);
    ifma_mont_sqr(sum.X, q_plus, mc);
    ifma_mont_sqr(u, q_minus, mc);
    ifma_mont_mul(sum.Z, u, xdiff, mc);
    xz_dbl_from_sums(p, a24, plus, minus, q_plus, q_minus, tmp, mc);
}

/* Curve construction is shared by the two point schedules. When the affine
   difference is requested, preserve the ladder's behavior at setup failures.
   PRAC delegates those batches to the ladder so exceptional outputs agree. */
static bool setup_curves(const soa_xz &start, uint64_t *a24, uint64_t *xdiff,
                         const uint64_t sigmas[IFMA_LANES], bool init_start,
                         const ifma_ctx_t *mc)
{
    bool invertible = true;
    mpz_t u, v, t, num, den, X0, Z0, Amz, inv;
    mpz_inits(u, v, t, num, den, X0, Z0, Amz, inv, NULL);
    for (unsigned k = 0; k < IFMA_LANES; k++) {
        mont_set_sigma(u, sigmas[k]);
        mpz_mul(u, u, u);
        mpz_sub_ui(u, u, 5);
        mont_set_sigma(v, sigmas[k]);
        mpz_mul_ui(v, v, 4);
        mpz_sub(t, v, u);
        mpz_powm_ui(num, t, 3, mc->N);
        mpz_mul_ui(t, u, 3);
        mpz_add(t, t, v);
        mpz_mul(num, num, t);
        mpz_mod(num, num, mc->N);
        mpz_powm_ui(den, u, 3, mc->N);
        mpz_mul_ui(den, den, 4);
        mpz_mul(den, den, v);
        mpz_mod(den, den, mc->N);
        if (mpz_invert(inv, den, mc->N) == 0) {
            invertible = false;
            mpz_set_ui(inv, 0);
        }
        mpz_mul(num, num, inv);
        mpz_sub_ui(num, num, 2);
        mpz_mod(Amz, num, mc->N);
        mpz_powm_ui(X0, u, 3, mc->N);
        mpz_powm_ui(Z0, v, 3, mc->N);

        if (init_start) {
            ifma_from_mpz_lane(start.X, k, X0, mc);
            ifma_from_mpz_lane(start.Z, k, Z0, mc);
        }

        mpz_add_ui(num, Amz, 2);
        mpz_set_ui(inv, 4);
        mpz_invert(inv, inv, mc->N);
        mpz_mul(num, num, inv);
        mpz_mod(num, num, mc->N);
        ifma_from_mpz_lane(a24, k, num, mc);

        if (xdiff) {
            mpz_invert(inv, Z0, mc->N);
            mpz_mul(num, X0, inv);
            mpz_mod(num, num, mc->N);
            ifma_from_mpz_lane(xdiff, k, num, mc);
        } else if (invertible) {
            // A24=0 or 1 modulo any factor gives a singular Montgomery curve.
            // Keep those batches on the legacy ladder's exceptional path.
            mpz_sub_ui(t, num, 1);
            mpz_mul(t, t, num);
            mpz_gcd(t, t, mc->N);
            invertible = mpz_cmp_ui(t, 1) == 0;
        }
    }
    mpz_clears(u, v, t, num, den, X0, Z0, Amz, inv, NULL);
    return invertible;
}

static void export_point(const soa_xz &point, mpz_t *out_x, mpz_t *out_gcd,
                         const ifma_ctx_t *mc)
{
    mpz_t X, Z, inv;
    mpz_inits(X, Z, inv, NULL);
    for (unsigned k = 0; k < IFMA_LANES; k++) {
        ifma_to_mpz_lane(Z, point.Z, k, mc);
        mpz_gcd(out_gcd[k], Z, mc->N);
        if (mpz_sgn(Z) != 0 && mpz_invert(inv, Z, mc->N)) {
            ifma_to_mpz_lane(X, point.X, k, mc);
            mpz_mul(out_x[k], X, inv);
            mpz_mod(out_x[k], out_x[k], mc->N);
        } else {
            mpz_set_ui(out_x[k], 0);
        }
    }
    mpz_clears(X, Z, inv, NULL);
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
size_t mont_soa_state_words(const mont_soa_ctx_t *c)
{
    return 4 * 8 * c->n;
}

/* The batch ladder, now interruptible: see mont_soa_stage1_bits_ex() in the header.
   start_bit == 0 keeps the original behaviour bit-for-bit; the only extra code on
   that path is one `i > start_bit` comparison per checkpoint chunk. */
int mont_soa_stage1_bits_ex(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                            const uint64_t sigmas[IFMA_LANES],
                            size_t start_bit, uint64_t *state, size_t *out_bitnum,
                            mpz_t *out_x, mpz_t *out_gcd,
                            mont_soa_progress_fn cb, void *cb_ctx, size_t chunk_bits)
{
    if (start_bit > nbits) return MONT_SOA_ERROR;
    if (start_bit > 0 && !state) return MONT_SOA_ERROR;
    const ifma_ctx_t *mc = &c->mc;
    const size_t n = c->n, lw = 8 * n;
    uint64_t *pool = c->pool;

    soa_xz R0 = { pool + 0 * lw, pool + 1 * lw };
    soa_xz R1 = { pool + 2 * lw, pool + 3 * lw };
    soa_xz T  = { pool + 4 * lw, pool + 5 * lw };
    uint64_t *a24 = pool + 6 * lw, *xdiff = pool + 7 * lw;
    uint64_t *A_ = pool + 8 * lw, *B_ = pool + 9 * lw, *E_ = pool + 10 * lw, *tt = pool + 11 * lw;
    uint64_t *t0 = pool + 12 * lw, *t1 = pool + 13 * lw;
    uint64_t *tmp = pool + 14 * lw;      /* scratch for the sum's candidate r-N */

    /* Resume: the checkpoint buffer uses exactly the pool element layout (lane-SoA,
       same limb count, same field representation), so restoring p0/p1 is a memcpy. */
    if (start_bit > 0) {
        memcpy(R0.X, state + 0 * lw, lw * sizeof(uint64_t));
        memcpy(R0.Z, state + 1 * lw, lw * sizeof(uint64_t));
        memcpy(R1.X, state + 2 * lw, lw * sizeof(uint64_t));
        memcpy(R1.Z, state + 3 * lw, lw * sizeof(uint64_t));
    }

    setup_curves(R0, a24, xdiff, sigmas, start_bit == 0, mc);

    /* R1 = 2*R0, then the branch-free ladder over the shared exponent bits */
    size_t i;
    if (start_bit == 0) {
        xz_dbl(R1, R0, a24, A_, B_, E_, tt, tmp, mc);
        i = 1;                                     /* bits[0] is the implicit top bit */
    } else {
        i = start_bit;                             /* state already holds p0/p1 */
    }

    soa_xz *p0 = &R0, *p1 = &R1, *pt = &T;
    int paused = 0;
    /* Checkpoint cadence as a countdown rather than  i % chunk  : the ladder runs
       one integer division per bit otherwise, which is a real cost against ~10
       field multiplications. */
    const size_t ck_stride = (cb && chunk_bits) ? chunk_bits : 0;
    size_t next_ck = ck_stride ? (start_bit + ck_stride) : 0;   /* 0 = never */
    for (; i < nbits; i++) {
        if (ck_stride && i == next_ck) {
            next_ck += ck_stride;
            memcpy(state + 0 * lw, p0->X, lw * sizeof(uint64_t));
            memcpy(state + 1 * lw, p0->Z, lw * sizeof(uint64_t));
            memcpy(state + 2 * lw, p1->X, lw * sizeof(uint64_t));
            memcpy(state + 3 * lw, p1->Z, lw * sizeof(uint64_t));
            if (out_bitnum) *out_bitnum = i;
            if (cb(cb_ctx, i)) { paused = 1; break; }
        }
        if (bits[i]) {
            xz_dbl_add(*p1, *p0, *pt, a24, xdiff, A_, B_, E_, tt, t0, t1, tmp, mc);
            soa_xz *sw = p0; p0 = pt; pt = sw;
        } else {
            xz_dbl_add(*p0, *p1, *pt, a24, xdiff, A_, B_, E_, tt, t0, t1, tmp, mc);
            soa_xz *sw = p1; p1 = pt; pt = sw;
        }
    }

    if (paused) return MONT_SOA_PAUSED;

    export_point(*p0, out_x, out_gcd, mc);
    if (out_bitnum) *out_bitnum = nbits;
    return MONT_SOA_DONE;
}

int mont_soa_stage1_bits(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                         const uint64_t sigmas[IFMA_LANES],
                         mpz_t *out_x, mpz_t *out_gcd)
{
    return mont_soa_stage1_bits_ex(c, bits, nbits, sigmas, 0, NULL, NULL,
                                   out_x, out_gcd, NULL, NULL, 0);
}

int mont_soa_stage1_prac_ex(mont_soa_ctx_t *c, const EcmPracPrime *primes, size_t count,
                           const uint64_t sigmas[IFMA_LANES],
                           size_t start, uint64_t *state, size_t *out_offset,
                           mpz_t *out_x, mpz_t *out_gcd,
                           mont_soa_progress_fn cb, void *cb_ctx, size_t chunk_primes)
{
    const size_t stride = cb ? chunk_primes : 0;
    if (!primes || !count || start > count || ((start || stride) && !state))
        return MONT_SOA_ERROR;

    const ifma_ctx_t *mc = &c->mc;
    const size_t lw = IFMA_LANES * c->n;
    const size_t bytes = lw * sizeof(uint64_t);
    auto field = [&](size_t i) { return c->pool + i * lw; };
    soa_xz a{field(0), field(1)}, b{field(2), field(3)};
    soa_xz diff{field(4), field(5)}, t{field(6), field(7)};
    uint64_t *a24 = field(8);

    if (start > 0) {
        memcpy(a.X, state, bytes);
        memcpy(a.Z, state + lw, bytes);
    }
    if (!setup_curves(a, a24, nullptr, sigmas, start == 0, mc))
        return MONT_SOA_REPLAY;

    auto dbl = [&](const soa_xz &out, const soa_xz &in) {
        xz_dbl(out, in, a24, field(9), field(10), field(11), field(12), field(15), mc);
    };
    auto dadd = [&](const soa_xz &out, const soa_xz &lhs, const soa_xz &rhs,
                    const soa_xz &difference) {
        xz_add(out, lhs, rhs, difference.X, difference.Z,
               field(9), field(10), field(11), field(12), field(13), field(14), field(15), mc);
    };

    // At a prime boundary only A survives. Swapping point roles moves their
    // pointers, so a PRAC rule never copies a field element to change a role.
    size_t next = start + stride;
    for (size_t i = start; i < count; ++i) {
        if (stride && i == next) {
            next += stride;
            memcpy(state, a.X, bytes);
            memcpy(state + lw, a.Z, bytes);
            memset(state + 2 * lw, 0, 2 * bytes);
            if (out_offset) *out_offset = i;
            if (cb(cb_ctx, i)) return MONT_SOA_PAUSED;
        }
        const EcmPracPrime &prime = primes[i];
        for (uint32_t rep = 0; rep < prime.repetitions; ++rep) {
            if (prime.p == 2) {
                dbl(a, a);
                continue;
            }

            memcpy(diff.X, a.X, bytes);
            memcpy(diff.Z, a.Z, bytes);
            dbl(b, a);
            uint32_t e = prime.p - prime.d;
            uint32_t d = prime.d - e;
            // The planner maintains d*A + e*B = p*P and Diff = ±(A-B).
            while (d != e) {
                if (d < e) {
                    std::swap(d, e);
                    std::swap(a, b);
                }
                if (uint64_t(d) * 100 <= uint64_t(e) * 296) {
                    dadd(t, a, b, diff);
                    std::swap(b, diff);
                    std::swap(b, t);
                    d -= e;
                } else if ((d & 1) == (e & 1)) {
                    dadd(t, a, b, diff);
                    dbl(a, a);
                    std::swap(b, t);
                    d = (d - e) / 2;
                } else if (!(d & 1)) {
                    dadd(t, a, diff, b);
                    dbl(a, a);
                    std::swap(diff, t);
                    d /= 2;
                } else {
                    dadd(t, b, diff, a);
                    dbl(b, b);
                    std::swap(diff, t);
                    e /= 2;
                }
            }
            dadd(t, a, b, diff);
            std::swap(a, t);
        }
    }
    export_point(a, out_x, out_gcd, mc);
    if (out_offset) *out_offset = count;
    return MONT_SOA_DONE;
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
