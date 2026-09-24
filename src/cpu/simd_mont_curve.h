/* ---------------------------------------------------------------------------
 * simd_mont_curve.h -- batched (8 lanes) Suyama-sigma Montgomery stage 1.
 *
 * Same math and conventions as the scalar path in ecm_mont_cpu.{h,cpp}:
 *   u = sigma^2-5, v = 4*sigma, A = (v-u)^3(3u+v)/(4u^3v) - 2, a24 = (A+2)/4,
 *   start (X:Z) = (u^3:v^3), [s]P by the x-only Montgomery ladder, hit <=> gcd(Z,N)>1.
 *
 * Batch layout is "lane = curve" (SoA), exactly like simd_edwards:
 * one field element is uint64_t e[8*n], column i = lane vector at e + 8*i.
 *
 * Because every lane multiplies by the SAME exponent s (one B1 per batch), the bit
 * sequence is identical for all lanes: the ladder needs no divergence handling and
 * not even a cswap -- all lanes execute the same branch.  That is the whole reason
 * this collection of kernels is so SIMD friendly.
 *
 * Verify against the scalar path with tools/bench/mont_simd_verify.cpp.
 * ------------------------------------------------------------------------- */
#ifndef SIMD_MONT_CURVE_H
#define SIMD_MONT_CURVE_H

#include "simd_mont_ifma.h"
#include <gmp.h>
#include <stdint.h>

typedef struct {
    ifma_ctx_t mc;       /* field layer: Montgomery CIOS or Mersenne fold */
    size_t     n;        /* 52-bit limbs */
    uint64_t  *pool;     /* scratch: 22 field elements */
    size_t     pool_elems;
} mont_soa_ctx_t;

/* field_mode: IFMA_FIELD_MONT | IFMA_FIELD_MERS | IFMA_FIELD_AUTO */
int  mont_soa_init(mont_soa_ctx_t *c, const mpz_t N, int field_mode);
void mont_soa_clear(mont_soa_ctx_t *c);

/* Run [s]P for the 8 curves given by sigmas[0..7].
 *   out_x[k]   : normalized Montgomery x = X/Z (only meaningful when out_gcd[k] == 1)
 *   out_gcd[k] : gcd(Z, N) for lane k; > 1 means a stage-1 hit (== N means the point
 *                is the identity mod N, i.e. no usable factor)
 * Returns 0 on success, -1 on field-layer init failure. */
int mont_soa_stage1(mont_soa_ctx_t *c, const mpz_t s, const uint64_t sigmas[IFMA_LANES],
                    mpz_t *out_x, mpz_t *out_gcd);

/* Same, with the exponent handed over as a pre-expanded MSB-first bit array
   (mont_expand_bits() in ecm_mont_cpu.h).  loops and worker threads should use this
   one: the bit array is derived from (B1, torsion) only, so it is built once per
   task and shared by every batch/thread. */
int mont_soa_stage1_bits(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                         const uint64_t sigmas[IFMA_LANES], mpz_t *out_x, mpz_t *out_gcd);

/* ---------------------------------------------------------------------------
 * Interruptible batch (mid-stage-1 checkpoints, docs/ECM_Montgomery_STAGE1.md §17).
 *
 * Every lane walks the SAME exponent bits in lockstep, so one bit offset
 * describes the whole batch: a paused batch resumes with start_bit and the
 * per-lane p0/p1 pair.  State layout (a caller-allocated buffer of
 * mont_soa_state_words() uint64_t, lane-SoA like every other field element):
 *
 *     word  0 * lw : X0     p0 = [k]P      (lw = 8*n)
 *     word  1 * lw : Z0
 *     word  2 * lw : X1     p1 = [k+1]P
 *     word  3 * lw : Z1
 *
 * on return MONT_SOA_PAUSED the buffer holds the state at *out_bitnum.
 * Returns MONT_SOA_DONE / MONT_SOA_PAUSED / MONT_SOA_ERROR. */
typedef int (*mont_soa_progress_fn)(void *ctx, size_t bitnum);

#define MONT_SOA_DONE   0
#define MONT_SOA_PAUSED 1
#define MONT_SOA_ERROR  (-1)

size_t mont_soa_state_words(const mont_soa_ctx_t *c);

int mont_soa_stage1_bits_ex(mont_soa_ctx_t *c, const uint8_t *bits, size_t nbits,
                            const uint64_t sigmas[IFMA_LANES],
                            size_t start_bit, uint64_t *state, size_t *out_bitnum,
                            mpz_t *out_x, mpz_t *out_gcd,
                            mont_soa_progress_fn cb, void *cb_ctx, size_t chunk_bits);

/* Field ops per point operation (reporting / cost accounting). */
void mont_soa_op_counts(uint64_t *muls_per_bit_x1000);

#endif /* SIMD_MONT_CURVE_H */
