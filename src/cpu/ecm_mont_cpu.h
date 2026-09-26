/* ---------------------------------------------------------------------------
 * ecm_mont_cpu.h -- Suyama-sigma (Prime95 sigma_type = 1) Montgomery-curve
 * ECM stage 1, scalar reference implementation.
 *
 * Math (all conventions pinned by experiment; see docs/ECM_Montgomery_STAGE1.md):
 *   u = sigma^2 - 5              v = 4*sigma
 *   A = (v-u)^3 (3u+v) / (4 u^3 v) - 2        (mod N)   [Montgomery coefficient]
 *   a24 = (A+2)/4                                       [doubling constant]
 *   start point (X:Z) = (u^3 : v^3)
 *   exponent  s = torsion * lcm(1..B1)     torsion = 1 matches gmp-ecm param 0
 *                                          torsion = 12 matches Prime95 choose12
 *   stage 1: [s]P by the x-only Montgomery ladder (one xDBLADD per bit);
 *            factor found <=> gcd(Z, N) > 1
 *
 * Field arithmetic: MPN, via the generic Montgomery layer in
 * src/cpu/ecm_edwards_mont.h (GMP REDC).  The header was written for the Edwards
 * path but is curve-agnostic.
 * ------------------------------------------------------------------------- */
#ifndef ECM_MONT_CPU_H
#define ECM_MONT_CPU_H

#include <gmp.h>
#include <stdint.h>

#include <string>

/* s = torsion * lcm(1..B1) with lcm built from prime powers (gmp-ecm convention
   when torsion == 1).  Returns s_bits. */
size_t mont_build_s(mpz_t s, uint64_t B1, uint64_t torsion, std::string *detail = nullptr);

/* mpz_set_ui() takes `unsigned long`, which is 32 bits on Windows, so passing a
   64-bit sigma through it silently TRUNCATES: sigma = 2^62 became 0, which makes
   v = 4*sigma = 0, the curve coefficient A degenerate (A = -2, a singular curve)
   and every curve return the identity (gcd == N, no factor) with X = 0x0 in the
   save file.  Always go through this helper for sigmas. */
static inline void mont_set_sigma(mpz_t r, uint64_t sigma)
{
    mpz_set_ui(r, (unsigned long)(sigma >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(sigma & 0xFFFFFFFFull));
}

/* Expand the exponent into a byte array of single bits, most significant first:
   bits[0] is the top set bit of s, bits[nbits-1] is bit 0.  Returns the malloc'd
   array (NULL when s == 0) and writes its length to *out_nbits.
 *
 * The ladder walks every bit of s, and s depends only on (B1, torsion) -- not on N
 * and not on sigma -- so this is done once per task and shared by every curve and
 * every SIMD batch/worker thread.  The bit loop then reads one byte instead of
 * calling mpz_tstbit (which is O(1) but still a call plus an mpz struct deref).
 * Caller frees with free(). */
uint8_t *mont_expand_bits(const mpz_t s, size_t *out_nbits);

/* sigma -> Montgomery coefficient A and the starting projective point (X0:Z0),
   all plain-domain integers mod N (Z0 = v^3, not normalised: the caller needs the
   affine x only for the differential-addition constant). */
void mont_suyama_curve(mpz_t A, mpz_t X0, mpz_t Z0, uint64_t sigma, const mpz_t N);

/* One curve: Qx/Qz return the plain-domain point [s]P (Qz == 0 mod N means the
   point is the identity, i.e. a hit mod every prime factor).  factor receives
   gcd(Qz, N) and the return value is
     1  : non-trivial factor found (1 < gcd < N), stored in `factor`
     0  : no factor (gcd == 1, or gcd == N)
    -1  : internal error (N too large for the field layer). */
int mont_stage1_curve(mpz_t factor, mpz_t Qx, mpz_t Qz,
                      const mpz_t N, uint64_t sigma, const mpz_t s);

/* Same, but the normalized x = Qx/Qz (what the reference save files store) and
   gcd(Qz, N).  Returns as above; x is set even when no factor is found. */
int mont_stage1_curve_x(mpz_t factor, mpz_t x, const mpz_t N, uint64_t sigma, const mpz_t s);

/* Same as mont_stage1_curve()/mont_stage1_curve_x(), but the ladder consumes a
   pre-expanded bit array (mont_expand_bits) instead of the mpz_t exponent, so the
   per-bit cost has no mpz call in it.  These are the entry points used in loops. */
int mont_stage1_curve_bits(mpz_t factor, mpz_t Qx, mpz_t Qz, const mpz_t N,
                           uint64_t sigma, const uint8_t *bits, size_t nbits);
int mont_stage1_curve_bits_x(mpz_t factor, mpz_t x, const mpz_t N,
                             uint64_t sigma, const uint8_t *bits, size_t nbits);

/* ---------------------------------------------------------------------------
 * Mid-ladder checkpoint support (docs/ECM_Montgomery_STAGE1.md §17).
 *
 * The ladder invariant at the top of every iteration is
 *
 *     p0 = [k]P        p1 = [k+1]P            k = bits consumed so far
 *
 * so (bitnum = k, p0, p1) is a *complete* resume point: the curve constants are
 * not state at all -- a24 and xdiff are recomputed from sigma, which the
 * checkpoint file stores next to the point.  Nothing else in the ladder is
 * stateful (no FFT scratch, no window table), which is why this path needs no
 * bit-offset cleverness like the NAF-based Edwards ladder does.
 * ------------------------------------------------------------------------- */
typedef struct {
    size_t bitnum;      /* exponent bits already consumed (0 = fresh curve) */
    mpz_t  X0, Z0;      /* p0 = [k]P    (plain domain, mod N) */
    mpz_t  X1, Z1;      /* p1 = [k+1]P  (plain domain, mod N) */
} mont_ladder_state_t;

void mont_ladder_state_init(mont_ladder_state_t *st);
void mont_ladder_state_clear(mont_ladder_state_t *st);

/* Progress / pause callback, called every `chunk_bits` exponent bits.  `st`
 * describes the state at the pause point (bits[0..bitnum-1] consumed); return
 * non-zero to pause there.  It is NOT called after the last bit: a caller that
 * wants 100% progress marks the curve done from the return code instead. */
typedef int (*mont_progress_fn)(void *ctx, const mont_ladder_state_t *st);

#define MONT_LADDER_HIT    1
#define MONT_LADDER_MISS   0
#define MONT_LADDER_ERROR  (-1)
#define MONT_LADDER_PAUSED 2      /* *st holds the resume point */

/* Ladder that can both start and stop in the middle.
 *   start_bit == 0 : fresh curve built from sigma (st is only written on pause)
 *   start_bit >  0 : resume; *st must already hold the state at start_bit
 *                     (start_bit == nbits is legal and just returns that state)
 * Returns MONT_LADDER_HIT / _MISS / _ERROR / _PAUSED. */
int mont_stage1_curve_bits_ex(mpz_t factor, mpz_t Qx, mpz_t Qz, const mpz_t N,
                              uint64_t sigma, const uint8_t *bits, size_t nbits,
                              size_t start_bit, mont_ladder_state_t *st,
                              mont_progress_fn cb, void *cb_ctx, size_t chunk_bits);

#endif /* ECM_MONT_CPU_H */
