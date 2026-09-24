/* ---------------------------------------------------------------------------
 * ecm_stage1_exp.h -- the stage-1 exponent  s = torsion * lcm(1..B1).
 *
 * ONE builder for every method that needs it (GPU batch, CPU Edwards, CPU
 * Montgomery).  It exists as a shared module because the obvious loop --
 *
 *     for each prime power q <= B1:  mpz_mul_ui(s, s, q)
 *
 * is quadratic in the number of primes: at B1 = 1e7 that is 620k multiplications
 * of a 1-limb value into an accumulator that grows to ~225k limbs, i.e. ~7e10
 * limb operations.  Measured on the Ryzen AI 9 HX 370 box: **29 s of pure startup
 * before the first ladder bit ran** (reported by the user for a queue task at
 * B1 = 1e7, while the GPU path needed ~5 s for B1 = 1.1e8 -- the GPU path already
 * used a product tree, and the Montgomery path did not).
 *
 * The fix is to build the same product with a binary counter: prime powers are
 * merged pairwise ("slot j holds a product of 2^j of them"), so every mpz_mul has
 * two operands of comparable size and GMP's Karatsuba/Toom/FFT paths apply.  This
 * is the technique the GPU path and the Edwards path already used; it is now the
 * only implementation.
 *
 * Exactness is not affected: multiplication is exact and commutative, so the
 * result is the SAME integer -- only the order of the multiplications changes.
 * tools/test/stage1_exp_check.cpp verifies that (old loop vs this builder, plus
 * independent p-adic valuation checks).
 * ------------------------------------------------------------------------- */
#ifndef ECM_STAGE1_EXP_H
#define ECM_STAGE1_EXP_H

#include <gmp.h>
#include <stdint.h>

/* s = torsion * prod_{p <= B1} p^floor(log_p B1),  torsion 0 is treated as 1.
   Returns false only when B1 is out of range (> 5e9) or the sieve cannot be
   allocated; `s` is then left at `torsion` (never a wrong product). */
bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion);

#endif /* ECM_STAGE1_EXP_H */
