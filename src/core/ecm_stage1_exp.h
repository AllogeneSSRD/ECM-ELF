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
   allocated; `s` is then left at `torsion` (never a wrong product).

   COST (measured, single thread, this project's dev box):

     B1        bits      before 2026-09-26   now
     1e6       1.44 M       0.015 s        0.012 s
     1e7      14.4 M        0.20 s         0.17 s
     1e8     144.3 M        5.2-5.4 s      2.95 s
     2.6e8   375.1 M       16.5 s         10.7 s     <- production B1

   What the 2026-09-26 rewrite changed (all measured, see the .cpp):
     * odds-only SEGMENTED sieve instead of a 260 MB full-range byte sieve (0.14 s of marking);
     * primes are folded into a small (512-bit) 32-bit-limb accumulator with a hand-rolled
       in-place mul instead of one mpz_mul_ui chain per prime (GMP allocates+copies per call);
     * only p <= sqrt(B1) needs the prime-power loop (one division per prime saved);
     * the counter slots are combined PAIRWISE instead of largest-first sequentially.

   The remaining 10.7 s at B1 = 2.6e8 is inherent to a 375 Mbit product: one GMP multiply of
   two 187.5 Mbit values alone is 0.87 s and the tree's upper levels need several (folds 7.5 s
   + final reduction 2.8 s; sieve and per-prime work together are only ~0.4 s).  Since each
   queue task used to repeat that build, the value is cached on disk as well: a validated
   cache load is 0.26 s (42x faster) -- see ecm_stage1_exp_cache.h for the validation chain. */
bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion);

#endif /* ECM_STAGE1_EXP_H */
