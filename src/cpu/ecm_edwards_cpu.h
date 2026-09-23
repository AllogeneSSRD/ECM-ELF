#pragma once
// Edwards (Atkin-Morain, a=1, Z/2xZ/8) stage-1 CPU backend, mpz-based.
//
// Correctness-first reference implementation; the scalar multiplier is plain
// double-and-add (NAF is a later performance step). The stage-1 result is the
// Montgomery point (Qx:Qz) = (z+y : z-y) of [s]P, plus a gcd factor check.

#include <stdint.h>
#include <gmp.h>

// Run one Atkin-Morain Edwards stage-1 curve.
//
//   factor : receives the non-trivial factor when found (may be NULL).
//   Qx, Qz : receive the Montgomery point (z+y, z-y) mod N when non-NULL
//            (for later save/checkpoint; may be NULL).
//   N      : composite to factor.
//   sigma  : 64-bit curve parameter (same value Prime95 uses for sigma_type=0).
//   s      : stage-1 exponent (must already be 48 * lcm(1..B1)).
//
// Returns:
//   1  : non-trivial factor found (1 < gcd(Qz,N) < N), stored in `factor`.
//   0  : no factor (gcd(Qz,N) == 1, or gcd == N meaning [s]P is identity mod N).
//   -1 : internal error.
int edwards_stage1_curve(mpz_t factor, mpz_t Qx, mpz_t Qz,
                         const mpz_t N, uint64_t sigma, const mpz_t s);

// Set the w-NAF window size (default 4; must be >= 2). Larger w -> fewer
// additions in the scalar multiply, but a 2^(w-2)-entry dictionary.
// Provided for tuning/benchmarking.
void edwards_set_naf_w(int w);
