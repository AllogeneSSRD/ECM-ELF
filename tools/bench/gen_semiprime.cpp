/* ---------------------------------------------------------------------------
 * gen_semiprime.cpp -- print a semiprime of a requested bit size, for benchmarks.
 *
 * Why: a stage-1 run STOPS as soon as a curve finds a factor, so timing a
 * throughput baseline on M3001 (which has small factors) truncates the run and
 * silently invalidates the measurement.  A semiprime p*q with both factors near
 * half the size has no chance of being smooth up to B1, so every curve runs to
 * completion and the wall time measures arithmetic only.
 *
 * build: tools\build_tool.bat tools\bench\gen_semiprime.cpp
 * usage: build_vs18\tools\gen_semiprime.exe <bits> [seed]
 * ------------------------------------------------------------------------- */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const unsigned bits = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 10) : 3001;
    const unsigned long seed = (argc > 2) ? strtoul(argv[2], NULL, 10) : 1;

    mpz_t p, q, N;
    mpz_inits(p, q, N, NULL);

    /* Both factors get exactly bits/2 bits (top bit at bits/2 - 1), so N comes out
       at `bits` bits - the same CGBN container a Mersenne of that size would pick.
       Both are odd, which is also what CGBN requires of the modulus. */
    mpz_set_ui(p, seed);
    mpz_mul_2exp(p, p, bits / 4);
    mpz_setbit(p, bits / 2 - 1);
    mpz_nextprime(p, p);

    mpz_set_ui(q, seed + 7);
    mpz_mul_2exp(q, q, bits / 4);
    mpz_setbit(q, bits / 2 - 1);
    mpz_add_ui(q, q, 11);
    mpz_nextprime(q, q);

    mpz_mul(N, p, q);
    gmp_printf("%Zd\n", N);

    mpz_clears(p, q, N, NULL);
    return 0;
}
