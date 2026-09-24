/* ecm_stage1_exp.cpp -- see ecm_stage1_exp.h for why this is a product tree. */
#include "ecm_stage1_exp.h"

#include <vector>

/* mpz_set_ui() takes `unsigned long`, which is 32 bits on Windows, so a 64-bit
   prime power must be assembled from its halves (same trap as mont_set_sigma:
   sigma = 2^62 silently became 0 through mpz_set_ui). */
static void set_u64(mpz_t r, uint64_t v)
{
    mpz_set_ui(r, (unsigned long)(v >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

/* Binary counter height: one slot per bit of the prime-power count.  2^40 prime
   powers is far beyond any B1 this program accepts (5e9 -> 2.3e8 primes). */
#define EXP_SLOTS 40

bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion)
{
    mpz_set_ui(s, torsion ? (unsigned long)torsion : 1);
    if (B1 < 2) return true;
    if (B1 > 5000000000ull) return false;

    std::vector<char> sieve;
    try {
        sieve.assign((size_t)B1 + 1u, 1);
    } catch (...) {
        return false;                       /* out of memory: caller reports it */
    }
    sieve[0] = 0;
    if (B1 >= 1) sieve[1] = 0;
    /* mark from p*p: the smaller multiples were already marked by smaller primes */
    for (uint64_t p = 2; p * p <= B1; ++p) {
        if (!sieve[(size_t)p]) continue;
        for (uint64_t q = p * p; q <= B1; q += p) sieve[(size_t)q] = 0;
    }

    mpz_t acc[EXP_SLOTS];
    mpz_t ppz;
    for (unsigned j = 0; j < EXP_SLOTS; ++j) mpz_init_set_ui(acc[j], 1);
    mpz_init(ppz);

    for (uint64_t p = 2; p <= B1; ++p) {
        if (!sieve[(size_t)p]) continue;

        uint64_t pp = p;                    /* highest power of p that is <= B1 */
        while (pp <= B1 / p) pp *= p;
        set_u64(ppz, pp);

        /* carry the new prime power up until an empty slot accepts it */
        unsigned j = 0;
        while (j + 1 < EXP_SLOTS && mpz_cmp_ui(acc[j], 1) != 0) {
            mpz_mul(ppz, ppz, acc[j]);
            mpz_set_ui(acc[j], 1);
            ++j;
        }
        mpz_set(acc[j], ppz);
    }

    /* combine the slots, largest partial product first so the final
       multiplications stay balanced as well */
    for (int j = EXP_SLOTS - 1; j >= 0; --j) {
        if (mpz_cmp_ui(acc[j], 1) != 0) mpz_mul(s, s, acc[j]);
        mpz_clear(acc[j]);
    }
    mpz_clear(ppz);
    return true;
}
