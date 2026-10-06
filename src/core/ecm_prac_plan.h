#ifndef ECM_PRAC_PLAN_H
#define ECM_PRAC_PLAN_H
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <gmp.h>

// Version 1: param0 DBL=5, projective DADD=6; Prime95 ten seeds/search=7.
// Persistent descriptors, not an expanded operation stream. All fields are u32.
struct EcmPracPrime { uint32_t p, d, repetitions, work; };
static_assert(sizeof(EcmPracPrime) == 16, "PRAC descriptor ABI");
struct EcmPracPlan {
    std::vector<EcmPracPrime> primes;
    uint64_t work = 0, identity = 0;
    std::string status;
};
uint64_t ecm_prac_hash(const void *data, size_t bytes, uint64_t seed = 14695981039346656037ull);
uint64_t ecm_prac_hash_mpz(mpz_srcptr n);
EcmPracPlan ecm_prac_build(uint32_t B1, uint32_t torsion, mpz_srcptr s,
                          const std::string &cache_dir);
#endif
