#ifndef ECM_PRAC_PLAN_H
#define ECM_PRAC_PLAN_H
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <gmp.h>

// Version 1: param0 DBL=5, projective DADD=6; Prime95 ten seeds/search=7.
// Persistent descriptors are shared by the CPU and CUDA executors.

#include "descriptor.h"
struct EcmPracPlan {
    std::vector<EcmPracPrime> primes;
    uint64_t work = 0, identity = 0;
    bool cache_hit = false;
    bool cache_written = false;
    std::string status;
};
uint64_t ecm_prac_hash(const void *data, size_t bytes, uint64_t seed = 14695981039346656037ull);
uint64_t ecm_prac_hash_mpz(mpz_srcptr n);
EcmPracPlan ecm_prac_build(uint32_t B1, uint32_t torsion, mpz_srcptr s,
                          const std::string &cache_dir, unsigned max_threads = 0);
#endif
