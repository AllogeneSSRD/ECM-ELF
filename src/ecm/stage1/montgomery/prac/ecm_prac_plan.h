#ifndef ECM_PRAC_PLAN_H
#define ECM_PRAC_PLAN_H
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <gmp.h>

#include "descriptor.h"
struct EcmPracCost {
    uint64_t dbl = 0, add = 0, work = 0;
};

struct EcmPracPlan {
    std::vector<EcmPracPrime> primes;
    EcmPracCost ladder, prac;
    uint64_t work = 0, identity = 0;
    bool cache_hit = false;
    bool cache_written = false;
    std::string status;
};
uint64_t ecm_prac_hash(const void *data, size_t bytes, uint64_t seed = 14695981039346656037ull);
uint64_t ecm_prac_hash_mpz(mpz_srcptr n);
unsigned ecm_prac_search_width(const char *option);
std::string ecm_prac_cache_summary(const std::string &file);
EcmPracPlan ecm_prac_build(uint32_t B1, uint32_t torsion, mpz_srcptr s,
                          const std::string &cache_dir, unsigned max_threads = 0,
                          unsigned search = 7);
#endif
