#pragma once
#include "ecm/stage1/task.h"
#include "ecm/io/save.h"

namespace ecm_stage1 {
// Borrowed inputs; exponent and returned naming data belong to the caller.
bool compute_batch_s(mpz_t, double, unsigned torsion = 1, const std::string &cache_dir = {});
uint64_t random_sigma_u64();
bool fixed_sigma_batch_exceeds_prime95_range(uint64_t, uint32_t);
ecm_save::Naming save_naming(const Stage1Options &);
} // namespace ecm_stage1
