#pragma once
#include "ecm/service/stage1/task.h"

namespace ecm_stage1 {
// Existing host preparation/naming policy. Inputs are borrowed for the call;
// scalar output and returned names are owned by the caller.
bool compute_batch_s(mpz_t,double,unsigned torsion=1,const std::string& cache_dir={});
uint64_t random_sigma_u64();
std::string stage1_save_prefix(const Stage1Options &);
std::string stage1_prefixed_save(const std::string &,const std::string &);
std::string mont_format_save_name(const std::string &,mpz_srcptr,double,const Stage1Options *options=nullptr);
bool fixed_sigma_batch_exceeds_prime95_range(uint64_t,uint32_t);
bool ensure_dir(const std::string &);
// Shared clock/console operations keep existing CPU/GPU callback timing.
long long stage1_now_ms();
bool stdout_is_tty_local();
}
