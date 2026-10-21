#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <gmp.h>
#include "prac/domain.h"
namespace ecm_cuda_mont_host {
// Read at the original execution boundaries, rather than caching process state.
const char *stage1_algorithm(); // nullptr after the original validation diagnostic
struct PracPolicy {
    int mode;
    bool compact,outlined,single_add,single_compact;
    uint32_t requested_tpi;
};
PracPolicy read_prac_policy(bool);
const char *prac_constants_policy(bool,int &,uint32_t,uint32_t,mpz_srcptr,size_t);
std::string prac_plan_cache();
double prac_target_ms(bool);
double prac_sample_limit();
bool gpu_dump_enabled();
struct PracWindow {
    std::string position;
    uint64_t first = 0, count = 0, chunk = 0;
    uint32_t warmup = 2;
    double seconds = 6;
    bool dump = false;
    bool enabled() const { return !position.empty(); }
};
PracWindow prac_window_settings(bool,uint64_t);
}
