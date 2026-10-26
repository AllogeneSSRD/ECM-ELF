#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <map>
#include <gmp.h>
#include "prac/domain.h"
namespace ecm_cuda_mont_host {
// Own one run's environment values, preserving unset versus explicit empty.
// Parsing/resolution keeps its existing scalar/tier boundaries; no process cache.
struct MontRequestedSpec {
    std::map<std::string,std::string> values;
    std::string default_plan_cache;
    const char *get(const char *key) const {
        const auto found=values.find(key);return found==values.end()?nullptr:found->second.c_str();
    }
};
MontRequestedSpec capture_mont_request(const char *algorithm=nullptr, const std::string& default_cache_dir={});
const char *stage1_algorithm(const MontRequestedSpec &);
struct PracPolicy {
    int mode;
    bool compact,outlined,single_add,single_compact;
    uint32_t requested_tpi;
};
PracPolicy read_prac_policy(const MontRequestedSpec &,bool);
const char *prac_constants_policy(const MontRequestedSpec &,bool,int &,uint32_t,uint32_t,mpz_srcptr,size_t);
std::string prac_plan_cache(const MontRequestedSpec &);
double prac_target_ms(const MontRequestedSpec &,bool);
double prac_sample_limit(const MontRequestedSpec &);
bool gpu_dump_enabled(const MontRequestedSpec &);
struct PracWindow {
    std::string position;
    uint64_t first = 0, count = 0, chunk = 0;
    uint32_t warmup = 2;
    double seconds = 6;
    bool dump = false;
    bool enabled() const { return !position.empty(); }
};
PracWindow prac_window_settings(const MontRequestedSpec &,bool,uint64_t);
}
