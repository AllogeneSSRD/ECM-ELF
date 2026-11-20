#pragma once
#include <gmp.h>
#include <cstdint>
#include <vector>
#include "ecm/stage2_gw_abi.h"

namespace ecm_gwpoly {
inline constexpr uint64_t max_b2=1000000000000000ull;
inline constexpr int automatic_tree=-1, segmented_tree=-2;
inline constexpr unsigned max_tune_candidates=16;
struct Options {
    uint64_t d=0,exponent=0,memory_mib=512;
    unsigned threads=1;
    int (*stop)()=nullptr;
    bool check=false;
    ecm_gw_stage2_progress_fn progress=nullptr;
    void* progress_context=nullptr;
    int f_tree_splits=automatic_tree;
    unsigned tune_candidate_limit=0; // Zero uses the ABI 4 search.
    bool dense_d=false;
};
struct Stats : ecm_gw_stage2_stats {
    ecm_gw_stage2_phases phases{};
};
using TuneCandidate = ecm_gw_stage2_tune_plan;
// Baseline first, followed by admitted D and F-tree alternatives.
std::vector<TuneCandidate> tune_candidates(mpz_srcptr n,uint64_t b1,uint64_t b2,const Options&);
const char* cpu();
const char* identity();
const char* load_error();
unsigned stage2_abi();
// x and a24 are residues modulo the actual N; they are never modified. True
// means f is a proper divisor of N. Interruption and saturated GCD throw.
bool stage2(mpz_t f,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &options,Stats &stats);
void selftest(unsigned threads);
const char *version();
}
