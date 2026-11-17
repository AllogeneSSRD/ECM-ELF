#pragma once
#include <gmp.h>
#include <cstdint>
#include "ecm/stage2_gw_abi.h"

namespace ecm_gwpoly {
inline constexpr uint64_t max_b2=1000000000000000ull;
struct Options {
    uint64_t d=0,exponent=0,memory_mib=512;
    unsigned threads=1;
    int (*stop)()=nullptr;
    bool check=false;
};
using Stats = ecm_gw_stage2_stats;
const char* cpu();
const char* identity();
const char* load_error();
// x and a24 are residues modulo the actual N; they are never modified. True
// means f is a proper divisor of N. Interruption and saturated GCD throw.
bool stage2(mpz_t f,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &options,Stats &stats);
void selftest(unsigned threads);
const char *version();
}
