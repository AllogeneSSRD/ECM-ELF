#pragma once
#include <gmp.h>
#include <cstdint>

namespace ecm_gwpoly {
inline constexpr uint64_t max_b2=1000000000000000ull;
struct Options {
    uint64_t d=0,exponent=0,memory_mib=512;
    unsigned threads=1;
    int (*stop)()=nullptr;
    bool check=false;
};
struct Stats {
    uint64_t d=0,poly_size=0,fft_length=0,blocks=0,giants=0,polymults=0,checked_values=0;
    double init_seconds=0,main_seconds=0,max_roundoff=0,estimated_memory_mib=0;
    double point_seconds=0,tree_seconds=0,fold_seconds=0,descent_seconds=0;
};
// x and a24 are residues modulo the actual N; they are never modified. True
// means f is a proper divisor of N. Interruption and saturated GCD throw.
bool stage2(mpz_t f,const mpz_t n,const mpz_t x,const mpz_t a24,
            uint64_t b1,uint64_t b2,const Options &options,Stats &stats);
void selftest(unsigned threads);
const char *version();
}
