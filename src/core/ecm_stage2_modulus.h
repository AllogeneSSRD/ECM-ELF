#pragma once
#include <gmp.h>
#include <cstddef>
#include <string>
#include "ecm_stage2_geometry.h"

namespace ecm_stage2 {
// Arithmetic residues are canonical modulo N (the carrier). Mathematical
// inversions, zero/unit predicates and factors belong to target(). The legacy
// member name N is retained for the polynomial engine's arithmetic-only sites.
struct ModulusContext {
    mpz_t N, target_N;
    size_t S=0, W=0;
    unsigned carrier_exponent=0;

    ModulusContext() { mpz_inits(N,target_N,nullptr); }
    ~ModulusContext() { mpz_clears(N,target_N,nullptr); }
    ModulusContext(const ModulusContext&)=delete;
    ModulusContext& operator=(const ModulusContext&)=delete;

    // Arithmetic fixtures may initialize N directly without a separate target.
    mpz_srcptr target() const { return mpz_sgn(target_N)>0 ? target_N : N; }
    size_t target_bits() const { return mpz_sizeinbase(target(),2); }
    bool lifted() const { return mpz_cmp(N,target())!=0; }
    bool target_zero(mpz_srcptr value) const {
        return mpz_sgn(value)==0 || (lifted() && mpz_divisible_p(value,target()));
    }
    bool configure(const char *text,int base,unsigned exponent,std::string &error) {
        if(!text || mpz_set_str(target_N,text,base)) { error="invalid target N";return false; }
        const auto bits=mpz_sizeinbase(target_N,2);
        if(mpz_cmp_ui(target_N,3)<=0 || !mpz_odd_p(target_N) || bits>max_input_bits) {
            error="target N must be odd, >3 and at most 16384 bits";return false;
        }
        if(exponent) {
            if(exponent<2 || exponent>max_input_bits || exponent<bits) {
                error="carrier exponent must cover target N and be at most 16384";return false;
            }
            mpz_set_ui(N,1);mpz_mul_2exp(N,N,exponent);mpz_sub_ui(N,N,1);
            if(!mpz_divisible_p(N,target_N)) {
                error="target N does not divide the requested Mersenne carrier";return false;
            }
        } else mpz_set(N,target_N);
        carrier_exponent=exponent;S=mpz_sizeinbase(N,2);W=(S+63)/64;
        return true;
    }
};
} // namespace ecm_stage2
