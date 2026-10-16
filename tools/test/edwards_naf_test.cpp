#include "ecm_edwards_naf.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static unsigned cases=0,negative=0,cross_limb=0;
static void check(const mpz_t scalar) {
    mpz_t rebuilt,term;mpz_inits(rebuilt,term,nullptr);
    for(unsigned w=2;w<=12;++w) {
        EcmEdwardsNaf plan(scalar,w);mpz_set_ui(rebuilt,0);unsigned previous=0,seen=0;
        for(uint32_t i=0;i<plan.bits;++i) {
            const auto d=plan.digit(i);if(!d)continue;
            if(!(d&1) || std::abs(d)>=(1<<(w-1)) || (seen && i-previous<w))
                throw std::runtime_error("NAF digit range/spacing failed");
            mpz_set_si(term,d);mpz_mul_2exp(term,term,i);mpz_add(rebuilt,rebuilt,term);
            previous=i;++seen;if(d<0)++negative;if(i%32+w-1>32)++cross_limb;
        }
        if(mpz_cmp(rebuilt,scalar) || seen!=plan.additions || !plan.digit(plan.bits-1))
            throw std::runtime_error("NAF scalar reconstruction failed");
        ++cases;
    }
    mpz_clears(rebuilt,term,nullptr);
}
int main() {
    try {
        mpz_t scalar;mpz_init(scalar);
        for(unsigned k=1;k<4096;++k){mpz_set_ui(scalar,k);check(scalar);}
        for(unsigned bits: {31,32,33,63,64,65,127,128,129,1023,1024,1025,17408}) {
            mpz_set_ui(scalar,1);mpz_mul_2exp(scalar,scalar,bits);check(scalar);
            mpz_sub_ui(scalar,scalar,1);check(scalar);mpz_add_ui(scalar,scalar,2);check(scalar);
        }
        gmp_randstate_t rng;gmp_randinit_default(rng);gmp_randseed_ui(rng,20261016);
        for(unsigned i=0;i<100;++i){mpz_urandomb(scalar,rng,1+i*137);mpz_setbit(scalar,0);check(scalar);}
        gmp_randclear(rng);mpz_clear(scalar);
        if(!negative || !cross_limb)throw std::runtime_error("missing signed/cross-limb coverage");
        std::printf("PASS cases=%u negative_digits=%u cross_limb_codes=%u windows=2..12\n",cases,negative,cross_limb);
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
