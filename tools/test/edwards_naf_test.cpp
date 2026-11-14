#include "ecm/stage1/scalar.h"
#include "ecm/stage1/edwards/cpu/ecm_edwards_cpu.h"
#include "../../src/ecm/stage1/edwards/cuda/edwards_slice.h"
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>

static unsigned cases=0,negative=0,cross_limb=0;
static void slices() {
    EcmEdwardsSlice fixed(113);
    for(double ms:{0.0,0.01,10000.0}) {
        fixed.observe(113,ms);
        if(fixed.columns(1000)!=113 || fixed.columns(7)!=7)
            throw std::runtime_error("fixed slice contract changed");
    }
    EcmEdwardsSlice adaptive(0);
    // Change simulated device cost by two orders of magnitude. Scheduling
    // must converge without skipped/duplicated columns or zero-size launches.
    for(double cost:{0.1,5.0,0.05,0.5}) {
        uint32_t done=0;
        while(done<100000) {
            const auto size=adaptive.columns(100000-done);
            if(!size || size>4096 || size>100000-done)
                throw std::runtime_error("adaptive slice made invalid progress");
            adaptive.observe(size,cost*size);done+=size;
        }
        const auto ms=cost*adaptive.columns(100000);
        if(ms<80 || ms>120)throw std::runtime_error("adaptive slice failed to converge");
    }
    std::printf("PASS fixed/adaptive slices: progress, tails, changing device cost\n");
}
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
static void cpu_cache() {
    constexpr uint64_t b1 = 1000;
    constexpr int w = 8;
    mpz_t scalar, next_scalar, n, x, z, factor, reference_x, reference_z, reference_factor;
    mpz_inits(scalar, next_scalar, n, x, z, factor, reference_x, reference_z, reference_factor, nullptr);
    if (!ecm_build_lcm_exponent(scalar, b1, 48) ||
        !ecm_build_lcm_exponent(next_scalar, 2 * b1, 48))
        throw std::runtime_error("fixture scalar construction failed");
    if (edwards_cached_naf(b1, w)) throw std::runtime_error("unexpected warm NAF cache");
    auto a = std::async(std::launch::async, [&] { return edwards_prepare_naf(scalar, b1, w); });
    auto b = std::async(std::launch::async, [&] { return edwards_prepare_naf(scalar, b1, w); });
    const auto plan = a.get();
    if (!plan || plan != b.get() || plan != edwards_cached_naf(b1, w))
        throw std::runtime_error("identical CPU NAF requests did not share their plan");
    if (plan->packed() || plan->bytes() > edwards_naf_bytes(mpz_sizeinbase(scalar, 2)))
        throw std::runtime_error("CPU NAF allocation exceeds the preflight estimate");
    const auto packed = edwards_prepare_naf(scalar, 0, w, true);
    if (!packed->packed() || packed->size() != plan->size() ||
        packed->bytes() > edwards_naf_bytes(mpz_sizeinbase(scalar, 2), true))
        throw std::runtime_error("CPU NAF packed allocation or shape differs from its estimate");
    for (size_t bit = 0; bit < packed->size(); ++bit)
        if ((*packed)[bit] != (*plan)[bit]) throw std::runtime_error("CPU NAF packed digit differs");
    const auto changed_window = edwards_prepare_naf(scalar, b1, w + 1);
    if (changed_window == plan || edwards_cached_naf(b1, w) ||
        edwards_cached_naf(b1, w + 1) != changed_window)
        throw std::runtime_error("CPU NAF window key or single-entry eviction failed");
    const auto changed_bound = edwards_prepare_naf(next_scalar, 2 * b1, w + 1);
    if (changed_bound == changed_window || edwards_cached_naf(b1, w + 1))
        throw std::runtime_error("CPU NAF bound key failed");
    if (edwards_prepare_naf(scalar, 0, w) == plan || edwards_cached_naf(0, w))
        throw std::runtime_error("arbitrary scalar unexpectedly used the B1 cache");

    mpz_setbit(n, 127);
    mpz_sub_ui(n, n, 1);
    const auto run = [&](int window, const EdwardsNafPlan* digits, mpz_t f, mpz_t qx, mpz_t qz) {
        return edwards_stage1_curve_progress(f, qx, qz, n, 17, scalar, 0,
                                             nullptr, nullptr, nullptr, digits, window);
    };
    const int rc = run(w, nullptr, reference_factor, reference_x, reference_z);
    if (rc != 0 || run(w, plan.get(), factor, x, z) != rc ||
        mpz_cmp(factor, reference_factor) || mpz_cmp(x, reference_x) || mpz_cmp(z, reference_z))
        throw std::runtime_error("CPU NAF shared and local plans disagree");
    if (run(w, packed.get(), factor, x, z) != rc ||
        mpz_cmp(factor, reference_factor) || mpz_cmp(x, reference_x) || mpz_cmp(z, reference_z))
        throw std::runtime_error("CPU NAF packed and dense endpoints differ");
    if (run(w + 1, changed_window.get(), factor, x, z) != rc)
        throw std::runtime_error("CPU NAF alternate window result differs");
    mpz_mul(x, x, reference_z);
    mpz_mul(z, z, reference_x);
    mpz_sub(x, x, z);
    if (!mpz_divisible_p(x, n)) throw std::runtime_error("CPU NAF alternate window endpoint differs");
    mpz_clears(scalar, next_scalar, n, x, z, factor, reference_x, reference_z, reference_factor, nullptr);
    std::printf("PASS CPU NAF shared identity, key changes, eviction and packed/dense endpoints\n");
}
int main() {
    try {
        slices();
        cpu_cache();
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
