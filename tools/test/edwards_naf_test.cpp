#include "ecm/stage1/scalar.h"
#include "ecm/stage1/edwards/cpu/ecm_edwards_cpu.h"
#include "../../src/ecm/curve/edwards.h"
#include "../../src/ecm/stage1/edwards/cuda/edwards_slice.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

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

static void check_mersenne_fold() {
    using ecm_edwards::Integer;
    for (unsigned bits : {5, 31, 61, 64, 65, 127, 4001, 65521, 262139}) {
        Integer modulus, input, actual, expected;
        mpz_setbit(modulus.v, bits);
        mpz_sub_ui(modulus.v, modulus.v, 1);
        ecm_edwards::MersenneConstruction worker(modulus.v);
        const auto check = [&] {
            for (int sign : {1, -1}) {
                mpz_mul_si(actual.v, input.v, sign);
                mpz_mod(expected.v, actual.v, modulus.v);
                worker.reduce(actual.v);
                if (mpz_cmp(actual.v, expected.v))
                    throw std::runtime_error("Mersenne construction reduction differs at width " +
                                             std::to_string(bits));
            }
        };
        check();
        mpz_sub_ui(input.v, modulus.v, 1);
        check();
        mpz_set(input.v, modulus.v);
        check();
        mpz_add_ui(input.v, modulus.v, 1);
        check();
        mpz_mul(input.v, modulus.v, modulus.v);
        check();
        mpz_sub_ui(input.v, input.v, 1);
        check();
        mpz_set_ui(input.v, 0);
        mpz_setbit(input.v, 3 * bits + 17);
        mpz_sub_ui(input.v, input.v, 1);
        check();
    }
}

struct ConstructionOutput {
    ecm_edwards::Integer d, x, y, a24, factor;
};

static std::array<unsigned, 3> compare_construction(mpz_srcptr n, const std::vector<uint64_t>& sigmas, unsigned carrier = 0) {
    using ecm_edwards::Setup;
    ecm_edwards::MersenneConstruction worker(n,carrier);
    ConstructionOutput generic, optimized;
    ecm_edwards::Integer chart, denominator, reciprocal, expected_chart;
    std::array<unsigned, 3> counts{};
    for (const uint64_t sigma : sigmas) {
        const auto expected = ecm_edwards::construct(generic.d.v, generic.x.v, generic.y.v, generic.a24.v,
                                                     generic.factor.v, sigma, n);
        const auto actual = ecm_edwards::construct(optimized.d.v, optimized.x.v, optimized.y.v,
                                                   optimized.a24.v, optimized.factor.v, sigma, n, &worker, nullptr, chart.v);
        if (expected == Setup::Ready) {
            mpz_ui_sub(denominator.v, 1, generic.y.v);
            mpz_set_si(expected_chart.v, -1);
            if (mpz_invert(reciprocal.v, denominator.v, n)) {
                mpz_add_ui(expected_chart.v, generic.y.v, 1);
                mpz_mul(expected_chart.v, expected_chart.v, reciprocal.v);
                mpz_mod(expected_chart.v, expected_chart.v, n);
            }
            if (mpz_cmp(chart.v, expected_chart.v))
                throw std::runtime_error("Batched Montgomery chart differs at sigma " + std::to_string(sigma));
        }
        if (actual != expected || mpz_cmp(optimized.factor.v, generic.factor.v) ||
            (actual == Setup::Ready &&
             (mpz_cmp(optimized.d.v, generic.d.v) || mpz_cmp(optimized.x.v, generic.x.v) ||
              mpz_cmp(optimized.y.v, generic.y.v) || mpz_cmp(optimized.a24.v, generic.a24.v)))) {
            throw std::runtime_error("Mersenne curve or factor differs at width " +
                                     std::to_string(mpz_sizeinbase(n, 2)) + ", sigma " + std::to_string(sigma));
        }
        if (actual == Setup::Factor &&
            (mpz_cmp_ui(optimized.factor.v, 1) <= 0 || mpz_cmp(optimized.factor.v, n) >= 0 ||
             !mpz_divisible_p(n, optimized.factor.v)))
            throw std::runtime_error("Mersenne construction reported an invalid factor");
        ++counts[static_cast<unsigned>(actual)];
    }
    return counts;
}

static void check_mersenne_construction() {
    check_mersenne_fold();
    std::vector<uint64_t> sigmas{UINT64_MAX, UINT64_MAX - 1, 0, UINT64_MAX - 1, uint64_t(1) << 63,
                                (uint64_t(1) << 63) - 1, uint64_t(1) << 63, 20260922, 20260922, 1};
    std::mt19937_64 rng(16319);
    for (unsigned i = 0; i < 32; ++i)
        sigmas.push_back(rng());
    for (uint64_t sigma = 20260896; sigma <= 20261152; ++sigma)
        sigmas.push_back(sigma);
    for (uint64_t sigma = 1; sigma < 128; ++sigma)
        sigmas.push_back(sigma);

    std::array<unsigned, 3> counts{};
    for (unsigned bits : {5, 6, 8, 11, 15, 21, 25, 61}) {
        ecm_edwards::Integer modulus;
        mpz_setbit(modulus.v, bits);
        mpz_sub_ui(modulus.v, modulus.v, 1);
        const auto result = compare_construction(modulus.v, sigmas);
        for (size_t i = 0; i < counts.size(); ++i)
            counts[i] += result[i];
    }
    for (unsigned bits : {12u, 8192u}) {
        ecm_edwards::Integer target;
        mpz_setbit(target.v,bits);
        mpz_sub_ui(target.v,target.v,1);
        mpz_divexact_ui(target.v,target.v,3);
        compare_construction(target.v,{26,27,28,UINT64_MAX,UINT64_MAX-1},bits);
    }
    // 255 divides 2^8192-1. Exercise the wide projective map cheaply,
    // including products that hide different nonunits in their denominators.
    ecm_edwards::Integer composite;
    mpz_set_ui(composite.v,255);
    compare_construction(composite.v,sigmas,8192);
    if (!counts[0] || !counts[1] || !counts[2])
        throw std::runtime_error("missing ready, factor or degenerate construction coverage");

    ecm_edwards::Integer modulus;
    mpz_setbit(modulus.v, 127);
    mpz_sub_ui(modulus.v, modulus.v, 1);
    auto reverse = sigmas;
    std::reverse(reverse.begin(), reverse.end());
    auto first = std::async(std::launch::async, [&] { return compare_construction(modulus.v, sigmas); });
    auto second = std::async(std::launch::async, [&] { return compare_construction(modulus.v, reverse); });
    if (first.get() != second.get())
        throw std::runtime_error("independent construction workers changed lane classification");

    mpz_set_ui(modulus.v, 0);
    mpz_setbit(modulus.v, 65521);
    mpz_sub_ui(modulus.v, modulus.v, 1);
    compare_construction(modulus.v, {20260922, 20260923, 20260924, UINT64_MAX, UINT64_MAX - 1, 0, 20260922});
    std::printf("PASS Mersenne Edwards construction and signed fold: ready=%u factors=%u degenerate=%u\n",
                counts[0], counts[1], counts[2]);
}

int main() {
    try {
        slices();
        cpu_cache();
        check_mersenne_construction();
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
