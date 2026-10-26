#include "sigma.h"

#include <gmp.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ecm_stage1 {

bool sigma_batch_fits(uint64_t first, uint32_t curves, uint64_t minimum,
                      uint64_t maximum) noexcept {
    return curves != 0 && first >= minimum && first <= maximum &&
           uint64_t(curves - 1) <= maximum - first;
}

bool montgomery_sigma_batch_fits(uint64_t first, uint32_t curves, int param) noexcept {
    switch (param) {
    case 0:
        return sigma_batch_fits(first, curves);
    case 2:
        return sigma_batch_fits(first, curves, 2, UINT32_MAX);
    case 3:
        return sigma_batch_fits(first, curves, 1, UINT32_MAX);
    default:
        return false;
    }
}

uint64_t random_sigma53() {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned)time(nullptr));
        seeded = true;
    }
    uint64_t sigma;
    do {
        uint32_t hi = 0, lo = 0;
        sigma = ((uint64_t)(rand() & 0x1F)) << 48;
        sigma += ((uint64_t)(rand() & 0xFFFF)) << 32;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        const unsigned __int64 tsc = __rdtsc();
        hi = (uint32_t)(tsc >> 32);
        lo = (uint32_t)tsc;
#endif
        sigma += (uint64_t)(lo ^ hi ^ ((uint32_t)rand() << 16));
    } while (sigma <= 5);
    return sigma;
}

uint32_t random_sigma32(uint32_t curves, uint32_t minimum) {
    if (!sigma_batch_fits(minimum, curves, 1, UINT32_MAX))
        throw std::invalid_argument("sigma: empty uint32 batch range");

    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    unsigned long seed = (unsigned long)time(nullptr);
#ifdef _WIN32
    seed ^= (unsigned long)_getpid() * 0x9e3779b9ul;
#else
    seed ^= (unsigned long)getpid() * 0x9e3779b9ul;
#endif
    seed ^= (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    gmp_randseed_ui(rng, seed);

    // The inclusive last starting value is 2^32 - curves. Subtracting
    // minimum - 1 restricts the sampler without retrying invalid PARAM2 seeds.
    mpz_t range, value;
    mpz_init_set_ui(range, UINT32_MAX - (curves - 1) - minimum + 1);
    mpz_init(value);
    mpz_urandomm(value, rng, range);
    const uint32_t sigma = uint32_t(mpz_get_ui(value)) + minimum;
    mpz_clears(range, value, nullptr);
    gmp_randclear(rng);
    return sigma;
}

uint64_t select_curve_sigma(uint64_t saved, bool fixed, uint64_t first, uint32_t index) {
    if (saved)
        return saved;
    if (!fixed)
        return random_sigma53();
    if (uint64_t(index) > UINT64_MAX - first)
        throw std::invalid_argument("sigma + curve index exceeds 2^64-1");
    return first + index;
}

uint64_t select_batch_sigma(uint64_t saved, bool fixed, uint64_t first,
                          uint32_t curves, int param) {
    uint64_t sigma = saved;
    if (!sigma) {
        if (fixed)
            sigma = first;
        else if (param == 0)
            sigma = random_sigma53();
        else if (param == 2 || param == 3)
            sigma = random_sigma32(curves, param == 2 ? 2 : 1);
        else
            throw std::invalid_argument("sigma: unsupported Montgomery param");
    }
    if (!montgomery_sigma_batch_fits(sigma, curves, param))
        throw std::invalid_argument("sigma: invalid Montgomery batch range");
    return sigma;
}

} // namespace ecm_stage1

extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves) {
    if (curves == 0 || curves == UINT32_MAX)
        return 2u;
    return ecm_stage1::random_sigma32(curves);
}
