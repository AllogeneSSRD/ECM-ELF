#include "ecm/integer.h"
#include "ecm/stage1/scalar.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    const auto usage = [] {
        std::fputs("Usage: naf_work B1 W [W ...]\nB1: 2..5000000000. W: 0 (binary x-only) or 2..12.\n", stderr);
        return 2;
    };
    if (argc < 3) return usage();

    const auto parse = [](std::string_view text, auto& value) {
        const char* end = text.data() + text.size();
        const auto result = std::from_chars(text.data(), end, value);
        return result.ec == std::errc{} && result.ptr == end;
    };
    uint64_t b1 = 0;
    if (!parse(argv[1], b1) || b1 < 2 || b1 > 5000000000ULL) return usage();
    std::vector<unsigned> windows;
    for (int i = 2; i < argc; ++i) {
        unsigned window = 0;
        if (!parse(argv[i], window) || window == 1 || window > 12) return usage();
        windows.push_back(window);
    }

    ecm_edwards::Integer scalar;
    if (!ecm_build_lcm_exponent(scalar.v, b1, 48)) {
        std::fputs("Could not construct the Edwards Stage1 scalar.\n", stderr);
        return 1;
    }
    const size_t scalar_bits = mpz_sizeinbase(scalar.v, 2);
    if (scalar_bits > std::numeric_limits<uint32_t>::max() - 63) {
        std::fputs("Scalar exceeds the CUDA Stage1 progress range.\n", stderr);
        return 1;
    }

    // The x-only ladder starts with the leading bit already folded into the
    // initial points. Every remaining bit has a 6M + 4S step.
    const uint64_t binary_steps = scalar_bits - 1;
    const uint64_t binary_nonzero = mpz_popcount(scalar.v) - 1;
    const uint64_t baseline = 10 * binary_steps;
    std::puts("b1,naf,steps,nonzero,nonzero_percent,mad_per_step,total_mad,percent_of_w0");
    for (unsigned window : windows) {
        uint64_t steps = binary_steps, nonzero = binary_nonzero, work = baseline;
        if (window) {
            const EcmEdwardsNaf naf(scalar.v, window);
            steps = naf.bits;
            nonzero = naf.additions;
            work = 7 * steps + 8 * nonzero;
        }
        std::printf("%llu,%u,%llu,%llu,%.4f,%.4f,%llu,%.4f\n",
                    static_cast<unsigned long long>(b1), window,
                    static_cast<unsigned long long>(steps), static_cast<unsigned long long>(nonzero),
                    100.0 * nonzero / steps, double(work) / steps, static_cast<unsigned long long>(work),
                    100.0 * work / baseline);
    }
}
