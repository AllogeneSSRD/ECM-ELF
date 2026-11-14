#include "ecm/stage1/montgomery/prac/ecm_prac_plan.h"
#include "io/exponent_cache.h"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
uint32_t parse_uint(std::string_view text, const char* name, uint32_t low, uint32_t high) {
    uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < low || value > high)
        throw std::runtime_error(std::string("invalid ") + name + ": " + std::string(text));
    return value;
}

struct Scalar {
    Scalar() { mpz_init(value); }
    ~Scalar() { mpz_clear(value); }
    mpz_t value;
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--summary") {
            std::cout << ecm_prac_cache_summary(argv[2]) << '\n';
            return 0;
        }
        unsigned search = 7;
        int first = 1;
        if (argc >= 3 && std::string_view(argv[1]) == "--search") {
            search = ecm_prac_search_width(argv[2]);
            first += 2;
        }
        if (argc - first < 4) {
            std::cerr << "Usage: prepare_prac_cache [--search 1..50] <cache-dir> <1|12> <threads:1..32> <B1> [B1 ...]\n"
                         "       prepare_prac_cache --summary <prac-v2-file>\n";
            return 1;
        }
        const auto dir = std::filesystem::absolute(std::filesystem::u8path(argv[first]));
        const auto torsion = parse_uint(argv[first + 1], "lcm", 1, 12);
        if (torsion != 1 && torsion != 12)
            throw std::runtime_error("lcm must be 1 or 12");
        const auto threads = parse_uint(argv[first + 2], "threads", 1, 32);
        std::vector<uint32_t> bounds;
        for (int i = first + 3; i < argc; ++i)
            bounds.push_back(parse_uint(argv[i], "B1", 2, UINT32_MAX));

        std::filesystem::create_directories(dir);
        Scalar scalar;
        for (const auto bound : bounds) {
            std::cout << "B1=" << bound << " lcm=" << torsion << '\n' << std::flush;
            std::string detail;
            if (!ecm_build_lcm_exponent_cached(scalar.value, bound, torsion, dir.u8string(), &detail))
                throw std::runtime_error(detail);
            std::cout << "  exponent: " << detail << '\n' << std::flush;
            {
                const auto plan = ecm_prac_build(bound, torsion, scalar.value, dir.u8string(), threads, search);
                std::cout << "  PRAC: " << plan.status << " identity=" << plan.identity << '\n' << std::flush;
                if (plan.cache_hit && !plan.cache_written)
                    continue;
                if (!plan.cache_written)
                    throw std::runtime_error("could not write the PRAC cache entry to " + dir.u8string());
            }
            const auto check = ecm_prac_build(bound, torsion, scalar.value, dir.u8string(), threads, search);
            if (!check.cache_hit)
                throw std::runtime_error("could not reload the PRAC cache entry from " + dir.u8string());
            std::cout << "  verified: " << check.status << '\n' << std::flush;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
