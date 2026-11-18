#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ecm_cpu_tune {
using ThreadCounts = std::array<unsigned, 2>;

struct ThreadLayout {
    ThreadCounts threads{}, smt_extra{};
};

struct Key {
    std::string cpu, gw_version;
    ThreadLayout layout;
    bool mersenne = false;
    uint64_t width = 0, b1 = 0, b2 = 0;
};

struct Candidate {
    uint64_t d = 0, poly_size = 0, fft_length = 0, fft_exponent = 0, max_exponent = 0;
    unsigned f_tree_splits = 0;
    double planned_mib = 0;
    std::string fft_layout, isa;
};

// Candidates are already admitted for this modulus and budget. The static
// planner's winner comes first and breaks ties in exploration and selection.
uint64_t select(const std::filesystem::path&, const Key&, const std::vector<Candidate>&,
                uint64_t min_repeats, bool explore);
void observe(const std::filesystem::path&, const Key&, const Candidate&, double seconds,
             uint64_t sample_limit);
} // namespace ecm_cpu_tune
