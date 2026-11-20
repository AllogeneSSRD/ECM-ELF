#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ecm_cpu_tune {
using ThreadCounts = std::array<unsigned, 2>;

struct ThreadLayout {
    ThreadCounts threads{}, smt_extra{};
};

struct CpuIdentity {
    std::string name, supported;
};

CpuIdentity cpu_identity(std::string_view description);

struct Key {
    CpuIdentity cpu;
    std::string gw_version;
    ThreadLayout layout;
    bool mersenne = false;
    uint64_t width = 0, b1 = 0, b2 = 0;
};

struct Candidate {
    uint64_t d = 0, poly_size = 0, fft_length = 0, fft_exponent = 0, max_exponent = 0;
    unsigned f_tree_splits = 0;
    double planned_mib = 0;
    std::string fft_layout, isa;
    uint64_t actual_b2 = 0; // Runtime candidate only; persisted in Key::b2.
};

// Candidates are already admitted for this modulus and budget. The static
// planner's winner comes first and breaks ties in exploration and selection.
std::optional<std::size_t> select(const std::filesystem::path&, const Key&, const std::vector<Candidate>&,
                                  uint64_t min_repeats, bool explore);
void observe(const std::filesystem::path&, const Key&, const Candidate&, double seconds,
             uint64_t sample_limit);
} // namespace ecm_cpu_tune
