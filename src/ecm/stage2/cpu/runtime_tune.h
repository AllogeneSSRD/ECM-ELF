#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ecm_cpu_tune {
struct Key {
    std::string cpu, build;
    uint64_t bits = 0, exponent = 0, b1 = 0, b2 = 0, memory_mib = 0, threads = 0;
    std::string affinity;
};

struct Candidate {
    uint64_t d = 0, poly_size = 0, fft_length = 0;
};

struct Sample {
    double seconds = 0, init_seconds = 0, main_seconds = 0;
    double baby_seconds = 0, ftree_seconds = 0, reciprocal_seconds = 0, giant_seconds = 0;
    double gtree_seconds = 0, fold_seconds = 0, scale_seconds = 0, descent_seconds = 0;
    double leaf_gcd_seconds = 0, estimated_memory_mib = 0;
    uint64_t blocks = 0, giants = 0;
    std::string fft_description, isa;
};

// Candidates are already admitted for this modulus and budget. The static
// planner's winner comes first and breaks ties in exploration and selection.
uint64_t select(const std::filesystem::path&, const Key&, const std::vector<Candidate>&,
                uint64_t min_repeats, bool explore);
void observe(const std::filesystem::path&, const Key&, const Candidate&, const Sample&,
             uint64_t sample_limit);
} // namespace ecm_cpu_tune
