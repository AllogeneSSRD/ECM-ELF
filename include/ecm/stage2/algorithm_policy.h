#pragma once

#include "ecm/application/configuration/bit_width_policy.h"
#include <stdexcept>

namespace ecm_stage2 {

enum class Algorithm { Gwnum, CpuNtt, CpuPlain, Cpu, Gpu, Cuda, OpenCl };

inline const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
    case Algorithm::Gwnum: return "gwnum";
    case Algorithm::CpuNtt: return "cpu-ntt";
    case Algorithm::CpuPlain: return "cpu-plain";
    case Algorithm::Cpu: return "cpu";
    case Algorithm::Gpu: return "gpu";
    case Algorithm::Cuda: return "cuda";
    case Algorithm::OpenCl: return "opencl";
    }
    throw std::logic_error("invalid Stage2 algorithm");
}

struct AlgorithmRule {
    Algorithm algorithm;
    std::size_t first_bit;
    std::optional<std::size_t> end_bit;

    bool matches(std::size_t bits) const {
        return bits >= first_bit && (!end_bit || bits < *end_bit);
    }
};

inline std::vector<AlgorithmRule> parse_algorithm_policy(std::string_view text) {
    auto normalized = ecm_config::bit_width_policy::normalize_policy(text);
    for (char& ch : normalized)
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    const auto parse = [](std::string_view name, Algorithm& algorithm) {
        for (const auto value : {Algorithm::Gwnum, Algorithm::CpuNtt, Algorithm::CpuPlain,
                                 Algorithm::Cpu, Algorithm::Gpu, Algorithm::Cuda, Algorithm::OpenCl}) {
            if (name == algorithm_name(value)) {
                algorithm = value;
                return true;
            }
        }
        return false;
    };
    Algorithm single;
    if (parse(normalized, single)) return {{single, 1, std::nullopt}};
    std::vector<AlgorithmRule> rules;
    if (!ecm_config::bit_width_policy::parse_policy_ranges<Algorithm>(normalized, parse, rules))
        throw std::runtime_error("invalid stage2_algorithm: expected an algorithm or an ordered list such as gwnum[4096:], gpu[:16385], cpu[]");
    return rules;
}

} // namespace ecm_stage2
