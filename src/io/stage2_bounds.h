#pragma once

#include "config_format.h"

namespace ecm_stage2 {
struct Bounds {
    uint64_t lower = 0;
    uint64_t upper = 0;
};

inline Bounds parse_bounds(const std::string& text, const char* label) {
    const auto value = ecm_config::trim(text);
    for (auto dash = value.find('-', 1); dash != value.npos; dash = value.find('-', dash + 1)) {
        if (value[dash - 1] == 'e' || value[dash - 1] == 'E')
            continue;
        const auto lower = ecm_config::unsigned_integer(value.substr(0, dash), label);
        const auto upper = ecm_config::unsigned_integer(value.substr(dash + 1), label);
        if (lower < 2 || upper <= lower)
            throw std::runtime_error(std::string(label) + " interval requires 2 <= lower < upper");
        return {lower, upper};
    }
    return {0, ecm_config::unsigned_integer(value, label)};
}
}
