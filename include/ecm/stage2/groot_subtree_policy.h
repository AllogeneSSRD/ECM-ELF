#pragma once

#include "ecm/application/configuration/bit_width_policy.h"
#include <cstddef>
#include <stdexcept>

namespace ecm_stage2 {

struct GrootSubtreeRule {
    int mode;
    std::size_t first_bit;
    std::optional<std::size_t> end_bit;
};

inline std::vector<GrootSubtreeRule> parse_groot_subtree_policy(std::string_view value) {
    using namespace ecm_config::bit_width_policy;
    std::vector<GrootSubtreeRule> rules;
    const auto parse=[](std::string_view mode,int &out) {
        if(mode=="auto" || mode=="-1")out=-1;
        else if(mode=="0" || mode=="1")out=mode.front()-'0';
        else return false;
        return true;
    };
    if(!parse_policy_ranges<int>(normalize_policy(value),parse,rules,true))
        throw std::runtime_error("invalid G-tree subtree policy; expected auto, -1, 0, 1 or a bit-slice list");
    return rules;
}

inline int groot_subtree_mode(const std::vector<GrootSubtreeRule> &rules,std::size_t target_bits) {
    for(const auto &rule:rules)
        if(target_bits>=rule.first_bit && (!rule.end_bit || target_bits<*rule.end_bit))return rule.mode;
    return -1;
}

inline constexpr bool groot_subtree_supported(unsigned arithmetic_bits,bool mersenne) {
    return mersenne && arithmetic_bits>=2 && arithmetic_bits<=72*64;
}

// A worker can run curves with different widths without changing its process environment.
inline thread_local bool groot_subtree_enabled=false;
struct GrootSubtreeScope {
    bool previous;
    explicit GrootSubtreeScope(bool enabled):previous(groot_subtree_enabled) {groot_subtree_enabled=enabled;}
    ~GrootSubtreeScope() {groot_subtree_enabled=previous;}
};

} // namespace ecm_stage2
