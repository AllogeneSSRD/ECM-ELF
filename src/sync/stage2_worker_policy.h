#pragma once

#include "ecm/application/configuration/bit_width_policy.h"
#include "io/worker_list.h"

struct Stage2WorkerRule {
    std::vector<int> workers;
    std::size_t first_bit;
    std::optional<std::size_t> end_bit;
};

inline bool stage2_worker_slices(const std::string& spec) {
    return spec.find_first_of("[]") != spec.npos;
}

inline bool parse_stage2_worker_ids(std::string_view text, std::vector<int>& workers) {
    int ignored = 0;
    return ecm_workers::parse_list(std::string(text), 1000000, workers, ignored) &&
           !ignored && !workers.empty();
}

inline bool parse_stage2_worker_slices(const std::string& spec, std::vector<Stage2WorkerRule>& rules) {
    using namespace ecm_config::bit_width_policy;
    return parse_policy_ranges<std::vector<int>>(normalize_policy(spec), parse_stage2_worker_ids, rules, true);
}

inline std::vector<int> stage2_workers_for_bits(const std::vector<Stage2WorkerRule>& rules, std::size_t bits) {
    std::vector<int> workers;
    for (const auto& rule : rules)
        if (bits >= rule.first_bit && (!rule.end_bit || bits < *rule.end_bit))
            workers.insert(workers.end(), rule.workers.begin(), rule.workers.end());
    std::sort(workers.begin(), workers.end());
    workers.erase(std::unique(workers.begin(), workers.end()), workers.end());
    return workers;
}
