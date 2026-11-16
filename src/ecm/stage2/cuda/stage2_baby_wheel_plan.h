#pragma once
#include "stage2_baby_wheel_launch.h"
#include <vector>

namespace ecm_stage2::baby_wheel {

struct Plan {
    std::vector<Word> seeds;
    std::vector<Segment> segments;
    std::vector<unsigned> slots;
};

// Keep caller order in slots. Grouping by residue is only for seed/chain scheduling.
inline Plan make_plan(const std::vector<Word>& indices, unsigned wheel, unsigned length) {
    Plan plan;
    std::vector<std::vector<unsigned>> classes(wheel);
    for (size_t slot = 0; slot < indices.size(); ++slot)
        classes[indices[slot] % wheel].push_back(unsigned(slot));
    plan.seeds.push_back(wheel);
    for (const auto& group : classes) {
        size_t start = 0;
        while (start < group.size()) {
            const auto first = indices[group[start]];
            size_t end = start + 1;
            while (end < group.size() && (indices[group[end]] - first) / wheel < length) ++end;
            const unsigned span = unsigned((indices[group[end - 1]] - first) / wheel + 1);
            const Segment seg{unsigned(plan.slots.size()), unsigned(plan.seeds.size()), span};
            plan.segments.push_back(seg);
            plan.slots.resize(plan.slots.size() + span, skip);
            for (; start < end; ++start)
                plan.slots[seg.walk_begin + (indices[group[start]] - first) / wheel] = group[start];
            plan.seeds.push_back(first);
            if (span > 1) plan.seeds.push_back(first + wheel);
        }
    }
    return plan;
}

}
