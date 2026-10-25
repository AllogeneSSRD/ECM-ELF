#pragma once
#include <cstdint>
#include <stdexcept>

namespace ecm_cuda_mont_host {
// Ladder layout belongs to the parametrization, not to the chosen bit tier.
// Resident/PRAC have their own layout contract and must not borrow this one.
struct LadderConstruction {
    uint32_t param, words, first_x, second_x;
    bool full_sigma;
};

inline LadderConstruction ladder_construction(uint32_t param) {
    switch (param) {
    case 0:
        return {0, 7, 3, 5, true};
    case 2:
        return {2, 7, 3, 5, false};
    case 3:
        return {3, 5, 1, 3, false};
    default:
        throw std::invalid_argument("Montgomery param must be 0, 2 or 3");
    }
}
} // namespace ecm_cuda_mont_host
