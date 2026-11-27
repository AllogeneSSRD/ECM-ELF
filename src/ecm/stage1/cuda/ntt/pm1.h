#pragma once
#include "ecm/stage1/task.h"
#include <functional>

namespace ecm_mersenne_ntt {

// k includes the leading bit. Residues use ceil(p/32) little-endian words.
struct Pm1Progress {
    uint64_t k = 1;
    std::vector<uint32_t> residue;
    uint32_t errors = 0;
};

// The callback receives only verified endpoints. Return false to stop there.
Pm1Progress pm1_stage1(int device, unsigned p, const std::vector<uint32_t>& exponent,
                       Pm1Progress start, const ecm_stage1::Stage1NttOptions& options,
                       const std::function<bool(const Pm1Progress&)>& verified,
                       const std::vector<uint32_t>* extension_base = nullptr);
}
