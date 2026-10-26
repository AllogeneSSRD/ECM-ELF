#pragma once
#include <gmp.h>
#include <cstdint>
#include <vector>

// s = torsion * product(p^floor(log_p B1), p <= B1); torsion 0 means 1.
// False for B1 > 5e9 or allocation failure; s remains at torsion on failure.
bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion);

// Width-w NAF, MSB-first progress columns. Nonzero and code bit planes are
// ordinary host values used to prepare CPU/GPU layouts; w is in [2,12].
struct EcmEdwardsNaf {
    unsigned window;
    uint32_t bits=0, additions=0;
    std::vector<uint32_t> nonzero,codes;
    static uint32_t read(const std::vector<uint32_t> &words,uint64_t bit,unsigned width);
    explicit EcmEdwardsNaf(const mpz_t scalar,unsigned w);
    int digit(uint32_t bit) const;
};
