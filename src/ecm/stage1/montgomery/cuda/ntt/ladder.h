#pragma once
#include <cstdint>
#include <vector>

namespace ecm_mersenne_ntt {
// Initial slots (slot-major, then curve, then little-endian uint32 limbs):
// X(P), Z(P), X(2P), Z(2P), a24, affine x(P). All canonical mod 2^p-1.
// Bits are MSB first, omitting the leading 1. Result contains both X/Z pairs.
struct Run {
    std::vector<uint32_t> points;
    std::vector<double> device_ms, wall_ms;
    double setup_seconds = 0, readback_seconds = 0;
    unsigned length = 0, digit_bits = 0, passes = 0, tile = 0;
};
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint8_t>& bits, unsigned window, unsigned samples);
} // namespace ecm_mersenne_ntt
