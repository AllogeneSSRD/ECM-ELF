#pragma once
#include <algorithm>
#include <cstdint>
#include <numeric>

namespace ecm_gwpoly {
struct GiantRange {
    uint64_t original_first, first, last;
    uint64_t size() const { return first <= last ? last - first + 1 : 0; }
};

// Lift discarded witnesses by powers of the first odd m coprime to D.
// The upper grid is inclusive. Only relocate if it saves a whole G block.
inline GiantRange giant_range(uint64_t b1, uint64_t last, uint64_t d, uint64_t p, bool relocate=true) {
    const auto original = std::max<uint64_t>(1, b1 > d/2 ? (b1-d/2)/d : 1);
    GiantRange range{original, original, last};
    if (!relocate || last <= original) return range;
    uint64_t m = 3;
    while (std::gcd(m, d) != 1) m += 2;
    const auto suffix = (last + (m+1)/2) / m;
    if (suffix > original && (last-suffix)/p < (last-original)/p) range.first = suffix;
    return range;
}

// At a fixed wheel, fill only the last block already needed by the request.
// Moving the endpoint can move the relocated start as well. Each iteration
// advances by the remaining deficit and never crosses the target block count.
// Zero means that the first full block endpoint lies beyond the allowed bound.
inline uint64_t full_block_b2(uint64_t b1, uint64_t requested, uint64_t d, uint64_t p, uint64_t limit) {
    if (requested > limit) return 0;
    auto last = (requested + d/2) / d;
    auto tail = giant_range(b1,last,d,p).size() % p;
    if (!tail) return requested;
    const auto max_last = (limit + d/2) / d;
    do {
        if (p-tail > max_last-last) return 0;
        last += p-tail;
        tail = giant_range(b1,last,d,p).size() % p;
    } while (tail);
    // All integers through this endpoint still map to the same final centre.
    return std::min(limit, last*d + d/2 - 1);
}
} // namespace ecm_gwpoly
