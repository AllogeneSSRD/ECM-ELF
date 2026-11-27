#pragma once
#include "ntt_length.h"

namespace ecm_stage1 {

struct NttSize {
    unsigned length, linear, weighted, edwards;
};

// End of the continuous production range for each transform family.
// Zero denotes a family absent from the automatic transform-size table.
inline constexpr NttSize ntt_sizes[] = {
    {2, 0, 62, 62}, {4, 0, 120, 122}, {8, 93, 240, 240},
    {16, 210, 464, 472}, {32, 450, 928, 928}, {64, 899, 1792, 1824},
    {128, 1827, 3584, 3584}, {256, 3556, 6912, 7040},
    {512, 7140, 13824, 13824}, {stage1_ntt_cubic_length, 0, 19968, 20481},
    {1024, 13797, 26624, 27136},
    {2048, 27621, 53248, 53248}, {2560, 0, 64000, 66048},
    {4096, 53222, 102400, 104448}, {5120, 0, 128000, 129536},
    {8192, 106470, 204800, 204800}, {10240, 0, 245760, 253952},
    {16384, 204775, 393216, 401408}, {20480, 0, 491520, 497665},
    {32768, 409575, 786432, 786432}, {40960, 0, 942080, 974848},
    {65536, 786408, 1507328, 1540096}, {81920, 0, 1884160, 1908736},
    {131072, 1572840, 3014656, 3014656}, {163840, 0, 3604480, 3735552},
    {262144, 3014633, 5242880, 5242880}, {524288, 5242880, 0, 0}
};

constexpr unsigned ntt_max_exp(unsigned length, bool weighted, bool native) {
    for (const auto& size : ntt_sizes)
        if (size.length == length)
            return weighted && length==stage1_ntt_max_length ? stage1_ntt_max_exponent :
                   weighted ? (native ? size.edwards : size.weighted) : size.linear;
    // Retired small radix-5 sizes can still occur in archived tune files.
    if (weighted && stage1_ntt_radix5(length)) {
        for (unsigned bits=31; bits; --bits) {
            const unsigned long long digit=(1ull<<bits)-1;
            if (digit*digit <= 0xffffffff00000000ull / (native ? length : 2*length))
                return length*bits;
        }
    }
    return 0;
}

} // namespace ecm_stage1
