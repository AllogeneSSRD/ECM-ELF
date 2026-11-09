#pragma once
#include <algorithm>

namespace ecm_mersenne_ntt {

struct Profile {
    int tile, outer;
    unsigned window;
    bool warp_tail;
    const char* source;
};

inline Profile default_profile(int log_length, unsigned requested) {
    return {std::min(log_length, requested == 2 ? 10 : 12), 4, 16, true,
            requested == 2 ? "split_tile" : requested == 1 ? "full_tile" : "portable"};
}

inline Profile laptop_4060_profile(int log_length, unsigned curves) {
    // Production device-event medians, B1=256. Keep only improvements of at
    // least 3% over full_tile. Each row applies to its measured batch size.
    struct Measured {
        int log_length;
        unsigned curves;
        int tile, outer;
        unsigned window;
        bool warp_tail;
    };
    constexpr Measured tuned[] = {
        {11, 1,  8, 4, 32, true},
        {12, 1,  8, 4, 16, true},
        {12, 4, 10, 4, 16, true},
        {13, 1, 10, 2, 32, true},
        {13, 4, 10, 4, 32, true},
    };
    for (const auto& row : tuned)
        if (log_length == row.log_length && curves == row.curves)
            return {row.tile, row.outer, row.window, row.warp_tail, "rtx4060_laptop"};

    auto profile = default_profile(log_length, 0);
    // The probe screened all batch sizes at B1=13 and confirmed the selected
    // tile at B1=41. Use these tile-only choices for the unmeasured batches.
    if (curves != 1 && curves != 4 && curves != 8 && log_length >= 11 && log_length <= 14) {
        constexpr int tiles[][8] = {
            { 8,  8, 11, 11, 11, 11, 11, 11},
            { 8, 10,  8, 10, 12, 10, 10, 10},
            {10, 12, 10, 12, 10, 12, 12, 12},
            {12, 10, 12, 12, 10, 10, 12, 10},
        };
        const auto tile = tiles[log_length - 11][curves - 1];
        if (tile != profile.tile) {
            profile.tile = tile;
            profile.source = "rtx4060_laptop_probe";
        }
    }
    return profile;
}

} // namespace ecm_mersenne_ntt
