#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// A positive request preserves the caller's exact slice boundaries. Zero uses
// measured slices near 100 ms; only scheduling changes, never scalar position.
class EcmEdwardsSlice {
    uint32_t requested_,next_=32;
public:
    explicit EcmEdwardsSlice(uint32_t requested):requested_(requested){}
    uint32_t columns(uint32_t remaining) const {
        return (std::min)(remaining,requested_?requested_:next_);
    }
    void observe(uint32_t columns,double milliseconds) {
        if(requested_ || !columns || !std::isfinite(milliseconds) || milliseconds<=0)return;
        // Ignore the short final tail; limit growth while the device warms up.
        if(columns!=next_)return;
        const double estimate=columns*100.0/milliseconds;
        const double bounded=(std::clamp)(estimate,double((std::max)(16u,next_/4)),
                                                   double((std::min)(4096u,next_*4)));
        next_=uint32_t(bounded+0.5);
    }
};
