#pragma once
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace stage2_schedule {
inline unsigned queue_phase_mask() {
    constexpr unsigned default_mask=57;
    const auto *value=std::getenv("ECM_STAGE2_QUEUE_PHASE_MASK");
    if(!value)return default_mask;
    const std::string_view text(value);
    unsigned mask=0;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),mask);
    if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || mask>127)
        throw std::runtime_error("ECM_STAGE2_QUEUE_PHASE_MASK must be an integer from 0 to 127");
    return mask;
}

inline bool whole_giant_loop_requested() {
    const auto *value=std::getenv("ECM_STAGE2_GIANT_GATE");
    return value && std::string_view(value)=="loop";
}
}
