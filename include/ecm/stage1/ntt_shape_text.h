#pragma once

#include <string>
#include <string_view>

namespace ecm_stage1 {

enum class NttPfaCode : char { None = '0', Direct = '1', Coalesced = '2', FusedOuter = '3', Unknown = '?' };

inline std::string ntt_shape_text(bool weighted, std::string_view tile, std::string_view forward,
                                  std::string_view inverse, NttPfaCode pfa) {
    std::string shape = weighted ? "1:" : "0:";
    shape += tile;
    shape += ':';
    shape += forward;
    shape += ':';
    shape += inverse;
    shape += ':';
    shape += static_cast<char>(pfa);
    return shape;
}

} // namespace ecm_stage1
