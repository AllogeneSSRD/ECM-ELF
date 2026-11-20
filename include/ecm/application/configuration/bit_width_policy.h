#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ecm_config::bit_width_policy {

inline std::string normalize_policy(std::string_view value) {
    std::string normalized;
    while (!value.empty()) {
        if (value.substr(0, 3) == "\xEF\xBC\x9A") { // Full-width colon.
            normalized += ':';
            value.remove_prefix(3);
            continue;
        }
        if (value.substr(0, 3) == "\xE3\x80\x80") { // Ideographic space.
            value.remove_prefix(3);
            continue;
        }
        if (std::string_view(" \t\r\n\f\v").find(value.front()) == std::string_view::npos)
            normalized += value.front();
        value.remove_prefix(1);
    }
    return normalized;
}

template<class Algorithm, class Rule, class Parse>
bool parse_policy_ranges(std::string_view rest, Parse parse, std::vector<Rule>& rules,
                         bool allow_unsliced = false) {
    const auto parse_bit = [](std::string_view text, std::size_t& bits) {
        const auto end = text.data() + text.size();
        const auto parsed = std::from_chars(text.data(), end, bits);
        return parsed.ec == std::errc{} && parsed.ptr == end && bits > 0;
    };
    for (;;) {
        const auto comma = rest.find(',');
        const auto item = rest.substr(0, comma);
        const auto bracket = item.find('[');
        if (item.empty() || (bracket == item.npos ? !allow_unsliced : item.back() != ']'))
            return false;
        Algorithm algorithm{};
        if (!parse(item.substr(0, bracket), algorithm))
            return false;
        Rule rule{algorithm, 1, std::nullopt};
        const auto range = bracket == item.npos ? std::string_view{} :
                                                   item.substr(bracket + 1, item.size() - bracket - 2);
        if (!range.empty()) {
            const auto colon = range.find(':');
            if (colon == std::string_view::npos ||
                (colon && !parse_bit(range.substr(0, colon), rule.first_bit)))
                return false;
            const auto end = range.substr(colon + 1);
            if (!end.empty()) {
                std::size_t bits = 0;
                if (!parse_bit(end, bits) || bits <= rule.first_bit) return false;
                rule.end_bit = bits;
            }
        }
        rules.push_back(rule);
        if (comma == std::string_view::npos) break;
        rest.remove_prefix(comma + 1);
    }
    return true;
}

} // namespace
