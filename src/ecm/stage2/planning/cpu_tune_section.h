#pragma once

#include <string_view>

namespace ecm_stage2::tune {
inline constexpr std::string_view cpu_section = "[cpu_stage2]";
inline constexpr std::string_view cpu_host_prefix = "[cpu_stage2.host.";
inline constexpr std::string_view legacy_cpu_section = "[[cpu_stage2]]";

inline bool has_tune_header(std::string_view line, std::string_view header) {
    if (line.substr(0, header.size()) != header) return false;
    const auto tail = line.find_first_not_of(" \t\r", header.size());
    return tail == line.npos || line[tail] == '#';
}

inline bool is_cpu_host_section(std::string_view line) {
    if (line.substr(0, cpu_host_prefix.size()) != cpu_host_prefix) return false;
    const auto end = line.find(']', cpu_host_prefix.size());
    if (end == line.npos) return false;
    const auto id = line.substr(cpu_host_prefix.size(), end - cpu_host_prefix.size());
    return !id.empty() && id.find_first_not_of("0123456789") == id.npos &&
           has_tune_header(line, line.substr(0, end + 1));
}

inline bool is_cpu_tune_section(std::string_view line) {
    return has_tune_header(line, cpu_section) || is_cpu_host_section(line) ||
           has_tune_header(line, legacy_cpu_section);
}

inline bool is_tune_section(std::string_view line) {
    // Portable tune table names are bare words. Matrix lines begin with
    // quoted headers, numeric cells or nested arrays.
    if (line.size() < 3 || line.front() != '[') return false;
    if (line[1] == '[') line.remove_prefix(1);
    const char first = line[1];
    return (first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') || first == '_';
}
} // namespace ecm_stage2::tune
