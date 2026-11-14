#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__unix__)
#include <unistd.h>
#endif

namespace ecm_edwards_memory {
constexpr uint64_t mib = 1024 * 1024;
constexpr uint64_t mb = 1000 * 1000;

inline uint64_t configured_budget(uint64_t available, unsigned limit_mb) {
    return limit_mb ? (std::min)(available, uint64_t(limit_mb) * mb) : available;
}

inline uint64_t scalar_preparation_bytes(uint64_t bound) {
    // Budget the GMP product tree, multiplication scratch and cache I/O before
    // the exact bit count is known. Stage1 supports B1 up to 5e9.
    return 4 * mib + 2 * bound;
}

inline uint64_t batch_host_bytes(uint32_t curves, size_t modulus_bits) {
    // Curve outcomes and serialized checkpoint/result buffers can coexist.
    return 2 * mib + uint64_t(curves) * (512 + 16 * ((uint64_t(modulus_bits) + 7) / 8));
}

inline std::optional<uint64_t> host_free_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX state{};
    state.dwLength = sizeof(state);
    if (GlobalMemoryStatusEx(&state))
        return (std::min)(state.ullAvailPhys, state.ullAvailPageFile);
#elif defined(__linux__)
    std::ifstream status("/proc/meminfo");
    std::string line;
    while (std::getline(status, line)) {
        constexpr std::string_view label = "MemAvailable:";
        if (line.compare(0, label.size(), label) == 0) {
            const auto first = line.find_first_not_of(" \t", label.size());
            uint64_t kib = 0;
            if (first != std::string::npos &&
                std::from_chars(line.data() + first, line.data() + line.size(), kib).ec == std::errc{})
                return kib * 1024;
            break;
        }
    }
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages >= 0 && page_size > 0)
        return uint64_t(pages) * uint64_t(page_size);
#elif defined(__APPLE__)
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    vm_size_t page_size = 0;
    const auto rc = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
    const auto page_rc = host_page_size(host, &page_size);
    mach_port_deallocate(mach_task_self(), host);
    if (rc == KERN_SUCCESS && page_rc == KERN_SUCCESS)
        return (uint64_t(stats.free_count) + stats.inactive_count) * page_size;
#endif
    return std::nullopt;
}

inline uint64_t usable_bytes(uint64_t free_bytes, uint64_t minimum_reserve) {
    const auto reserve = (std::max)(minimum_reserve, free_bytes / 5);
    return free_bytes > reserve ? free_bytes - reserve : 0;
}

// Limits are optional upper bounds for reproducing memory pressure without allocating it.
inline uint64_t cap_budget(uint64_t budget, const char* limit_name) {
    const char* text = std::getenv(limit_name);
    if (!text || !*text)
        return budget;
    const std::string_view value(text);
    uint64_t limit = 0;
    const auto end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, limit);
    if (parsed.ec != std::errc{} || parsed.ptr != end || limit > std::numeric_limits<uint64_t>::max() / mib)
        return budget;
    return (std::min)(budget, limit * mib);
}
} // namespace ecm_edwards_memory
