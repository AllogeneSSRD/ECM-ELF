#pragma once

#include "../memory.h"
#include "../../cuda_runtime.h"

namespace ecm_cuda_stage1 {

constexpr size_t edwards_device_report_bytes = 32;

inline size_t edwards_device_bytes(uint32_t container_bits, uint32_t count,
                                   uint32_t scalar_bits, unsigned window) {
    if (!container_bits || container_bits % 32 || !count || !scalar_bits ||
        (window && (window < 2 || window > 12)))
        throw std::invalid_argument("invalid CUDA Edwards memory estimate inputs");
    const size_t points_per_curve = window ? 14 + 4 * (size_t(1) << (window - 2)) : 10;
    const size_t points = checked_size({count, container_bits / 8, points_per_curve});
    const auto scalar_bytes = window ? 8 * ((uint64_t(scalar_bits) + 32) / 32 + 1)
                                     : 4 * ((uint64_t(scalar_bits) + 31) / 32);
    if (points > (std::numeric_limits<size_t>::max)() - scalar_bytes - edwards_device_report_bytes)
        throw std::length_error("CUDA Edwards device memory estimate overflow");
    return points + size_t(scalar_bytes) + edwards_device_report_bytes;
}

inline size_t edwards_device_budget(size_t free_bytes) {
    return size_t(ecm_edwards_memory::usable_bytes(free_bytes, 16 * ecm_edwards_memory::mib));
}

inline uint64_t edwards_host_budget() {
    using namespace ecm_edwards_memory;
    const auto free = host_free_bytes();
    return cap_budget(free ? usable_bytes(*free, 32 * mib) : (std::numeric_limits<uint64_t>::max)(),
                      "ECM_EDWARDS_HOST_MEMORY_MIB");
}

inline std::optional<unsigned> edwards_device_window(uint32_t container_bits, uint32_t count,
                                                     uint32_t scalar_bits, unsigned requested_window,
                                                     bool can_lower, size_t budget) {
    for (unsigned w = requested_window;; --w) {
        if (edwards_device_bytes(container_bits, count, scalar_bits, w) <= budget)
            return w;
        if (!can_lower || w <= 2)
            return std::nullopt;
    }
}

// Peak additional bulk host buffers after the scalar and curve state already exist.
inline uint64_t edwards_runtime_host_bytes(uint32_t container_bits, uint32_t count,
                                           uint32_t scalar_bits, bool naf) {
    const uint64_t words = (uint64_t(scalar_bits) + 31) / 32;
    const uint64_t data = uint64_t(count) * (container_bits / 8);
    const uint64_t packed = 8 * ((uint64_t(scalar_bits) + 32) / 32 + 1);
    return naf ? packed + (std::max)(4 * words, 14 * data) : 4 * words + 10 * data;
}

} // namespace ecm_cuda_stage1
