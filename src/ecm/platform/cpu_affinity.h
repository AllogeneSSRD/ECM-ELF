#pragma once

#include "io/config_format.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace ecm_platform {

// Apply the worker's CPU pool before creating computation threads. Legacy
// Stage1 per-thread affinity uses its own ordering and is resolved separately.
inline unsigned apply_worker_cpu_affinity(const std::string& spec, bool log = true) {
    auto cpus = ecm_config::affinity_indices(spec);
    if (cpus.empty()) return 0;
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    const auto unavailable = [](unsigned cpu) {
        return std::runtime_error("shared.cpu_affinity: logical CPU " + std::to_string(cpu) +
                                  " is unavailable to this worker");
    };
    const char* method;
#if defined(_WIN32)
    const auto process = GetCurrentProcess();
    if (GetActiveProcessorGroupCount() == 1) {
        DWORD_PTR allowed = 0, active = 0, mask = 0;
        if (!GetProcessAffinityMask(process, &allowed, &active))
            throw std::runtime_error("shared.cpu_affinity: cannot query process affinity");
        for (const auto cpu : cpus) {
            if (cpu >= sizeof(mask) * 8 || !(allowed & (DWORD_PTR{1} << cpu)))
                throw unavailable(cpu);
            mask |= DWORD_PTR{1} << cpu;
        }
        if (!SetProcessAffinityMask(process, mask))
            throw std::runtime_error("shared.cpu_affinity: cannot set process affinity (Windows error " +
                                     std::to_string(GetLastError()) + ")");
        method = "process_mask";
    } else {
        // CPU indices are group*64 + processor index. CPU Set IDs are separate
        // Windows identifiers and must be discovered before setting the pool.
        ULONG bytes = 0;
        GetSystemCpuSetInformation(nullptr, 0, &bytes, process, 0);
        std::vector<unsigned char> storage(bytes);
        if (!bytes || !GetSystemCpuSetInformation(
                reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage.data()), bytes, &bytes, process, 0))
            throw std::runtime_error("shared.cpu_affinity: cannot enumerate Windows CPU Sets");
        std::map<unsigned, ULONG> ids;
        for (ULONG at = 0; at < bytes;) {
            const auto& item = *reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(storage.data() + at);
            if (!item.Size || item.Size > bytes - at)
                throw std::runtime_error("shared.cpu_affinity: invalid Windows CPU Set record");
            if (item.Type == CpuSetInformation && (!item.CpuSet.Allocated || item.CpuSet.AllocatedToTargetProcess))
                ids.emplace(unsigned(item.CpuSet.Group) * 64 + item.CpuSet.LogicalProcessorIndex, item.CpuSet.Id);
            at += item.Size;
        }
        std::vector<ULONG> selected;
        for (const auto cpu : cpus) {
            const auto id = ids.find(cpu);
            if (id == ids.end()) throw unavailable(cpu);
            selected.push_back(id->second);
        }
        if (!SetProcessDefaultCpuSets(process, selected.data(), static_cast<ULONG>(selected.size())))
            throw std::runtime_error("shared.cpu_affinity: cannot set default CPU Sets (Windows error " +
                                     std::to_string(GetLastError()) + ")");
        method = "process_default_cpu_sets";
    }
#elif defined(__linux__)
    cpu_set_t allowed, mask;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("shared.cpu_affinity: cannot query scheduler affinity");
    CPU_ZERO(&mask);
    for (const auto cpu : cpus) {
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) throw unavailable(cpu);
        CPU_SET(cpu, &mask);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask))
        throw std::runtime_error("shared.cpu_affinity: cannot set scheduler affinity");
    method = "inherited_thread_mask";
#else
    throw std::runtime_error("shared.cpu_affinity is unavailable on this platform");
#endif
    if (log) {
        std::cout << "worker_cpu_affinity: method=" << method << " logical_cpus=";
        for (size_t i = 0; i < cpus.size(); ++i) std::cout << (i ? "," : "") << cpus[i];
        std::cout << std::endl;
    }
    return static_cast<unsigned>(cpus.size());
}

} // namespace ecm_platform
