#pragma once

#include "ecm/platform/cpu_affinity.h"
#include "runtime_tune.h"
#include <optional>
#include <set>

namespace ecm_cpu_tune {
struct LogicalCpu {
    unsigned id, core, efficiency_class;
};

// One logical CPU from each physical core first, then explicitly allowed SMT
// siblings. A homogeneous CPU uses the first slot of the P/E pair.
inline std::vector<unsigned> choose_cpus(std::vector<LogicalCpu> available,
                                         const std::vector<unsigned>& requested,
                                         unsigned count, unsigned highest, ThreadLayout& layout) {
    if (available.empty()) throw std::runtime_error("no available CPU Stage2 cores");
    if (!requested.empty()) {
        for (auto id : requested)
            if (std::none_of(available.begin(), available.end(), [id](const auto& cpu) { return cpu.id == id; }))
                throw std::runtime_error("CPU Stage2 logical CPU " + std::to_string(id) + " is unavailable");
        available.erase(std::remove_if(available.begin(), available.end(), [&](const auto& cpu) {
            return std::find(requested.begin(), requested.end(), cpu.id) == requested.end();
        }), available.end());
    }
    std::sort(available.begin(), available.end(), [](const auto& a, const auto& b) {
        if (a.efficiency_class != b.efficiency_class) return a.efficiency_class > b.efficiency_class;
        return a.id < b.id;
    });
    layout = {};
    std::vector<unsigned> selected;
    std::set<unsigned> physical;
    std::vector<LogicalCpu> siblings;
    const auto add = [&](const LogicalCpu& cpu, bool sibling) {
        const auto slot = cpu.efficiency_class == highest ? 0 : 1;
        selected.push_back(cpu.id);
        ++layout.threads[slot];
        if (sibling) ++layout.smt_extra[slot];
    };
    for (const auto& cpu : available) {
        if (selected.size() == count) break;
        if (physical.insert(cpu.core).second) add(cpu, false);
        else if (!requested.empty()) siblings.push_back(cpu);
    }
    for (const auto& cpu : siblings) {
        if (selected.size() == count) break;
        add(cpu, true);
    }
    return selected;
}

#if defined(_WIN32)
inline std::optional<ThreadLayout> bind_worker(const std::string& spec, uint64_t& threads) {
    if (ecm_config::trim(spec) == "none") return std::nullopt;
    const auto requested = ecm_config::affinity_indices(spec);
    try {
        const auto process = GetCurrentProcess();
        const auto groups = GetActiveProcessorGroupCount();
        std::vector<USHORT> allowed_groups(groups);
        USHORT allowed_count = groups;
        if (!GetProcessGroupAffinity(process, &allowed_count, allowed_groups.data()))
            throw std::runtime_error("cannot query process processor groups");
        allowed_groups.resize(allowed_count);
        DWORD_PTR allowed = 0, active = 0;
        if (allowed_count == 1 && !GetProcessAffinityMask(process, &allowed, &active))
            throw std::runtime_error("cannot query process affinity");

        ULONG set_count = 0;
        GetProcessDefaultCpuSets(process, nullptr, 0, &set_count);
        std::vector<ULONG> default_sets(set_count);
        if (set_count && !GetProcessDefaultCpuSets(process, default_sets.data(), set_count, &set_count))
            throw std::runtime_error("cannot query default CPU sets");
        default_sets.resize(set_count);
        ULONG bytes = 0;
        GetSystemCpuSetInformation(nullptr, 0, &bytes, process, 0);
        std::vector<unsigned char> storage(bytes);
        if (!bytes || !GetSystemCpuSetInformation(
                reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage.data()), bytes, &bytes, process, 0))
            throw std::runtime_error("cannot enumerate CPU sets");

        std::vector<LogicalCpu> available;
        unsigned highest = 0;
        for (ULONG at = 0; at < bytes;) {
            const auto& item = *reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(storage.data() + at);
            if (!item.Size || item.Size > bytes - at)
                throw std::runtime_error("invalid CPU set record");
            at += item.Size;
            if (item.Type != CpuSetInformation) continue;
            const auto& cpu = item.CpuSet;
            highest = (std::max)(highest, unsigned(cpu.EfficiencyClass));
            if ((cpu.Allocated && !cpu.AllocatedToTargetProcess) ||
                std::find(allowed_groups.begin(), allowed_groups.end(), cpu.Group) == allowed_groups.end() ||
                (allowed && !(allowed & (DWORD_PTR{1} << cpu.LogicalProcessorIndex))) ||
                (!default_sets.empty() && std::find(default_sets.begin(), default_sets.end(), cpu.Id) == default_sets.end()))
                continue;
            available.push_back({unsigned(cpu.Group) * 64 + cpu.LogicalProcessorIndex,
                                 unsigned(cpu.Group) * 256 + cpu.CoreIndex, cpu.EfficiencyClass});
        }
        ThreadLayout layout;
        const auto selected = choose_cpus(std::move(available), requested, static_cast<unsigned>(threads), highest, layout);
        std::string mask;
        for (auto cpu : selected) mask += (mask.empty() ? "" : ",") + std::to_string(cpu);
        ecm_platform::apply_worker_cpu_affinity(mask);
        if (selected.size() != threads) std::cerr<<"CPU Stage2: using "<<selected.size()
            <<" of "<<threads<<" requested threads in the available CPU pool\n";
        threads = selected.size();
        std::cout<<"CPU Stage2: P/E threads=["<<layout.threads[0]<<','<<layout.threads[1]
                 <<"] SMT extra=["<<layout.smt_extra[0]<<','<<layout.smt_extra[1]<<"]\n";
        return layout;
    } catch (const std::exception& e) {
        if (!requested.empty()) throw;
        threads = 1;
        std::cerr<<"CPU Stage2: "<<e.what()<<"; using one thread with tuning disabled\n";
        return std::nullopt;
    }
}
#elif defined(__linux__)
inline std::optional<ThreadLayout> bind_worker(const std::string& spec, uint64_t& threads) {
    const auto selected = ecm_platform::apply_worker_cpu_affinity(spec);
    if (selected && threads > selected) {
        std::cerr << "CPU Stage2: using " << selected << " of " << threads
                  << " requested threads in the configured CPU pool\n";
        threads = selected;
    }
    return std::nullopt;
}
#endif
} // namespace ecm_cpu_tune
