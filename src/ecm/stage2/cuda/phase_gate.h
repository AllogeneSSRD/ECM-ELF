#pragma once

#include "host_wait.h"
#include "../schedule_policy.h"
#include "../../cuda/ntt/host_trace.h"
#include <charconv>
#include <optional>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#endif

namespace stage2_schedule {

enum class Work { setup, baby, f_tree, inverse, giant, g_tree_fold, descent, gcd, finalize, giant_loop };

inline unsigned gate_bit(Work work) {
    if(work==Work::giant_loop)return gate_bit(Work::giant)|gate_bit(Work::g_tree_fold);
    return work == Work::setup || work == Work::finalize ? 0 : 1u << (unsigned(work) - 1);
}

struct Options {
    bool trace = stage2_wait::option("ECM_STAGE2_PHASE_TRACE", false) || stage2_trace::enabled();
    unsigned mask = 0;
    std::string mutex_name;

    Options() {
        const char* value = std::getenv("ECM_STAGE2_PHASE_GATE");
        if (!value || !*value) return;
#ifndef _WIN32
        throw std::runtime_error("Stage2 phase gate is available on Windows");
#else
        const std::string_view spec(value);
        const auto colon = spec.find(':');
        if (colon == spec.npos)
            throw std::runtime_error("ECM_STAGE2_PHASE_GATE requires mask:group");
        const auto parsed = std::from_chars(spec.data(), spec.data() + colon, mask);
        const auto group = spec.substr(colon + 1);
        if (parsed.ec != std::errc{} || parsed.ptr != spec.data() + colon || mask > 127 ||
            group.empty() || group.size() > 48 ||
            group.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_") != group.npos)
            throw std::runtime_error("invalid Stage2 phase gate mask or group");
        mutex_name = "Local\\ECM-ELF.Stage2.Phase." + std::string(group);
#endif
    }
};

inline const Options& options() {
    static const Options value;
    return value;
}

inline bool gate_whole_giant_loop() {
    constexpr unsigned loop_mask=8|16;
    return (options().mask & loop_mask)==loop_mask && whole_giant_loop_requested();
}

// Experimental one-workspace lease. Acquire it before any phase mutex.
class WorkspaceLease {
#ifdef _WIN32
    HANDLE mutex_=nullptr;
    bool held_=false;
    static void mark(const char *event) {
        const auto time=std::chrono::steady_clock::now().time_since_epoch();
        stage2_log::print(stage2_log::debug,"stage2_workspace_lease: event=%s ns=%lld\n",event,
            (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(time).count());
    }
    void acquire() {
        const auto status=WaitForSingleObject(mutex_,INFINITE);
        if(status==WAIT_ABANDONED)ReleaseMutex(mutex_);
        if(status!=WAIT_OBJECT_0)throw std::runtime_error("cannot acquire Stage2 workspace lease");
        held_=true;
        mark("acquire");
    }
#endif
public:
    WorkspaceLease() {
#ifdef _WIN32
        if(!stage2_wait::option("ECM_STAGE2_FTREE_LEASE",false) || options().mutex_name.empty())return;
        mutex_=CreateMutexA(nullptr,FALSE,(options().mutex_name+".workspace").c_str());
        if(!mutex_)throw std::runtime_error("cannot create Stage2 workspace lease");
#endif
    }
    WorkspaceLease(const WorkspaceLease&)=delete;
    WorkspaceLease& operator=(const WorkspaceLease&)=delete;
    bool deferred() const {
#ifdef _WIN32
        return mutex_ && !held_;
#else
        return false;
#endif
    }
    void acquire_heavy() {
#ifdef _WIN32
        if(deferred())acquire();
#endif
    }
    ~WorkspaceLease() {
#ifdef _WIN32
        if(mutex_) {
            if(held_) {
                try {mark("release");}catch(const std::exception &) {}
                ReleaseMutex(mutex_);
            }
            CloseHandle(mutex_);
        }
#endif
    }

};

// One timeline per engine section, reused across giant chunks and G-tree
// batches. The experiment shares only the gate, never CUDA buffers or state.
class Phases {
    using Clock = std::chrono::steady_clock;
    const Options& options_ = options();
    std::optional<stage2_trace::Range> range_;
    const char* name_ = nullptr;
    Clock::time_point queued_, started_;
#ifdef _WIN32
    HANDLE mutex_ = nullptr;
    cudaEvent_t drained_ = nullptr;
    bool held_ = false;

    void acquire() {
        if (!mutex_) {
            mutex_ = CreateMutexA(nullptr, FALSE, options_.mutex_name.c_str());
            if (!mutex_) throw std::runtime_error("cannot open Stage2 phase gate");
            const auto status = cudaEventCreateWithFlags(&drained_, cudaEventBlockingSync | cudaEventDisableTiming);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        const stage2_trace::Range waiting("stage2.phase.wait");
        const auto status = WaitForSingleObject(mutex_, INFINITE);
        if (status == WAIT_ABANDONED) {
            ReleaseMutex(mutex_);
            throw std::runtime_error("Stage2 phase gate owner exited during GPU work");
        }
        if (status != WAIT_OBJECT_0) throw std::runtime_error("cannot wait for Stage2 phase gate");
        held_ = true;
    }
#endif

    static long long ns(Clock::duration value) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
    }

public:
    Phases() = default;
    Phases(const Phases&) = delete;
    Phases& operator=(const Phases&) = delete;

    void begin(Work work,bool use_gate=true) {
        finish();
        const bool gate = use_gate && (options_.mask & gate_bit(work)) != 0;
        if (!options_.trace && !gate) return;
        static constexpr const char* names[] = {
            "stage2.phase.setup", "stage2.phase.baby", "stage2.phase.f_tree",
            "stage2.phase.inverse", "stage2.phase.giant", "stage2.phase.g_tree_fold",
            "stage2.phase.descent", "stage2.phase.gcd", "stage2.phase.finalize", "stage2.phase.giant_loop"
        };
        queued_ = Clock::now();
#ifdef _WIN32
        if (gate) acquire();
#endif
        started_ = Clock::now();
        name_ = names[unsigned(work)];
        range_.emplace(name_);
    }

    void finish() noexcept {
        if (!name_) return;
        bool gated = false;
#ifdef _WIN32
        gated = held_;
        if (held_) {
            // Giant and polynomial work in these phases uses the legacy stream.
            // The next worker may enter only after the preceding device work.
            auto status = cudaEventRecord(drained_, nullptr);
            if (status == cudaSuccess) status = cudaEventSynchronize(drained_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "Stage2 phase fence failed: %s\n", cudaGetErrorString(status));
                std::exit(3);
            }
        }
#endif
        const auto end = Clock::now();
        range_.reset();
#ifdef _WIN32
        if (held_) {
            if (!ReleaseMutex(mutex_)) {
                std::fputs("Cannot release Stage2 phase gate\n", stderr);
                std::exit(3);
            }
            held_ = false;
        }
#endif
        try {
            stage2_log::print(stage2_log::debug,
                "stage2_phase_window: phase=%s start_ns=%lld end_ns=%lld wait_ns=%lld gated=%u\n",
                name_, ns(started_.time_since_epoch()), ns(end.time_since_epoch()),
                ns(started_ - queued_), unsigned(gated));
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Stage2 phase trace failed: %s\n", error.what());
            std::exit(3);
        }
        name_ = nullptr;
    }

    ~Phases() {
        finish();
#ifdef _WIN32
        if (drained_) cudaEventDestroy(drained_);
        if (mutex_) CloseHandle(mutex_);
#endif
    }
};

} // namespace stage2_schedule
