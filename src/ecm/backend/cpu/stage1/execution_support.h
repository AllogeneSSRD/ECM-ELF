#pragma once
#include "ecm/core/stage1/selection.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace indicators {
class ProgressBar;
}

namespace ecm_cpu_stage1 {
extern std::mutex g_stage1_out_mutex;
extern std::atomic<uint32_t> g_stage1_done;
int to_ifma_field_mode(ecm_stage1::SimdReduction);
int driver_simd_isa_ok();
void apply_thread_affinity(const std::vector<unsigned>&, uint32_t thread);
void stage1_progress_set(double done, uint32_t total);

// Own the optional terminal widget; worker callbacks only borrow its lifetime.
class Stage1Progress {
    indicators::ProgressBar* bar_ = nullptr;

public:
    Stage1Progress();
    ~Stage1Progress();
    Stage1Progress(const Stage1Progress&) = delete;
    Stage1Progress& operator=(const Stage1Progress&) = delete;
    void finish() noexcept;
};
} // namespace ecm_cpu_stage1
