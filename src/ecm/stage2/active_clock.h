#pragma once

#include <chrono>

namespace ecm_stage2::timing {

// These waits run on the curve worker's host thread. CUDA completion waits
// still count because the device is doing work for the curve.
using Clock = std::chrono::steady_clock;
inline thread_local Clock::duration lock_wait{};

inline void record_lock_wait(Clock::time_point started) {
    lock_wait += Clock::now() - started;
}

inline Clock::time_point active_now() {
    return Clock::now() - lock_wait;
}

} // namespace ecm_stage2::timing
