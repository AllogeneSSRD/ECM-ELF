#pragma once
#include <cstdint>

// Host-only policy, independent of CUDA/Windows so the exact retry/error paths
// can be tested with a deterministic clock and event backend.
namespace stage2_wait {
struct Counters {
    uint64_t calls=0, ready=0, pending=0, queries=0, sleeps=0, sleep_us=0, fallbacks=0;
    double seconds=0;
};

template<class Backend>
auto complete_event(Backend &backend, bool adaptive, Counters &stats) -> decltype(backend.query()) {
    ++stats.calls;
    if(!adaptive)return backend.synchronize();
    auto status=backend.query();++stats.queries;
    if(status==backend.success) {++stats.ready;return backend.synchronize();}
    if(status!=backend.not_ready)return status;
    ++stats.pending;
    unsigned delay_us=32;
    do {
        if(!backend.pause(delay_us)) {++stats.fallbacks;return backend.synchronize();}
        ++stats.sleeps;stats.sleep_us+=delay_us;
        status=backend.query();++stats.queries;
        if(delay_us<256)delay_us*=2;
    } while(status==backend.not_ready);
    // Retain the real CUDA completion/visibility boundary. Prediction and sleep
    // never authorize consuming or reusing a buffer. Errors are returned intact.
    return status==backend.success ? backend.synchronize() : status;
}
}
