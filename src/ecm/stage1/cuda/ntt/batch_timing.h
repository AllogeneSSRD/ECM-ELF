#pragma once

#include "ecm/platform/logging.h"
#include <chrono>

namespace ecm_mersenne_ntt {

inline void log_batch_span(const char* phase, std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now()) {
    using namespace std::chrono;
    ecm_ts_fprintf(stdout,"stage1_ntt_span: phase=%s start_us=%lld end_us=%lld\n",phase,
        static_cast<long long>(duration_cast<microseconds>(start.time_since_epoch()).count()),
        static_cast<long long>(duration_cast<microseconds>(end.time_since_epoch()).count()));
}

} // namespace ecm_mersenne_ntt
