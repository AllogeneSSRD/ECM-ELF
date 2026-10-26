#pragma once
#include <cstdint>
#include <string>

namespace ecm_stage1_gpu_progress {
// Shared Stage1 stdout protocol. Percent is clamped to [0,100].
std::string line(double pct,uint64_t done,uint64_t delta,const char *unit,
                 double per_curve_s,double elapsed_s,double remaining_s);

// Baseline starts after preparation; callers exclude checkpoint I/O.
class SpeedWindow {
    double samples_[50]{};
    double sum_=0;
    unsigned count_=0,next_=0;
public:
    void add(uint64_t delta,double milliseconds);
    double speed() const;
    double per_curve(uint64_t total,uint32_t curves) const;
    double remaining(uint64_t total,uint64_t done) const;
};
}

namespace ecm_stage1 {
long long stage1_now_ms();
bool stdout_is_tty_local();
}
