#pragma once
#include <cstdint>
#include <deque>
#include <string>

namespace ecm_stage1_gpu_progress {
// Shared Stage1 stdout protocol. Percent is clamped to [0,100].
std::string line(double pct,uint64_t done,uint64_t delta,const char *unit,
                 double per_curve_s,double elapsed_s,double remaining_s);

// Display policy is set once after worker configuration is resolved.
void set_average_seconds(double seconds);
double average_seconds();

// Baseline starts after preparation; callers exclude checkpoint I/O. Samples
// carry their measured duration: a time window never assumes a slice cadence.
class SpeedWindow {
    struct Sample { double work, milliseconds; };
    std::deque<Sample> samples_;
    double window_ms_, work_=0, milliseconds_=0;
public:
    explicit SpeedWindow(double seconds=average_seconds());
    void add(double delta,double milliseconds);
    double speed() const;
    double per_curve(uint64_t total,uint32_t curves) const;
    double remaining(uint64_t total,uint64_t done) const;
};
}

namespace ecm_stage1 {
long long stage1_now_ms();
bool stdout_is_tty_local();
}
