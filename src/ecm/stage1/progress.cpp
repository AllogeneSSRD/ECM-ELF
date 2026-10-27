#include "ecm/stage1/progress.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace ecm_stage1_gpu_progress {
std::string line(double pct,uint64_t done,uint64_t delta,const char *unit,
                        double per_curve_s,double elapsed_s,double remaining_s) {
    pct=(std::max)(0.0,(std::min)(100.0,pct));
    const int filled=int(40*pct/100);
    char bar[41];
    for(int i=0;i<40;++i)bar[i]=i<filled?'=':' ';
    bar[40]='\0';
    if(filled>0 && filled<40)bar[filled-1]='>';
    char text[384];
    std::snprintf(text,sizeof(text),
        "GPU: [%s] %.1f%%  %llu, +%llu %s (~%.2f s/curve)  elapsed %.1fs  remaining %.1fs",
        bar,pct,static_cast<unsigned long long>(done),static_cast<unsigned long long>(delta),
        unit,per_curve_s,elapsed_s,remaining_s);
    return text;
}

namespace {
double configured_average_seconds=0;
double checked_seconds(double seconds) {
    if(!std::isfinite(seconds) || seconds<0 || seconds>60)
        throw std::invalid_argument("progress_average_seconds must be finite and within [0,60]");
    return seconds;
}
}

void set_average_seconds(double seconds) {configured_average_seconds=checked_seconds(seconds);}
double average_seconds() {return configured_average_seconds;}
SpeedWindow::SpeedWindow(double seconds):window_ms_(checked_seconds(seconds)*1000) {}

void SpeedWindow::add(double delta,double milliseconds) {
    if(delta<=0 || !std::isfinite(delta) || milliseconds<=0 || !std::isfinite(milliseconds))return;
    if(!std::isfinite(delta/milliseconds))return;
    if(window_ms_==0) {
        work_=delta;milliseconds_=milliseconds;
        return;
    }
    // A slice longer than the window represents the whole visible interval.
    if(milliseconds>=window_ms_) {
        samples_.clear();
        delta*=window_ms_/milliseconds;milliseconds=window_ms_;
        work_=milliseconds_=0;
    }
    samples_.push_back({delta,milliseconds});
    work_+=delta;milliseconds_+=milliseconds;
    while(milliseconds_>window_ms_ && !samples_.empty()) {
        auto& oldest=samples_.front();
        const double excess=milliseconds_-window_ms_;
        if(oldest.milliseconds<=excess) {
            work_-=oldest.work;milliseconds_-=oldest.milliseconds;
            samples_.pop_front();
        } else {
            const double removed=oldest.work*(excess/oldest.milliseconds);
            oldest.work-=removed;oldest.milliseconds-=excess;
            work_-=removed;milliseconds_=window_ms_;
        }
    }
}
double SpeedWindow::speed() const {return milliseconds_>0?work_/milliseconds_:0;}
double SpeedWindow::per_curve(uint64_t total,uint32_t curves) const {
    return speed()>0 && curves?double(total)/(speed()*curves*1000):0;
}
double SpeedWindow::remaining(uint64_t total,uint64_t done) const {
    return speed()>0 && done<total?double(total-done)/(speed()*1000):0;
}
}

namespace ecm_stage1 {
long long stage1_now_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool stdout_is_tty_local(void) {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

}
