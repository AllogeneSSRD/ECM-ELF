#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

// GPU Stage1 stdout protocol, shared by Montgomery and Edwards.
namespace ecm_stage1_gpu_progress {
inline std::string line(double pct,uint64_t done,uint64_t delta,const char *unit,
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

// Average recent slice speeds, in work units/ms (same 50-sample window as CUDA Mont).
// Callers establish the baseline after preparation and exclude checkpoint I/O.
class SpeedWindow {
    double samples_[50]{};
    double sum_=0;
    unsigned count_=0,next_=0;
public:
    void add(uint64_t delta,double milliseconds) {
        if(!delta || milliseconds<=0 || !std::isfinite(milliseconds))return;
        const double speed=double(delta)/milliseconds;
        if(!std::isfinite(speed))return;
        if(count_<50)++count_;else sum_-=samples_[next_];
        samples_[next_]=speed;sum_+=speed;next_=(next_+1)%50;
    }
    double speed() const {return count_?sum_/count_:0;}
    double per_curve(uint64_t total,uint32_t curves) const {
        return speed()>0 && curves?double(total)/(speed()*curves*1000):0;
    }
    double remaining(uint64_t total,uint64_t done) const {
        return speed()>0 && done<total?double(total-done)/(speed()*1000):0;
    }
};
}
