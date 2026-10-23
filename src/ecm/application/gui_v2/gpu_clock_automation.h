#pragma once
#include "gpu_clocks.h"
#include "../gui/platform.h"
#include <ctime>

namespace ecmgui_v2 {
enum class ClockAutoReason {None,Startup,Hourly,Deviation};
inline const char* clock_auto_key(ClockAutoReason reason){
    return reason==ClockAutoReason::Startup?"gpu.clock_auto_startup":reason==ClockAutoReason::Hourly?"gpu.clock_auto_hourly":"gpu.clock_auto_deviation";
}
inline long long clock_local_hour(){
    const auto now=std::time(nullptr);std::tm local{};localtime_s(&local,&now);
    // Local-hour boundary expressed as epoch seconds, including fractional
    // timezone offsets and the two distinct hours during a DST fall-back.
    return static_cast<long long>(now)-local.tm_min*60-local.tm_sec;
}
struct ClockAutoPolicy {
    ClockRange range;
    bool enabled=false,startup=true;
    long long checked_hour=-1;
    unsigned long long previous_sample=0,outside_since=0;
    bool outside=false,fired=false;
    void reset_violation(){previous_sample=0;outside_since=0;outside=false;fired=false;}
    ClockAutoReason evaluate(unsigned long long now,long long hour,const ecmgui::GpuSample& sample,bool busy){
        if(!enabled||!range.valid()){reset_violation();return ClockAutoReason::None;}
        if(!busy&&(startup||hour!=checked_hour)){
            const auto reason=startup?ClockAutoReason::Startup:ClockAutoReason::Hourly;
            startup=false;checked_hour=hour;reset_violation();return reason;
        }
        if(!sample.valid||!sample.clock_sm_mhz||sample.ts_ms>now||now-sample.ts_ms>1500){reset_violation();return ClockAutoReason::None;}
        if(sample.ts_ms==previous_sample)return ClockAutoReason::None;
        if(previous_sample&&(sample.ts_ms<previous_sample||sample.ts_ms-previous_sample>1500))reset_violation();
        previous_sample=sample.ts_ms;
        if(sample.clock_sm_mhz>=unsigned(range.low)&&sample.clock_sm_mhz<=unsigned(range.high)){
            outside_since=0;outside=false;fired=false;return ClockAutoReason::None;
        }
        if(!outside){outside_since=sample.ts_ms;outside=true;}
        if(!busy&&!fired&&sample.ts_ms-outside_since>=5000){fired=true;return ClockAutoReason::Deviation;}
        return ClockAutoReason::None;
    }
};
}
