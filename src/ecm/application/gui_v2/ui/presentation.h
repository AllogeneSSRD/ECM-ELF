#pragma once
#include <cmath>
#include <limits>
#include <cstdio>
#include <string>
namespace ecmgui_v2::ui {
inline double memory_occupancy(unsigned used_mib,unsigned total_mib) {
    return total_mib&&used_mib<=total_mib?100.*used_mib/total_mib:std::numeric_limits<double>::quiet_NaN();
}
inline double power_value(double watts,double limit,bool valid=true) {
    // Legacy GUI's enforced-power-limit rule, applied per sample/device.
    // Missing values stay gaps; do not clamp spikes into apparently valid data.
    const bool excessive=std::isfinite(limit)&&limit>0&&watts>3.*limit;
    return valid&&std::isfinite(watts)&&watts>=0&&!excessive?watts:std::numeric_limits<double>::quiet_NaN();
}
inline std::string power_readout(double watts,double limit,bool valid=true) {
    const double value=power_value(watts,limit,valid);char current[48],maximum[48];
    if(std::isfinite(value))std::snprintf(current,sizeof(current),"%.1f",value);else std::snprintf(current,sizeof(current),"--");
    if(std::isfinite(limit)&&limit>0)std::snprintf(maximum,sizeof(maximum),"%.0f",limit);else std::snprintf(maximum,sizeof(maximum),"--");
    return std::string(current)+" / "+maximum+" W";
}
inline double percent(double probability){return probability*100.;}
inline double probability(double percent){return percent/100.;}
}
