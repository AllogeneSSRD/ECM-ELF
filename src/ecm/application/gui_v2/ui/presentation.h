#pragma once
#include <cmath>
#include <limits>
namespace ecmgui_v2::ui {
inline double memory_occupancy(unsigned used_mib,unsigned total_mib) {
    return total_mib&&used_mib<=total_mib?100.*used_mib/total_mib:std::numeric_limits<double>::quiet_NaN();
}
inline double percent(double probability){return probability*100.;}
inline double probability(double percent){return percent/100.;}
}
