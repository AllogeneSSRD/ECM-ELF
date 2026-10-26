#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ecm_stage2 { namespace timing {
constexpr const char *contract="exclusive_engine_v1";
constexpr size_t count=10;
constexpr const char *names[count]={"shape","setup","baby","ftree","main_setup",
    "inverse_setup","giant_loop","descent","accum","finalize"};
struct PhaseTimes {
    std::array<double,count> seconds{};
    bool complete=false;
    double total()const {double sum=0;for(auto value:seconds)sum+=value;return sum;}
};
struct Boundaries {
    double shape=0,init_begin=0,baby_begin=0,ftree_begin=0,init_end=0;
    double main_begin=0,inverse_begin=0,loop_begin=0,loop_end=0,accum_begin=0,accum_end=0,main_end=0;
};
// These partitions cover the existing engine total. Planning, saved Stage1 and
// process startup remain outside it. No legacy nested timer enters this sum.
inline PhaseTimes partition(const Boundaries &b) {
    PhaseTimes result;
    const std::array<double,4> init={b.init_begin,b.baby_begin,b.ftree_begin,b.init_end};
    const std::array<double,7> main={b.main_begin,b.inverse_begin,b.loop_begin,b.loop_end,b.accum_begin,b.accum_end,b.main_end};
    auto ordered=[](const auto &v) {
        for(auto x:v)if(!std::isfinite(x) || x<0)return false;
        return std::is_sorted(v.begin(),v.end());
    };
    if(!std::isfinite(b.shape) || b.shape<0 || !ordered(init) || !ordered(main) || b.main_begin<b.init_end)return result;
    result.seconds={b.shape,b.baby_begin-b.init_begin,b.ftree_begin-b.baby_begin,b.init_end-b.ftree_begin,
        b.inverse_begin-b.main_begin,b.loop_begin-b.inverse_begin,b.loop_end-b.loop_begin,
        b.accum_begin-b.loop_end,b.accum_end-b.accum_begin,b.main_end-b.accum_end};
    const double expected=b.shape+(b.init_end-b.init_begin)+(b.main_end-b.main_begin);
    if(!std::isfinite(expected) || expected<=0 || std::abs(result.total()-expected)>1e-9*std::max(1.,expected))return PhaseTimes{};
    result.complete=true;return result;
}
} }
