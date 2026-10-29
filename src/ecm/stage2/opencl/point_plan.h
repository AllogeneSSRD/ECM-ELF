#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace ecm_opencl::stage2 {
// Feedback from useful ladder work only: no warmup or repeated points. The
// first command keeps the qualified seed size. Promotion fills at most one
// group per compute unit; an observed slow command suppresses later growth.
// This is a soft promotion limit, not a watchdog or free-memory guarantee.
struct PointPlan {
    std::size_t seed=256,group=32,units=0,ceiling=256,samples=0;
    double seconds_per_bit=0;
    bool enabled=false;
    static constexpr double target_seconds=1.0;
    static constexpr double margin=1.15;

    PointPlan(std::size_t initial,std::size_t local,std::size_t cu,bool active)
        :seed(initial),group(local),units(cu),enabled(active&&local&&cu) {
        ceiling=enabled?std::max(seed,std::min(std::size_t(4096),units*group)):seed;
    }
    std::size_t chunk(unsigned bits)const {
        return enabled&&samples&&seconds_per_bit*bits*margin<=target_seconds?ceiling:seed;
    }
    void observe(double seconds,unsigned bits,std::size_t count) {
        // Partial work-groups underestimate full-lane private-memory pressure.
        if(!enabled||!bits||count<group||!(seconds>0)||!std::isfinite(seconds))return;
        const auto groups=(count+group-1)/group,waves=(groups+units-1)/units;
        seconds_per_bit=std::max(seconds_per_bit,seconds/(bits*waves));++samples;
    }
    std::size_t admit(std::size_t wanted,std::size_t remaining,std::size_t free_points)const {
        const auto count=std::min(wanted,remaining);
        // Promotion must not fail an allocation that the seed size could fit.
        // Existing seed/manual requests retain Runtime's hard-budget checks.
        return enabled?std::max(std::min(seed,remaining),std::min(count,free_points)):count;
    }
    std::uint64_t window(std::uint64_t g_batch,std::uint64_t cap)const {
        if(!enabled)return g_batch;
        cap=std::min(cap,std::uint64_t(65536));
        const auto factor=ceiling/std::gcd(std::uint64_t(ceiling),g_batch);
        // Prefer a common multiple so neither G batches nor full point waves
        // acquire tails. Avoid overflowing lcm when a tight budget cannot fit.
        if(g_batch<=cap/factor)return g_batch*factor;
        return std::max(g_batch,(cap/g_batch)*g_batch);
    }
};
} // namespace ecm_opencl::stage2
