#pragma once
#include "width_prediction.h"
#include "runtime_cache.h"

namespace ecm_stage2 { namespace tune { namespace portable {
struct RuntimeCandidate {
    Shape shape;
    RuntimeStatistics statistics;
    unsigned priority=3; // carrier default, ordinary default, historical D, model D
    bool probe_allowed=true;
};
inline std::string runtime_pause_scope(const std::string &context,const Shape &reference,
                                      const std::vector<RuntimeCandidate> &candidates) {
    std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)<<context<<'\n';
    auto shape=[&](const Shape &s){for(auto value:s.key())out<<value<<',';out<<'\n';};
    shape(reference);
    for(const auto &c:candidates)if(c.shape.carrier==reference.carrier) {
        shape(c.shape);out<<c.priority<<','<<c.probe_allowed<<','<<c.statistics.count<<','
            <<c.statistics.seconds<<','<<c.statistics.stddev<<','<<c.statistics.preferred_order<<'\n';
    }
    return out.str();
}
struct RuntimeChoice {
    size_t index=0;
    bool available=false,probe=false;
};
struct RuntimeSearchAttempts {
    size_t steps=0,shapes=0;unsigned admissions=0;
    explicit RuntimeSearchAttempts(size_t count):shapes(count){}
    bool next(){return admissions<16 && steps++<=shapes;}
};
struct RuntimeSearchBudget {
    static constexpr unsigned initial_candidates=4,extended_candidates=8,regions=3;
    static constexpr double max_extra_seconds=120;
    unsigned candidates=initial_candidates;
    double spent=0,remaining=max_extra_seconds,reference=0;
    bool permits(double upper,Word samples,Word queue_remaining=0)const {
        if(queue_remaining && queue_remaining<4)return false;
        const double needed=double(samples<2?2-samples:0)*std::max(0.,upper-reference);
        return reference>0 && remaining>0 && needed<=remaining;
    }
};
inline RuntimeSearchBudget runtime_search_budget(const std::vector<DSearchNode> &nodes,Word baseline,Word sample_limit) {
    RuntimeSearchBudget out;Word trusted_count=0;
    for(const auto &n:nodes)if(n.samples>=2 && n.seconds>0) {
        trusted_count=std::max(trusted_count,n.samples);
        if(!out.reference || n.seconds<out.reference)out.reference=n.seconds;
    }
    if(trusted_count>=std::min(Word(8),sample_limit))out.candidates=RuntimeSearchBudget::extended_candidates;
    for(const auto &n:nodes)if(n.d!=baseline && n.samples && n.seconds>out.reference)
        out.spent+=n.samples*(n.seconds-out.reference);
    out.remaining=std::max(0.,RuntimeSearchBudget::max_extra_seconds-out.spent);return out;
}
// Candidates have already passed current arithmetic and live memory admission.
// Trial curves are useful Stage2 work, never an unreported duplicate benchmark.
inline RuntimeChoice runtime_choice(const std::vector<RuntimeCandidate> &candidates,bool explore=true) {
    RuntimeChoice out;unsigned priority=~0u;
    for(size_t i=0;i<candidates.size();++i) {
        const auto &c=candidates[i];
        if(explore && c.probe_allowed && c.statistics.count<2 && c.priority<priority) {
            out={i,true,true};priority=c.priority;
        }
    }
    if(out.available)return out;
    // Repeated, admitted production observations are ranked by their mean.
    // Dispersion is diagnostic; it must not veto a measured improvement.
    // Preserve the actual incumbent only on an exactly equal mean, then use a
    // full-shape tie break so candidate iteration order never changes the route.
    auto better=[&](size_t a,size_t b) {
        const auto &x=candidates[a],&y=candidates[b];
        if(x.statistics.seconds!=y.statistics.seconds)return x.statistics.seconds<y.statistics.seconds;
        if(x.statistics.preferred_order!=y.statistics.preferred_order)
            return x.statistics.preferred_order>y.statistics.preferred_order;
        return x.shape.key()<y.shape.key();
    };
    for(size_t i=0;i<candidates.size();++i) {
        const auto &s=candidates[i].statistics;
        if(s.count<2 || !(s.seconds>0) || !std::isfinite(s.seconds))continue;
        if(!out.available || better(i,out.index))out={i,true,false};
    }
    return out;
}
} } }
