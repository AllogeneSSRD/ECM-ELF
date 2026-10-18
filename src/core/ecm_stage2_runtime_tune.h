#pragma once
#include "ecm_stage2_width_prediction.h"

namespace ecm_stage2 { namespace tune { namespace portable {
struct RuntimeCandidate {
    Shape shape;
    RuntimeStatistics statistics;
    unsigned priority=3; // carrier default, ordinary default, historical D, model D
};
struct RuntimeChoice {
    size_t index=0;
    bool available=false,probe=false;
};
// Candidates have already passed current arithmetic and live memory admission.
// Trial curves are useful Stage2 work, never an unreported duplicate benchmark.
inline RuntimeChoice runtime_choice(const std::vector<RuntimeCandidate> &candidates,bool explore=true) {
    RuntimeChoice out;unsigned priority=~0u;
    for(size_t i=0;i<candidates.size();++i) {
        const auto &c=candidates[i];
        if(explore && c.statistics.count<2 && c.priority<priority) {
            out={i,true,true};priority=c.priority;
        }
    }
    if(out.available)return out;
    Word latest=0;
    auto fallback=[](unsigned p){return p==1?0u:p+1;};
    for(size_t i=0;i<candidates.size();++i) {
        const auto &c=candidates[i];if(c.statistics.count<2)continue;
        if(!out.available || c.statistics.preferred_order>latest ||
           (!latest && fallback(c.priority)<fallback(candidates[out.index].priority))) {
            out={i,true,false};latest=c.statistics.preferred_order;
        }
    }
    if(!out.available)return out;
    // Keep the successful incumbent when the observed bands overlap. Default D
    // has no special privilege after a measured path has been established.
    for(size_t i=0;i<candidates.size();++i)if(candidates[i].statistics.count>=2 &&
        LocalSearch::improves(candidates[i].statistics.timing(),candidates[out.index].statistics.timing()))out.index=i;
    return out;
}
} } }
