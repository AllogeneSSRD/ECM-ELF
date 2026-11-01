#pragma once
#include "width_prediction.h"
#include "runtime_bounds.h"
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
    for(const auto &c:candidates)if(c.shape.carrier==reference.carrier && c.shape.groot_balanced==reference.groot_balanced) {
        shape(c.shape);out<<c.priority<<','<<c.probe_allowed<<','<<c.statistics.count<<','
            <<c.statistics.seconds<<','<<c.statistics.stddev<<','<<c.statistics.preferred_order<<'\n';
    }
    return out.str();
}
struct RuntimeChoice {
    size_t index=0;
    bool available=false,probe=false;
};
struct BabyBackendChoice {bool available=false;unsigned mode=0;double distance=0;};
// A paired comparison within one backend-width/point-batch region is reusable
// across neighboring D. Never compare an unpaired fast D against another D.
inline BabyBackendChoice qualified_baby_backend(const Estimator &engine,const Shape &reference,Word minimum=2) {
    std::map<Shape::Key,std::array<const Fields*,2>> pairs;
    const auto cap=optional_uint(engine.condition,"env_ladder_cap",8192);
    const auto bucket=point_batch_bucket(d_catalog::degree(reference.d),cap);
    for(const auto &row:engine.profile.samples)if(measured(row) && uint(row,"repeats")>=minimum &&
        matches(engine.profile.conditions.at(uint(row,"condition")),engine.condition,Ignore::parse(""))) {
        auto s=engine.shape(row);
        if(s.target_bits!=reference.target_bits || s.arithmetic_bits!=reference.arithmetic_bits || s.carrier!=reference.carrier ||
           s.mersenne!=reference.mersenne || s.b2!=reference.b2 || s.giant_coop!=reference.giant_coop ||
           s.chain_block!=reference.chain_block || (s.baby_coop!=0 && s.baby_coop!=32) ||
           point_batch_bucket(d_catalog::degree(s.d),cap)!=bucket)continue;
        const auto mode=s.baby_coop;s.baby_coop=0;auto key=s.key();key[3]=0;
        auto &pair=pairs[key];const auto at=mode==32?1:0;
        if(!pair[at] || uint(row,"repeats")>uint(*pair[at],"repeats"))pair[at]=&row;
    }
    BabyBackendChoice result;
    for(const auto &entry:pairs)if(entry.second[0] && entry.second[1]) {
        const auto &a=*entry.second[0],&b=*entry.second[1];
        const double distance=std::abs(std::log2(double(d_catalog::degree(uint(a,"d")))/d_catalog::degree(reference.d)));
        if(!result.available || distance<result.distance) {
            result.available=true;result.distance=distance;
            result.mode=real(b,"mean_seconds")+optional_real(b,"point_check_seconds")<
                real(a,"mean_seconds")+optional_real(a,"point_check_seconds")?32:0;
        }
    }
    return result;
}
// One alternative at the fastest repeated scalar incumbent per arithmetic
// family. Keep D/layout/chain fixed, rather than multiplying the D catalogue.
inline std::vector<Shape> runtime_baby_trials(const std::vector<RuntimeCandidate> &all,Word minimum=2) {
    std::map<unsigned,const RuntimeCandidate*> best;
    for(const auto &c:all)if(!c.shape.baby_coop && cooperative_point_width(c.shape.arithmetic_bits) && d_catalog::degree(c.shape.d)>=512 &&
        c.statistics.count>=minimum && c.statistics.seconds>0) {
        auto &at=best[c.shape.carrier];if(!at || c.statistics.seconds<at->statistics.seconds)at=&c;
    }
    std::vector<Shape> out;
    for(const auto &entry:best) {
        auto s=entry.second->shape;s.baby_coop=32;
        bool found=false;for(const auto &c:all)found|=c.shape.key()==s.key();
        if(!found)out.push_back(s);
    }
    return out;
}
// At most one new chain trial per arithmetic family: the fastest repeated
// 64-point incumbent, keeping its D/layout/residency fixed. Never expand the
// D catalogue by chain length. Full-engine measurements include seed cost and
// the actual device's parallel efficiency; operation counts alone cannot pick L.
inline std::vector<Shape> runtime_chain_trials(const std::vector<RuntimeCandidate> &all,Word minimum=2) {
    std::map<unsigned,const RuntimeCandidate*> best;
    for(const auto &c:all)if(c.shape.chain_block==64 && c.statistics.count>=minimum && c.statistics.seconds>0) {
        GiantWork work;
        if(!giant_work(c.shape.b2/c.shape.d+2,c.shape.d,c.shape.giant_chunk,c.shape.chain_min,
            c.shape.force_ladder,work,d_catalog::degree(c.shape.d),c.shape.tail_balance) || !work.chain_points)continue;
        auto &at=best[c.shape.carrier];if(!at || c.statistics.seconds<at->statistics.seconds)at=&c;
    }
    std::vector<Shape> out;
    for(const auto &entry:best) {
        auto s=entry.second->shape;s.chain_block=32;
        bool found=false;for(const auto &c:all)found|=c.shape.key()==s.key();
        if(!found)out.push_back(s);
    }
    return out;
}
struct RuntimeSearchAttempts {
    size_t steps=0,shapes=0;unsigned admissions=0,limit=16;
    explicit RuntimeSearchAttempts(size_t count,unsigned maximum=16):shapes(count),limit(maximum){}
    bool next(){return admissions<limit && steps++<=shapes;}
};
struct RuntimeAllocation {
    bool ordinary_reduced=false;double advantage=0;
    unsigned ordinary_candidates=2,ordinary_admissions=4;
};
inline RuntimeAllocation runtime_allocation(const std::vector<RuntimeCandidate> &candidates,Word minimum=2) {
    RuntimeAllocation out;const RuntimeStatistics *ordinary=nullptr,*carrier=nullptr;
    for(const auto &c:candidates)if(c.statistics.count>=minimum && c.statistics.seconds>0) {
        auto &best=c.shape.carrier?carrier:ordinary;
        if(!best || c.statistics.seconds<best->seconds)best=&c.statistics;
    }
    if(!ordinary || !carrier)return out;
    const double low=ordinary->seconds-2*ordinary->stddev,high=carrier->seconds+2*carrier->stddev;
    if(low>0){out.advantage=1-high/low;out.ordinary_reduced=high<=.80*low;}
    return out;
}
struct RuntimeSearchBudget {
    static constexpr unsigned initial_candidates=4,extended_candidates=8,regions=3;
    static constexpr double max_extra_seconds=120;
    unsigned candidates=initial_candidates;Word minimum=2;
    double spent=0,remaining=max_extra_seconds,reference=0;
    bool permits(double upper,Word samples,Word queue_remaining=0)const {
        if(queue_remaining && queue_remaining<2*minimum)return false;
        const double needed=double(samples<minimum?minimum-samples:0)*std::max(0.,upper-reference);
        return reference>0 && remaining>0 && needed<=remaining;
    }
};
inline RuntimeSearchBudget runtime_search_budget(const std::vector<DSearchNode> &nodes,Word baseline,Word sample_limit,Word minimum=2) {
    RuntimeSearchBudget out;out.minimum=minimum;Word trusted_count=0;
    for(const auto &n:nodes)if(n.samples>=minimum && n.seconds>0) {
        trusted_count=std::max(trusted_count,n.samples);
        if(!out.reference || n.seconds<out.reference)out.reference=n.seconds;
    }
    if(trusted_count>=std::min(Word(8),sample_limit))out.candidates=RuntimeSearchBudget::extended_candidates;
    for(const auto &n:nodes)if(n.d!=baseline && n.samples && n.seconds>out.reference)
        out.spent+=n.samples*(n.seconds-out.reference);
    out.remaining=std::max(0.,RuntimeSearchBudget::max_extra_seconds-out.spent);return out;
}
inline RuntimeSearchBudget runtime_family_budget(const std::vector<RuntimeCandidate> &all,
        const Shape &reference,Word sample_limit,const RuntimeAllocation &allocation,Word minimum=2) {
    RuntimeSearchBudget out;out.minimum=minimum;Word trusted=0;unsigned outside=0;
    const bool legacy=legacy_groot_layout((reference.arithmetic_bits+63)/64,
        reference.b2/reference.d+2,d_catalog::degree(reference.d));
    for(const auto &c:all)if(c.shape.carrier==reference.carrier && c.shape.chain_block==reference.chain_block && c.statistics.count>=minimum) {
        trusted=std::max(trusted,c.statistics.count);
        if(!out.reference || c.statistics.seconds<out.reference)out.reference=c.statistics.seconds;
    }
    if(trusted>=std::min(Word(8),sample_limit))out.candidates=RuntimeSearchBudget::extended_candidates;
    if(!reference.carrier && allocation.ordinary_reduced)out.candidates=allocation.ordinary_candidates;
    for(const auto &c:all)if(c.shape.carrier==reference.carrier && c.shape.chain_block==reference.chain_block && c.statistics.count) {
        const bool extra=c.shape.d!=reference.d || c.shape.groot_balanced!=legacy;
        if(extra && (c.shape.groot_balanced!=reference.groot_balanced || c.shape.d==reference.d))++outside;
        if(extra && c.statistics.seconds>out.reference)out.spent+=c.statistics.count*(c.statistics.seconds-out.reference);
    }
    out.candidates=outside<out.candidates?out.candidates-outside:0;
    out.remaining=std::max(0.,RuntimeSearchBudget::max_extra_seconds-out.spent);return out;
}
// Candidates have already passed current arithmetic and live memory admission.
// Trial curves are useful Stage2 work, never an unreported duplicate benchmark.
inline RuntimeChoice runtime_choice(const std::vector<RuntimeCandidate> &candidates,bool explore=true,Word minimum=2) {
    RuntimeChoice out;unsigned priority=~0u;
    for(size_t i=0;i<candidates.size();++i) {
        const auto &c=candidates[i];
        // Complete the configured observations before another family; on equal
        // priority probe carrier work first. Neither rule changes winner ranking.
        const bool tie=out.available && c.priority==priority &&
            (c.statistics.count>candidates[out.index].statistics.count ||
             (c.statistics.count==candidates[out.index].statistics.count &&
              bool(c.shape.carrier)>bool(candidates[out.index].shape.carrier)));
        if(explore && c.probe_allowed && c.statistics.count<minimum && (c.priority<priority || tie)) {
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
        if(x.statistics.comparison_seconds()!=y.statistics.comparison_seconds())return x.statistics.comparison_seconds()<y.statistics.comparison_seconds();
        if(x.statistics.preferred_order!=y.statistics.preferred_order)
            return x.statistics.preferred_order>y.statistics.preferred_order;
        return x.shape.key()<y.shape.key();
    };
    for(size_t i=0;i<candidates.size();++i) {
        const auto &s=candidates[i].statistics;
        if(s.count<minimum || !(s.seconds>0) || !std::isfinite(s.seconds))continue;
        if(!out.available || better(i,out.index))out={i,true,false};
    }
    return out;
}
} } }
