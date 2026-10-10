#pragma once
#include "ecm_stage2_tune_prediction.h"
#include "ecm_stage2_requests.h"
#include <tuple>

namespace ecm_stage2 { namespace tune {
constexpr const char *component_prediction_model="phase_ntt_loop_v1";
using NttMeasurements=std::map<std::pair<Word,Word>,double>;
struct NttReferenceBin { Word calls=0,pairs=0; };
struct NttReferencePhase {
    Word calls=0,pairs=0,missing_bins=0;
    double matched_seconds=0;
    bool complete=false;
};
struct NttReferences {
    Word p=0,points=0,budget=0,chunk_max=0;
    int bits=0;unsigned buffers=0;bool physical=false,valid=false,complete=false;
    std::map<std::tuple<unsigned,Word,Word>,NttReferenceBin> bins;
    std::array<NttReferencePhase,RequestPhaseCount> phases{};
};
// Exact physical batches, including tails. No interpolation in length/slices,
// allocation, memory admission, or expansion of repeated middle G roots.
// query must be the engine's packing query, not an approximate length formula.
template<class Query> bool ntt_references(Word p,Word points,int bits,Query query,
        Word budget,unsigned buffers,bool physical,Word chunk_max,
        const NttMeasurements &measurements,NttReferences &out) {
    out=NttReferences{};
    if(bits<2 || bits>max_input_bits || !budget || (buffers!=2 && buffers!=3))return false;
    RequestProgram program;
    if(!request_program(p,points,program) || !program.supported)return false;
    out.p=p;out.points=points;out.bits=bits;out.budget=budget;out.buffers=buffers;
    out.physical=physical;out.chunk_max=chunk_max;
    for(const auto &block:program.blocks)for(const auto &r:block.requests) {
        Word n=0,slots=0;const Word operand=std::max(r.ma,r.mb);
        if(!query(operand,bits,&n,&slots) || n<8 || n>(1ull<<27) || (n&(n-1)) || slots!=2*operand-1)return false;
        Word c=chunk_slices(n,slots,r.pairs,budget,physical?buffers:3,physical);
        if(chunk_max)c=std::min(c,chunk_max);
        for(const auto batch: {std::make_pair(c,r.pairs/c),std::make_pair(r.pairs%c,Word(r.pairs%c!=0))}) {
            if(!batch.second)continue;
            Word calls=0,pairs=0;
            if(!multiply(batch.second,block.repeat,calls) || !multiply(calls,batch.first,pairs))return false;
            auto &bin=out.bins[{r.phase,n,batch.first}];
            if(!add(bin.calls,calls,bin.calls) || !add(bin.pairs,pairs,bin.pairs))return false;
        }
    }
    for(const auto &entry:out.bins) {
        const auto phase=std::get<0>(entry.first);
        const auto n=std::get<1>(entry.first),slices=std::get<2>(entry.first);
        auto &summary=out.phases[phase];const auto &bin=entry.second;
        if(!add(summary.calls,bin.calls,summary.calls) || !add(summary.pairs,bin.pairs,summary.pairs))return false;
        const auto measured=measurements.find({n,slices});
        if(measured==measurements.end()) {++summary.missing_bins;continue;}
        if(!(measured->second>0) || !std::isfinite(measured->second))return false;
        summary.matched_seconds+=double(bin.calls)*measured->second;
        if(!std::isfinite(summary.matched_seconds))return false;
    }
    out.complete=true;
    for(auto &phase:out.phases){phase.complete=phase.missing_bins==0;out.complete&=phase.complete;}
    out.valid=true;return true;
}
inline bool same_reference_policy(const NttReferences &a,const NttReferences &b) {
    return a.p==b.p && a.bits==b.bits && a.budget==b.budget && a.buffers==b.buffers &&
        a.physical==b.physical && a.chunk_max==b.chunk_max;
}
inline double loop_reference(const NttReferences &reference) {
    if(!reference.valid || !reference.complete)return 0;
    return reference.phases[RequestGtrees].matched_seconds+reference.phases[RequestFold].matched_seconds;
}
inline std::string component_scope(const Fields &sample) {
    auto scope=b2_scope(sample);
    for(const char *key:{"giant_chunk_points","giant_chain_min","giant_force_ladder"})scope+=required(sample,key)+":";
    return scope;
}
struct ComponentAnchor {const Fields *sample=nullptr;const NttReferences *reference=nullptr;};
struct ComponentModel {
    B2Features coefficients{};
    double fixed_seconds=0;
    B2Prediction evidence;
    Word d=0,chunk=0,minimum=0,ladder_low=0,ladder_high=0;
    bool force=false,qualified=false;
    std::string scope;
    NttReferences policy;
};
// Paired subtraction is per repetition, before the fixed-cost median. This
// prevents nested legacy component timers from being added to engine time.
inline std::vector<double> paired_fixed_costs(const Fields &sample) {
    const auto total=array(required(sample,"seconds"));
    if(required(sample,"phase_accounting")!=std::string("\"")+timing::contract+"\"")
        throw std::runtime_error("component model requires exclusive paired phases");
    validate_paired_costs(sample,total);
    const auto loop=array(required(sample,"phase_giant_loop_samples"));
    std::vector<double> fixed;
    for(size_t i=0;i<total.size();++i) {
        const auto value=total[i]-loop[i];
        if(!std::isfinite(value) || value<0)throw std::runtime_error("invalid paired fixed cost");
        fixed.push_back(value);
    }
    return fixed;
}
inline bool prepare_component_model(std::vector<ComponentAnchor> anchors,ComponentModel &out) {
    out=ComponentModel{};
    if(anchors.size()<7 || anchors.size()>128)return false;
    for(const auto &a:anchors)if(!a.sample || !a.reference)return false;
    std::sort(anchors.begin(),anchors.end(),[](const ComponentAnchor &a,const ComponentAnchor &b){return uint(*a.sample,"b2")<uint(*b.sample,"b2");});
    out.scope=component_scope(*anchors.front().sample);out.policy=*anchors.front().reference;
    out.policy.bins.clear();out.policy.phases={}; // Store only packing/chunk policy, not a query result.
    const auto &first=*anchors.front().sample;
    out.d=uint(first,"d");out.chunk=uint(first,"giant_chunk_points");
    out.minimum=uint(first,"giant_chain_min");out.force=uint(first,"giant_force_ladder")!=0;
    out.ladder_low=std::numeric_limits<Word>::max();
    std::vector<B2Features> x;std::vector<double> totals;std::vector<std::vector<double>> fixed;
    Word previous=0;size_t chains=0,ladders=0;double noise=0;
    for(const auto &anchor:anchors) {
        const auto &s=*anchor.sample;const auto &r=*anchor.reference;
        validate_sample(s);
        const Word points=uint(s,"giant_points"),steps=uint(s,"giant_ladder_steps");
        const double reference=loop_reference(r);
        if(component_scope(s)!=out.scope || !uint(s,"fold_resident") || !uint(s,"frontier_resident") ||
           points<=previous || points>(1ull<<53) || steps>(1ull<<53) ||
           !same_reference_policy(r,out.policy) || r.points!=points || r.p!=uint(s,"p") ||
           r.bits!=uint(s,"arithmetic_bits") || !(reference>0) || !std::isfinite(reference))return false;
        if(steps){++ladders;out.ladder_low=std::min(out.ladder_low,steps);out.ladder_high=std::max(out.ladder_high,steps);}
        else ++chains;
        previous=points;
        x.push_back({1.,reference,double(steps),double(uint(s,"giant_ladder_chunks"))});
        totals.push_back(real(s,"median_seconds"));fixed.push_back(paired_fixed_costs(s));
        noise=std::max(noise,real(s,"mad_seconds"));
    }
    if(previous<2*uint(first,"giant_points") || chains<3 || ladders<3)return false;
    auto fit=[&](size_t omitted,double &base,B2Features &coefficients) {
        std::vector<double> values;
        for(size_t i=0;i<fixed.size();++i)if(i!=omitted)values.insert(values.end(),fixed[i].begin(),fixed[i].end());
        base=median(values);std::vector<double> y;
        for(size_t i=0;i<totals.size();++i){const auto residual=totals[i]-base;
            if(i!=omitted && (!(residual>0) || !std::isfinite(residual)))return false;y.push_back(residual);}
        return fit_route_cost(x,y,omitted,4,coefficients);
    };
    double absolute=0,relative=0;
    for(size_t i=0;i<x.size();++i) {
        double base=0;B2Features coef{};if(!fit(i,base,coef))return false;
        double predicted=base;for(unsigned j=0;j<4;++j)predicted+=coef[j]*x[i][j];
        const double difference=std::abs(predicted-totals[i]);
        absolute=std::max(absolute,difference);relative=std::max(relative,difference/totals[i]);
    }
    out.evidence={0,noise,absolute,relative,uint(first,"b2"),uint(*anchors.back().sample,"b2"),(Word)anchors.size()};
    if(relative>b2_holdout_error_limit || !fit(anchors.size(),out.fixed_seconds,out.coefficients))return false;
    out.qualified=true;return true;
}
inline bool predict_component(const ComponentModel &model,Word b2,const NttReferences &reference,B2Prediction &out) {
    out=B2Prediction{};
    if(!model.qualified || b2<=model.evidence.low || b2>=model.evidence.high ||
       !same_reference_policy(model.policy,reference) || reference.points!=b2/model.d+2)return false;
    const double seconds=loop_reference(reference);
    GiantWork route;
    if(!(seconds>0) || reference.points>(1ull<<53) ||
       !giant_work(reference.points,model.d,model.chunk,model.minimum,model.force,route) ||
       (route.ladder_steps && (route.ladder_steps<model.ladder_low || route.ladder_steps>model.ladder_high)))return false;
    const B2Features x{1.,seconds,double(route.ladder_steps),double(route.ladder_chunks)};
    double total=model.fixed_seconds;for(unsigned i=0;i<4;++i)total+=x[i]*model.coefficients[i];
    if(!(total>0) || !std::isfinite(total))return false;
    out=model.evidence;out.seconds=total;return true;
}
} }
