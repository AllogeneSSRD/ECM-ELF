#pragma once
#include "ecm_stage2_tune_prediction.h"
#include <set>
#include <functional>

namespace ecm_stage2 { namespace tune {
struct AutoRequest {
    Word b1=0,minimum=0,maximum=0;
    double stage1_seconds=0,adjust=1;
};
struct AutoCandidate {
    const Fields *sample=nullptr;
    Word b2=0,low=0,high=0;
    double seconds=0,mad_seconds=0,guarded_seconds=0,benefit=0,score=0;
    B2Prediction prediction;
    bool limited=false;
};
struct AdditionalPrediction {
    std::function<bool(const std::vector<const Fields*>&)> prepare;
    std::function<bool(const std::vector<const Fields*>&,Word,B2Prediction&)> predict;
};
inline double relative_ecm_benefit(Word b1,Word b2) {
    const double a=1.96617-0.06781*std::log10((double)b1);
    return 0.11343+0.88657*std::pow(std::log10((double)b2/(double)b1)/2,a);
}
// Samples have passed the native reader and the caller's target/device/policy
// and explicit-option filters. Costs are performance estimates, not admission.
inline std::vector<AutoCandidate> auto_candidates(const std::vector<const Fields*> &samples,
                                                bool opted,const AutoRequest &request,const AdditionalPrediction &additional={}) {
    if(request.b1<2 || !(request.stage1_seconds>0) || !std::isfinite(request.stage1_seconds) ||
       !(request.adjust>0) || !std::isfinite(request.adjust))
        throw std::runtime_error("full ECM Auto B2 requires positive Stage1 seconds per curve and ratio adjustment");
    std::map<std::string,std::vector<const Fields*>> groups;
    Word measured_low=(Word)INT64_MAX-8192,measured_high=0;
    for(const auto *sample:samples) {
        if(uint(*sample,"b1")!=request.b1 || !uint(*sample,"fold_resident") || !uint(*sample,"frontier_resident"))continue;
        groups[b2_scope(*sample)].push_back(sample);
        measured_low=std::min(measured_low,uint(*sample,"b2"));
        measured_high=std::max(measured_high,uint(*sample,"b2"));
    }
    if(groups.empty())throw std::runtime_error("no matching full ECM Auto B2 scope");
    Word low=measured_low,high=measured_high;
    if(request.minimum) {
        if(request.minimum<low || request.minimum>high)throw std::runtime_error("Auto B2 lower bound outside measured scope");
        low=request.minimum;
    }
    if(request.maximum) {
        if(request.maximum>high || request.maximum<low)throw std::runtime_error("Auto B2 upper bound outside measured scope");
        high=request.maximum;
    }
    std::vector<AutoCandidate> result;
    for(const auto &group:groups) {
        const auto &anchors=group.second;
        Word first=(Word)INT64_MAX-8192,last=0;
        for(const auto *s:anchors){first=std::min(first,uint(*s,"b2"));last=std::max(last,uint(*s,"b2"));}
        const Word lo=std::max(first,low),hi=std::min(last,high);
        if(lo>hi)continue;
        std::set<Word> points;
        for(const auto *s:anchors)if(uint(*s,"b2")>=lo && uint(*s,"b2")<=hi)points.insert(uint(*s,"b2"));
        B2Model model;
        const bool route_ready=opted && prepare_b2_model(anchors,model);
        const bool component_ready=additional.prepare && additional.prepare(anchors);
        const bool model_ready=route_ready || component_ready;
        if(model_ready) {
            points.insert(lo);points.insert(hi);
            for(int index=1;index<64;++index) {
                const double x=std::exp(std::log((double)lo)+std::log((double)hi/(double)lo)*index/64.);
                points.insert(x<=(double)lo?lo:x>=(double)hi?hi:(Word)std::floor(x+.5));
            }
            // Cost is constant within each integer I plateau. Include both
            // sides of nearby plateaus rather than only rounded log samples.
            const auto initial=points;const Word d=uint(*anchors.front(),"d");
            auto add=[&](Word value){if(value>=lo && value<=hi)points.insert(value);};
            for(Word point:initial) {
                const Word base=(point/d)*d;
                if(base)add(base-1);add(base);add(base+d-1);add(base+d);
            }
        }
        for(Word b2:points) {
            AutoCandidate candidate;candidate.b2=b2;candidate.low=lo;candidate.high=hi;
            const Fields *exact=nullptr;
            for(const auto *s:anchors)if(uint(*s,"b2")==b2){exact=s;break;}
            if(exact) {
                candidate.sample=exact;candidate.seconds=real(*exact,"median_seconds");
                candidate.mad_seconds=real(*exact,"mad_seconds");
                candidate.guarded_seconds=candidate.seconds+2*candidate.mad_seconds;
            } else {
                const bool component=component_ready && additional.predict && additional.predict(anchors,b2,candidate.prediction);
                if(!component && (!route_ready || !predict_b2(model,b2,candidate.prediction)))continue;
                candidate.sample=anchors.front();candidate.seconds=candidate.prediction.seconds;
                candidate.mad_seconds=candidate.prediction.mad_seconds;
                candidate.guarded_seconds=candidate.seconds+candidate.prediction.error_seconds+2*candidate.mad_seconds;
            }
            candidate.benefit=relative_ecm_benefit(request.b1,b2);
            candidate.score=candidate.benefit/(request.stage1_seconds+request.adjust*candidate.guarded_seconds);
            const Word d=uint(*candidate.sample,"d");
            candidate.limited=b2/d==lo/d || b2/d==hi/d;
            if(!(candidate.score>0) || !std::isfinite(candidate.score) || !std::isfinite(candidate.guarded_seconds))continue;
            result.push_back(candidate);
        }
    }
    std::stable_sort(result.begin(),result.end(),[](const AutoCandidate &a,const AutoCandidate &b){
        if(a.score!=b.score)return a.score>b.score;
        if(a.guarded_seconds!=b.guarded_seconds)return a.guarded_seconds<b.guarded_seconds;
        return a.b2<b.b2;
    });
    if(result.empty())throw std::runtime_error("no measured or qualified Auto B2 candidate inside requested bounds");
    return result;
}
} }
