#pragma once
#include "ecm_stage2_tune_ecm.h"

namespace ecm_stage2 { namespace tune {
constexpr double b2_holdout_error_limit=0.08;
struct B2Prediction {
    double seconds=0,mad_seconds=0,error_seconds=0,max_relative_error=0;
    Word low=0,high=0,samples=0;
};
inline std::string b2_scope(const Fields &sample) {
    std::string key;
    for(const char *field:{"target_bits","arithmetic_bits","carrier_exponent","modulus_kind","b1","d"})
        key+=required(sample,field)+":";
    return key;
}
inline bool predicts_b2(const EcmProfile &profile) {
    const auto i=profile.profile.find("prediction_model");
    return i!=profile.profile.end() && i->second==std::string("\"")+b2_prediction_model+"\"";
}
// A performance estimate, never a mathematical or memory admission proof.
// All source scopes have already passed EcmProfile's independent validation.
inline bool predict_b2(std::vector<const Fields*> samples,Word b2,B2Prediction &out) {
    out=B2Prediction{};
    if(samples.size()<3 || samples.size()>128)return false;
    std::sort(samples.begin(),samples.end(),[](const Fields *a,const Fields *b){return uint(*a,"b2")<uint(*b,"b2");});
    const auto scope=b2_scope(*samples.front());
    const Word low=uint(*samples.front(),"b2"),high=uint(*samples.back(),"b2"),d=uint(*samples.front(),"d");
    if(b2<=low || b2>=high)return false;
    std::vector<double> x,y;double noise=0;
    Word previous=0;
    for(const auto *sample:samples) {
        const auto points=uint(*sample,"giant_points");
        if(b2_scope(*sample)!=scope || !uint(*sample,"fold_resident") || !uint(*sample,"frontier_resident") ||
           points<=previous || points>(1ull<<53))return false;
        previous=points;x.push_back((double)points);y.push_back(real(*sample,"median_seconds"));
        noise=std::max(noise,real(*sample,"mad_seconds"));
    }
    if(x.back()<2*x.front())return false;
    auto fit=[&](size_t omitted,double &fixed,double &rate) {
        double mx=0,my=0;size_t count=0;
        for(size_t i=0;i<x.size();++i)if(i!=omitted){mx+=x[i];my+=y[i];++count;}
        mx/=count;my/=count;double xx=0,xy=0;
        for(size_t i=0;i<x.size();++i)if(i!=omitted){const auto delta=x[i]-mx;xx+=delta*delta;xy+=delta*(y[i]-my);}
        if(!(xx>0))return false;rate=xy/xx;fixed=my-rate*mx;
        return std::isfinite(rate) && std::isfinite(fixed) && rate>=0 && fixed>=0;
    };
    double error=0,relative=0;
    for(size_t i=0;i<x.size();++i) {
        double fixed=0,rate=0;if(!fit(i,fixed,rate))return false;
        const auto difference=std::abs(fixed+rate*x[i]-y[i]);
        error=std::max(error,difference);relative=std::max(relative,difference/y[i]);
    }
    if(relative>b2_holdout_error_limit)return false;
    double fixed=0,rate=0;if(!fit(samples.size(),fixed,rate))return false;
    const auto points=b2/d+2;const auto seconds=fixed+rate*(double)points;
    if(!std::isfinite(seconds) || !(seconds>0))return false;
    out={seconds,noise,error,relative,low,high,(Word)samples.size()};return true;
}
} }
