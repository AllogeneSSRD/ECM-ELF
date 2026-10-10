#pragma once
#include "ecm_stage2_tune_ecm.h"
#include <utility>

namespace ecm_stage2 { namespace tune {
constexpr double b2_holdout_error_limit=0.08;
struct B2Prediction {
    double seconds=0,mad_seconds=0,error_seconds=0,max_relative_error=0;
    Word low=0,high=0,samples=0;
    const char *model=b2_prediction_model;
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
// Qualification depends only on the anchors, not on the requested B2. Retain
// coefficients per caller-owned group so an Auto B2 grid does not repeat every
// leave-one-out fit. This carries no device/free-memory admission state.
struct B2Model {
    std::array<double,4> coefficients{};
    Word d=0,chunk=0,minimum=0,ladder_low=0,ladder_high=0;
    bool force=false,has_chain=false,has_ladder=false;
    B2Prediction evidence;
    bool qualified=false;
};
using B2Features=std::array<double,4>;
// Four small, scaled columns. Enumerate nonnegative active sets; singular sets
// are refused. This is fitting performance data, never changing ECM arithmetic.
inline bool fit_route_cost(const std::vector<B2Features> &x,const std::vector<double> &y,
                           size_t omitted,unsigned dimensions,B2Features &best) {
    B2Features scale{};for(const auto &row:x)for(unsigned j=0;j<dimensions;++j)scale[j]=std::max(scale[j],row[j]);
    double best_error=std::numeric_limits<double>::infinity();bool found=false;
    for(unsigned mask=1;mask<(1u<<dimensions);++mask) {
        // Preserve the strict nonnegative intercept/rate contract for chain-only data.
        if(dimensions==2 && mask!=3)continue;
        unsigned columns[4]{},n=0;bool supported=true;
        for(unsigned j=0;j<dimensions;++j)if(mask&(1u<<j)) {
            if(!(scale[j]>0)){supported=false;break;}columns[n++]=j;
        }
        if(!supported)continue;
        double a[4][5]{};
        for(size_t r=0;r<x.size();++r)if(r!=omitted)for(unsigned i=0;i<n;++i) {
            const double v=x[r][columns[i]]/scale[columns[i]];
            for(unsigned j=0;j<n;++j)a[i][j]+=v*x[r][columns[j]]/scale[columns[j]];
            a[i][n]+=v*y[r];
        }
        for(unsigned i=0;i<n;++i) {
            unsigned pivot=i;for(unsigned k=i+1;k<n;++k)if(std::abs(a[k][i])>std::abs(a[pivot][i]))pivot=k;
            if(std::abs(a[pivot][i])<1e-10){supported=false;break;}
            for(unsigned j=0;j<=n;++j)std::swap(a[i][j],a[pivot][j]);
            const double diagonal=a[i][i];for(unsigned j=i;j<=n;++j)a[i][j]/=diagonal;
            for(unsigned k=0;k<n;++k)if(k!=i) {
                const double v=a[k][i];for(unsigned j=i;j<=n;++j)a[k][j]-=v*a[i][j];
            }
        }
        if(!supported)continue;
        B2Features coefficients{};
        for(unsigned i=0;i<n;++i) {
            if(!std::isfinite(a[i][n]) || a[i][n]<-1e-10){supported=false;break;}
            coefficients[columns[i]]=std::max(0.,a[i][n])/scale[columns[i]];
        }
        if(!supported)continue;
        double error=0;for(size_t r=0;r<x.size();++r)if(r!=omitted) {
            double predicted=0;for(unsigned j=0;j<dimensions;++j)predicted+=x[r][j]*coefficients[j];
            const double residual=predicted-y[r];error+=residual*residual;
        }
        if(error<best_error){best_error=error;best=coefficients;found=true;}
    }
    return found;
}
// A performance estimate, never a mathematical or memory admission proof.
// All source scopes have already passed EcmProfile's independent validation.
inline bool prepare_b2_model(std::vector<const Fields*> samples,B2Model &model) {
    model=B2Model{};
    if(samples.size()<3 || samples.size()>128)return false;
    std::sort(samples.begin(),samples.end(),[](const Fields *a,const Fields *b){return uint(*a,"b2")<uint(*b,"b2");});
    const auto scope=b2_scope(*samples.front());
    const Word low=uint(*samples.front(),"b2"),high=uint(*samples.back(),"b2"),d=uint(*samples.front(),"d");
    if(!d)return false;
    if(!samples.front()->count("giant_work_model"))return false;
    model.d=d;model.chunk=uint(*samples.front(),"giant_chunk_points");
    model.minimum=uint(*samples.front(),"giant_chain_min");model.force=uint(*samples.front(),"giant_force_ladder")!=0;
    std::vector<B2Features> x;std::vector<double> y;double noise=0;
    size_t chain_anchors=0,ladder_anchors=0;
    model.ladder_low=std::numeric_limits<Word>::max();
    Word previous=0;
    for(const auto *sample:samples) {
        const auto points=uint(*sample,"giant_points");
        if(b2_scope(*sample)!=scope || !uint(*sample,"fold_resident") || !uint(*sample,"frontier_resident") ||
           points<=previous || points>(1ull<<53) || !sample->count("giant_work_model") ||
           uint(*sample,"giant_chunk_points")!=model.chunk || uint(*sample,"giant_chain_min")!=model.minimum ||
           (uint(*sample,"giant_force_ladder")!=0)!=model.force)return false;
        const auto steps=uint(*sample,"giant_ladder_steps"),chunks=uint(*sample,"giant_ladder_chunks");
        if(steps>(1ull<<53))return false;
        if(steps) {
            ++ladder_anchors;model.ladder_low=std::min(model.ladder_low,steps);model.ladder_high=std::max(model.ladder_high,steps);
        } else ++chain_anchors;
        previous=points;x.push_back({1.,(double)points,(double)steps,(double)chunks});y.push_back(real(*sample,"median_seconds"));
        noise=std::max(noise,real(*sample,"mad_seconds"));
    }
    if(x.back()[1]<2*x.front()[1])return false;
    model.has_chain=chain_anchors!=0;model.has_ladder=ladder_anchors!=0;
    // A mixed model must keep both branches represented in every leave-one-out fit.
    // All-ladder groups remain exact-only until their distinct cost is calibrated.
    if(!model.has_chain || (model.has_ladder && (chain_anchors<3 || ladder_anchors<3 || samples.size()<7)))return false;
    const unsigned dimensions=model.has_ladder?4:2;
    double error=0,relative=0;
    for(size_t i=0;i<x.size();++i) {
        B2Features coefficients{};if(!fit_route_cost(x,y,i,dimensions,coefficients))return false;
        double estimated=0;for(unsigned j=0;j<dimensions;++j)estimated+=coefficients[j]*x[i][j];
        const auto difference=std::abs(estimated-y[i]);
        error=std::max(error,difference);relative=std::max(relative,difference/y[i]);
    }
    if(relative>b2_holdout_error_limit)return false;
    if(!fit_route_cost(x,y,samples.size(),dimensions,model.coefficients))return false;
    model.evidence={0,noise,error,relative,low,high,(Word)samples.size()};
    model.qualified=true;return true;
}
inline bool predict_b2(const B2Model &model,Word b2,B2Prediction &out) {
    out=B2Prediction{};
    if(!model.qualified || b2<=model.evidence.low || b2>=model.evidence.high)return false;
    const auto points=b2/model.d+2;GiantWork work;
    if(points>(1ull<<53) || !giant_work(points,model.d,model.chunk,model.minimum,model.force,work))return false;
    if(work.ladder_steps && (!model.has_ladder || work.ladder_steps<model.ladder_low || work.ladder_steps>model.ladder_high))return false;
    const B2Features features{1.,(double)points,(double)work.ladder_steps,(double)work.ladder_chunks};
    double seconds=0;for(unsigned j=0;j<4;++j)seconds+=model.coefficients[j]*features[j];
    if(!std::isfinite(seconds) || !(seconds>0))return false;
    out=model.evidence;out.seconds=seconds;return true;
}
inline bool predict_b2(std::vector<const Fields*> samples,Word b2,B2Prediction &out) {
    out=B2Prediction{};
    if(samples.size()<3 || samples.size()>128)return false;
    Word low=uint(*samples.front(),"b2"),high=low;
    for(const auto *sample:samples) {
        low=std::min(low,uint(*sample,"b2"));high=std::max(high,uint(*sample,"b2"));
    }
    if(b2<=low || b2>=high)return false;
    B2Model model;
    return prepare_b2_model(std::move(samples),model) && predict_b2(model,b2,out);
}
} }
