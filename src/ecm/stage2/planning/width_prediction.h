#pragma once
#include "portable_model.h"
#include "width_reference.h"

namespace ecm_stage2 { namespace tune { namespace portable {
struct WidthEstimate : Estimate {
    unsigned arithmetic_low=0,arithmetic_high=0;
    Word runtime_samples=0;
    bool region_covered=false;
    const char *uncertainty="analytic_prior";
};
// Arithmetic-width cost atlas. No modulus value, sigma, B1 or carrier exponent
// identity is a lookup key. Actual packing, routes and residency are recomputed.
class WidthPredictor {
    Estimator &engine;
    const Fields *excluded=nullptr;
    struct Anchor {const Fields *sample;Shape shape;Work work;double correction;};
    using Region=std::array<Word,13>;
    mutable std::map<Region,std::vector<Anchor>> regions;
    static double ratio(double a,double b){return b>0?a/b:1.;}
    static Shape reference(const WidthReference &r) {
        Shape s;s.target_bits=r.target_bits;s.arithmetic_bits=r.arithmetic_bits;
        s.carrier=r.mersenne?r.arithmetic_bits:0;s.mersenne=r.mersenne;
        s.b1=r.b1;s.b2=r.b2;s.d=r.d;s.giant_chunk=r.giant_chunk;s.batch_bytes=r.batch_bytes;return s;
    }
    static Region region(const Shape &s,const Work &w) {
        Region out{};const auto topology=interval(s,w).key;
        std::copy(topology.begin(),topology.end(),out.begin());
        out[8]=s.mersenne;out[9]=s.physical;out[10]=s.buffers;
        out[11]=s.short_batch_tune;out[12]=s.short_batch_hash;return out;
    }
    static double geometry_distance(const Work &a,const Work &b) {
        return std::abs(std::log2(double(a.p)/b.p))+
            .25*std::abs(std::log2(double(a.points)/b.points));
    }
    // Positive operation mixture, not a fitted or independently validated rate.
    double basic_from(const Shape &s,const Work &w,const WidthReference &r)const {
        Work ref;if(!engine.cached_work(reference(r),ref))return 0;
        const double nt=s.mersenne?.85:.65,red=s.mersenne?.05:.27;
        const double transfer=s.mersenne?.06:.04;
        auto poly=[&](unsigned phase) {
            return nt*ratio(w.ntt[phase],ref.ntt[phase])+red*ratio(w.reduce[phase],ref.reduce[phase])+
                transfer*ratio(w.transfer[phase],ref.transfer[phase])+.04*ratio(w.calls[phase],ref.calls[phase]);
        };
        const double fixed=std::max(.002,r.total-r.baby-r.ftree-r.inverse-r.descent-r.accum-r.loop);
        double extra_transfer=0;
        for(unsigned phase=0;phase<RequestPhaseCount;++phase)
            if(ref.transfer[phase]==0)extra_transfer+=w.transfer[phase]/(6.*1024*1024*1024);
        const double nested=r.giant+r.trees+r.fold;
        const double loop=nested>0?(r.giant*ratio(w.giant,ref.giant)+r.trees*poly(RequestGtrees)+r.fold*poly(RequestFold))/nested:
            poly(RequestGtrees);
        return fixed+extra_transfer+r.baby*ratio(w.baby,ref.baby)+r.ftree*poly(RequestFtree)+r.inverse*poly(RequestInverse)+
            r.descent*poly(RequestDescent)+r.accum*ratio(w.accum,ref.accum)+r.loop*loop;
    }
public:
    explicit WidthPredictor(Estimator &e,const Fields *omit=nullptr):engine(e),excluded(omit){}
    bool has_width_measurement(const Shape &s)const {
        Work current;if(!engine.cached_work(s,current) || current.p<512 || !current.exact_topology)return false;
        for(const auto &sample:engine.profile.samples)if(engine.eligible(sample,s)) {
            const auto before=engine.shape(sample);Work w;
            if(before.arithmetic_bits==s.arithmetic_bits && engine.cached_work(before,w) &&
               w.p>=512 && w.exact_topology && region(before,w)==region(s,current))return true;
        }
        return false;
    }
    double basic(const Shape &s,const Work &w)const {
        const WidthReference *low=nullptr,*high=nullptr;
        for(const auto &r:width_references)if(r.mersenne==s.mersenne) {
            if(r.arithmetic_bits<=s.arithmetic_bits && (!low || r.arithmetic_bits>low->arithmetic_bits))low=&r;
            if(r.arithmetic_bits>=s.arithmetic_bits && (!high || r.arithmetic_bits<high->arithmetic_bits))high=&r;
        }
        const auto a=low?low:high,b=high?high:low;if(!a || !b)return 0;
        const double first=basic_from(s,w,*a),last=basic_from(s,w,*b);if(!(first>0 && last>0))return 0;
        if(a->arithmetic_bits==b->arithmetic_bits)return first;
        const double fraction=std::log(double(s.arithmetic_bits)/a->arithmetic_bits)/
                              std::log(double(b->arithmetic_bits)/a->arithmetic_bits);
        return std::exp((1-fraction)*std::log(first)+fraction*std::log(last));
    }
    bool predict(const Shape &s,WidthEstimate &out)const {
        out=WidthEstimate{};Work w;if(!engine.cached_work(s,w))return false;
        const auto runtime=engine.runtime_statistics(s);
        if(runtime.count>=2 && w.p>=512 && w.exact_topology) {
            out.seconds=runtime.seconds;out.stddev=runtime.timing().stddev;
            out.runtime_samples=runtime.count;out.exact=out.region_covered=true;
            out.anchors=runtime.count;out.arithmetic_low=out.arithmetic_high=s.arithmetic_bits;
            out.relative_error=0;out.rank=out.seconds+2*out.stddev;
            out.model="runtime_shape_mean_v2";out.uncertainty="observed_runtime_stddev";return true;
        }
        if(const auto measured=engine.observed(s,excluded)) {
            out.seconds=real(*measured,"mean_seconds");out.stddev=real(*measured,"stddev_seconds");
            out.anchor=measured;out.anchors=1;out.exact=true;out.region_covered=true;
            out.arithmetic_low=out.arithmetic_high=s.arithmetic_bits;out.model="exact_shape_measurement_v1";
            const bool same=matches(engine.profile.conditions.at(uint(*measured,"condition")),engine.condition,
                                    Ignore::parse("environment,memory"));
            out.relative_error=same?0:.15;out.uncertainty=same?"sample_stddev":"changed_hardware_prior";
            out.rank=out.seconds*(1+out.relative_error)+2*out.stddev;return true;
        }
        out.seconds=basic(s,w);if(!(out.seconds>0) || !std::isfinite(out.seconds))return false;
        double distance=100,geometry=100;
        for(const auto &r:width_references)if(r.mersenne==s.mersenne) {
            if(r.arithmetic_bits<=s.arithmetic_bits)out.arithmetic_low=std::max(out.arithmetic_low,r.arithmetic_bits);
            if(r.arithmetic_bits>=s.arithmetic_bits && (!out.arithmetic_high || r.arithmetic_bits<out.arithmetic_high))
                out.arithmetic_high=r.arithmetic_bits;
            Work ref;const auto origin=reference(r);if(!engine.cached_work(origin,ref))continue;
            const double d=std::abs(std::log2(double(s.arithmetic_bits)/r.arithmetic_bits));
            if(d<distance) {distance=d;geometry=geometry_distance(w,ref);}
            out.region_covered|=d<=1 && region(origin,ref)==region(s,w);
        }
        if(!out.arithmetic_low)out.arithmetic_low=out.arithmetic_high;
        if(!out.arithmetic_high)out.arithmetic_high=out.arithmetic_low;
        out.model="basic_stage2_width_v1";out.extrapolated=true;
        out.relative_error=std::min(.8,.25+.12*distance+.04*geometry);
        auto at=regions.find(region(s,w));
        if(at==regions.end()) {
            std::vector<Anchor> anchors;
            for(const auto &sample:engine.profile.samples)if(&sample!=excluded && engine.eligible(sample,s)) {
                const auto shape=engine.shape(sample);Work before;
                if(!engine.cached_work(shape,before) || before.p<512 || !before.exact_topology ||
                   region(shape,before)!=region(s,w))continue;
                const double baseline=basic(shape,before);
                if(baseline>0 && std::isfinite(baseline))anchors.push_back({&sample,shape,before,real(sample,"mean_seconds")/baseline});
            }
            at=regions.emplace(region(s,w),std::move(anchors)).first;
        }
        const Anchor *low=nullptr,*high=nullptr;
        auto better=[&](const Anchor &a,const Anchor *old) {
            return !old || geometry_distance(a.work,w)<geometry_distance(old->work,w);
        };
        for(const auto &a:at->second) {
            ++out.anchors;
            if(a.shape.arithmetic_bits<=s.arithmetic_bits &&
               (!low || a.shape.arithmetic_bits>low->shape.arithmetic_bits ||
                (a.shape.arithmetic_bits==low->shape.arithmetic_bits && better(a,low))))low=&a;
            if(a.shape.arithmetic_bits>=s.arithmetic_bits &&
               (!high || a.shape.arithmetic_bits<high->shape.arithmetic_bits ||
                (a.shape.arithmetic_bits==high->shape.arithmetic_bits && better(a,high))))high=&a;
        }
        if(low || high) {
            const auto a=low?low:high,b=high?high:low;
            const double fraction=a->shape.arithmetic_bits==b->shape.arithmetic_bits?0:
                std::log(double(s.arithmetic_bits)/a->shape.arithmetic_bits)/
                std::log(double(b->shape.arithmetic_bits)/a->shape.arithmetic_bits);
            out.seconds*=std::exp((1-fraction)*std::log(a->correction)+fraction*std::log(b->correction));
            out.arithmetic_low=a->shape.arithmetic_bits;out.arithmetic_high=b->shape.arithmetic_bits;
            out.anchor=a->sample;out.region_covered=true;out.extrapolated=!(low && high);
            out.model=out.extrapolated?"width_region_extrapolation_v1":"width_region_interpolation_v1";
            const double distance=std::max(std::abs(std::log2(double(s.arithmetic_bits)/a->shape.arithmetic_bits)),
                                           std::abs(std::log2(double(s.arithmetic_bits)/b->shape.arithmetic_bits)));
            out.relative_error=std::min(.8,.15+.10*distance+.04*std::max(geometry_distance(w,a->work),geometry_distance(w,b->work)));
            // Dispersion is model disagreement, never independent validation.
            if(a!=b)out.relative_error=std::max(out.relative_error,
                .5*std::abs(a->correction-b->correction)/std::max(a->correction,b->correction));
            out.stddev=std::max(real(*a->sample,"stddev_seconds")/real(*a->sample,"mean_seconds"),
                             real(*b->sample,"stddev_seconds")/real(*b->sample,"mean_seconds"))*out.seconds;
            out.uncertainty="width_distance_and_anchor_dispersion";
            if(!matches(engine.profile.conditions.at(uint(*a->sample,"condition")),engine.condition,
                        Ignore::parse("environment,memory")) ||
               !matches(engine.profile.conditions.at(uint(*b->sample,"condition")),engine.condition,
                        Ignore::parse("environment,memory")))out.relative_error=std::min(.8,out.relative_error+.15);
        }
        out.rank=out.seconds*(1+out.relative_error)+2*out.stddev;
        return out.seconds>0 && std::isfinite(out.rank);
    }
    static double lower(const WidthEstimate &e){return std::max(0.,e.seconds*(1-e.relative_error)-2*e.stddev);}
    static bool improves(const WidthEstimate &a,const WidthEstimate &b) {
        return a.region_covered && b.region_covered && a.rank<.98*lower(b);
    }
};
} } }
