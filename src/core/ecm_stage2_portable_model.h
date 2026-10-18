#pragma once
#include "ecm_stage2_tune_portable.h"
#include "ecm_stage2_tune_components.h"

namespace ecm_stage2 { namespace tune { namespace portable {
struct Shape {
    unsigned target_bits=0,arithmetic_bits=0,carrier=0;
    Word b1=0,b2=0,d=0,batch_bytes=64ull<<20,chunk_max=0;
    unsigned buffers=3;bool physical=false,fold=true,frontier=true,mersenne=false;
    bool short_batch_tune=false;
    Word short_batch_hash=0;
    Word giant_chunk=0,chain_min=32768;bool force_ladder=false;
    using Key=std::array<Word,18>;
    Key key()const {
        return {target_bits,arithmetic_bits,carrier,b1,b2,d,batch_bytes,chunk_max,buffers,
            Word(physical),Word(fold),Word(frontier),Word(mersenne),Word(short_batch_tune),
            short_batch_hash,giant_chunk,chain_min,Word(force_ladder)};
    }
};
struct Work {
    std::array<double,RequestPhaseCount> ntt{},reduce{},transfer{},calls{};
    std::array<Word,RequestPhaseCount> largest_phase_ntt{};
    double baby=0,giant=0,accum=0;
    Word p=0,points=0,largest_ntt=0;bool exact_topology=true;
};
// A cost estimate, not an arithmetic, memory, or primality check. NTT values
// outside the measured (length,batch) cells are explicitly extrapolated.
template<class Query,class NttCost> bool work(const Shape &s,Query query,NttCost ntt_cost,Work &out,
        const ShortNttBatchPolicy *short_ntt=nullptr) {
    out=Work{};
    if(s.target_bits<2 || s.arithmetic_bits<s.target_bits || s.arithmetic_bits>16384 ||
       s.b1<2 || s.b2<=s.b1 || s.d<6 || s.d%2 || !s.batch_bytes)return false;
    out.p=d_catalog::degree(s.d);out.points=s.b2/s.d+2;
    RequestProgram program;if(!request_program(out.p,std::max(out.points,out.p+1),program,
        giant_tree_balanced((s.arithmetic_bits+63)/64,out.points,out.p)))return false;
    if(out.points<=out.p) {
        // G=1 has a local root division/inverse. Use a conservative full-degree
        // cost surrogate, without claiming the resident request topology.
        out.exact_topology=false;
        for(auto &block:program.blocks)for(auto &r:block.requests)
            if(r.phase==RequestFold)r.pairs=0;
    }
    const double words=(s.arithmetic_bits+63)/64.,target_words=(s.target_bits+63)/64.;
    // Generic long division scales with dependent limb MACs; Mersenne folding
    // scales with input words. Coefficients are anchored to full-engine phases.
    const double reduction=s.mersenne?words:words*words;
    for(const auto &block:program.blocks)for(const auto &r:block.requests) {
        if(!r.pairs)continue;
        Word n=0,slots=0;const Word operand=std::max(r.ma,r.mb);
        if(!query(operand,s.arithmetic_bits,&n,&slots) || n<8 || n>(1ull<<27) || slots!=2*operand-1)return false;
        out.largest_ntt=std::max(out.largest_ntt,n);
        out.largest_phase_ntt[r.phase]=std::max(out.largest_phase_ntt[r.phase],n);
        Word c=planned_chunk_slices(n,slots,r.pairs,s.batch_bytes,s.physical?s.buffers:3,s.physical,
            s.chunk_max,s.short_batch_tune?short_ntt:nullptr);
        for(auto batch:{std::make_pair(c,r.pairs/c),std::make_pair(r.pairs%c,Word(r.pairs%c!=0))})if(batch.second) {
            const double calls=double(batch.second)*block.repeat;
            out.ntt[r.phase]+=calls*ntt_cost(n,batch.first);
            out.calls[r.phase]+=calls;
        }
        const double pairs=double(r.pairs)*block.repeat;
        out.reduce[r.phase]+=pairs*r.count*reduction;
        const bool host=r.input==RequestHost || (r.phase==RequestFold && !s.fold) ||
            (r.phase==RequestDescent && !s.frontier);
        if(host)out.transfer[r.phase]+=pairs*(r.ma+r.mb+r.count)*words*8;
    }
    out.baby=out.p*words*words*std::log2(double(s.d));
    GiantWork route;
    if(s.giant_chunk && giant_work(out.points,s.d,s.giant_chunk,s.chain_min,s.force_ladder,route))
        out.giant=words*words*(6.*route.chain_points+12.*route.ladder_steps);
    else out.giant=out.points*words*words*12.*std::log2(double(s.b2));
    out.accum=out.p*target_words*target_words;
    return true;
}
// Coarse regions, not an identity for all requests or a memory admission proof.
struct NttInterval {
    std::array<Word,8> key{};
    bool operator<(const NttInterval &other)const{return key<other.key;}
};
inline NttInterval interval(const Shape &s,const Work &w) {
    Word levels=0,pad=1;while(pad<w.p){pad*=2;++levels;}
    GiantWork giant;Word route=0;
    if(s.giant_chunk && giant_work(w.points,s.d,s.giant_chunk,s.chain_min,s.force_ladder,giant))
        route=giant.chain_points?(giant.ladder_steps?2:1):3;
    return {{w.largest_phase_ntt[RequestFold],w.largest_phase_ntt[RequestInverse],
        w.largest_phase_ntt[RequestFtree],levels,Word(s.fold),Word(s.frontier),route,
        w.largest_phase_ntt[RequestDescent]}};
}
inline double relative_work(const Work &a,const Work &b) {
    auto ratio=[](double x,double y){return y>0?x/y:1.;};
    auto poly=[&](unsigned ph){return .55*ratio(a.ntt[ph],b.ntt[ph])+
        .40*ratio(a.reduce[ph],b.reduce[ph])+.03*ratio(a.transfer[ph],b.transfer[ph])+
        .02*ratio(a.calls[ph],b.calls[ph]);};
    return .44*poly(RequestGtrees)+.19*poly(RequestFold)+.12*ratio(a.giant,b.giant)+
        .07*ratio(a.baby,b.baby)+.05*poly(RequestFtree)+.05*poly(RequestInverse)+
        .06*poly(RequestDescent)+.02*ratio(a.accum,b.accum);
}
struct Estimate {
    double seconds=0,stddev=0,relative_error=0.25,rank=0;
    Word anchors=0;bool validated=false,exact=false,extrapolated=false;
    const Fields *anchor=nullptr;
    const char *model="portable_components_v2";
};
struct RuntimeStatistics {
    Word count=0,preferred_order=0;
    double seconds=0,stddev=0;
    LocalSearch::Timing timing()const {return {seconds,stddev};}
};
class Estimator {
public:
    using Query=std::function<bool(Word,int,Word*,Word*)>;
    const Profile &profile;Fields condition;Ignore ignore;Query query;
    std::string runtime_version;
    std::map<std::pair<Word,Word>,double> ntt;
    ShortNttBatchPolicy short_ntt;
    // Each estimator requires an immutable profile while ranking. A caller
    // that updates the profile constructs a new estimator before using caches.
    mutable std::map<std::array<Word,18>,std::pair<bool,Work>> work_cache;
    std::set<const Fields*> profile_samples;
    mutable std::map<const Fields*,Shape> sample_shapes;
    mutable std::map<Word,bool> matching_conditions;
    struct Fit {bool valid=false;B2Features coefficients{};double error=0;Word low=0,high=0;};
    mutable std::map<std::array<Word,13>,Fit> fit_cache;
    Estimator(const Profile &p,Fields current,Ignore ign,Query q):profile(p),condition(std::move(current)),ignore(ign),query(std::move(q)) {
        for(const auto &s:p.samples)profile_samples.insert(&s);
        for(const auto &s:p.ntt)if(ntt_matches(p.conditions.at(uint(s,"condition")),condition,ignore)) {
            const auto key=ntt_scope(s);const double v=real(s,"mean_seconds");
            auto i=ntt.find(key);if(i==ntt.end() || v<i->second)ntt[key]=v;
            short_ntt.observe(key.first,key.second,v);
        }
    }
    double ntt_cost(Word n,Word batch)const {
        auto exact=ntt.find({n,batch});if(exact!=ntt.end())return exact->second;
        if(const double short_cost=short_ntt.cost(n,batch))return short_cost;
        double distance=std::numeric_limits<double>::infinity(),result=0;
        for(const auto &a:ntt) {
            const double d=std::abs(std::log2(double(n)/a.first.first))+0.5*std::abs(std::log2(double(batch)/a.first.second));
            if(d<distance){distance=d;result=a.second*double(batch)/a.first.second*
                double(n)/a.first.first*std::log2(double(n))/std::log2(double(a.first.first));}
        }
        // Relative work units only when component data is absent. A full-engine
        // anchor still determines seconds; this value is never called measured.
        return result>0?result:double(n)*batch*std::log2(double(n));
    }
    bool cached_work(const Shape &s,Work &out)const {
        const auto key=s.key();
        auto at=work_cache.find(key);
        if(at==work_cache.end()) {Work w;const bool ok=work(s,query,[&](Word n,Word b){return ntt_cost(n,b);},w,&short_ntt);
            at=work_cache.emplace(key,std::make_pair(ok,std::move(w))).first;}
        out=at->second.second;return at->second.first;
    }
    Shape shape(const Fields &s)const {
        const auto old=sample_shapes.find(&s);if(old!=sample_shapes.end())return old->second;
        Shape x;x.target_bits=unsigned(uint(s,"target_bits"));x.arithmetic_bits=unsigned(uint(s,"arithmetic_bits"));
        x.carrier=unsigned(uint(s,"carrier_exponent"));x.b1=uint(s,"b1");x.b2=uint(s,"b2");x.d=uint(s,"d");
        x.fold=uint(s,"fold_resident")!=0;x.frontier=uint(s,"frontier_resident")!=0;x.mersenne=text(s,"modulus_kind")=="mersenne";
        x.giant_chunk=optional_uint(s,"giant_chunk_points");x.chain_min=optional_uint(s,"giant_chain_min",32768);
        x.force_ladder=optional_uint(s,"giant_force_ladder")!=0;
        const auto &c=profile.conditions.at(uint(s,"condition"));
        x.batch_bytes=optional_uint(c,"env_s4_batch_mb",64)*1048576;
        x.buffers=optional_uint(c,"env_workspace_reuse_bq")?2:3;
        x.physical=optional_uint(c,"env_s4_workspace_budget")!=0;x.chunk_max=optional_uint(c,"env_s4_chunk_max");
        x.short_batch_tune=optional_uint(c,"env_s4_short_batch_tune")!=0;
        x.short_batch_hash=optional_uint(c,"env_s4_short_batch_hash");
        // Temporary Fields objects can reuse the same stack address. Cache
        // only records owned by this estimator's immutable profile.
        if(profile_samples.count(&s))sample_shapes.emplace(&s,x);return x;
    }
    bool eligible(const Fields &s,const Shape &x)const {
        if(!measured(s))return false;
        const auto id=uint(s,"condition");auto at=matching_conditions.find(id);
        if(at==matching_conditions.end())at=matching_conditions.emplace(id,matches(profile.conditions.at(id),condition,ignore)).first;
        if(!at->second)return false;
        const auto a=shape(s);
        if(a.mersenne!=x.mersenne)return false;
        if(!ignore.memory && (a.fold!=x.fold || a.frontier!=x.frontier))return false;
        return true;
    }
    RuntimeStatistics runtime_statistics(const Shape &x,bool complete_parameters=false)const {
        RuntimeStatistics out;if(runtime_version.empty())return out;
        double m2=0;
        for(const auto &s:profile.samples)if(runtime_observation(s) && measured(s) &&
            s.count("program_version") && text(s,"program_version")==runtime_version &&
            matches(profile.conditions.at(uint(s,"condition")),condition,Ignore::parse(""))) {
            const auto a=shape(s);auto left=a.key(),right=x.key();if(!complete_parameters)left[3]=right[3]=0;
            if(left!=right)continue;
            const auto n=uint(s,"repeats");const double delta=real(s,"mean_seconds")-out.seconds;
            const double sd=real(s,"stddev_seconds");
            m2+=double(n-1)*sd*sd+delta*delta*double(out.count)*double(n)/double(out.count+n);
            out.seconds+=delta*double(n)/double(out.count+n);out.count+=n;
            out.preferred_order=std::max(out.preferred_order,optional_uint(s,"preferred_order"));
        }
        out.stddev=out.count>1?std::sqrt(std::max(0.,m2/double(out.count-1))):0;return out;
    }
    bool fitted(const Shape &x,const Work &target,double &seconds,double &error,const Fields *omit=nullptr)const {
        const std::array<Word,13> key={x.target_bits,x.arithmetic_bits,Word(x.mersenne),Word(bool(x.carrier)),
            Word(x.physical),x.batch_bytes,x.buffers,x.chunk_max,Word(x.fold),Word(x.frontier),
            x.short_batch_hash,Word(x.short_batch_tune),Word(reinterpret_cast<std::uintptr_t>(omit))};
        auto encode=[](const Work &w) {
            double ntt=0,reduce=0;
            for(unsigned ph=0;ph<RequestPhaseCount;++ph){ntt+=w.ntt[ph];reduce+=w.reduce[ph];}
            return B2Features{1.,ntt,reduce,w.baby+w.giant+w.accum};
        };
        auto dot=[](const B2Features &a,const B2Features &b){double s=0;for(unsigned j=0;j<4;++j)s+=a[j]*b[j];return s;};
        auto at=fit_cache.find(key);
        if(at==fit_cache.end()) {
            Fit fit;fit.low=UINT64_MAX;
            std::vector<B2Features> features;std::vector<double> times;
            std::set<Word> ds,bounds;
            for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
                const auto a=shape(s);
                if(a.target_bits!=x.target_bits || a.arithmetic_bits!=x.arithmetic_bits ||
                   a.physical!=x.physical || a.batch_bytes!=x.batch_bytes || a.buffers!=x.buffers ||
                   a.chunk_max!=x.chunk_max || a.fold!=x.fold || a.frontier!=x.frontier ||
                   a.short_batch_hash!=x.short_batch_hash ||
                   a.short_batch_tune!=x.short_batch_tune || d_catalog::degree(a.d)<512)continue;
                Work w;if(!cached_work(a,w) || !w.exact_topology)continue;
                features.push_back(encode(w));times.push_back(real(s,"mean_seconds"));
                ds.insert(a.d);bounds.insert(a.b2);fit.low=std::min(fit.low,w.p);fit.high=std::max(fit.high,w.p);
            }
            // Qualification is unchanged: independent leave-one-out errors,
            // six curves, three D, two B2; never substitute training residuals.
            if(features.size()>=6 && features.size()<=128 && ds.size()>=3 && bounds.size()>=2) {
                bool ok=true;B2Features coefficients{};
                for(size_t i=0;i<features.size();++i) {
                    if(!fit_route_cost(features,times,i,4,coefficients)){ok=false;break;}
                    fit.error=std::max(fit.error,std::abs(dot(features[i],coefficients)-times[i])/times[i]);
                }
                const double limit=optional_real(profile.metadata,"error_limit",.08);
                if(ok && limit>0 && limit<=1 && fit.error<=limit &&
                   fit_route_cost(features,times,features.size(),4,fit.coefficients))fit.valid=true;
            }
            at=fit_cache.emplace(key,fit).first;
        }
        const auto &fit=at->second;
        if(!fit.valid || target.p<fit.low/2 || target.p>2*fit.high)return false;
        error=fit.error;seconds=dot(encode(target),fit.coefficients);
        return seconds>0 && std::isfinite(seconds);
    }
    double scaled(const Fields &a,const Shape &x,const Work &w)const {
        auto before=shape(a);
        if(before.short_batch_hash!=x.short_batch_hash)before.short_batch_tune=false;
        Work old;if(!cached_work(before,old))return 0;
        auto ratio=[](double x,double y){return y>0?x/y:1.;};
        auto phase=[&](const char *key,double scale){return optional_real(a,key)*scale;};
        double seconds=phase("phase_shape_seconds",1)+phase("phase_setup_seconds",ratio(double(x.arithmetic_bits),uint(a,"arithmetic_bits")))+
            phase("phase_baby_seconds",ratio(w.baby,old.baby))+phase("phase_accum_seconds",ratio(w.accum,old.accum))+
            phase("phase_finalize_seconds",1)+phase("phase_main_setup_seconds",1);
        auto poly_ratio=[&](unsigned ph) {
            // Separate operation/transfer features; sparse anchors cannot identify
            // every coefficient. Use an explicitly estimated positive mixture.
            // Sparse launch-bound probes do not identify the large-transform
            // arithmetic rate. Do not scale their fixed launch latency with
            // coefficient work; use an explicit conservative model instead.
            return .55*ratio(w.ntt[ph],old.ntt[ph])+.40*ratio(w.reduce[ph],old.reduce[ph])+
                .03*ratio(w.transfer[ph],old.transfer[ph])+.02*ratio(w.calls[ph],old.calls[ph]);
        };
        seconds+=phase("phase_ftree_seconds",poly_ratio(RequestFtree))+
            phase("phase_inverse_setup_seconds",poly_ratio(RequestInverse))+
            phase("phase_descent_seconds",poly_ratio(RequestDescent));
        const double giant=optional_real(a,"giant_seconds"),trees=optional_real(a,"gtrees_seconds"),fold=optional_real(a,"fold_seconds");
        const double nested=giant+trees+fold;
        const double loop_scale=nested>0?(giant*ratio(w.giant,old.giant)+trees*poly_ratio(RequestGtrees)+fold*poly_ratio(RequestFold))/nested:
            poly_ratio(RequestGtrees);
        seconds+=phase("phase_giant_loop_seconds",loop_scale);
        if(!(seconds>0)) {
            double before=old.baby+old.giant+old.accum,after=w.baby+w.giant+w.accum;
            for(unsigned ph=0;ph<RequestPhaseCount;++ph){before+=old.ntt[ph]+old.reduce[ph];after+=w.ntt[ph]+w.reduce[ph];}
            seconds=real(a,"mean_seconds")*ratio(after,before);
        }
        return seconds;
    }
    bool predict(const Shape &x,Estimate &out,const Fields *omit=nullptr)const {
        out=Estimate{};Work w;if(!cached_work(x,w))return false;
        double nearest=std::numeric_limits<double>::infinity();
        for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
            const auto a=shape(s);const double distance=2*std::abs(std::log2(double(x.arithmetic_bits)/a.arithmetic_bits))+
                std::abs(std::log2(double(w.p)/d_catalog::degree(a.d)))+
                .25*std::abs(std::log2(double(x.b2)/a.b2));
            ++out.anchors;
            if(distance<nearest) {nearest=distance;out.anchor=&s;}
        }
        if(!out.anchor)return false;
        const auto &a=*out.anchor;const auto sh=shape(a);
        out.exact=!runtime_observation(a) && x.target_bits==sh.target_bits && x.arithmetic_bits==sh.arithmetic_bits && x.d==sh.d &&
            x.b2==sh.b2 && x.batch_bytes==sh.batch_bytes && x.buffers==sh.buffers &&
            x.physical==sh.physical && x.chunk_max==sh.chunk_max && x.giant_chunk==sh.giant_chunk &&
            x.chain_min==sh.chain_min && x.force_ladder==sh.force_ladder &&
            x.fold==sh.fold && x.frontier==sh.frontier &&
            x.short_batch_tune==sh.short_batch_tune && x.short_batch_hash==sh.short_batch_hash &&
            matches(profile.conditions.at(uint(a,"condition")),condition,Ignore::parse(""));
        out.seconds=out.exact?real(a,"mean_seconds"):scaled(a,x,w);
        double fit_error=0,fit_seconds=0;bool fit=false;
        if(!out.exact && fitted(x,w,fit_seconds,fit_error,omit)) {
            fit=true;out.seconds=fit_seconds;out.model="portable_nonnegative_v1";
        }
        if(!out.exact) {
            std::vector<const Fields*> same;
            for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
                const auto sh=shape(s);
                if(sh.target_bits==x.target_bits && sh.arithmetic_bits==x.arithmetic_bits && sh.d==x.d &&
                   sh.batch_bytes==x.batch_bytes && sh.buffers==x.buffers && sh.chunk_max==x.chunk_max &&
                   sh.physical==x.physical && sh.giant_chunk==x.giant_chunk && sh.chain_min==x.chain_min &&
                   sh.force_ladder==x.force_ladder && sh.fold==x.fold && sh.frontier==x.frontier &&
                   sh.short_batch_tune==x.short_batch_tune && sh.short_batch_hash==x.short_batch_hash)same.push_back(&s);
            }
            std::sort(same.begin(),same.end(),[](const Fields *a,const Fields *b){return uint(*a,"b2")<uint(*b,"b2");});
            const Fields *lo=nullptr,*hi=nullptr;
            for(auto s:same){if(uint(*s,"b2")<=x.b2)lo=s;if(uint(*s,"b2")>=x.b2 && !hi)hi=s;}
            // Interpolate only inside observed B2 points on this actual arithmetic
            // and tree shape. Outside, retain the route/component extrapolation.
            if(lo && hi && lo!=hi && uint(*hi,"b2")>uint(*lo,"b2")) {
                const double f=double(x.b2-uint(*lo,"b2"))/double(uint(*hi,"b2")-uint(*lo,"b2"));
                out.seconds=real(*lo,"mean_seconds")+(real(*hi,"mean_seconds")-real(*lo,"mean_seconds"))*f;
                out.model="same_shape_b2_interpolation_v1";
                out.stddev=std::max(real(*lo,"stddev_seconds"),real(*hi,"stddev_seconds"));
            }
        }
        out.stddev=std::max(out.stddev,real(a,"stddev_seconds")*out.seconds/real(a,"mean_seconds"));
        out.validated=out.exact && optional_uint(a,"validation_count")>0;
        out.relative_error=fit?fit_error:out.validated?optional_real(a,"validation_max_relative_error"):.25;
        out.extrapolated=nearest>0 || !w.exact_topology;
        if(!out.exact && !fit)out.relative_error=std::max(out.relative_error,.08+.05*nearest);
        out.rank=out.seconds*(1+out.relative_error)+2*out.stddev;
        return out.seconds>0 && std::isfinite(out.rank);
    }
    // Production D replacement requires measurements at this width/B2/path.
    // B1 is deliberately absent: it is not a tune eligibility dimension.
    const Fields *observed(const Shape &x,const Fields *omit=nullptr)const {
        const Fields *best=nullptr;
        for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
            // A single production curve calibrates a width region. It does not
            // establish an exact paired benchmark with zero dispersion.
            if(runtime_observation(s))continue;
            const auto a=shape(s);
            if(a.target_bits!=x.target_bits || a.arithmetic_bits!=x.arithmetic_bits || a.carrier!=x.carrier || a.d!=x.d ||
               a.b2!=x.b2 || a.batch_bytes!=x.batch_bytes || a.physical!=x.physical ||
               a.buffers!=x.buffers || a.chunk_max!=x.chunk_max || a.giant_chunk!=x.giant_chunk ||
               a.force_ladder!=x.force_ladder || a.chain_min!=x.chain_min ||
               a.fold!=x.fold || a.frontier!=x.frontier || a.short_batch_tune!=x.short_batch_tune ||
               a.short_batch_hash!=x.short_batch_hash)continue;
            if(!best || real(s,"mean_seconds")<real(*best,"mean_seconds"))best=&s;
        }
        return best;
    }
};
} } }
