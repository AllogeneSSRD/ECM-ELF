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
};
struct Work {
    std::array<double,RequestPhaseCount> ntt{},reduce{},transfer{},calls{};
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
    out.p=cost::phi(s.d)/2;out.points=s.b2/s.d+2;
    RequestProgram program;if(!request_program(out.p,std::max(out.points,out.p+1),program))return false;
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
struct Estimate {
    double seconds=0,mad=0,relative_error=0.25,rank=0;
    Word anchors=0;bool validated=false,exact=false,extrapolated=false;
    const Fields *anchor=nullptr;
    const char *model="portable_components_v2";
};
class Estimator {
public:
    using Query=std::function<bool(Word,int,Word*,Word*)>;
    const Profile &profile;Fields condition;Ignore ignore;Query query;
    std::map<std::pair<Word,Word>,double> ntt;
    ShortNttBatchPolicy short_ntt;
    Estimator(const Profile &p,Fields current,Ignore ign,Query q):profile(p),condition(std::move(current)),ignore(ign),query(std::move(q)) {
        for(const auto &s:p.ntt)if(ntt_matches(p.conditions.at(uint(s,"condition")),condition,ignore)) {
            const auto key=ntt_scope(s);const double v=real(s,"median_seconds");
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
    Shape shape(const Fields &s)const {
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
        x.short_batch_hash=optional_uint(c,"env_s4_short_batch_hash");return x;
    }
    bool eligible(const Fields &s,const Shape &x)const {
        if(!measured(s) || !matches(profile.conditions.at(uint(s,"condition")),condition,ignore))return false;
        const auto a=shape(s);
        if(a.mersenne!=x.mersenne || bool(a.carrier)!=bool(x.carrier))return false;
        if(!ignore.memory && (a.fold!=x.fold || a.frontier!=x.frontier))return false;
        return true;
    }
    bool fitted(const Shape &x,const Work &target,double &seconds,double &error,const Fields *omit=nullptr)const {
        std::vector<B2Features> features;std::vector<double> times;
        std::set<Word> ds,bounds;Word low_p=UINT64_MAX,high_p=0;
        auto encode=[](const Work &w) {
            double ntt=0,reduce=0;
            for(unsigned ph=0;ph<RequestPhaseCount;++ph){ntt+=w.ntt[ph];reduce+=w.reduce[ph];}
            return B2Features{1.,ntt,reduce,w.baby+w.giant+w.accum};
        };
        for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
            const auto a=shape(s);
            if(a.target_bits!=x.target_bits || a.arithmetic_bits!=x.arithmetic_bits ||
               a.physical!=x.physical || a.batch_bytes!=x.batch_bytes || a.buffers!=x.buffers ||
               a.chunk_max!=x.chunk_max || a.fold!=x.fold || a.frontier!=x.frontier ||
               a.short_batch_hash!=x.short_batch_hash ||
               a.short_batch_tune!=x.short_batch_tune || cost::phi(a.d)/2<512)continue;
            Work w;if(!work(a,query,[&](Word n,Word b){return ntt_cost(n,b);},w,&short_ntt) || !w.exact_topology)continue;
            features.push_back(encode(w));times.push_back(real(s,"median_seconds"));
            ds.insert(a.d);bounds.insert(a.b2);low_p=std::min(low_p,w.p);high_p=std::max(high_p,w.p);
        }
        // An intercept separates setup from variable work. Qualification uses
        // held-out full-engine curves, never a residual against fitted rows.
        if(features.size()<6 || features.size()>128 || ds.size()<3 || bounds.size()<2 ||
            target.p<low_p/2 || target.p>2*high_p)return false;
        auto dot=[](const B2Features &a,const B2Features &b){double s=0;for(unsigned j=0;j<4;++j)s+=a[j]*b[j];return s;};
        error=0;B2Features coefficients{};
        for(size_t i=0;i<features.size();++i) {
            if(!fit_route_cost(features,times,i,4,coefficients))return false;
            error=std::max(error,std::abs(dot(features[i],coefficients)-times[i])/times[i]);
        }
        const double limit=optional_real(profile.metadata,"error_limit",.08);
        if(!(limit>0) || limit>1 || error>limit || !fit_route_cost(features,times,features.size(),4,coefficients))return false;
        seconds=dot(encode(target),coefficients);return seconds>0 && std::isfinite(seconds);
    }
    double scaled(const Fields &a,const Shape &x,const Work &w)const {
        auto before=shape(a);
        if(before.short_batch_hash!=x.short_batch_hash)before.short_batch_tune=false;
        Work old;if(!work(before,query,[&](Word n,Word b){return ntt_cost(n,b);},old,&short_ntt))return 0;
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
            seconds=real(a,"median_seconds")*ratio(after,before);
        }
        return seconds;
    }
    bool predict(const Shape &x,Estimate &out,const Fields *omit=nullptr)const {
        out=Estimate{};Work w;if(!work(x,query,[&](Word n,Word b){return ntt_cost(n,b);},w,&short_ntt))return false;
        double nearest=std::numeric_limits<double>::infinity();
        for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
            const auto a=shape(s);const double distance=2*std::abs(std::log2(double(x.arithmetic_bits)/a.arithmetic_bits))+
                std::abs(std::log2(double(w.p)/(cost::phi(a.d)/2)))+
                .25*std::abs(std::log2(double(x.b2)/a.b2));
            ++out.anchors;
            if(distance<nearest) {nearest=distance;out.anchor=&s;}
        }
        if(!out.anchor)return false;
        const auto &a=*out.anchor;const auto sh=shape(a);
        out.exact=x.target_bits==sh.target_bits && x.arithmetic_bits==sh.arithmetic_bits && x.d==sh.d &&
            x.b2==sh.b2 && x.batch_bytes==sh.batch_bytes && x.buffers==sh.buffers &&
            x.physical==sh.physical && x.chunk_max==sh.chunk_max && x.giant_chunk==sh.giant_chunk &&
            x.chain_min==sh.chain_min && x.force_ladder==sh.force_ladder &&
            x.fold==sh.fold && x.frontier==sh.frontier &&
            x.short_batch_tune==sh.short_batch_tune && x.short_batch_hash==sh.short_batch_hash &&
            matches(profile.conditions.at(uint(a,"condition")),condition,Ignore::parse(""));
        out.seconds=out.exact?real(a,"median_seconds"):scaled(a,x,w);
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
                out.seconds=real(*lo,"median_seconds")+(real(*hi,"median_seconds")-real(*lo,"median_seconds"))*f;
                out.model="same_shape_b2_interpolation_v1";
                out.mad=std::max(real(*lo,"mad_seconds"),real(*hi,"mad_seconds"));
            }
        }
        out.mad=std::max(out.mad,real(a,"mad_seconds")*out.seconds/real(a,"median_seconds"));
        out.validated=out.exact && optional_uint(a,"validation_count")>0;
        out.relative_error=fit?fit_error:out.validated?optional_real(a,"validation_max_relative_error"):.25;
        out.extrapolated=nearest>0 || !w.exact_topology;
        if(!out.exact && !fit)out.relative_error=std::max(out.relative_error,.08+.05*nearest);
        out.rank=out.seconds*(1+out.relative_error)+2*out.mad;
        return out.seconds>0 && std::isfinite(out.rank);
    }
    // Production D replacement requires measurements at this width/B2/path.
    // B1 is deliberately absent: it is not a tune eligibility dimension.
    const Fields *observed(const Shape &x)const {
        const Fields *best=nullptr;
        for(const auto &s:profile.samples)if(eligible(s,x)) {
            const auto a=shape(s);
            if(a.target_bits!=x.target_bits || a.arithmetic_bits!=x.arithmetic_bits || a.d!=x.d ||
               a.b2!=x.b2 || a.batch_bytes!=x.batch_bytes || a.physical!=x.physical ||
               a.buffers!=x.buffers || a.chunk_max!=x.chunk_max || a.giant_chunk!=x.giant_chunk ||
               a.force_ladder!=x.force_ladder || a.chain_min!=x.chain_min ||
               a.fold!=x.fold || a.frontier!=x.frontier || a.short_batch_tune!=x.short_batch_tune ||
               a.short_batch_hash!=x.short_batch_hash)continue;
            if(!best || real(s,"median_seconds")<real(*best,"median_seconds"))best=&s;
        }
        return best;
    }
};
} } }
