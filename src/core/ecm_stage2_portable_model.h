#pragma once
#include "ecm_stage2_tune_portable.h"
#include "ecm_stage2_tune_components.h"

namespace ecm_stage2 { namespace tune { namespace portable {
struct Shape {
    unsigned target_bits=0,arithmetic_bits=0,carrier=0;
    Word b1=0,b2=0,d=0,batch_bytes=64ull<<20,chunk_max=0;
    unsigned buffers=3;bool physical=false,fold=true,frontier=true,mersenne=false;
    Word giant_chunk=0,chain_min=32768;bool force_ladder=false;
};
struct Work {
    std::array<double,RequestPhaseCount> ntt{},reduce{},transfer{},calls{};
    double baby=0,giant=0,accum=0;
    Word p=0,points=0,largest_ntt=0;bool exact_topology=true;
};
// A cost estimate, not an arithmetic, memory, or primality check. NTT values
// outside the measured (length,batch) cells are explicitly extrapolated.
template<class Query,class NttCost> bool work(const Shape &s,Query query,NttCost ntt_cost,Work &out) {
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
        Word c=chunk_slices(n,slots,r.pairs,s.batch_bytes,s.physical?s.buffers:3,s.physical);
        if(s.chunk_max)c=std::min(c,s.chunk_max);
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
    const char *model="portable_components_v1";
};
class Estimator {
public:
    using Query=std::function<bool(Word,int,Word*,Word*)>;
    const Profile &profile;Fields condition;Ignore ignore;Query query;
    std::map<std::pair<Word,Word>,double> ntt;
    Estimator(const Profile &p,Fields current,Ignore ign,Query q):profile(p),condition(std::move(current)),ignore(ign),query(std::move(q)) {
        for(const auto &s:p.ntt)if(matches(p.conditions.at(uint(s,"condition")),condition,ignore)) {
            const auto key=ntt_scope(s);const double v=real(s,"median_seconds");
            auto i=ntt.find(key);if(i==ntt.end() || v<i->second)ntt[key]=v;
        }
    }
    double ntt_cost(Word n,Word batch)const {
        auto exact=ntt.find({n,batch});if(exact!=ntt.end())return exact->second;
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
        x.physical=optional_uint(c,"env_s4_workspace_budget")!=0;x.chunk_max=optional_uint(c,"env_s4_chunk_max");return x;
    }
    bool eligible(const Fields &s,const Shape &x)const {
        if(!measured(s) || !matches(profile.conditions.at(uint(s,"condition")),condition,ignore))return false;
        const auto a=shape(s);
        if(a.mersenne!=x.mersenne || bool(a.carrier)!=bool(x.carrier))return false;
        if(!ignore.memory && (a.fold!=x.fold || a.frontier!=x.frontier))return false;
        return true;
    }
    double scaled(const Fields &a,const Shape &x,const Work &w)const {
        Work old;if(!work(shape(a),query,[&](Word n,Word b){return ntt_cost(n,b);},old))return 0;
        auto ratio=[](double x,double y){return y>0?x/y:1.;};
        auto phase=[&](const char *key,double scale){return optional_real(a,key)*scale;};
        double seconds=phase("phase_shape_seconds",1)+phase("phase_setup_seconds",ratio(double(x.arithmetic_bits),uint(a,"arithmetic_bits")))+
            phase("phase_baby_seconds",ratio(w.baby,old.baby))+phase("phase_accum_seconds",ratio(w.accum,old.accum))+
            phase("phase_finalize_seconds",1)+phase("phase_main_setup_seconds",ratio(w.giant,old.giant));
        auto poly_ratio=[&](unsigned ph) {
            // Separate operation/transfer features; sparse anchors cannot identify
            // every coefficient. Use an explicitly estimated positive mixture.
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
        out=Estimate{};Work w;if(!work(x,query,[&](Word n,Word b){return ntt_cost(n,b);},w))return false;
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
            x.b2==sh.b2 && x.b1==sh.b1 && x.batch_bytes==sh.batch_bytes && x.fold==sh.fold && x.frontier==sh.frontier &&
            matches(profile.conditions.at(uint(a,"condition")),condition,Ignore::parse(""));
        out.seconds=out.exact?real(a,"median_seconds"):scaled(a,x,w);
        if(!out.exact) {
            std::vector<const Fields*> same;
            for(const auto &s:profile.samples)if(&s!=omit && eligible(s,x)) {
                const auto sh=shape(s);
                if(sh.target_bits==x.target_bits && sh.arithmetic_bits==x.arithmetic_bits && sh.d==x.d &&
                   sh.batch_bytes==x.batch_bytes && sh.buffers==x.buffers && sh.chunk_max==x.chunk_max &&
                   sh.physical==x.physical && sh.fold==x.fold && sh.frontier==x.frontier)same.push_back(&s);
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
        out.validated=optional_uint(a,"validation_count")>0 && x.target_bits==sh.target_bits &&
            x.arithmetic_bits==sh.arithmetic_bits && x.d==sh.d && x.b2==sh.b2;
        out.relative_error=out.validated?optional_real(a,"validation_max_relative_error"):.25;
        out.extrapolated=nearest>0 || !w.exact_topology;
        if(!out.exact)out.relative_error=std::max(out.relative_error,.08+.05*nearest);
        out.rank=out.seconds*(1+out.relative_error)+2*out.mad;
        return out.seconds>0 && std::isfinite(out.rank);
    }
};
} } }
