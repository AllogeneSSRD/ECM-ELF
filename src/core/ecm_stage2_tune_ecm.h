#pragma once
#include "ecm_stage2_tune_format.h"
#include "ecm_stage2_cost_profile.h"
#include "ecm_stage2_phase_times.h"
#include "ecm_stage2_giant_work.h"
#include "ecm_stage2_tune_grid.h"
#include <gmp.h>
#include <limits>

namespace ecm_stage2 { namespace tune {
constexpr const char *b2_prediction_model="giant_route_cost_v2";
constexpr const char *legacy_b2_prediction_model="linear_giant_points_v1";
constexpr const char *component_prediction_model="phase_ntt_loop_v1";
constexpr const char *ntt_cost_accounting="cuda_events_two_forward_product_inverse_v1";
struct EcmEffort {
    std::vector<unsigned> exponents;
    std::vector<Word> d,b2;
    int repeats=0;
    Word max_batches=0;
    unsigned tail_samples=0;
};
inline EcmEffort ecm_effort(int level) {
    if(level<1 || level>10)throw std::runtime_error("tune-level must be 1..10");
    // Known Mersenne prime exponents. Each level retains all previous inputs.
    const unsigned primes[]={521,2203,4423,9689,1279,3217,11213,607,2281,4253,127,107,9941};
    const Word ds[]={30030,60060,120120,180180,210210,360360,570570,690690,810810,1021020,1141140,1381380,1711710,2282280};
    const Word bounds[]={2600000000ull,26000000000ull,260000000000ull,2600000000000ull,8000000000000ull};
    EcmEffort e;
    e.exponents.assign(primes,primes+std::min(13,level+3));
    e.d.assign(ds,ds+level+2+(level>=7?2:0));
    const int intervals=(level-1)/2;
    // Keep low-I support at intermediate effort, too. A mixed route fit with
    // only the two endpoints and midpoint can reject the lowest chain anchor.
    const int subdivisions=level<3?1:4;
    e.b2.push_back(bounds[0]);
    for(int interval=0;interval<intervals;++interval) {
        for(int step=1;step<subdivisions;++step) {
            const double fraction=(double)step/subdivisions;
            const auto point=(Word)std::llround((double)bounds[interval]*
                std::pow((double)bounds[interval+1]/bounds[interval],fraction));
            e.b2.push_back(point);
        }
        e.b2.push_back(bounds[interval+1]);
    }
    e.repeats=2*level+1;e.max_batches=64+16*(level-1);e.tail_samples=level>=3?3:0;return e;
}
struct Mpz {
    mpz_t v;Mpz(){mpz_init(v);}~Mpz(){mpz_clear(v);}
    Mpz(const Mpz&)=delete;Mpz& operator=(const Mpz&)=delete;
};
struct Point {Mpz x,z;};
// Small, plain-GMP preparation only; not a production Stage1 implementation or
// a Stage1 timing calibration. a24=(A+2)/4, Q=[lcm(1..20)]P, sigma=26.
inline std::string benchmark_point(const std::string &n_hex) {
    Mpz n,u,v,t,a24,den,inv,xdiff,scalar;
    if(mpz_set_str(n.v,n_hex.c_str(),16) || mpz_cmp_ui(n.v,3)<=0)
        throw std::runtime_error("invalid benchmark modulus");
    mpz_set_ui(u.v,26*26-5);mpz_set_ui(v.v,4*26);
    mpz_sub(t.v,v.v,u.v);mpz_powm_ui(a24.v,t.v,3,n.v);
    mpz_mul_ui(t.v,u.v,3);mpz_add(t.v,t.v,v.v);mpz_mul(a24.v,a24.v,t.v);
    mpz_powm_ui(den.v,u.v,3,n.v);mpz_mul(den.v,den.v,v.v);mpz_mul_ui(den.v,den.v,16);
    if(!mpz_invert(inv.v,den.v,n.v))throw std::runtime_error("degenerate benchmark curve");
    mpz_mul(a24.v,a24.v,inv.v);mpz_mod(a24.v,a24.v,n.v);
    Point p,r0,r1,sum,doubled;
    mpz_powm_ui(p.x.v,u.v,3,n.v);mpz_powm_ui(p.z.v,v.v,3,n.v);
    if(!mpz_invert(inv.v,p.z.v,n.v))throw std::runtime_error("nonunit benchmark point");
    mpz_mul(xdiff.v,p.x.v,inv.v);mpz_mod(xdiff.v,xdiff.v,n.v);
    auto set=[](Point &r,const Point &q){mpz_set(r.x.v,q.x.v);mpz_set(r.z.v,q.z.v);};
    auto dbl=[&](Point &r,const Point &q) {
        Mpz a,b,e,w;
        mpz_add(a.v,q.x.v,q.z.v);mpz_mul(a.v,a.v,a.v);mpz_mod(a.v,a.v,n.v);
        mpz_sub(b.v,q.x.v,q.z.v);mpz_mul(b.v,b.v,b.v);mpz_mod(b.v,b.v,n.v);
        mpz_sub(e.v,a.v,b.v);mpz_mul(r.x.v,a.v,b.v);mpz_mod(r.x.v,r.x.v,n.v);
        mpz_mul(w.v,a24.v,e.v);mpz_add(w.v,w.v,b.v);mpz_mul(r.z.v,e.v,w.v);mpz_mod(r.z.v,r.z.v,n.v);
    };
    auto plus=[&](Point &r,const Point &a,const Point &b) {
        Mpz t0,t1,t2,t3;
        mpz_add(t0.v,a.x.v,a.z.v);mpz_sub(t1.v,b.x.v,b.z.v);mpz_mul(t0.v,t0.v,t1.v);
        mpz_sub(t2.v,a.x.v,a.z.v);mpz_add(t3.v,b.x.v,b.z.v);mpz_mul(t2.v,t2.v,t3.v);
        mpz_add(t1.v,t0.v,t2.v);mpz_sub(t3.v,t0.v,t2.v);
        mpz_mul(r.x.v,t1.v,t1.v);mpz_mod(r.x.v,r.x.v,n.v);
        mpz_mul(r.z.v,t3.v,t3.v);mpz_mul(r.z.v,r.z.v,xdiff.v);mpz_mod(r.z.v,r.z.v,n.v);
    };
    mpz_set_ui(scalar.v,1);for(unsigned i=2;i<=20;++i)mpz_lcm_ui(scalar.v,scalar.v,i);
    set(r0,p);dbl(r1,p);
    for(int i=(int)mpz_sizeinbase(scalar.v,2)-2;i>=0;--i) {
        plus(sum,r0,r1);
        if(mpz_tstbit(scalar.v,i)){dbl(doubled,r1);set(r0,sum);set(r1,doubled);}
        else {dbl(doubled,r0);set(r1,sum);set(r0,doubled);}
    }
    if(!mpz_invert(inv.v,r0.z.v,n.v))throw std::runtime_error("benchmark Stage1 point is infinity");
    mpz_mul(t.v,r0.x.v,inv.v);mpz_mod(t.v,t.v,n.v);
    auto hex=[](mpz_srcptr z){std::string s(mpz_sizeinbase(z,16)+2,'\0');mpz_get_str(&s[0],16,z);s.resize(std::char_traits<char>::length(s.c_str()));return s;};
    return hex(t.v);
}
inline std::string prime_point(unsigned exponent,std::string &n_hex) {
    const unsigned allowed[]={107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213};
    if(std::find(std::begin(allowed),std::end(allowed),exponent)==std::end(allowed))
        throw std::runtime_error("ECM tune exponent is not in the known-prime catalogue");
    Mpz n;mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,exponent);mpz_sub_ui(n.v,n.v,1);
    n_hex.resize(mpz_sizeinbase(n.v,16)+2);mpz_get_str(&n_hex[0],16,n.v);
    n_hex.resize(std::char_traits<char>::length(n_hex.c_str()));return benchmark_point(n_hex);
}
using Fields=std::map<std::string,std::string>;
inline Fields legacy_environment(const std::string &text) {
    if(text.size()<2 || text.front()!='"' || text.back()!='"')throw std::runtime_error("invalid legacy tune environment");
    Fields result;size_t a=1;
    while(a<text.size()-1) {
        const auto b=text.find(';',a),eq=text.find('=',a);
        if(b==text.npos || eq==text.npos || eq>=b || text.compare(a,4,"NTT_")!=0)throw std::runtime_error("invalid legacy tune setting");
        auto key=text.substr(a+4,eq-a-4);for(char &c:key)c=(char)std::tolower((unsigned char)c);
        auto value=text.substr(eq+1,b-eq-1);if(value.empty())value="\"\"";
        const auto parsed=fields("{\""+key+"\":"+value+"}");
        if(!result.emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate legacy tune setting");a=b+1;
    }
    return result;
}
inline const std::string &required(const Fields &f,const char *key) {
    auto i=f.find(key);if(i==f.end())throw std::runtime_error(std::string("missing ECM tune field: ")+key);return i->second;
}
inline Word uint(const Fields &f,const char *key){return cost::integer(required(f,key));}
inline double real(const Fields &f,const char *key){return cost::number(required(f,key));}
inline bool matches_giant_work_policy(const Fields &sample,Word chunk,Word minimum,bool force) {
    return !sample.count("giant_work_model") || (uint(sample,"giant_chunk_points")==chunk &&
        uint(sample,"giant_chain_min")==minimum && (uint(sample,"giant_force_ladder")!=0)==force);
}
inline std::vector<double> array(const std::string &s) {
    if(s.size()<3 || s.front()!='[' || s.back()!=']')throw std::runtime_error("empty/invalid ECM tune array");
    std::vector<double> v;size_t a=1;
    while(a<s.size()-1) {
        const auto b=s.find(',',a);auto n=s.substr(a,(b==s.npos?s.size()-1:b)-a);
        const auto first=n.find_first_not_of(" \t");const auto last=n.find_last_not_of(" \t");
        if(first==n.npos)throw std::runtime_error("invalid ECM tune array item");
        v.push_back(cost::number(n.substr(first,last-first+1)));
        if(b==s.npos)break;a=b+1;if(a==s.size()-1)throw std::runtime_error("trailing ECM tune comma");
    }
    return v;
}
inline double median(std::vector<double> v) {
    if(v.empty())throw std::runtime_error("empty ECM tune samples");std::sort(v.begin(),v.end());
    return v.size()%2?v[v.size()/2]:v[v.size()/2-1]+(v[v.size()/2]-v[v.size()/2-1])/2;
}
inline double mad(const std::vector<double> &v) {const auto m=median(v);auto a=v;for(auto &x:a)x=std::abs(x-m);return median(a);}
inline std::pair<Word,Word> ntt_scope(const Fields &s) {
    return {uint(s,"length"),uint(s,"batch")};
}
inline std::string ntt_section(const Fields &s) {
    const auto key=ntt_scope(s);return "ntt.length_"+std::to_string(key.first)+".slices_"+std::to_string(key.second);
}
inline void validate_ntt_measurement(const Fields &s) {
    const auto key=ntt_scope(s);const auto n=key.first,b=key.second;
    if(n<8 || n>(1ull<<27) || (n&(n-1)) || !b || b>65535 || uint(s,"bad") ||
       required(s,"status")!="\"measured\"" || required(s,"unit")!="\"field_convolution\"" ||
       required(s,"reference_kind")!="\"gmp_3x3_distinct_constant_v1\"" || uint(s,"verified_words_per_sample")!=n*b)
        throw std::runtime_error("invalid/unchecked NTT measurement");
    Word log=0;for(auto size=n;size>1;size>>=1)++log;
    if(uint(s,"log2_length")!=log)throw std::runtime_error("inconsistent NTT length");
    const auto seconds=array(required(s,"seconds"));const auto m=median(seconds);
    auto close=[](double a,double b){return std::abs(a-b)<=1e-9*std::max(std::abs(a),std::abs(b));};
    if(seconds.size()!=uint(s,"repeats") || seconds.size()>1000 || *std::min_element(seconds.begin(),seconds.end())<=0 ||
       !close(m,real(s,"median_seconds")) || !close(double(b)/m,real(s,"conv_iter_per_s")) ||
       !close(1/m,real(s,"batch_iter_per_s")))throw std::runtime_error("inconsistent NTT samples/statistics");
}
inline Word environment_uint(const Fields &environment,const char *key,Word fallback) {
    const auto found=environment.find(key);return found==environment.end()?fallback:cost::integer(found->second);
}
inline Fields ntt_packing_policy(const Fields &environment) {
    const auto override=environment_uint(environment,"s4_batch_mb",0);
    const auto batch=override?override:((environment_uint(environment,"s4_hostpack",0) ||
        !environment_uint(environment,"s4_pack_direct",1))?32:64);
    if(!batch || batch>std::numeric_limits<Word>::max()/1048576)throw std::runtime_error("invalid NTT chunk budget");
    return {{"accounting",std::string("\"")+ntt_cost_accounting+"\""},
        {"batch_bytes",std::to_string(batch*1048576)},
        {"buffers",environment_uint(environment,"arena_workspace_pool",1) && environment_uint(environment,"workspace_reuse_bq",0)?"2":"3"},
        {"physical_chunks",environment_uint(environment,"s4_workspace_budget",0)?"1":"0"},
        {"chunk_max",std::to_string(environment_uint(environment,"s4_chunk_max",0))}};
}
inline std::string sample_scope(const Fields &sample) {
    std::string key;
    for(const char *field:{"target_bits","arithmetic_bits","carrier_exponent","modulus_kind","b1","b2","d"})
        key+=required(sample,field)+":";
    return key;
}
inline void validate_paired_costs(const Fields &f,const std::vector<double> &totals) {
    auto paired=[&](const std::string &samples,const std::string &middle) {
        auto values=array(required(f,samples.c_str()));
        if(values.size()!=totals.size() || *std::min_element(values.begin(),values.end())<0 ||
            std::abs(median(values)-real(f,middle.c_str()))>1e-9*std::max(1.,median(values)))
            throw std::runtime_error("inconsistent paired ECM cost samples");
        return values;
    };
    auto close=[](double a,double b){return std::abs(a-b)<=1e-9*std::max({1.,a,b});};
    if(f.count("phase_accounting")) {
        if(required(f,"phase_accounting")!=std::string("\"")+timing::contract+"\"")throw std::runtime_error("unsupported ECM phase accounting");
        const auto init=paired("init_samples","init_seconds"),main=paired("main_samples","main_seconds");
        std::array<std::vector<double>,timing::count> phases;
        for(size_t i=0;i<timing::count;++i) {
            const auto key=std::string("phase_")+timing::names[i];
            phases[i]=paired(key+"_samples",key+"_seconds");
        }
        for(size_t r=0;r<totals.size();++r) {
            double pre=0,post=0;for(size_t i=0;i<4;++i)pre+=phases[i][r];for(size_t i=4;i<timing::count;++i)post+=phases[i][r];
            if(!close(pre,init[r]) || !close(post,main[r]) || !close(pre+post,totals[r]))
                throw std::runtime_error("ECM exclusive phases do not conserve engine time");
        }
    } else {
        for(const auto &field:f)if(field.first.compare(0,6,"phase_")==0 || field.first=="init_samples" || field.first=="main_samples")
            throw std::runtime_error("paired ECM phases missing accounting contract");
    }
    if(f.count("worker_accounting")) {
        if(required(f,"worker_accounting")!="\"spawn_wait_exit_v1\"")throw std::runtime_error("unsupported ECM worker accounting");
        const auto workers=paired("worker_samples","worker_seconds"),overheads=paired("worker_overhead_samples","worker_overhead_seconds");
        if(std::abs(mad(workers)-real(f,"worker_mad_seconds"))>1e-9*std::max(1.,mad(workers)))
            throw std::runtime_error("inconsistent ECM worker spread");
        for(size_t i=0;i<workers.size();++i)if(!close(workers[i],totals[i]+overheads[i]))
            throw std::runtime_error("ECM worker and engine timing mismatch");
    } else for(const auto &field:f)if(field.first.compare(0,7,"worker_")==0)
        throw std::runtime_error("ECM worker samples missing accounting contract");
}
inline void validate_sample(const Fields &f) {
    const auto bits=uint(f,"target_bits"),s=uint(f,"arithmetic_bits"),p=uint(f,"carrier_exponent");
    const auto b1=uint(f,"b1"),b2=uint(f,"b2"),d=uint(f,"d"),leaves=uint(f,"p");
    if(bits<2 || bits>16384 || s<bits || s>16384 || (p && p!=s) ||
       b1<2 || b2<=b1 || b2>(Word)INT64_MAX-8192 || d<6 || d%2 || d>200000000 ||
       leaves!=cost::phi(d)/2 || uint(f,"giant_points")!=b2/d+2 || uint(f,"giant_points")<=leaves ||
       uint(f,"clean")!=1 || uint(f,"hits") || uint(f,"bad") || !uint(f,"selftest_cases") || !uint(f,"checked") ||
       uint(f,"fold_resident")>1 || uint(f,"frontier_resident")>1 || !uint(f,"required_free_bytes"))
        throw std::runtime_error("invalid/unchecked ECM tune scope");
    const auto kind=required(f,"modulus_kind");
    if(kind!="\"mersenne\"" && kind!="\"generic\"")throw std::runtime_error("invalid ECM arithmetic kind");
    if(p && kind!="\"mersenne\"")throw std::runtime_error("carrier must use Mersenne arithmetic");
    if(!p && bits!=s)throw std::runtime_error("ordinary ECM tune width mismatch");
    const auto v=array(required(f,"seconds"));
    if(v.size()!=uint(f,"repeats") || v.size()>1000 || *std::min_element(v.begin(),v.end())<=0 ||
       std::abs(median(v)-real(f,"median_seconds"))>1e-9*std::max(1.,median(v)) ||
       std::abs(mad(v)-real(f,"mad_seconds"))>1e-9*std::max(1.,mad(v)))
        throw std::runtime_error("inconsistent ECM tune repetitions/statistics");
    for(const char *phase:{"init_seconds","main_seconds","giant_seconds","gtrees_seconds","fold_seconds","descent_seconds","inverse_seconds","accum_seconds"})real(f,phase);
    validate_paired_costs(f,v);
    if(f.count("giant_work_model")) {
        if(required(f,"giant_work_model")!="\"chunk_routes_v1\"" || uint(f,"giant_force_ladder")>1)
            throw std::runtime_error("unsupported giant work contract");
        GiantWork work;
        const auto chunk=uint(f,"giant_chunk_points");
        if(!chunk || chunk%leaves || !giant_work(uint(f,"giant_points"),d,chunk,
            uint(f,"giant_chain_min"),uint(f,"giant_force_ladder")!=0,work) ||
            uint(f,"giant_chain_points")!=work.chain_points || uint(f,"giant_ladder_points")!=work.ladder_points ||
            uint(f,"giant_chain_chunks")!=work.chain_chunks || uint(f,"giant_ladder_chunks")!=work.ladder_chunks ||
            uint(f,"giant_ladder_steps")!=work.ladder_steps)
            throw std::runtime_error("inconsistent giant route work");
    } else for(const auto &field:f)if(field.first.compare(0,12,"giant_chain_")==0 ||
        field.first.compare(0,13,"giant_ladder_")==0 || field.first=="giant_chunk_points" || field.first=="giant_force_ladder")
        throw std::runtime_error("giant route work missing contract");
}
inline std::string ecm_table(const Fields &f,size_t index) {
    validate_sample(f);std::ostringstream out;out<<"\n[ecm.sample_"<<index<<"]\n";
    for(const auto &x:f)out<<x.first<<" = "<<x.second<<'\n';return out.str();
}
// Bounded reader for the generated flat TOML subset; no dependency on Python.
// Additional scalar performance fields are retained for extensions. Required
// schema, completion, scopes and duplicate keys are checked independently.
struct EcmProfile {
    Fields profile,device,policy,environment,summary;
    std::vector<Fields> samples;
    Fields ntt_policy;
    std::map<std::pair<Word,Word>,Fields> ntt_samples;
    static EcmProfile load(const std::filesystem::path &path) {
        std::error_code error;const auto size=std::filesystem::file_size(path,error);
        // Level 10 has up to 3094 scopes, each with 21 paired cost samples.
        if(error || size>64*1048576)throw std::runtime_error("ECM tune profile missing or exceeds 64MiB");
        std::ifstream in(path,std::ios::binary);EcmProfile p;Fields *table=nullptr;
        std::set<std::string> sections;std::string line;
        std::map<std::string,Fields> ntt_tables;
        auto trim=[](std::string s){const auto a=s.find_first_not_of(" \t\r\n");if(a==s.npos)return std::string{};return s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);};
        bool first_line=true;
        while(std::getline(in,line)) {
            if(first_line && line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);first_line=false;
            line=trim(line);if(line.empty() || line[0]=='#')continue;
            if(line[0]=='[') {
                if(line.back()!=']')throw std::runtime_error("invalid ECM tune table");
                const auto name=line.substr(1,line.size()-2);
                if(!sections.insert(name).second)throw std::runtime_error("duplicate ECM tune table");
                if(name=="profile")table=&p.profile;else if(name=="device")table=&p.device;
                else if(name=="policy")table=&p.policy;else if(name=="summary")table=&p.summary;
                else if(name=="policy.environment")table=&p.environment;
                else if(name=="ntt.policy")table=&p.ntt_policy;
                else if(name.compare(0,11,"ntt.length_")==0)table=&ntt_tables[name];
                else if(name.compare(0,11,"ecm.sample_")==0) {
                    cost::integer(name.substr(11));if(p.samples.size()>=4096)throw std::runtime_error("too many ECM tune samples");
                    p.samples.emplace_back();table=&p.samples.back();
                } else throw std::runtime_error("unsupported ECM tune table");
            } else {
                const auto eq=line.find('=');if(!table || eq==line.npos)throw std::runtime_error("invalid ECM tune row");
                const auto key=trim(line.substr(0,eq)),value=trim(line.substr(eq+1));
                const auto parsed=fields("{\""+key+"\":"+value+"}");
                if(!table->emplace(key,parsed.at(key)).second)throw std::runtime_error("duplicate ECM tune key");
            }
        }
        if(!in.eof() || (uint(p.profile,"format")!=2 && uint(p.profile,"format")!=3 && uint(p.profile,"format")!=4) || required(p.profile,"unit")!="\"full_stage2\"" ||
           uint(p.profile,"algorithm_revision")!=1 || uint(p.summary,"complete")!=1 || uint(p.summary,"failed") ||
           uint(p.summary,"measured")!=p.samples.size() || p.samples.empty())throw std::runtime_error("incomplete/unsupported ECM tune profile");
        if(uint(p.profile,"effort_level")<1 || uint(p.profile,"effort_level")>10 ||
           !uint(p.profile,"repeats") || uint(p.profile,"repeats")>1000 || uint(p.profile,"warmups")!=1)
            throw std::runtime_error("invalid ECM tune effort metadata");
        if(p.profile.count("prediction_model") && required(p.profile,"prediction_model")!=std::string("\"")+b2_prediction_model+"\"" &&
           required(p.profile,"prediction_model")!=std::string("\"")+legacy_b2_prediction_model+"\"")
            throw std::runtime_error("unsupported ECM tune prediction model");
        const bool sampled=p.profile.count("sampling_model")!=0;
        if(sampled!=bool(p.profile.count("tail_samples")) || (sampled &&
           (required(p.profile,"sampling_model")!=std::string("\"")+tail_sampling_model+"\"" || uint(p.profile,"tail_samples")>16)))
            throw std::runtime_error("invalid ECM tune sampling metadata");
        if(!cost::hex(required(p.device,"uuid_hex").substr(1,32),32) || required(p.device,"uuid_hex").size()!=34)
            throw std::runtime_error("invalid ECM tune device");
        for(const char *key:{"sm_major","sm_minor","cuda_runtime","cuda_driver","gl_fixed_mode","outer_unroll_u","add_sub_mask"})uint(p.device,key);
        for(const char *key:{"batch_mb","arena_mb","fold_mb"})uint(p.policy,key);
        if(uint(p.profile,"format")==2) {
            if(sections.count("policy.environment"))throw std::runtime_error("mixed tune environment schemas");
            p.environment=legacy_environment(required(p.policy,"environment"));
        } else {
            if(!sections.count("policy.environment") || p.policy.count("environment"))throw std::runtime_error("missing/mixed named tune environment");
            if(uint(p.profile,"max_batches")>1048576)throw std::runtime_error("invalid tune batch limit");
        }
        if(uint(p.profile,"format")==4) {
            for(const auto &setting:p.environment)cost::integer(setting.second);
            if(required(p.profile,"component_model")!=std::string("\"")+component_prediction_model+"\"" ||
               p.ntt_policy!=ntt_packing_policy(p.environment) || ntt_tables.empty() || ntt_tables.size()>65536 ||
               uint(p.summary,"ntt_measured")!=ntt_tables.size())throw std::runtime_error("incomplete ECM/NTT component profile");
            for(const auto &entry:ntt_tables) {
                validate_ntt_measurement(entry.second);
                if(entry.first!=ntt_section(entry.second) || !p.ntt_samples.emplace(ntt_scope(entry.second),entry.second).second)
                    throw std::runtime_error("duplicate/mismatched embedded NTT shape");
            }
        } else if(!ntt_tables.empty() || !p.ntt_policy.empty() || p.profile.count("component_model") || p.summary.count("ntt_measured"))
            throw std::runtime_error("embedded NTT measurements require ECM format 4");
        std::set<std::string> scopes;
        for(const auto &s:p.samples) {
            validate_sample(s);if(uint(s,"repeats")!=uint(p.profile,"repeats"))throw std::runtime_error("inconsistent ECM profile repetitions");
            if(s.count("sampling_source")) {
                const auto &source=required(s,"sampling_source");
                if(!sampled || (source!="\"base\"" && source!="\"ladder_tail\"" && source!="\"chain_anchor\""))
                    throw std::runtime_error("invalid ECM tune sampling source");
                if(source!="\"base\"") {
                    if(!uint(p.profile,"tail_samples") || !s.count("giant_work_model"))throw std::runtime_error("adaptive sample lacks route policy");
                    if((source=="\"ladder_tail\"")!=(uint(s,"giant_ladder_steps")!=0))throw std::runtime_error("adaptive sample route mismatch");
                }
            }
            if(uint(p.profile,"format")>=3 && uint(p.profile,"max_batches") &&
                (uint(s,"giant_points")+uint(s,"p")-1)/uint(s,"p")>uint(p.profile,"max_batches"))throw std::runtime_error("ECM sample exceeds declared batch limit");
            if(!scopes.insert(sample_scope(s)).second)throw std::runtime_error("duplicate ECM tune measurement scope");
        }
        return p;
    }
    bool matches(const EcmStage2DeviceInfo &d,Word batch,Word arena,Word fold,const Fields &env,unsigned add_sub_mask)const {
        return required(device,"uuid_hex")==std::string("\"")+d.uuid_hex+"\"" && uint(device,"sm_major")==d.major &&
            uint(device,"sm_minor")==d.minor && uint(device,"cuda_runtime")==d.runtime && uint(device,"cuda_driver")==d.driver &&
            uint(device,"gl_fixed_mode")==d.fixed_mode && uint(device,"outer_unroll_u")==d.outer_unroll_u &&
            uint(device,"add_sub_mask")==add_sub_mask && uint(policy,"batch_mb")==batch && uint(policy,"arena_mb")==arena &&
            uint(policy,"fold_mb")==fold && environment==env;
    }
};
// Merge only comparable measurements. Repeated scopes use the last supplied
// profile, rather than mixing trials collected under unknown thermal conditions.
inline EcmProfile merge_ecm_profiles(const std::vector<EcmProfile> &inputs) {
    if(inputs.empty() || inputs.size()>64)throw std::runtime_error("merge requires 1..64 ECM tune profiles");
    EcmProfile result=inputs.front();result.policy.erase("environment");result.samples.clear();result.ntt_samples.clear();
    std::map<std::string,size_t> positions;Word effort=0,limit=0,skipped=0,replaced=0,tail_samples=0;bool unlimited=false,sampled=false;
    for(const auto &input:inputs) {
        if(input.profile.count("prediction_model"))result.profile["prediction_model"]=required(input.profile,"prediction_model");
        auto policy=input.policy;policy.erase("environment");
        if(input.device!=result.device || policy!=result.policy || input.environment!=result.environment)
            throw std::runtime_error("ECM tune merge device or memory/backend policy mismatch");
        for(const char *field:{"unit","algorithm_revision","repeats","warmups"})
            if(required(input.profile,field)!=required(result.profile,field))
                throw std::runtime_error(std::string("ECM tune merge metadata mismatch: ")+field);
        effort=std::max(effort,uint(input.profile,"effort_level"));
        if(input.profile.count("sampling_model")) {sampled=true;tail_samples=std::max(tail_samples,uint(input.profile,"tail_samples"));}
        if(uint(input.profile,"format")==2 || !uint(input.profile,"max_batches"))unlimited=true;
        else limit=std::max(limit,uint(input.profile,"max_batches"));
        const auto skipped_field=input.summary.find("skipped");
        const auto count=skipped_field==input.summary.end()?0:cost::integer(skipped_field->second);
        if(count>std::numeric_limits<Word>::max()-skipped)throw std::runtime_error("merged skip count overflow");
        skipped+=count;
        for(const auto &sample:input.samples) {
            const auto key=sample_scope(sample);const auto existing=positions.find(key);
            if(existing==positions.end()) {
                if(result.samples.size()>=4096)throw std::runtime_error("merged ECM tune exceeds 4096 samples");
                positions.emplace(key,result.samples.size());result.samples.push_back(sample);
            } else {result.samples[existing->second]=sample;++replaced;}
        }
        for(const auto &entry:input.ntt_samples) {
            const auto found=result.ntt_samples.find(entry.first);
            if(found!=result.ntt_samples.end() && found->second!=entry.second)
                throw std::runtime_error("ECM merge has ambiguous NTT measurement");
            result.ntt_samples[entry.first]=entry.second;
        }
    }
    result.profile["format"]="3";result.profile["effort_level"]=std::to_string(effort);
    result.profile["max_batches"]=std::to_string(unlimited?0:limit);
    if(sampled){result.profile["sampling_model"]=std::string("\"")+tail_sampling_model+"\"";result.profile["tail_samples"]=std::to_string(tail_samples);}
    result.summary={{"complete","1"},{"failed","0"},{"measured",std::to_string(result.samples.size())},
                    {"skipped",std::to_string(skipped)},{"merged_profiles",std::to_string(inputs.size())},
                    {"replaced_scopes",std::to_string(replaced)}};
    if(!result.ntt_samples.empty()) {
        result.profile["format"]="4";result.profile["component_model"]=std::string("\"")+component_prediction_model+"\"";
        result.ntt_policy=ntt_packing_policy(result.environment);result.summary["ntt_measured"]=std::to_string(result.ntt_samples.size());
    } else {result.ntt_policy.clear();result.profile.erase("component_model");}
    return result;
}
inline std::string ecm_profile_text(const EcmProfile &profile) {
    std::ostringstream out;out<<"# Full ECM Stage2 measurements. Seconds; bytes.\n";
    auto emit=[&](const char *name,const Fields &fields) {
        out<<'\n'<<'['<<name<<"]\n";for(const auto &field:fields)out<<field.first<<" = "<<field.second<<'\n';
    };
    emit("profile",profile.profile);emit("device",profile.device);emit("policy",profile.policy);
    emit("policy.environment",profile.environment);
    for(size_t i=0;i<profile.samples.size();++i)out<<ecm_table(profile.samples[i],i);
    if(!profile.ntt_samples.empty()) {
        emit("ntt.policy",profile.ntt_policy);
        for(const auto &entry:profile.ntt_samples)emit(ntt_section(entry.second).c_str(),entry.second);
    }
    emit("summary",profile.summary);return out.str();
}
} }
