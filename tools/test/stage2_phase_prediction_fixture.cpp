#include "../../src/ecm/stage2/planning/runtime_tune.h"
#include <iostream>

namespace p=ecm_stage2::tune::portable;
using ecm_stage2::Word;
using ecm_stage2::tune::Fields;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}

// Independent exact packing oracle for CPU model tests. Production still uses
// ntt_shape_query; the Goldilocks coefficient bound is evaluated with GMP here.
static bool packing(Word m,int bits,Word *n,Word *slots) {
    static std::map<std::pair<Word,int>,Word> cache;
    const auto key=std::make_pair(m,bits);auto at=cache.find(key);
    if(at==cache.end()) {
        Word log=1;while((Word(1)<<log)<m)++log;
        const Word slot=2*Word(bits)+log;Word length=0;
        mpz_t a,b,prime;mpz_inits(a,b,prime,nullptr);
        mpz_set_str(prime,"18446744069414584321",10);
        for(int digit=62;digit>=1;--digit) {
            const Word stride=(slot+digit-1)/digit,terms=m*stride;
            mpz_set_ui(a,1);mpz_mul_2exp(a,a,digit);mpz_sub_ui(a,a,1);mpz_mul(a,a,a);
            mpz_import(b,1,-1,sizeof(terms),0,0,&terms);mpz_mul(a,a,b);
            if(mpz_cmp(a,prime)>=0)continue;
            length=2;while(length<2*m*stride+1)length*=2;
            if(length>(Word(1)<<29))length=0;
            break;
        }
        mpz_clears(a,b,prime,nullptr);at=cache.emplace(key,length).first;
    }
    *n=at->second;*slots=2*m-1;return *n!=0;
}
static p::Shape candidate(p::Shape s,Word d) {
    s.d=d;const Word degree=ecm_stage2::d_catalog::degree(d);
    const Word capacity=(Word(256)<<20)/(((s.arithmetic_bits+63)/64)*16);
    s.giant_chunk=(capacity/degree+(capacity%degree!=0))*degree;
    s.canonicalize_point_policy();return s;
}
static p::Profile fixture() {
    p::Profile profile;profile.condition({{"uuid_hex","\"fixture\""},{"sm_major","8"},{"sm_minor","9"},
        {"cuda_driver","13030"},{"cuda_runtime","13030"},{"gl_fixed_mode","3"},{"outer_unroll_u","0"},{"add_sub_mask","1"},
        {"env_s4_batch_mb","256"},{"env_giant_chain_block","32"},{"env_s4_check_every","8"}});
    Fields a={{"condition","0"},{"target_bits","12285"},{"arithmetic_bits","12511"},
        {"carrier_exponent","12511"},{"modulus_kind","\"mersenne\""},{"b1","50000"},{"b2","273142759890"},
        {"d","360360"},{"p","34560"},{"fold_resident","1"},{"frontier_resident","1"},
        {"execution_path","\"resident\""},{"source","\"measured\""},{"benchmark_kind","\"runtime_save\""},
        {"clean","1"},{"bad","0"},{"hits","0"},{"selftest_cases","1"},{"checked","96"},
        {"repeats","2"},{"mean_seconds","100"},{"stddev_seconds","1"},
        {"giant_chunk_points","103680"},{"giant_chain_min","8192"},{"giant_force_ladder","0"},
        {"groot_balanced","1"},{"giant_chain_block","32"},{"baby_coop","32"},{"giant_coop","32"},{"seed_coop","32"},
        {"runtime_sample_id","\"phase-fixture\""},{"giant_seconds","8"},{"gtrees_seconds","32"},{"fold_seconds","14"}};
    const double phases[]={.1,.5,6,10,.1,5,55,20,3,.3};
    for(size_t i=0;i<ecm_stage2::timing::count;++i)a[std::string("phase_")+ecm_stage2::timing::names[i]+"_seconds"]=p::numeric(phases[i]);
    profile.update(a);return profile;
}
static void tests() {
    auto profile=fixture();auto condition=profile.conditions.at(0);
    p::Estimator e(profile,condition,p::Ignore::parse(""),packing);
    const auto anchor=e.shape(profile.samples[0]),target=candidate(anchor,510510);
    p::Work w,before;require(e.cached_work(target,w) && e.cached_work(anchor,before),"work unavailable");
    require(w.largest_ntt==(Word(1)<<27) && before.largest_ntt==w.largest_ntt,"packing cliff changed");
    p::WidthPredictor predictor(e,nullptr,true);p::WidthEstimate estimate;
    predictor.predict(target,estimate);
    if(std::string(estimate.model)!="same_width_phase_projection_v1") {
        std::cerr<<"phase test: "<<estimate.model<<" eligible="<<e.eligible(profile.samples[0],target)<<" old/new intervals=";
        for(auto key:p::interval(anchor,before).key)std::cerr<<key<<',';std::cerr<<" / ";
        for(auto key:p::interval(target,w).key)std::cerr<<key<<',';std::cerr<<'\n';
    }
    require(predictor.predict(target,estimate) && std::string(estimate.model)=="same_width_phase_projection_v1" &&
        !estimate.exact && !estimate.validated && !estimate.runtime_samples && estimate.relative_error>=.15 &&
        estimate.seconds==e.scaled(profile.samples[0],target,w),"exclusive phases were not projected");
    require(predictor.predict(anchor,estimate) && estimate.exact && estimate.seconds==100,"prediction overrode exact data");
    p::WidthPredictor default_predictor(e);
    require(default_predictor.predict(target,estimate) && std::string(estimate.model)!="same_width_phase_projection_v1",
        "phase nomination leaked into default/model-only selection");
    auto b1=target;b1.b1*=100;
    require(predictor.predict(b1,estimate) && std::string(estimate.model)=="same_width_phase_projection_v1","B1 blocked phase reuse");
    auto rejects=[&](p::Profile changed,p::Shape s,Fields current) {
        p::Estimator model(changed,std::move(current),p::Ignore::parse("gpu,environment,memory"),packing);
        p::WidthPredictor predict(model,nullptr,true);p::WidthEstimate value;
        require(!predict.predict(s,value) || std::string(value.model)!="same_width_phase_projection_v1","unsafe phase reuse");
    };
    auto changed=profile;changed.samples[0].erase("phase_baby_seconds");rejects(changed,target,condition);
    changed=profile;changed.samples[0]["phase_baby_seconds"]="60";rejects(changed,target,condition);
    auto foreign=condition;foreign["uuid_hex"]="\"foreign\"";rejects(profile,target,foreign);
    auto other=target;other.groot_balanced=false;rejects(profile,other,condition);
    other=target;other.fold=false;rejects(profile,other,condition);
    other=target;other.arithmetic_bits=12512;rejects(profile,other,condition);
    other=candidate(anchor,570570);require(e.cached_work(other,w) && w.largest_ntt==(Word(1)<<28),"expected NTT cliff missing");
    rejects(profile,other,condition);
    other=target;other.chain_block=64;rejects(profile,other,condition);
    other=target;other.giant_coop=16;rejects(profile,other,condition);
    other=target;other.b2*=4;rejects(profile,other,condition);
    // Preserve prediction caching and the measured-only stable winner rule.
    require(predictor.predict(target,estimate),"cached estimate missing");const auto seconds=estimate.seconds;
    require(predictor.predict(target,estimate) && estimate.seconds==seconds && predictor.prediction_hits>0,"cache changed estimate");
    auto measured=e.runtime_statistics(anchor);
    std::vector<p::RuntimeCandidate> routes{{anchor,measured,1},{target,{},3,false}};
    require(p::runtime_choice(routes,false).index==0,"unmeasured projection replaced stable route");
}
int main(int argc,char **argv) {
    try {
        tests();
        if(argc<3){std::cout<<"{\"complete\":true,\"phase_projection\":true,\"isolation\":true,\"native_packing_cliff\":true,\"measured_only_adoption\":true,\"CUDA_calls\":0}\n";return 0;}
        const auto profile=p::Profile::load(argv[2]);require(profile.samples.size()==1,"audit requires one training anchor");
        p::Estimator e(profile,profile.conditions.at(0),p::Ignore::parse(""),packing,true,1);
        const auto anchor=e.shape(profile.samples[0]);p::WidthPredictor predictor(e,nullptr,true);
        auto prior=profile;
        for(const auto name:ecm_stage2::timing::names)prior.samples[0].erase(std::string("phase_")+name+"_seconds");
        p::Estimator prior_engine(prior,prior.conditions.at(0),p::Ignore::parse(""),packing,true,1);
        p::WidthPredictor prior_predictor(prior_engine);
        std::cout<<"{\"complete\":true,\"prediction_only\":true,\"rows\":[";bool comma=false;
        for(const auto &entry:ecm_stage2::d_catalog::entries()) {
            auto s=candidate(anchor,entry.d);p::Work w;p::WidthEstimate value,old;
            if(!e.cached_work(s,w) || !w.exact_topology || w.p<512 || !predictor.predict(s,value))continue;
            require(prior_predictor.predict(s,old),"prior prediction missing");
            if(comma)std::cout<<',';comma=true;
            std::cout<<"{\"D\":"<<s.d<<",\"P\":"<<w.p<<",\"largest_ntt\":"<<w.largest_ntt<<",\"seconds\":"<<value.seconds
                <<",\"rank\":"<<value.rank<<",\"model\":"<<p::quoted(value.model)
                <<",\"prior_seconds\":"<<old.seconds<<",\"prior_rank\":"<<old.rank<<'}';
        }
        std::cout<<"]}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
