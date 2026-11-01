#include "../../src/ecm/stage2/planning/runtime_tune.h"
#include <iostream>
namespace p=ecm_stage2::tune::portable;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char **argv) {
    try {
        if(argc==4 && std::string(argv[1])=="--profile") {
            const auto profile=p::Profile::load(argv[2]);const auto requested=std::stoull(argv[3]);
            ecm_stage2::Word rows=0,count=0;
            for(const auto &sample:profile.samples)if(p::runtime_observation(sample)) {
                const auto condition=profile.conditions.at(ecm_stage2::tune::uint(sample,"condition"));
                p::Estimator engine(profile,condition,p::Ignore::parse(""),{},true,2);
                const auto shape=engine.shape(sample);
                if(!p::runtime_bound_matches(shape,requested,true))continue;
                const auto stats=engine.runtime_statistics(shape);
                require(stats.count>=ecm_stage2::tune::uint(sample,"repeats"),"current-device observations not recovered");
                ++rows;count+=stats.count;
            }
            require(rows && count,"no rounded production observation matched");
            std::cout<<"{\"complete\":true,\"matched_rows\":"<<rows<<",\"matched_observations\":"<<count<<",\"CUDA_calls\":0}\n";return 0;
        }
        require(argc==2,"need output directory");
        p::Shape original;original.target_bits=original.arithmetic_bits=8440;original.b1=20;
        original.b2=26000000;original.d=6006;original.giant_chunk=138240;original.chain_min=8192;
        auto a=original;require(p::runtime_bounds(a,original.b2,true),"cannot round first D");
        require(a.b2==51879828 && a.b2/a.d+2==8640,"wrong executed bound");
        require(p::runtime_bound_matches(a,original.b2,true),"actual observation not reusable");
        require(!p::runtime_bound_matches(a,original.b2,false),"unrounded request reused rounded timing");
        require(p::runtime_bound_matches(a,original.b2+1,true),"identical coverage plateau not reused");
        auto again=a;require(p::runtime_bounds(again,a.b2,true) && again.key()==a.key(),"rounding is not idempotent");
        auto b=original;b.d=6930;require(p::runtime_bounds(b,original.b2,true),"cannot round second D");
        require(b.b2!=a.b2 && b.b2>=original.b2 && a.key()!=b.key(),"D-specific bounds merged");
        require(b.b2/b.d+2==8640,"second D coverage differs");
        p::RuntimeCandidate first{a,{2,0,10,0},1,true},second{b,{2,0,9,0},2,true};
        require(p::runtime_choice({first,second},false,2).index==1,"faster admitted coverage not chosen");
        auto cap=original;require(!p::runtime_bounds(cap,original.b2,true,30000000),"AutoB2 cap bypassed");
        auto ladder=original;ladder.force_ladder=true;require(!p::runtime_bounds(ladder,original.b2,true),"forced ladder rounded");
        auto insufficient=original;insufficient.giant_chunk=720;require(!p::runtime_bounds(insufficient,original.b2,true),"short capacity rounded");
        auto large=original;require(!p::runtime_bounds(large,ecm_stage2::Word(INT64_MAX)-1,true),"bound limit bypassed");
        auto off=original;require(p::runtime_bounds(off,original.b2,false) && off.key()==original.key(),"default identity changed");
        auto coop=a;coop.baby_coop=32;require(p::runtime_bounds(coop,original.b2,true) && coop.key()!=a.key(),"backend identity erased");
        std::cout<<"{\"complete\":true,\"per_D_bounds\":true,\"exact_observation_identity\":true,\"coverage_plateau\":true,"
            "\"default_unchanged\":true,\"caps_and_failures\":true,\"CUDA_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
