#include "../../src/core/ecm_stage2_portable_model.h"
#include <iostream>
int main(int argc,char **argv) {
    try {
        namespace p=ecm_stage2::tune::portable;
        std::cout<<std::setprecision(17);
        if(argc==2 && std::string(argv[1])=="--short-batch") {
            ecm_stage2::ShortNttBatchPolicy policy;
            const auto n=1ull<<19;
            policy.observe(n,1,.000359424);policy.observe(n,4,.00114688);policy.observe(n,16,.00620544);
            auto require=[](bool ok){if(!ok)throw std::runtime_error("short batch policy assertion");};
            require(policy.select(n,64,16)==4);
            require(policy.select(n,64,1)==1);
            require(policy.select(n,64,3)==3);
            require(policy.select(n,64,4)==4);
            require(policy.select(1ull<<20,64,16)==16);
            require(policy.select(n,0,16)==16);
            require(policy.cost(n,0)==0 && policy.cost(n,2)>.000359424 && policy.cost(n,2)<.00114688);
            require(std::abs(policy.cost(n,32)-2*.00620544)<1e-15);
            for(unsigned pairs=1;pairs<=128;++pairs)for(unsigned limit=1;limit<=pairs;++limit) {
                auto chosen=policy.select(n,pairs,limit);
                require(chosen>=1 && chosen<=limit);
                if(chosen!=limit) {
                    auto total=[&](unsigned b){return (pairs/b)*policy.cost(n,b)+policy.cost(n,pairs%b);};
                    require(total(chosen)<.9*total(limit));
                }
            }
            require(ecm_stage2::planned_chunk_slices(n,1,64,1,3,true,0,&policy)==1);
            std::cout<<"true";
        } else if(argc==4 && std::string(argv[1])=="--neighbors") {
            auto indices=p::neighbors(std::stoul(argv[2]),std::stoul(argv[3]));
            std::cout<<'[';for(size_t i=0;i<indices.size();++i){if(i)std::cout<<',';std::cout<<indices[i];}std::cout<<']';
        } else if(argc==3 && std::string(argv[1])=="--grid") {
            const auto g=p::grid(std::stoul(argv[2]));
            auto list=[](const auto &v){for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}};
            std::cout<<"{\"widths\":[";list(g.widths);std::cout<<"],\"ds\":[";list(g.ds);
            std::cout<<"],\"b2s\":[";list(g.b2s);std::cout<<"],\"repeats\":"<<g.repeats<<"}\n";
        } else if(argc==3 && std::string(argv[1])=="--roundtrip") {
            std::cout<<p::Profile::load(argv[2]).serialize();
        } else if(argc==4 && std::string(argv[1])=="--update") {
            auto old=p::Profile::load(argv[2]);const auto added=p::Profile::load(argv[3]);
            for(auto s:added.samples){s["condition"]=std::to_string(old.condition(added.conditions.at(ecm_stage2::tune::uint(s,"condition"))));old.update(std::move(s));}
            std::cout<<old.serialize();
        } else if(argc==5 && std::string(argv[1])=="--matches") {
            const auto a=p::Profile::load(argv[2]),b=p::Profile::load(argv[3]);
            std::cout<<(p::matches(a.conditions.begin()->second,b.conditions.begin()->second,p::Ignore::parse(argv[4]))?"true":"false");
        } else if(argc==6 && std::string(argv[1])=="--csv") {
            const auto pred=ecm_stage2::stage1_cost::Table::load(argv[2]).predict(
                std::stoul(argv[3]),std::stod(argv[4]),std::stod(argv[5]));
            std::cout<<"{\"seconds\":"<<pred.seconds<<",\"container_bits\":"<<pred.shape.bits
                <<",\"tpi\":"<<pred.shape.tpi<<",\"source\":\""<<pred.source
                <<"\",\"crosses_tpi\":"<<(pred.crosses_tpi?"true":"false")<<"}\n";
        } else if(argc==6 && std::string(argv[1])=="--predict") {
            const auto profile=p::Profile::load(argv[2]);
            auto condition=profile.conditions.begin()->second;
            p::Estimator model(profile,condition,p::Ignore{},[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
                *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+7)/8))*n*=2;return true;});
            auto shape=model.shape(profile.samples.front());shape.target_bits=std::stoul(argv[3]);shape.arithmetic_bits=shape.target_bits;
            shape.b1=std::stoull(argv[4]);shape.b2=std::stoull(argv[5]);p::Estimate e;
            if(!model.predict(shape,e))throw std::runtime_error("no portable prediction");
            std::cout<<"{\"seconds\":"<<e.seconds<<",\"rank\":"<<e.rank<<",\"anchors\":"<<e.anchors
                <<",\"validated\":"<<(e.validated?"true":"false")<<"}\n";
        } else throw std::runtime_error("invalid fixture arguments");
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
