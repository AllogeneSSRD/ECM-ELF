#include "../../src/core/ecm_stage2_portable_model.h"
#include <iostream>
int main(int argc,char **argv) {
    try {
        namespace p=ecm_stage2::tune::portable;
        std::cout<<std::setprecision(17);
        if(argc==2 && std::string(argv[1])=="--columns") {
            std::cout<<'[';bool first=true;for(const auto *column:p::matrix_column_order){if(!first)std::cout<<',';first=false;std::cout<<p::quoted(column);}std::cout<<']';
        } else if(argc==2 && std::string(argv[1])=="--short-batch") {
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
        } else if(argc==2 && std::string(argv[1])=="--local-search") {
            auto require=[](bool ok){if(!ok)throw std::runtime_error("local search assertion");};
            p::LocalSearch valley(4,12);const double costs[]={20,18,12,8,9,11,14,16,18,20,22,24};
            std::set<size_t> visits;
            for(size_t i;(i=valley.next())<12;){require(visits.insert(i).second);valley.tested[i]={costs[i],0};}
            require(valley.center==3 && visits==std::set<size_t>({2,3,4,5}));
            p::LocalSearch noisy(2,5);noisy.tested[1]={9.9,.2};noisy.tested[2]={10,.2};noisy.tested[3]={11,0};
            require(noisy.next()==5 && noisy.center==2);
            p::LocalSearch edge(0,3);require(edge.next()==0);edge.rejected.insert(0);
            require(edge.next()==1);edge.rejected.insert(1);require(edge.next()==3);
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
        } else if(argc==3 && std::string(argv[1])=="--exact-shape") {
            const auto profile=p::Profile::load(argv[2]);
            p::Estimator model(profile,profile.conditions.begin()->second,p::Ignore{},
                [](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
                    *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+7)/8))*n*=2;return true;});
            auto shape=model.shape(profile.samples.front());p::Estimate e;
            shape.b1=1000;
            if(!model.predict(shape,e) || !e.exact)throw std::runtime_error("B1 blocked an exact shape");
            ++shape.buffers;
            if(!model.predict(shape,e) || e.exact)throw std::runtime_error("different buffers marked exact");
            shape=model.shape(profile.samples.front());++shape.giant_chunk;
            if(!model.predict(shape,e) || e.exact)throw std::runtime_error("different giant route marked exact");
            std::cout<<"true";
        } else if(argc==3 && std::string(argv[1])=="--fit-model") {
            auto profile=p::Profile::load(argv[2]);const auto seed=profile.samples.front();profile.samples.clear();
            auto query=[](ecm_stage2::Word m,int bits,ecm_stage2::Word *n,ecm_stage2::Word *slots){
                *slots=2*m-1;*n=8;while(*n<*slots*ecm_stage2::Word((bits+7)/8))*n*=2;return true;};
            auto cost=[](ecm_stage2::Word n,ecm_stage2::Word b){return double(n)*b*std::log2(double(n));};
            auto seconds=[](const p::Work &w){double ntt=0,reduce=0;
                for(unsigned ph=0;ph<ecm_stage2::RequestPhaseCount;++ph){ntt+=w.ntt[ph];reduce+=w.reduce[ph];}
                return .2+1e-10*ntt+1e-9*reduce+1e-10*(w.baby+w.giant+w.accum);};
            p::Estimator empty(profile,profile.conditions.begin()->second,p::Ignore{},query);
            for(auto d:{30030ull,60060ull,120120ull})for(auto b:{2600000000ull,26000000000ull,260000000000ull}) {
                auto sample=seed;sample["d"]=std::to_string(d);sample["p"]=std::to_string(ecm_stage2::cost::phi(d)/2);
                sample["b2"]=std::to_string(b);sample["stddev_seconds"]="0";
                p::Work w;if(!p::work(empty.shape(sample),query,cost,w))throw std::runtime_error("fit fixture shape");
                sample["mean_seconds"]=p::numeric(seconds(w));profile.update(sample);
            }
            p::Estimator model(profile,profile.conditions.begin()->second,p::Ignore{},query);
            auto target=model.shape(seed);target.d=90090;target.b2=26000000000;
            p::Estimate prediction;p::Work w;
            if(!p::work(target,query,cost,w) || !model.predict(target,prediction) ||
               std::string(prediction.model)!="portable_nonnegative_v1" ||
               std::abs(prediction.seconds-seconds(w))>1e-5*seconds(w) || prediction.relative_error>1e-5)
                throw std::runtime_error("qualified nonnegative model did not recover fixed cost");
            profile.samples[4]["mean_seconds"]=p::numeric(2*ecm_stage2::tune::real(profile.samples[4],"mean_seconds"));
            p::Estimator damaged(profile,profile.conditions.begin()->second,p::Ignore{},query);
            double value=0,error=0;if(damaged.fitted(target,w,value,error))throw std::runtime_error("bad holdout model qualified");
            std::cout<<"true";
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
