#include "../../src/core/ecm_stage2_tune_ecm.h"
#include "../../src/core/ecm_stage2_tune_prediction.h"
#include "../../src/core/ecm_stage2_tune_auto.h"
#include "../../src/core/ecm_stage1_tune_profile.h"
#include <iostream>
#include <chrono>
int main(int argc,char **argv) {
    try {
        namespace t=ecm_stage2::tune;
        if(argc==2 && std::string(argv[1])=="--tail-grid") {
            ecm_stage2::Word b1,d,p,chunk,minimum,batches;unsigned force,requested,count;
            while(std::cin>>b1>>d>>p>>chunk>>minimum>>force>>batches>>requested>>count) {
                if(count>64 || force>1)throw std::runtime_error("invalid grid fixture");
                std::vector<ecm_stage2::Word> bounds(count);for(auto &b:bounds)if(!(std::cin>>b))throw std::runtime_error("incomplete grid fixture");
                const auto grid=t::tune_tail_grid(bounds,b1,d,p,chunk,minimum,force!=0,batches,requested);
                std::cout<<"{\"valid\":"<<(grid.valid?"true":"false")<<",\"reason\":\""<<grid.reason
                    <<"\",\"low\":"<<grid.low<<",\"high\":"<<grid.high<<",\"chain_anchors\":"<<grid.existing_chain_anchors<<",\"points\":[";
                bool first=true;for(const auto &point:grid.points){if(!first)std::cout<<',';first=false;
                    std::cout<<"{\"b2\":"<<point.b2<<",\"source\":\""<<point.source<<"\"}";}
                std::cout<<"]}\n";
            }
        } else if(argc==3 && std::string(argv[1])=="--effort-tail") {
            std::cout<<t::ecm_effort(std::stoi(argv[2])).tail_samples<<'\n';
        } else if(argc==3 && std::string(argv[1])=="--effort-exponents") {
            for(const auto p:t::ecm_effort(std::stoi(argv[2])).exponents)std::cout<<p<<' ';
        } else if(argc==6 && std::string(argv[1])=="--work-policy") {
            const auto profile=t::EcmProfile::load(argv[2]);
            std::cout<<(t::matches_giant_work_policy(profile.samples.front(),std::stoull(argv[3]),
                std::stoull(argv[4]),std::stoul(argv[5])!=0)?1:0)<<'\n';
        } else if(argc==7 && std::string(argv[1])=="--giant-work") {
            ecm_stage2::GiantWork w;
            if(!ecm_stage2::giant_work(std::stoull(argv[2]),std::stoull(argv[3]),std::stoull(argv[4]),
                std::stoull(argv[5]),std::stoul(argv[6])!=0,w))throw std::runtime_error("invalid giant work");
            std::cout<<w.chain_points<<' '<<w.ladder_points<<' '<<w.chain_chunks<<' '<<w.ladder_chunks<<' '<<w.ladder_steps<<'\n';
        } else if(argc==14 && std::string(argv[1])=="--phases") {
            ecm_stage2::timing::Boundaries b;
            double *fields[]={&b.shape,&b.init_begin,&b.baby_begin,&b.ftree_begin,&b.init_end,
                &b.main_begin,&b.inverse_begin,&b.loop_begin,&b.loop_end,&b.accum_begin,&b.accum_end,&b.main_end};
            for(size_t i=0;i<12;++i)*fields[i]=std::stod(argv[i+2]);
            const auto phases=ecm_stage2::timing::partition(b);
            std::cout<<std::setprecision(17)<<"{\"complete\":"<<(phases.complete?"true":"false")<<",\"seconds\":[";
            for(size_t i=0;i<phases.seconds.size();++i){if(i)std::cout<<',';std::cout<<phases.seconds[i];}
            std::cout<<"],\"total\":"<<phases.total()<<"}\n";
        } else if(argc==3 && std::string(argv[1])=="--prime") {
            std::string n;const auto x=t::prime_point((unsigned)std::stoul(argv[2]),n);
            std::cout<<n<<'\n'<<x<<'\n';
        } else if(argc==3 && std::string(argv[1])=="--effort") {
            const auto e=t::ecm_effort(std::stoi(argv[2]));
            std::cout<<e.exponents.size()<<' '<<e.d.size()<<' '<<e.b2.size()<<' '<<e.repeats<<' '<<e.max_batches<<'\n';
        } else if(argc==3 && std::string(argv[1])=="--effort-d") {
            for(const auto d:t::ecm_effort(std::stoi(argv[2])).d)std::cout<<d<<' ';
        } else if(argc==3 && std::string(argv[1])=="--effort-b2") {
            for(const auto b2:t::ecm_effort(std::stoi(argv[2])).b2)std::cout<<b2<<' ';
        } else if(argc==4 && std::string(argv[1])=="--predict") {
            const auto profile=t::EcmProfile::load(argv[2]);std::vector<const t::Fields*> samples;
            for(const auto &sample:profile.samples)samples.push_back(&sample);
            t::B2Prediction prediction;
            const bool eligible=t::predicts_b2(profile) && t::predict_b2(samples,ecm_stage2::cost::integer(argv[3]),prediction);
            std::cout<<std::setprecision(17)<<"{\"eligible\":"<<(eligible?"true":"false")
                <<",\"seconds\":"<<prediction.seconds<<",\"error_seconds\":"<<prediction.error_seconds
                <<",\"mad_seconds\":"<<prediction.mad_seconds<<",\"max_relative_error\":"<<prediction.max_relative_error
                <<",\"samples\":"<<prediction.samples<<"}\n";
        } else if((argc==7 && (std::string(argv[1])=="--auto" || std::string(argv[1])=="--auto-grid")) ||
                  (argc==8 && std::string(argv[1])=="--auto-time")) {
            const auto profile=t::EcmProfile::load(argv[2]);std::vector<const t::Fields*> samples;
            for(const auto &sample:profile.samples)samples.push_back(&sample);
            const t::AutoRequest request{t::uint(profile.samples.front(),"b1"),ecm_stage2::cost::integer(argv[4]),
                ecm_stage2::cost::integer(argv[5]),std::stod(argv[3]),std::stod(argv[6])};
            const auto iterations=argc==8?ecm_stage2::cost::integer(argv[7]):1;
            if(!iterations || iterations>10000)throw std::runtime_error("invalid benchmark iterations");
            const auto start=std::chrono::steady_clock::now();
            std::vector<t::AutoCandidate> candidates;double checksum=0;
            for(uint64_t i=0;i<iterations;++i) {
                candidates=t::auto_candidates(samples,t::predicts_b2(profile),request);
                checksum+=candidates.front().score;
            }
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(std::string(argv[1])=="--auto-time") {
                std::cout<<std::setprecision(17)<<"{\"seconds\":"<<seconds<<",\"iterations\":"<<iterations
                    <<",\"candidates\":"<<candidates.size()<<",\"checksum\":"<<checksum<<"}\n";return 0;
            }
            if(std::string(argv[1])=="--auto-grid") {
                std::cout<<std::setprecision(17)<<'[';bool first=true;
                for(const auto &c:candidates) {
                    if(!first)std::cout<<',';first=false;
                    std::cout<<"{\"B2\":"<<c.b2<<",\"D\":"<<t::uint(*c.sample,"d")
                        <<",\"carrier\":"<<t::uint(*c.sample,"carrier_exponent")<<",\"seconds\":"<<c.seconds
                        <<",\"rank\":"<<c.guarded_seconds<<",\"K\":"<<c.benefit<<",\"score\":"<<c.score
                        <<",\"low\":"<<c.low<<",\"high\":"<<c.high<<",\"mad_seconds\":"<<c.mad_seconds
                        <<",\"fit_error_seconds\":"<<c.prediction.error_seconds
                        <<",\"fit_relative_error\":"<<c.prediction.max_relative_error
                        <<",\"fit_samples\":"<<c.prediction.samples
                        <<",\"predicted\":"<<(c.prediction.samples?"true":"false")
                        <<",\"limited\":"<<(c.limited?"true":"false")<<'}';
                }
                std::cout<<"]\n";return 0;
            }
            const auto &best=candidates.front();
            std::cout<<std::setprecision(17)<<"{\"B2\":"<<best.b2<<",\"D\":"<<t::uint(*best.sample,"d")
                <<",\"carrier\":"<<t::uint(*best.sample,"carrier_exponent")<<",\"seconds\":"<<best.seconds
                <<",\"rank\":"<<best.guarded_seconds<<",\"K\":"<<best.benefit<<",\"score\":"<<best.score
                <<",\"candidates\":"<<candidates.size()<<",\"predicted\":"<<(best.prediction.samples?"true":"false")
                <<",\"limited\":"<<(best.limited?"true":"false")<<"}\n";
        } else if(argc>=3 && std::string(argv[1])=="--merge") {
            std::vector<t::EcmProfile> inputs;
            for(int i=2;i<argc;++i)inputs.push_back(t::EcmProfile::load(argv[i]));
            std::cout<<t::ecm_profile_text(t::merge_ecm_profiles(inputs));
        } else if((argc==7 || argc==8) && std::string(argv[1])=="--stage1-cost") {
            const auto p=t::Stage1Profile::load(argv[2]);
            EcmStage2DeviceInfo d;std::strcpy(d.uuid_hex,"0123456789abcdef0123456789abcdef");
            d.major=8;d.minor=9;d.runtime=13030;d.driver=13030;
            std::cout<<std::setprecision(17)<<p.seconds(d,ecm_stage2::cost::integer(argv[3]),
                ecm_stage2::cost::integer(argv[4]),ecm_stage2::cost::integer(argv[5]),
                std::string("\"")+(argc==8?argv[7]:"mersenne")+"\"",argv[6])<<'\n';
        } else if(argc==3 && std::string(argv[1])=="--load") {
            const auto p=t::EcmProfile::load(argv[2]);
            EcmStage2DeviceInfo d;std::strcpy(d.uuid_hex,"0123456789abcdef0123456789abcdef");
            d.major=8;d.minor=9;d.runtime=13030;d.driver=13030;d.fixed_mode=3;
            if(!p.matches(d,256,6300,640,{{"xadd6","1"}},1))throw std::runtime_error("matching identity refused");
            if(p.matches(d,64,6300,640,{{"xadd6","1"}},1) || p.matches(d,256,6300,640,{{"xadd6","0"}},1))throw std::runtime_error("policy mismatch accepted");
            d.driver=13040;if(p.matches(d,256,6300,640,{{"xadd6","1"}},1))throw std::runtime_error("driver mismatch accepted");
            std::cout<<p.samples.size()<<'\n';
        } else return 2;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
    return 0;
}
