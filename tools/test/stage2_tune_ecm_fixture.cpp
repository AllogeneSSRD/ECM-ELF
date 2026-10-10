#include "../../src/core/ecm_stage2_tune_ecm.h"
#include "../../src/core/ecm_stage2_tune_prediction.h"
#include "../../src/core/ecm_stage2_tune_auto.h"
#include <iostream>
int main(int argc,char **argv) {
    try {
        namespace t=ecm_stage2::tune;
        if(argc==3 && std::string(argv[1])=="--prime") {
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
        } else if(argc==7 && std::string(argv[1])=="--auto") {
            const auto profile=t::EcmProfile::load(argv[2]);std::vector<const t::Fields*> samples;
            for(const auto &sample:profile.samples)samples.push_back(&sample);
            const t::AutoRequest request{t::uint(profile.samples.front(),"b1"),ecm_stage2::cost::integer(argv[4]),
                ecm_stage2::cost::integer(argv[5]),std::stod(argv[3]),std::stod(argv[6])};
            const auto candidates=t::auto_candidates(samples,t::predicts_b2(profile),request);
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
