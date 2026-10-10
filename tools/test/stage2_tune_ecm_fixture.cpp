#include "../../src/core/ecm_stage2_tune_ecm.h"
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
