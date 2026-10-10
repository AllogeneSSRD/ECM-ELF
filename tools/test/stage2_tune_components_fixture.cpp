// CPU-only conformance driver. Offline shape tables are frozen engine queries;
// production callers must supply ecm_cuda_stage2_shape_query instead.
#include "../../src/core/ecm_stage2_tune_components.h"
#include <iostream>
#include <iomanip>
using namespace ecm_stage2;
namespace t=ecm_stage2::tune;
static void check(bool value){if(!value)throw std::runtime_error("component fixture gate failed");}
struct Case {
    int sample=-1,bits=0;Word p=0,points=0,budget=0,limit=0,d=0,carrier=0,b2=0;
    unsigned buffers=0;bool physical=false;
    std::map<Word,std::pair<Word,Word>> shapes;
    t::NttReferences reference;
};
static void selftest() {
    auto query=[](Word m,int,Word *n,Word *slots){*n=8;while(*n<4*m)*n*=2;*slots=2*m-1;return true;};
    t::NttReferences r;t::NttMeasurements measurements;
    check(t::ntt_references(48,200,129,query,1024,2,true,0,measurements,r));
    check(r.valid && !r.complete && t::loop_reference(r)==0);
    for(const auto &bin:r.bins)measurements[{std::get<1>(bin.first),std::get<2>(bin.first)}]=.001;
    check(t::ntt_references(48,200,129,query,1024,2,true,0,measurements,r) && r.complete);
    measurements.begin()->second=-1;
    check(!t::ntt_references(48,200,129,query,1024,2,true,0,measurements,r));
    check(!t::ntt_references(48,48,129,query,1024,2,true,0,{},r));
    check(!t::ntt_references(48,200,129,query,0,2,true,0,{},r));
    check(!t::ntt_references(48,200,129,query,1024,4,true,0,{},r));
    check(!t::ntt_references(1,~Word(0),129,query,1024,2,true,0,{},r));
    auto invalid=[](Word,int,Word *n,Word *slots){*n=9;*slots=1;return true;};
    check(!t::ntt_references(48,200,129,invalid,1024,2,true,0,{},r));
    t::ComponentModel model;t::B2Prediction predicted;
    check(!t::prepare_component_model({},model) && !t::predict_component(model,100,r,predicted));
    std::cout<<"{\"complete\":true,\"gates\":10}\n";
}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--selftest"){selftest();return 0;}
        if(argc!=3)throw std::runtime_error("fixture requires ECM profile and offline cases");
        const auto profile=t::EcmProfile::load(argv[1]);std::ifstream in(argv[2]);
        size_t count=0;in>>count;if(!in || count>65536)throw std::runtime_error("invalid measurement count");
        t::NttMeasurements measurements;
        for(size_t i=0;i<count;++i){Word n=0,s=0;double seconds=0;in>>n>>s>>seconds;
            if(!in || !measurements.emplace(std::make_pair(n,s),seconds).second)throw std::runtime_error("duplicate/invalid measurement");}
        in>>count;if(!in || count>8192)throw std::runtime_error("invalid case count");
        std::vector<Case> cases(count);
        for(auto &c:cases) {
            int physical=0;size_t shapes=0;
            in>>c.sample>>c.bits>>c.p>>c.points>>c.budget>>c.buffers>>physical>>c.limit>>c.d>>c.carrier>>c.b2>>shapes;
            if(!in || (physical!=0 && physical!=1) || shapes>4096 || c.sample<-1 || c.sample>=(int)profile.samples.size())
                throw std::runtime_error("invalid offline case");
            c.physical=physical!=0;
            for(size_t i=0;i<shapes;++i){Word m=0,n=0,s=0;in>>m>>n>>s;
                if(!in || !c.shapes.emplace(m,std::make_pair(n,s)).second)throw std::runtime_error("invalid offline shape");}
            auto query=[&](Word m,int bits,Word *n,Word *slots){const auto found=c.shapes.find(m);
                if(bits!=c.bits || found==c.shapes.end())return false;*n=found->second.first;*slots=found->second.second;return true;};
            if(!t::ntt_references(c.p,c.points,c.bits,query,c.budget,c.buffers,c.physical,c.limit,measurements,c.reference))
                throw std::runtime_error("invalid native reference workload");
        }
        std::string trailing;if(in>>trailing)throw std::runtime_error("trailing offline input");
        std::map<std::string,std::vector<t::ComponentAnchor>> groups;
        for(const auto &c:cases)if(c.sample>=0) {
            const auto &sample=profile.samples[c.sample];
            if(c.d!=t::uint(sample,"d") || c.carrier!=t::uint(sample,"carrier_exponent") || c.b2!=t::uint(sample,"b2"))
                throw std::runtime_error("offline anchor scope mismatch");
            groups[t::component_scope(sample)].push_back({&sample,&c.reference});
        }
        std::map<std::string,t::ComponentModel> models;
        std::cout<<std::setprecision(17)<<"{\"models\":[";bool first=true;
        for(const auto &g:groups) {
            auto &m=models[g.first];t::prepare_component_model(g.second,m);
            if(!first)std::cout<<',';first=false;
            const auto &s=*g.second.front().sample;
            std::cout<<"{\"d\":"<<t::uint(s,"d")<<",\"carrier\":"<<t::uint(s,"carrier_exponent")
                <<",\"qualified\":"<<(m.qualified?"true":"false")<<",\"fixed_seconds\":"<<m.fixed_seconds
                <<",\"relative\":"<<m.evidence.max_relative_error<<",\"absolute\":"<<m.evidence.error_seconds
                <<",\"coefficients\":[";
            for(unsigned i=0;i<4;++i){if(i)std::cout<<',';std::cout<<m.coefficients[i];}std::cout<<"]}";
        }
        std::cout<<"],\"cases\":[";first=true;
        for(const auto &c:cases) {
            if(!first)std::cout<<',';first=false;
            const auto &r=c.reference;
            std::cout<<"{\"complete\":"<<(r.complete?"true":"false")<<",\"phases\":[";
            for(unsigned i=0;i<RequestPhaseCount;++i){if(i)std::cout<<',';const auto &p=r.phases[i];
                std::cout<<"{\"calls\":"<<p.calls<<",\"pairs\":"<<p.pairs<<",\"missing\":"<<p.missing_bins<<",\"seconds\":"<<p.matched_seconds<<'}';}
            std::cout<<"],\"bins\":[";bool bin_first=true;
            for(const auto &b:r.bins){if(!bin_first)std::cout<<',';bin_first=false;
                std::cout<<'['<<std::get<0>(b.first)<<','<<std::get<1>(b.first)<<','<<std::get<2>(b.first)<<','<<b.second.calls<<','<<b.second.pairs<<']';}
            std::cout<<']';
            if(c.sample<0)for(const auto &g:groups) {
                const auto &sample=*g.second.front().sample;
                if(t::uint(sample,"d")!=c.d || t::uint(sample,"carrier_exponent")!=c.carrier || t::uint(sample,"arithmetic_bits")!=c.bits)continue;
                t::B2Prediction prediction;
                const bool predicted=t::predict_component(models.at(g.first),c.b2,r,prediction);
                std::cout<<",\"predicted\":"<<(predicted?"true":"false")<<",\"seconds\":"<<prediction.seconds
                    <<",\"rank\":"<<prediction.seconds+prediction.error_seconds+2*prediction.mad_seconds;break;
            }
            std::cout<<'}';
        }
        std::cout<<"]}\n";return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
