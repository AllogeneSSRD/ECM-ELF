// Uses the exported production packing query. No device queries or curves.
#include "ecm_stage2_tune_components.h"
#include "ecm_stage2_tune_auto.h"
#include "ecm_cuda_stage2.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace t=ecm_stage2::tune;
using ecm_stage2::Word;
int main(int argc,char **argv) {
    try {
        if(argc!=4)throw std::runtime_error("requires profile, requests, rounds");
        const auto profile=t::EcmProfile::load(argv[1]);
        const int rounds=std::stoi(argv[3]);if(rounds<1 || rounds>100)throw std::runtime_error("invalid rounds");
        std::ifstream in(argv[2]);size_t count=0;in>>count;
        if(!in || count>256)throw std::runtime_error("invalid request count");
        struct Request {t::AutoRequest auto_request;Word d=0;int carrier=-1;};
        std::vector<Request> requests(count);
        for(auto &r:requests)in>>r.auto_request.b1>>r.auto_request.minimum>>r.auto_request.maximum
            >>r.auto_request.stage1_seconds>>r.auto_request.adjust>>r.d>>r.carrier;
        if(!in)throw std::runtime_error("invalid requests");std::string trailing;
        if(in>>trailing)throw std::runtime_error("trailing request data");
        std::cout<<std::setprecision(17)<<"{\"cases\":[";
        for(size_t ri=0;ri<requests.size();++ri) {
            const auto &r=requests[ri];std::vector<const t::Fields*> samples;
            for(const auto &s:profile.samples)if(t::uint(s,"b1")==r.auto_request.b1 &&
                (!r.d || t::uint(s,"d")==r.d) && (r.carrier<0 || t::uint(s,"carrier_exponent")==Word(r.carrier)))samples.push_back(&s);
            Word calls=0;double query_seconds=0;
            auto query=[&](Word p,int bits,Word *n,Word *slots) {
                ++calls;const auto start=std::chrono::steady_clock::now();
                const bool valid=ecm_cuda_stage2_shape_query(p,bits,n,slots);
                query_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                return valid;
            };
            std::vector<t::AutoCandidate> candidates;
            const auto start=std::chrono::steady_clock::now();
            for(int round=0;round<rounds;++round) {
                t::ComponentEstimator estimator(profile,query);
                const t::AdditionalPrediction additional{
                    [&](const auto &a){return estimator.prepare(a);},
                    [&](const auto &a,Word b2,t::B2Prediction &p){return estimator.predict(a,b2,p);}};
                candidates=t::auto_candidates(samples,t::predicts_b2(profile),r.auto_request,additional);
            }
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(ri)std::cout<<',';
            std::cout<<"{\"seconds\":"<<seconds/rounds<<",\"query_seconds\":"<<query_seconds/rounds
                <<",\"query_calls\":"<<calls/rounds<<",\"candidates\":[";
            bool first=true;
            for(const auto &c:candidates) {
                if(!first)std::cout<<',';first=false;
                std::cout<<"["<<t::uint(*c.sample,"d")<<','<<t::uint(*c.sample,"carrier_exponent")<<','
                    <<c.b2<<','<<c.low<<','<<c.high<<','<<c.seconds<<','<<c.mad_seconds<<','<<c.guarded_seconds
                    <<','<<c.benefit<<','<<c.score<<','<<int(c.limited)<<','<<c.prediction.seconds<<','
                    <<c.prediction.error_seconds<<','<<c.prediction.max_relative_error<<','<<c.prediction.samples
                    <<','<<'"'<<c.prediction.model<<'"'<<"]";
            }
            std::cout<<"]}";
        }
        std::cout<<"]}\n";return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
