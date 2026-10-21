#include "ecm/platform/process/curve_process.h"
#include "ecm/platform/process/stage2_curve_process.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
namespace fs=std::filesystem;
static void require(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
int main(int argc,char **argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--curve-worker") {
            int code=0;
            for(int i=2;i<argc;++i) {
                std::cout<<"wire:"<<argv[i]<<'\n';
                if(std::string(argv[i])=="--queue-receipt" && i+1<argc)code=std::stoi(argv[i+1]);
            }
            std::cout<<"{\"type\":\"tune_selection\",\"runtime_prepare_hints\":\"fixture\"}\n";
            std::cerr<<"fixture stderr tail";std::cout<<"fixture stdout tail";
            return code;
        }
        require(argc==2,"expected isolated output directory");
        const auto out=fs::absolute(argv[1]);require(!fs::exists(out),"output must be fresh");fs::create_directories(out);
        s2prod::CurveProcessRequest r;
        r.save=out/"point with spaces.save";r.results=out/"internal.jsonl";
        r.record={47,98432,7};r.console_budget={128,32};r.b2=40000;r.d=210;r.batch=16;r.device=9999;r.worker=2;
        r.carrier_exponent=127;r.has_carrier=true;r.short_batch_tune=true;r.pack_zero=true;
        r.tune_runtime=true;r.runtime_samples=10;r.runtime_remaining=3;r.runtime_prepare_cache=true;
        r.runtime_prepare_hints="prepare";r.runtime_route_hints="route";
        r.tune_profile="profile with spaces.toml";r.tune_ignore="gpu,driver";
        r.stage1_csv="cost.csv";r.condition_tag="a quoted \"tag\" with trailing \\";
        r.stage1_mhz=1000;r.factor_bits=32;r.inferred_carrier=127;r.auto_carrier=true;r.prediction_policy="paired";
        r.arena=128;r.owner_mb=32;r.log_level=1;r.engine_log_level=3;r.ntt_counts=true;r.relocation=true;
        r.submission="public.txt";r.submission_ini="config.toml";r.submission_exponent=127;r.submission_known={"101","103"};r.submission_aid="AID";
        r.factorize_hits=true;r.gp="gp.exe";r.factor_timeout=5;r.factor_only=true;r.publication_task="task";
        int observed=0;
        r.observe_selection=[&](const std::string &line){require(line.find("fixture")!=line.npos,"selection feedback missing");++observed;};
        for(int code:{0,3,17}) {
            r.receipt=std::to_string(code);r.log=out/(r.receipt+".log");
            require(s2prod::run_curve_process(r)==(code==0?0:code==3?3:1),"exit status mapping changed");
            std::ifstream stream(r.log,std::ios::binary);const std::string log((std::istreambuf_iterator<char>(stream)),{});
            for(const auto &value:{std::string("wire:"+r.condition_tag),std::string("wire:"+r.save.string()),
                    std::string("wire:--short-calibration\nwire:0"),std::string("wire:--record-offset\nwire:47"),
                    std::string("wire:--record-hash\nwire:98432"),std::string("wire:--device\nwire:9999"),
                    std::string("wire:--known-factor\nwire:101"),std::string("wire:--known-factor\nwire:103"),
                    std::string("fixture stderr tail"),std::string("fixture stdout tail")})
                require(log.find(value)!=log.npos,"wire quoting/options or final drain changed");
        }
        require(observed==3,"selection callback count changed");
        r.runtime_prepare_cache=false;r.receipt="0";r.log=out/"no-feedback.log";
        require(s2prod::run_curve_process(r)==0 && observed==3,"disabled cache received feedback");
        r.runtime_prepare_cache=true;r.observe_selection=[](const std::string &){throw std::runtime_error("feedback fixture failure");};
        r.log=out/"callback-failure.log";bool threw=false;
        try{s2prod::run_curve_process(r);}catch(const std::runtime_error &e){threw=std::string(e.what())=="feedback fixture failure";}
        require(threw,"callback exception swallowed");
        std::cout<<"{\"passed\":true,\"processes\":5,\"wire_checks\":30,\"cuda_devices\":0,\"engine_links\":0}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
