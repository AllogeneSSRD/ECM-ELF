#include "ecm/stage2/curve_process.h"
#include "ecm/stage2/process.h"
#include <cstdlib>
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
            s2prod::Handle start,done,feedback;
            std::string finalize;
            for(int i=2;i<argc;++i) {
                const std::string arg=argv[i];
                std::cout<<"wire:"<<arg<<'\n';
                if(i+1==argc)continue;
                if(arg=="--queue-receipt")code=std::stoi(argv[i+1]);
                if(arg=="--queue-gpu-start")start.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--queue-gpu-done")done.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--queue-gpu-input")feedback.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--tune-condition-tag")finalize=argv[i+1];
            }
            if(start.value!=INVALID_HANDLE_VALUE) {
                require(WaitForSingleObject(start.value,5000)==WAIT_OBJECT_0,"worker did not receive GPU admission");
                const auto *view=static_cast<const char*>(MapViewOfFile(feedback.value,FILE_MAP_READ,0,0,s2prod::queue_feedback_bytes));
                require(view && std::string(view)=="fixture feedback","worker received incorrect tune feedback");
                UnmapViewOfFile(view);
                std::cout<<"stage2_queue_observation: {\"verified\":1}"<<std::endl;
                require(SetEvent(done.value)!=0,"cannot signal fixture GPU completion");
                s2prod::Handle finish;
                finish.value=OpenEventA(SYNCHRONIZE,FALSE,finalize.c_str());
                require(finish.value && WaitForSingleObject(finish.value,5000)==WAIT_OBJECT_0,"fixture publication was not resumed");
            }
            std::cout<<"{\"type\":\"tune_selection\",\"runtime_prepare_hints\":\"fixture\"}\n";
            std::cerr<<"fixture stderr tail";std::cout<<"fixture stdout tail";
            return code;
        }
        require(argc==2,"expected isolated output directory");
        const auto out=fs::absolute(argv[1]);require(!fs::exists(out),"output must be fresh");fs::create_directories(out);
        using s2prod::queue_pipeline_level;
        require(_putenv_s("ECM_STAGE2_QUEUE_PIPELINE","")==0,"cannot clear fixture pipeline override");
        require(queue_pipeline_level(4096,1,false)==2 && queue_pipeline_level(4097,1,false)==0,"serial prefetch boundary changed");
        require(queue_pipeline_level(2048,2,false)==2 && queue_pipeline_level(2049,2,false)==0,"paired witness prefetch boundary changed");
        require(queue_pipeline_level(4096,2,true)==2 && queue_pipeline_level(4097,2,true)==0,"paired factor-only prefetch boundary changed");
        require(_putenv_s("ECM_STAGE2_QUEUE_PIPELINE","2")==0 && queue_pipeline_level(4096,2,false)==2,"explicit prefetch override ignored");
        require(_putenv_s("ECM_STAGE2_QUEUE_PIPELINE","0")==0 && queue_pipeline_level(318,1,false)==0,"explicit prefetch disable ignored");
        require(_putenv_s("ECM_STAGE2_QUEUE_PIPELINE","")==0,"cannot restore fixture pipeline policy");
        s2prod::CurveProcessRequest r;
        r.save=out/"point with spaces.save";r.results=out/"internal.jsonl";
        r.record={47,98432,7};r.console_budget={128,32};r.b2=40000;r.d=210;r.batch=16;r.device=9999;r.worker=2;
        r.carrier_exponent=127;r.has_carrier=true;r.short_batch_tune=true;r.pack_zero=true;r.weighted_ntt=true;
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
                    std::string("wire:--weighted-ntt\nwire:1"),
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
        r.queue_pipeline=true;r.runtime_prepare_cache=false;r.log=out/"pipeline.log";
        r.condition_tag="Local\\ECM-ELF.Stage2.ProcessFixture."+std::to_string(GetCurrentProcessId());
        s2prod::Handle publish;
        publish.value=CreateEventA(nullptr,TRUE,FALSE,r.condition_tag.c_str());
        require(publish.value!=nullptr,"cannot create fixture publication event");
        s2prod::CurveProcess pipeline(r);
        require(WaitForSingleObject(pipeline.gpu_done_handle(),50)==WAIT_TIMEOUT,"worker ran before GPU admission");
        pipeline.allow_gpu("fixture feedback");
        require(WaitForSingleObject(pipeline.gpu_done_handle(),5000)==WAIT_OBJECT_0,"GPU completion signal missing");
        require(!pipeline.poll() && pipeline.gpu_released(),"GPU completion required worker exit");
        require(pipeline.gpu_feedback().value_or("")=="{\"verified\":1}","verified observation was not read");
        require(SetEvent(publish.value)!=0,"cannot resume fixture publication");
        require(WaitForSingleObject(pipeline.handle(),5000)==WAIT_OBJECT_0,"fixture worker did not finish");
        require(pipeline.poll().value_or(-1)==0,"pipeline fixture failed");
        std::cout<<"{\"passed\":true,\"processes\":6,\"wire_checks\":47,\"cuda_devices\":0,\"engine_links\":0}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
