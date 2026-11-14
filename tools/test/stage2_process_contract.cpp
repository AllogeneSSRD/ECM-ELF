#include "ecm/stage2/curve_process.h"
#include "ecm/stage2/process.h"
#include "ecm/platform/log_file.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fs=std::filesystem;
static void require(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
int main(int argc,char **argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--curve-worker") {
            int code=0;
            s2prod::Handle start,done,feedback;
            std::string finalize;
            bool publication_seen=false;
            for(int i=2;i<argc;++i) {
                const std::string arg=argv[i];
                std::cout<<"wire:"<<arg<<'\n';
                if(i+1==argc)continue;
                if(arg=="--queue-receipt")code=std::stoi(argv[i+1]);
                if(arg=="--queue-gpu-start")start.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--queue-gpu-done")done.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--queue-gpu-input")feedback.value=reinterpret_cast<HANDLE>(uintptr_t(std::stoull(argv[i+1])));
                if(arg=="--tune-condition-tag")finalize=argv[i+1];
                require(arg!="--config","child received live config path");
                if(arg=="--curve-publication") {
                    const auto publication=s2prod::parse_curve_publication(argv[i+1]);
                    require(publication.path=="public output.txt" && publication.user==u8"用户 \"A\"\\" &&
                            publication.computer.empty(),"publication snapshot did not round-trip");
                    publication_seen=true;
                }
            }
            require(publication_seen,"child did not receive a publication snapshot");
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
        if(argc==4 && std::string(argv[1])=="--log-fixture-writer") {
            const int worker=std::stoi(argv[3]);
            for(int i=0;i<24;++i)
                ecm_append_worker_log(fs::u8path(argv[2]),worker,1,
                    "entry="+std::to_string(i)+" "+std::string(300,'x'));
            return 0;
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
        r.publication={"public output.txt",u8"用户 \"A\"\\",""};
        r.submission_exponent=127;r.submission_known={"101","103"};r.submission_aid="AID";
        r.factorize_hits=true;r.gp="gp.exe";r.factor_timeout=5;r.factor_only=true;r.publication_task="task";
        int observed=0;
        r.observe_selection=[&](const std::string &line){require(line.find("fixture")!=line.npos,"selection feedback missing");++observed;};
        r.log_max_kb=1;
        for(int code:{0,3,17}) {
            r.receipt=std::to_string(code);r.log=out/(r.receipt+".log");
            require(s2prod::run_curve_process(r)==(code==0?0:code==3?3:1),"exit status mapping changed");
            std::string log;
            const auto consume=[&](const fs::path &path) {
                std::ifstream stream(path,std::ios::binary);
                std::string line;
                while(std::getline(stream,line)) {
                    require(line.rfind("[Worker #2] [",0)==0 && line.size()>=34 &&
                            line[32]==']' && line[33]==' ',"worker prefix missing");
                    log+=line.substr(34)+'\n';
                }
                require(!stream.bad(),"cannot read worker log");
            };
            for(unsigned i=1;;++i) {
                const auto archive=out/(r.receipt+'.'+std::to_string(i)+".log");
                if(!fs::exists(archive))break;
                consume(archive);
            }
            consume(r.log);
            for(const auto &value:{std::string("wire:"+r.condition_tag),std::string("wire:"+r.save.string()),
                    std::string("wire:--short-calibration\nwire:0"),std::string("wire:--record-offset\nwire:47"),
                    std::string("wire:--record-hash\nwire:98432"),std::string("wire:--device\nwire:9999"),
                    std::string("wire:--weighted-ntt\nwire:1"),
                    std::string("wire:--known-factor\nwire:101"),std::string("wire:--known-factor\nwire:103"),
                    std::string("fixture stderr tail"),std::string("fixture stdout tail")})
                require(log.find(value)!=log.npos,"wire quoting/options or final drain changed");
        }
        require(observed==3,"selection callback count changed");
        require(fs::exists(out/"0.1.log"),"oversize log was not archived");
        const auto gap=out/"gap.log", older=out/"gap.9.log";
        std::ofstream(older)<<"preserve this archive";
        ecm_append_worker_log(gap,2,1,std::string(1100,'x')+"\nnext\n");
        require(fs::exists(out/"gap.10.log") && fs::file_size(older)==21,"existing archive maximum was ignored");
        std::ifstream active(gap);std::string next;
        std::getline(active,next);
        require(next.size()>5 && next.substr(next.size()-5)==" next","activity did not continue after rotation");
        const auto concurrent=out/"concurrent.log";
        s2prod::Handle writers[2];
        for(int i=0;i<2;++i) {
            auto command=L"\""+s2prod::executable().wstring()+L"\" --log-fixture-writer \""+
                concurrent.wstring()+L"\" "+std::to_wstring(i+1);
            STARTUPINFOW startup{};
            startup.cb=sizeof(startup);
            PROCESS_INFORMATION process{};
            require(CreateProcessW(nullptr,command.data(),nullptr,nullptr,FALSE,0,nullptr,nullptr,
                                   &startup,&process)!=FALSE,"cannot start concurrent logger");
            CloseHandle(process.hThread);
            writers[i].value=process.hProcess;
        }
        for(const auto &writer:writers) {
            DWORD status=1;
            require(WaitForSingleObject(writer.value,15000)==WAIT_OBJECT_0 &&
                    GetExitCodeProcess(writer.value,&status) && status==0,"concurrent logger failed");
        }
        bool recorded[2][24]{};
        int concurrent_lines=0;
        const auto verify_concurrent=[&](const fs::path &path) {
            std::ifstream stream(path,std::ios::binary);
            std::string line;
            while(std::getline(stream,line)) {
                require(line.size()>40 && line.rfind("[Worker #",0)==0 && line[32]==']' &&
                        line.substr(34,6)=="entry=","concurrent line framing changed");
                const auto worker=line[9]-'1';
                const auto ordinal=std::stoi(line.substr(40));
                require(worker>=0 && worker<2 && ordinal>=0 && ordinal<24 &&
                        !recorded[worker][ordinal],"duplicate or corrupt concurrent entry");
                recorded[worker][ordinal]=true;
                ++concurrent_lines;
            }
            require(!stream.bad(),"cannot read concurrent worker log");
        };
        for(int i=1;;++i) {
            const auto archive=out/("concurrent."+std::to_string(i)+".log");
            if(!fs::exists(archive))break;
            verify_concurrent(archive);
        }
        verify_concurrent(concurrent);
        require(concurrent_lines==48,"concurrent log records lost during rotation");
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
