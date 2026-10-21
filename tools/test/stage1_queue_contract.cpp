#include "ecm/service/work/queue_workflow.h"
#include "../../src/ecm/service/stage1/legacy_execution.h"
#include "../../src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace ecm_stage1;
namespace fs=std::filesystem;
static std::string mode;
static int reads=0,computations=0;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static std::string read(const fs::path &path){std::ifstream in(path);return {std::istreambuf_iterator<char>(in),{}};}
static void stop_now(){
    static HANDLE event=CreateEventW(nullptr,TRUE,FALSE,nullptr);
    require(event && ecm_stage1_set_stop_handle(reinterpret_cast<uintptr_t>(event)),"stop handle");
    require(SetEvent(event)!=FALSE,"stop event");
}
static const std::string &identity(){static const std::string value="queue-contract";return value;}
static int window(){return 8;}
// Only arithmetic is substituted. Task, checkpoint, result publication and queue
// file mechanisms are the production implementations, including failure paths.
static int execute(mpz_srcptr,double,double,uint32_t count,const std::string &save,bool,
                   const std::string &,Stage1ExecutionOptions &,LegacyStage1Result *out){
    ++computations;out->curves=count;
    out->sigmas=static_cast<uint64_t *>(std::malloc(sizeof(uint64_t)*count));
    require(out->sigmas!=nullptr,"sigma allocation");
    for(uint32_t i=0;i<count;++i)out->sigmas[i]=ecm_stage1_batch_sigma(i,26+i);
    require(ecm_stage1_write_blob(ecm_stage1_curve_record(0),"queue-contract-state"),"checkpoint write");
    if(mode=="stop"){stop_now();return ECM_ERROR;}
    if(mode=="prepare"){out->prepare_failed=true;return ECM_ERROR;}
    out->factors=static_cast<mpz_t *>(std::malloc(sizeof(mpz_t)*count));
    out->array_found=static_cast<int *>(std::malloc(sizeof(int)*count));
    require(out->factors && out->array_found,"factor allocation");
    for(uint32_t i=0;i<count;++i){
        const bool hit=i==0 && (mode=="factor" || mode=="publication");
        mpz_init_set_ui(out->factors[i],hit?(mode=="factor"?101:5):0);
        out->array_found[i]=hit?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
    }
    out->firstsigma=26;out->firstsigma64=26;out->ret=ECM_NO_FACTOR_FOUND;
    std::ofstream(save)<<"queue-contract-save\n";
    return out->ret;
}
int main(int argc,char **argv){
    try{
        require(argc==3,"usage: stage1_queue_contract MODE NEW_DIRECTORY");mode=argv[1];
        const fs::path dir=fs::absolute(argv[2]);require(!fs::exists(dir),"fresh directory required");
        fs::create_directories(dir);
        QueuePaths paths;paths.worktodo=(dir/"worktodo.txt").string();paths.finished=(dir/"finished.txt").string();
        paths.exe_dir=dir.string();paths.sync1=(dir/"sync").string();paths.full_sync=true;
        fs::create_directory(paths.sync1); // Application validates/creates sync destinations.
        const std::string ticket="ECM2=1,10,4,403,20,20,2,26",next="ECMSTAGE2=1,10,4,403,\"point.save\",1000,0,2";
        std::ofstream(paths.worktodo)<<"[Worker #1]\n"<<(mode=="advance_failure"?"ECM2=1,2,127,-1,20":ticket)<<'\n';
        if(mode=="finished_failure"){paths.finished=(dir/"blocked").string();fs::create_directory(paths.finished);}
        if(mode=="mark_failure")paths.worktodo=(dir/"missing-parent/worktodo.txt").string();
        if(mode=="stop_before")stop_now();
        NextQueueTask source=[&](QueueItem &item){
            ++reads;
            if(mode=="read_error")return QueueReadStatus::Error;
            if(mode=="empty" || reads>1)return QueueReadStatus::Empty;
            item.ticket=ticket;
            if(mode=="mark" || mode=="mark_failure")return QueueReadStatus::MarkError;
            mpz_set_ui(item.modulus,10403);item.b1=item.b2=20;item.curves=2;
            item.expression="10403";item.save_file=(dir/"point.save").string();item.published_line="completed receipt";
            item.options.execution.selection.target=ExecutionTarget::Cpu;
            item.options.execution.selection.curve=CurveModel::Montgomery;item.options.execution.selection.gpu_param=0;
            item.options.persistence.tmp_dir=dir.string();item.options.persistence.results_file=(dir/"internal.jsonl").string();
            item.options.persistence.submission_file=(dir/"results.json.txt").string();
            item.options.identity.sigma_fixed=true;item.options.identity.fixed_sigma64=26;
            if(mode=="auto_stage2" || mode=="factor")item.auto_stage2_line=next;
            return QueueReadStatus::Ready;
        };
        const int rc=run_queue_workflow(paths,source,{execute,identity,window});
        const auto todo=read(paths.worktodo),finished=read(paths.finished);
        const bool fatal=mode=="prepare" || mode=="publication" || mode=="stop" || mode=="stop_before" ||
                         mode=="read_error" || mode=="advance_failure";
        require(rc==(fatal?1:0),"workflow return policy");
        const bool no_compute=mode=="empty" || mode=="read_error" || mode=="mark" || mode=="mark_failure" || mode=="stop_before";
        require(computations==(no_compute?0:1),"compute count");
        if(mode=="stop_before")require(reads==0,"stopped workflow consumed source");
        if(mode=="mark" || mode=="finished_failure")require(todo.find("# ERROR")!=std::string::npos,"failed input not marked");
        if(mode=="prepare" || mode=="publication" || mode=="stop" || mode=="stop_before" || mode=="read_error" || mode=="empty"){
            require(todo.find(ticket)!=std::string::npos && finished.empty(),"failure advanced source or finished");
        }
        if(mode=="complete" || mode=="factor" || mode=="auto_stage2" || mode=="advance_failure"){
            require(finished=="completed receipt\n","finished receipt");
            require(fs::exists(dir/"results.json.txt") && fs::exists(dir/"internal.jsonl"),"task publication missing");
            require(!fs::exists(dir/".m0_s1_bit14_20.ckpt"),"completed checkpoint not sealed");
            if(mode=="advance_failure")require(todo.find("ECM2=1,2,127,-1,20")!=std::string::npos,"wrong ticket removed");
            else{
                require(todo.find(ticket)==std::string::npos,"completed source retained");
                require((todo.find(next)!=std::string::npos)==(mode=="auto_stage2"),"Stage2 replacement/factor policy");
                require(read(dir/"sync/point.save")=="queue-contract-save\n","save sync");
            }
        }
        if(mode=="stop" || mode=="prepare" || mode=="publication")require(fs::exists(dir/".m0_s1_bit14_20.ckpt"),"failure checkpoint lost");
        std::cout<<"queue_contract: mode="<<mode<<" reads="<<reads<<" computations="<<computations<<" passed=1\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
