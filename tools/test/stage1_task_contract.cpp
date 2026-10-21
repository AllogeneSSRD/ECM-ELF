#include "ecm/service/stage1/task.h"
#include "../../src/ecm/service/stage1/legacy_execution.h"
#include "../../src/core/ecm_stage1_checkpoint_runtime.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#endif

using namespace ecm_stage1;
static_assert(!std::is_copy_constructible_v<TaskResult>);
static_assert(std::is_nothrow_move_constructible_v<TaskResult>);
static_assert(std::is_nothrow_move_assignable_v<TaskResult>);
namespace fs=std::filesystem;
static std::string mode;
static int calls=0;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void stop_now() {
#ifdef _WIN32
    static HANDLE event=CreateEventW(nullptr,TRUE,FALSE,nullptr);
    require(event && ecm_stage1_set_stop_handle(reinterpret_cast<uintptr_t>(event)),"stop handle");
    require(SetEvent(event)!=FALSE,"signal stop");
#else
    ecm_stage1_install_stop_handler();std::raise(SIGINT);
#endif
}
static const std::string &identity() {static const std::string value="task-contract";return value;}
static int window() {return 8;}

// Computation is replaced at the private legacy seam. Real task sequencing,
// checkpoint locking, publication, result ownership and status mapping run.
static int execute(mpz_srcptr,double,double,uint32_t count,const std::string &,bool,
                   const std::string &,Stage1ExecutionOptions &options,LegacyStage1Result *out) {
    ++calls;
    require(options.stage1_curves==0 && options.stage1_t0_ms==0,"execution state reused");
    options.stage1_curves=count;options.stage1_t0_ms=123;
    if(mode=="throw")throw std::runtime_error("execution fixture");
    out->curves=count;
    out->sigmas=static_cast<uint64_t *>(std::malloc(sizeof(uint64_t)*count));
    require(out->sigmas!=nullptr,"sigma allocation");
    for(uint32_t i=0;i<count;++i)out->sigmas[i]=ecm_stage1_batch_sigma(i,26+i);
    require(ecm_stage1_write_blob(ecm_stage1_curve_record(0),"task-contract-state"),"checkpoint write");
    if(mode=="stop" || mode=="stop_publication") {
        stop_now();out->publish_failed=mode=="stop_publication";
        return ECM_ERROR; // Deliberately only sigmas: owner must release independently.
    }
    if(mode=="prepare") {out->prepare_failed=true;return ECM_ERROR;}
    auto *factors=static_cast<mpz_t *>(std::malloc(sizeof(mpz_t)*count));
    auto *found=static_cast<int *>(std::malloc(sizeof(int)*count));
    if(!factors || !found){std::free(factors);std::free(found);throw std::bad_alloc();}
    for(uint32_t i=0;i<count;++i) {
        const bool hit=i==0 && (mode=="factor" || mode=="publication");
        mpz_init_set_ui(factors[i],hit?(mode=="factor"?101:5):0);
        found[i]=hit?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
    }
    out->factors=factors;out->array_found=found;
    out->firstsigma=26;out->firstsigma64=26;
    out->ret=ECM_NO_FACTOR_FOUND; // A hit array may carry a factor independently of this code.
    return out->ret;
}

int main(int argc,char **argv) {
    try {
        require(argc==3,"usage: stage1_task_contract MODE NEW_DIRECTORY");mode=argv[1];
        const fs::path folder=fs::absolute(argv[2]);require(!fs::exists(folder),"fresh directory required");
        fs::create_directories(folder);
        Stage1Options options;options.execution.selection.target=ExecutionTarget::Cpu;
        options.execution.selection.curve=CurveModel::Montgomery;options.execution.selection.gpu_param=0;
        options.persistence.tmp_dir=folder.string();options.persistence.results_file=(folder/"internal.jsonl").string();
        options.persistence.submission_file=(folder/"results.json.txt").string();options.identity.sigma_fixed=true;options.identity.fixed_sigma64=26;
        const auto save=(folder/"point.save").string();const std::string expression="10403";
        const auto checkpoint=folder/".m0_s1_bit14_20.ckpt";
        if(mode=="bad_begin") {std::ofstream damaged(checkpoint,std::ios::binary);damaged<<"damaged-container";}
        mpz_t n;mpz_init_set_ui(n,10403);
        const TaskDependencies deps{execute,identity,window};
        auto result=run_stage1_task({n,20,20,2,save,false,expression,options},deps);
        mpz_clear(n); // Result factors/sigmas must not borrow the request's modulus.
        require(options.execution.stage1_threads==ecm_stage1::Stage1Options{}.execution.stage1_threads,"request changed");
        if(mode=="complete" || mode=="factor") {
            require(calls==1 && result.status()==TaskStatus::Completed,"completed status");
            require(result.failure_source()==FailureSource::None && result.task_return()==ECM_NO_FACTOR_FOUND,"completed code");
            require(result.curves()==2 && result.has_sigmas() && result.sigma(1)==27,"curve identity");
            require(!fs::exists(checkpoint) && fs::exists(options.persistence.results_file) && fs::exists(options.persistence.submission_file),"completed publication");
            auto moved=std::move(result);
            require(!result.has_factors() && moved.curves()==2 && moved.sigma(0)==26,"move ownership");
            if(mode=="factor")require(mpz_cmp_ui(moved.factor(0),101)==0 && moved.found(0)==ECM_FACTOR_FOUND_STEP1,"owned factor with legacy NF code");
        } else if(mode=="stop") {
            require(result.status()==TaskStatus::Stopped && result.failure_source()==FailureSource::None,"stopped status");
            require(result.has_sigmas() && !result.has_factors(),"independently owned sigma payload");
            require(fs::exists(checkpoint) && !fs::exists(options.persistence.submission_file),"stop published incomplete NF");
        } else if(mode=="publication" || mode=="stop_publication") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Publication,"publication priority/source");
            require(fs::exists(checkpoint),"publication failure lost checkpoint");
        } else if(mode=="prepare" || mode=="throw" || mode=="bad_begin") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Preparation,"legacy preparation mapping");
            if(mode=="bad_begin") {
                require(calls==0,"corrupt begin reached execution");
                require(fs::exists(checkpoint),"damaged evidence removed");
            }
        } else throw std::runtime_error("unknown mode");
        // Windows' no-share lock can be reacquired only after TaskSession cleanup.
#ifdef _WIN32
        const auto lock=checkpoint.string()+".lock";
        HANDLE handle=CreateFileA(lock.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_FLAG_DELETE_ON_CLOSE,nullptr);
        require(handle!=INVALID_HANDLE_VALUE,"task leaked batch lock");CloseHandle(handle);
#endif
        std::cout<<"task_contract: mode="<<mode<<" calls="<<calls<<" passed=1\n";return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
