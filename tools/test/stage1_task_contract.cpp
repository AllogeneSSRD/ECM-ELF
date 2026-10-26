#include "ecm/service/stage1/task.h"
#include "../../src/ecm/service/stage1/execution_contract.h"
#include "io/stage1_checkpoint.h"
#include "io/stage1_binary.h"
#include "ecm/io/save.h"
#include "io/files.h"
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
#ifdef ECM_TEST_HISTORY_SNAPSHOT
void test_history_swap(const fs::path&, const std::string&);
bool test_history_swapped();
#endif
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
static int legacy_execute(mpz_srcptr n,double,double,uint32_t count,const std::string &save,bool,
                   const std::string &,ExecutionOptions &options,LegacyExecutionResult *out,
                   ecm_stage1_checkpoint::Context& checkpoint) {
    ++calls;
    require(options.stage1_curves==0 && options.stage1_t0_ms==0,"execution state reused");
    options.stage1_curves=count;options.stage1_t0_ms=123;
    if(mode=="throw")throw std::runtime_error("execution fixture");
    out->curves=count;
    out->sigmas=static_cast<uint64_t *>(std::malloc(sizeof(uint64_t)*count));
    require(out->sigmas!=nullptr,"sigma allocation");
    for(uint32_t i=0;i<count;++i)out->sigmas[i]=checkpoint.sigma(i,26+i);
    require(checkpoint.write_blob(ecm_stage1_curve_record(0),"task-contract-state"),"checkpoint write");
    if(mode=="selection_snapshot") {
#ifdef _WIN32
        require(_putenv_s("ECM_GPU_STAGE1_ALGO","bad")==0,"environment mutation");
#else
        require(setenv("ECM_GPU_STAGE1_ALGO","bad",1)==0,"environment mutation");
#endif
        require(options.gpu_algorithm=="resident","execution selection did not retain snapshot");
        std::ifstream file(checkpoint.path(),std::ios::binary);
        const std::string bytes{std::istreambuf_iterator<char>(file),{}};
        ecm_stage1_binary::Reader in(bytes);in.take(8);in.u32();in.u32();const auto identity=in.take(in.u64());
        require(identity.size()>=13 && identity.substr(identity.size()-13)=="|gpu|resident","task identity differs from execution selection");
    }
    if(mode=="stop" || mode=="stop_publication") {
        stop_now();out->publish_failed=mode=="stop_publication";
        return ECM_ERROR; // Deliberately only sigmas: owner must release independently.
    }
    if(mode=="legacy_failure")return ECM_ERROR;
    if(mode=="prepare") {out->prepare_failed=true;return ECM_ERROR;}
    if(mode=="execution" || mode=="execution_stop") {
        if(mode=="execution_stop")stop_now();
        out->fail(FailureSource::Execution);
        return ECM_ERROR;
    }
    if(mode=="persistence" || mode=="persistence_stop") {
        if(mode=="persistence_stop")stop_now();
        out->fail(FailureSource::Persistence);return ECM_ERROR;
    }
    if(mode=="device" || mode=="device_stop") {
        if(mode=="device_stop")stop_now();
        out->fail(FailureSource::Device);return ECM_ERROR;
    }
    auto *factors=static_cast<mpz_t *>(std::malloc(sizeof(mpz_t)*count));
    auto *found=static_cast<int *>(std::malloc(sizeof(int)*count));
    if(!factors || !found){std::free(factors);std::free(found);throw std::bad_alloc();}
    for(uint32_t i=0;i<count;++i) {
        const bool hit=i==0 && (mode=="factor" || mode=="publication");
        mpz_init_set_ui(factors[i],hit?101:0);
        found[i]=hit?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
    }
    out->factors=factors;out->array_found=found;
    out->firstsigma=26;out->firstsigma64=26;
    out->ret=ECM_NO_FACTOR_FOUND; // A hit array may carry a factor independently of this code.
    if(mode=="concurrent_append") {
        ecm_edwards::Integer x;mpz_set_ui(x.v,1);
        require(ecm_save::publish(save,ecm_encode_mont_save(n,20,0,{{999,x.v}},"10403"),true),"concurrent append fixture");
    }
    if(mode=="save_failure")fs::create_directory(save);
    if(mode=="publication")fs::create_directory(options.persistence.submission_file);
    if(mode=="typed_complete" || mode=="typed_no_point" || mode=="typed_invalid") {
        using Kind=LegacyExecutionResult::CurveKind;
        out->curve_kinds.assign(count,mode=="typed_no_point"?Kind::CompletedWithoutPoint:Kind::Stage2Point);
        if(mode=="typed_invalid")out->curve_kinds[0]=Kind::Degenerate;
        out->completion=LegacyExecutionResult::Completion::Completed;
    }
    if(mode=="typed_partial_point") {
        using Kind=LegacyExecutionResult::CurveKind;
        out->curve_kinds={Kind::CompletedWithoutPoint,Kind::Stage2Point};
        mpz_set_ui(factors[1],1);out->completion=LegacyExecutionResult::Completion::Completed;
    }
    return out->ret;
}

static ComputationResult execute(const Stage1Request &r,ExecutionOptions &options,
                                  ecm_stage1_checkpoint::Context& checkpoint) {
    LegacyExecutionResult raw;
    const auto code=legacy_execute(r.modulus,r.b1,r.b2,r.curves,r.save_file,r.append_save,r.expression,options,&raw,checkpoint);
    return take_computation_result(std::move(raw),code,ecm_stage1_stop_requested()!=0);
}

int main(int argc,char **argv) {
    try {
        require(argc==3,"usage: stage1_task_contract MODE NEW_DIRECTORY");mode=argv[1];
        const fs::path folder=fs::absolute(argv[2]);require(!fs::exists(folder),"fresh directory required");
        fs::create_directories(folder);
        Stage1Options options;options.execution.selection.target=ExecutionTarget::Cpu;
        options.execution.selection.curve=CurveModel::Montgomery;options.execution.selection.gpu_param=0;
        if(mode=="typed_partial_point" || mode=="typed_no_point")options.execution.selection.curve=CurveModel::Edwards;
        if(mode=="selection_snapshot") {
            options.execution.selection.target=ExecutionTarget::Gpu;
#ifdef _WIN32
            require(_putenv_s("ECM_GPU_STAGE1_ALGO","resident")==0,"fixture environment");
#else
            require(setenv("ECM_GPU_STAGE1_ALGO","resident",1)==0,"fixture environment");
#endif
        }
        options.persistence.tmp_dir=folder.string();options.persistence.results_file=(folder/"internal.jsonl").string();
        options.persistence.submission_file=(folder/"results.json.txt").string();options.identity.sigma_fixed=true;options.identity.fixed_sigma64=26;
        const auto save=(folder/"point.save").string();const std::string expression="10403";
        if(mode=="alias_save_internal")options.persistence.results_file=(folder/"POINT.SAVE").string();
        if(mode=="alias_save_submission")options.persistence.submission_file=(folder/"POINT.SAVE").string();
        const auto checkpoint=folder/((mode=="typed_partial_point" || mode=="typed_no_point")?".e_s1_bit14_20.ckpt":".m0_s1_bit14_20.ckpt");
        if(mode=="bad_begin") {std::ofstream damaged(checkpoint,std::ios::binary);damaged<<"damaged-container";}
        mpz_t n;mpz_init_set_ui(n,10403);
#ifdef ECM_TEST_HISTORY_SNAPSHOT
        if(mode=="history_snapshot") {
            const auto source=folder/"legacy.save";
            ecm_edwards::Integer x;mpz_set_ui(x.v,1);
            std::ofstream(source)<<ecm_encode_mont_save(n,20,0,{{999,x.v}},expression);
            options.persistence.legacy_save_source=source.string();
            test_history_swap(source,ecm_edwards::save_line(n,x.v,888,20,false));
        }
#endif
        const TaskDependencies deps{execute,identity,window};
        auto result=run_stage1_task({n,20,20,2,save,false,expression,options},deps);
        mpz_clear(n); // Result factors/sigmas must not borrow the request's modulus.
        require(options.execution.stage1_threads==ecm_stage1::Stage1Options{}.execution.stage1_threads,"request changed");
        if(mode=="alias_save_internal" || mode=="alias_save_submission") {
            require(result.status()==TaskStatus::Failed && calls==0,"aliased save reached computation");
            require(!fs::exists(save) && !fs::exists(options.persistence.submission_file),"alias rejection modified output");
#ifdef ECM_TEST_HISTORY_SNAPSHOT
        } else if(mode=="history_snapshot") {
            require(test_history_swapped(),"source mutation point was not reached");
            require(result.status()==TaskStatus::Failed && calls==0,"unvalidated source snapshot reached computation");
            require(!fs::exists(save) || ecm_io::read_file(save).find("METHOD=ECM-EDWARDS-AM")==std::string::npos,
                    "copied source bytes differ from validated history");
#endif
        } else if(mode=="concurrent_append" || mode=="complete" || mode=="factor" || mode=="typed_complete" || mode=="typed_no_point" || mode=="typed_partial_point" || mode=="selection_snapshot") {
            require(calls==1 && result.status()==TaskStatus::Completed,"completed status");
            require(result.failure_source()==FailureSource::None && result.task_return()==ECM_NO_FACTOR_FOUND,"completed code");
            require(result.curves()==2 && result.has_sigmas() && result.sigma(1)==27,"curve identity");
            require(!fs::exists(checkpoint) && fs::exists(options.persistence.results_file) && fs::exists(options.persistence.submission_file),"completed publication");
            if(mode=="concurrent_append")require(result.save_range().skip==1 && ecm_save::record_count(save)==3,"concurrent history overwritten");
            if(mode=="typed_partial_point")require(result.save_range().valid && result.save_range().records.size()==1,"partial actual range");
            auto moved=std::move(result);
            require(!result.has_factors() && moved.curves()==2 && moved.sigma(0)==26,"move ownership");
            if(mode=="factor")require(mpz_cmp_ui(moved.factor(0),101)==0 && moved.found(0)==ECM_FACTOR_FOUND_STEP1,"owned factor with legacy NF code");
        } else if(mode=="stop") {
            require(result.status()==TaskStatus::Stopped && result.failure_source()==FailureSource::None,"stopped status");
            require(result.curves()==0,"incomplete compatibility payload became completed curve outcomes");
            require(fs::exists(checkpoint) && !fs::exists(options.persistence.submission_file),"stop published incomplete NF");
        } else if(mode=="publication" || mode=="stop_publication" || mode=="save_failure") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Publication,"publication priority/source");
            require(fs::exists(checkpoint),"publication failure lost checkpoint");
        } else if(mode=="execution" || mode=="execution_stop" || mode=="throw" || mode=="typed_invalid") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Execution,"explicit execution failure/stop priority");
            require((mode=="throw" || fs::exists(checkpoint)) && !fs::exists(options.persistence.submission_file),"execution failure lost recovery or published NF");
        } else if(mode=="persistence" || mode=="persistence_stop") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Persistence,"persistence failure/stop priority");
            require(fs::exists(checkpoint),"persistence failure lost checkpoint");
        } else if(mode=="device" || mode=="device_stop") {
            require(result.status()==TaskStatus::Failed && result.failure_source()==FailureSource::Device,"device failure/stop priority");
            require(fs::exists(checkpoint),"device failure lost checkpoint");
        } else if(mode=="legacy_failure" || mode=="prepare" || mode=="bad_begin") {
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
