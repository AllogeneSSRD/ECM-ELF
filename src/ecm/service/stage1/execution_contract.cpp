#include "execution_contract.h"
#include <cstdlib>
#include <utility>
#include <new>
#include <limits>
#include <stdexcept>
namespace ecm_stage1 {
ExecutionOptions::ExecutionOptions(const Stage1Options &options) : Stage1Options(options) {
    if(const auto *algorithm=std::getenv("ECM_GPU_STAGE1_ALGO"))gpu_algorithm=algorithm;
    if(gpu_algorithm.empty())gpu_algorithm="ladder";
}

ExecutionResult::ExecutionResult(ExecutionResult &&other) noexcept { swap(other); }

ExecutionResult &ExecutionResult::operator=(ExecutionResult &&other) noexcept {
        ExecutionResult incoming(std::move(other));
        swap(incoming);
        return *this;
    }

ExecutionResult::~ExecutionResult() {
        if(factors)for(uint32_t i=0;i<curves;++i)mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        free(sigmas); // May exist on a stopped CPU Montgomery task with no factors.
    }

void ExecutionResult::allocate_curves(uint32_t count) {
    if(!count || count>std::numeric_limits<size_t>::max()/sizeof(mpz_t) ||
       count>std::numeric_limits<size_t>::max()/sizeof(uint64_t))
        throw std::length_error("invalid Stage1 result allocation size");
    ExecutionResult incoming;
    incoming.curve_kinds.assign(count,CurveKind::Legacy);
    incoming.factors=static_cast<mpz_t *>(std::malloc(sizeof(mpz_t)*size_t(count)));
    incoming.array_found=static_cast<int *>(std::malloc(sizeof(int)*size_t(count)));
    incoming.sigmas=static_cast<uint64_t *>(std::malloc(sizeof(uint64_t)*size_t(count)));
    if(!incoming.factors || !incoming.array_found || !incoming.sigmas)throw std::bad_alloc();
    for(uint32_t i=0;i<count;++i) {
        mpz_init(incoming.factors[i]);incoming.array_found[i]=ECM_NO_FACTOR_FOUND;incoming.sigmas[i]=0;
    }
    incoming.curves=count;
    swap(incoming);
}

void ExecutionResult::set_curve(uint32_t lane,CurveKind kind,mpz_srcptr value,uint64_t sigma) {
    if(lane>=curves || curve_kinds.size()!=curves || !factors || !array_found || !sigmas)
        throw std::logic_error("unallocated Stage1 curve result");
    if(value)mpz_set(factors[lane],value);else mpz_set_ui(factors[lane],0);
    sigmas[lane]=sigma;curve_kinds[lane]=kind;
    array_found[lane]=kind==CurveKind::Factor?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
}
void ExecutionResult::validate_curves(mpz_srcptr n) const {
    if(curve_kinds.empty())return; // Legacy CPU/OpenCL ownership contract.
    if(curve_kinds.size()!=curves)throw std::logic_error("Stage1 curve result count mismatch");
    for(uint32_t i=0;i<curves;++i) {
        const auto kind=curve_kinds[i];if(kind==CurveKind::Legacy)continue;
        if(kind==CurveKind::Failed || kind==CurveKind::Degenerate)
            throw std::runtime_error("Stage1 completed with an invalid curve outcome");
        if(kind==CurveKind::Factor) {
            if(array_found[i]!=ECM_FACTOR_FOUND_STEP1 || mpz_cmp_ui(factors[i],1)<=0 ||
               mpz_cmp(factors[i],n)>=0 || !mpz_divisible_p(n,factors[i]))
                throw std::runtime_error("invalid Stage1 factor outcome");
        } else if(array_found[i]!=ECM_NO_FACTOR_FOUND || mpz_sgn(factors[i])<0 || mpz_cmp(factors[i],n)>=0 ||
                  (kind==CurveKind::CompletedWithoutPoint && mpz_sgn(factors[i])))
            throw std::runtime_error("invalid Stage1 point outcome");
    }
}

void ExecutionResult::fail(FailureSource source) noexcept {
    ret=ECM_ERROR;completion=Completion::Failed;failure_source=source;
    prepare_failed=source==FailureSource::Preparation;
    publish_failed=source==FailureSource::Publication;
}

void ExecutionResult::stop() noexcept {
    ret=ECM_ERROR;completion=Completion::Stopped;
}

void ExecutionResult::finish_task(int code,bool stopped) noexcept {
        task_return=code;
        // Publication/prepare failure retains the existing fatal queue policy,
        // even when a stop was requested at the same time.
        if(completion==Completion::Failed || prepare_failed || publish_failed) {
            status=TaskStatus::Failed;
            if(failure_source==FailureSource::None)
                failure_source=publish_failed?FailureSource::Publication:FailureSource::Preparation;
        } else if(ret==ECM_ERROR) {
            const bool interrupted=completion==Completion::Stopped || completion==Completion::SampleLimit ||
                (completion==Completion::Legacy && stopped);
            status=interrupted?TaskStatus::Stopped:TaskStatus::Failed;
            failure_source=interrupted?FailureSource::None:FailureSource::Execution;
        } else {
            status=TaskStatus::Completed;
            failure_source=FailureSource::None;
        }
    }

void ExecutionResult::swap(ExecutionResult &other) noexcept {
        using std::swap;
        swap(completion,other.completion);
        swap(status,other.status);swap(failure_source,other.failure_source);swap(task_return,other.task_return);
        swap(ret,other.ret);swap(prepare_failed,other.prepare_failed);swap(publish_failed,other.publish_failed);
        swap(curves,other.curves);swap(firstsigma,other.firstsigma);swap(firstsigma64,other.firstsigma64);
        swap(factors,other.factors);swap(array_found,other.array_found);swap(sigmas,other.sigmas);
        swap(curve_kinds,other.curve_kinds);
    }
}
