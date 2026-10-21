#include "execution_contract.h"
#include <cstdlib>
#include <utility>
#include <new>
#include <limits>
#include <stdexcept>
namespace ecm_stage1 {
ExecutionOptions::ExecutionOptions(const Stage1Options &options) : Stage1Options(options) {}

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
    }
}
