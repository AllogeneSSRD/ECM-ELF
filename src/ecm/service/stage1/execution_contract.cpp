#include "execution_contract.h"
#include <cstdlib>
#include <utility>
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

void ExecutionResult::finish_task(int code,bool stopped) noexcept {
        task_return=code;
        // Publication/prepare failure retains the existing fatal queue policy,
        // even when a stop was requested at the same time.
        if(prepare_failed || publish_failed) {
            status=TaskStatus::Failed;
            failure_source=publish_failed?FailureSource::Publication:FailureSource::Preparation;
        } else if(ret==ECM_ERROR) {
            status=stopped?TaskStatus::Stopped:TaskStatus::Failed;
            failure_source=stopped?FailureSource::None:FailureSource::Execution;
        } else {
            status=TaskStatus::Completed;
            failure_source=FailureSource::None;
        }
    }

void ExecutionResult::swap(ExecutionResult &other) noexcept {
        using std::swap;
        swap(status,other.status);swap(failure_source,other.failure_source);swap(task_return,other.task_return);
        swap(ret,other.ret);swap(prepare_failed,other.prepare_failed);swap(publish_failed,other.publish_failed);
        swap(curves,other.curves);swap(firstsigma,other.firstsigma);swap(firstsigma64,other.firstsigma64);
        swap(factors,other.factors);swap(array_found,other.array_found);swap(sigmas,other.sigmas);
    }
}
