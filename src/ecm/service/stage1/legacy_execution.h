#pragma once
#include "ecm/service/stage1/task.h"
#include "../execution/legacy/ecm.h"
#include <cstdlib>
#include <utility>

// Transitional seam for the complete existing CPU/GPU execution blocks.
// Its raw payload and mutable progress stay private to the implementation.
namespace ecm_stage1 {
using Stage1RunOptions=Stage1Options;
using Stage1TaskStatus=TaskStatus;
using Stage1FailureSource=FailureSource;

// Private mutable legacy execution state. The caller's resolved options remain immutable.
struct Stage1ExecutionOptions : Stage1RunOptions {
    explicit Stage1ExecutionOptions(const Stage1RunOptions &options) : Stage1RunOptions(options) {}
    long long stage1_t0_ms = 0;
    uint32_t stage1_curves = 0;
};

struct LegacyStage1Result {
    Stage1TaskStatus status = Stage1TaskStatus::Failed;
    Stage1FailureSource failure_source = Stage1FailureSource::None;
    int task_return = ECM_ERROR; // Preserve the task/legacy return-code distinction.
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    bool publish_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    uint64_t firstsigma64 = 0;       // full 64-bit sigma
    mpz_t *factors = nullptr;
    int *array_found = nullptr;
    LegacyStage1Result() = default;
    LegacyStage1Result(const LegacyStage1Result &) = delete;
    LegacyStage1Result &operator=(const LegacyStage1Result &) = delete;
    LegacyStage1Result(LegacyStage1Result &&other) noexcept { swap(other); }
    LegacyStage1Result &operator=(LegacyStage1Result &&other) noexcept {
        LegacyStage1Result incoming(std::move(other));
        swap(incoming);
        return *this;
    }
    ~LegacyStage1Result() {
        if(factors)for(uint32_t i=0;i<curves;++i)mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        free(sigmas); // May exist on a stopped CPU Montgomery task with no factors.
    }
    void finish_task(int code,bool stopped) noexcept {
        task_return=code;
        // Publication/prepare failure retains the existing fatal queue policy,
        // even when a stop was requested at the same time.
        if(prepare_failed || publish_failed) {
            status=Stage1TaskStatus::Failed;
            failure_source=publish_failed?Stage1FailureSource::Publication:Stage1FailureSource::Preparation;
        } else if(ret==ECM_ERROR) {
            status=stopped?Stage1TaskStatus::Stopped:Stage1TaskStatus::Failed;
            failure_source=stopped?Stage1FailureSource::None:Stage1FailureSource::Execution;
        } else {
            status=Stage1TaskStatus::Completed;
            failure_source=Stage1FailureSource::None;
        }
    }
    void swap(LegacyStage1Result &other) noexcept {
        using std::swap;
        swap(status,other.status);swap(failure_source,other.failure_source);swap(task_return,other.task_return);
        swap(ret,other.ret);swap(prepare_failed,other.prepare_failed);swap(publish_failed,other.publish_failed);
        swap(curves,other.curves);swap(firstsigma,other.firstsigma);swap(firstsigma64,other.firstsigma64);
        swap(factors,other.factors);swap(array_found,other.array_found);swap(sigmas,other.sigmas);
    }
    // Per-curve sigma of THIS batch (malloc'd, `curves` entries, owned by this result; may be
    // null when a path does not know them). Needed because one batch does NOT always
    // share a sigma sequence: the Edwards path draws a random 64-bit sigma per curve,
    // so "firstsigma + i" would be wrong there. Consumers: the hit lines (D3, see
    // docs/usage/GUI.md) and therefore the GUI's results.txt sigma list.
    uint64_t *sigmas = nullptr;
};

struct TaskResult::State : LegacyStage1Result {};

struct TaskDependencies {
    int (*execute)(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                   const std::string &,Stage1ExecutionOptions &,LegacyStage1Result *);
    const std::string &(*build_identity)();
    int (*default_naf_window)();
};

bool stage1_publish_factor(const Stage1Options &,mpz_srcptr,uint64_t,uint64_t,mpz_srcptr);
bool stage1_results_complete(const Stage1Options &,mpz_srcptr,uint64_t,bool,unsigned,
                             const std::vector<uint64_t> &);
} // namespace ecm_stage1
