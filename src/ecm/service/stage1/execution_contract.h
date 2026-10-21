#pragma once
#include "ecm/service/stage1/task.h"
#include "../execution/ecm.h"

// One private computation contract shared by CPU/GPU implementations.
// Its raw payload and mutable progress stay private to the implementation.
namespace ecm_stage1 {

// Private mutable execution state. The caller's resolved options remain immutable.
struct ExecutionOptions : Stage1Options {
    explicit ExecutionOptions(const Stage1Options &options);
    long long stage1_t0_ms = 0;
    uint32_t stage1_curves = 0;
};

struct ExecutionResult {
    // Legacy CPU/OpenCL collaborators may still report through the old flags.
    // CUDA workflows report failure or stop explicitly, before returning.
    enum class Completion { Legacy, Stopped, SampleLimit, Failed };
    Completion completion = Completion::Legacy;
    TaskStatus status = TaskStatus::Failed;
    FailureSource failure_source = FailureSource::None;
    int task_return = ECM_ERROR; // Preserve the task/legacy return-code distinction.
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    bool publish_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    uint64_t firstsigma64 = 0;       // full 64-bit sigma
    mpz_t *factors = nullptr;
    int *array_found = nullptr;
    ExecutionResult() = default;
    ExecutionResult(const ExecutionResult &) = delete;
    ExecutionResult &operator=(const ExecutionResult &) = delete;
    ExecutionResult(ExecutionResult &&other) noexcept;
    ExecutionResult &operator=(ExecutionResult &&other) noexcept;
    ~ExecutionResult();
    void fail(FailureSource source) noexcept;
    void stop() noexcept;
    void finish_task(int code,bool stopped) noexcept;
    void swap(ExecutionResult &other) noexcept;
    // Per-curve sigma of THIS batch (malloc'd, `curves` entries, owned by this result; may be
    // null when a path does not know them). Needed because one batch does NOT always
    // share a sigma sequence: the Edwards path draws a random 64-bit sigma per curve,
    // so "firstsigma + i" would be wrong there. Consumers: the hit lines (D3, see
    // docs/usage/GUI.md) and therefore the GUI's results.txt sigma list.
    uint64_t *sigmas = nullptr;
};

struct TaskDependencies {
    int (*execute)(mpz_srcptr,double,double,uint32_t,const std::string &,bool,
                   const std::string &,ExecutionOptions &,ExecutionResult *);
    const std::string &(*build_identity)();
    int (*default_naf_window)();
};

bool stage1_publish_factor(const Stage1Options &,mpz_srcptr,uint64_t,uint64_t,mpz_srcptr);
} // namespace ecm_stage1
