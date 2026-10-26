#pragma once
#include "ecm/stage1/selection.h"
#include <gmp.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ecm_save { struct Range; struct Naming; }

namespace ecm_stage1 {

// Resolved per-task values, separated by the responsibilities that consume them.
struct Stage1ExecutionPreferences {
    Stage1ExecutionPreferences();
    Selection selection{};
    uint32_t stage1_threads;
    std::vector<unsigned> affinity_cpus;
    int naf_w;
    int verbose=0;
    int device_index;
    std::string gpu_mul_path,gpu_sqr_path,gpu_add_path,gpu_sub_path,gpu_special_mult_path;
};

struct Stage1Identity {
    bool sigma_fixed=false;
    uint64_t fixed_sigma64=0;
    bool target_kbnc=false;
    double handoff_k=1.0;
    std::string handoff_k_str="1";
    uint32_t handoff_b=2,handoff_n=0;
    int32_t handoff_c=-1;
    std::string handoff_factors;
    std::string result_user,result_computer,result_aid;
};

struct Stage1Persistence {
    Stage1Persistence();
    std::string save_name_pattern;
    std::string legacy_save_source;
    std::string tmp_dir;
    std::string exponent_cache_dir; // Empty explicitly disables the optional cache.
    unsigned long ckpt_ms;
    unsigned ckpt_backups;
    std::string results_file="stage1_results.jsonl",submission_file="results.json.txt";
    bool retain_completed_checkpoint=false;
};

struct Stage1Options {
    Stage1ExecutionPreferences execution;
    Stage1Identity identity;
    Stage1Persistence persistence;
};

// Synchronous request: N, strings and resolved options are borrowed until
// run_stage1_task returns. Its owned result retains none of these references.
struct Stage1Request {
    mpz_srcptr modulus;
    double b1, b2;
    uint32_t curves;
    const std::string &save_file;
    bool append_save;
    const std::string &expression;
    const Stage1Options &options;
};

enum class TaskStatus { Completed, Stopped, Failed };
enum class FailureSource { None, Preparation, Execution, Publication, Persistence, Device };

struct TaskDependencies;

// Owns the per-curve payload. Getter pointers remain borrowed until this value
// is moved or destroyed; factor/found/sigma indices must be below curves().
class TaskResult {
    struct State;
    std::unique_ptr<State> data_;
    friend TaskResult run_stage1_task(const Stage1Request &,const TaskDependencies &);
public:
    TaskResult();
    ~TaskResult();
    TaskResult(TaskResult &&) noexcept;
    TaskResult &operator=(TaskResult &&) noexcept;
    TaskResult(const TaskResult &) = delete;
    TaskResult &operator=(const TaskResult &) = delete;
    TaskStatus status() const noexcept;
    FailureSource failure_source() const noexcept;
    int task_return() const noexcept;
    int legacy_return() const noexcept;
    uint32_t curves() const noexcept;
    uint32_t first_sigma() const noexcept;
    bool has_factors() const noexcept;
    bool has_found() const noexcept;
    bool has_sigmas() const noexcept;
    mpz_srcptr factor(uint32_t index) const noexcept;
    int found(uint32_t index) const noexcept;
    uint64_t sigma(uint32_t index) const noexcept;
    const ecm_save::Range &save_range() const noexcept;
    const std::string &completed_checkpoint() const noexcept;
};

TaskResult run_stage1_task(const Stage1Request &,const TaskDependencies &);

// Project the resolved task identity into the IO naming contract.
ecm_save::Naming save_naming(const Stage1Options &);

} // namespace ecm_stage1
