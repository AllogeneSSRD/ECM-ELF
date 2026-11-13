#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
namespace s2prod {
struct CurvePublication {
    std::string path, user, computer;
};

// One synchronous curve-process launch. Owns no CLI parser or compute state.
struct CurveProcessRequest {
    std::filesystem::path save,results,log;
    struct Record {uint64_t offset=0,hash=0,index=0;} record;
    struct ConsoleBudget {uint64_t arena=0,owner_mb=0;} console_budget;
    uint64_t b2=0,d=0,batch=0;
    int device=0,engine_log_level=0;
    uint64_t arena=0;
    bool auto_b2=0;
    bool auto_carrier=0;
    uint64_t auto_max=0;
    uint64_t auto_min=0;
    unsigned carrier_exponent=0;
    std::string condition_tag{};
    std::string cost_profile{};
    std::string debug_file{};
    bool debug_log=0;
    uint64_t factor_bits=0;
    bool factor_only=0;
    unsigned factor_timeout=0;
    bool factorize_hits=0;
    std::string gp{};
    bool gui_events=0;
    bool has_carrier=0;
    unsigned inferred_carrier=0;
    int log_level=0;
    bool ntt_counts=0;
    uint64_t owner_mb=0;
    bool pack_zero=0;
    bool weighted_ntt=false;
    std::string prediction_policy{};
    std::string publication_task{};
    double ratio_adjust=0;
    std::string receipt{};
    bool relocation=0;
    bool giant_tail_balance=false,b2_round_chain=false;
    std::string giant_chain="64";
    std::string baby_backend="scalar",giant_backend="scalar",seed_backend="scalar",reduce_backend="scalar";
    unsigned reduce_threads=128,reduce_kernel=0;
    bool runtime_prepare_cache=0;
    std::string runtime_prepare_hints{};
    uint64_t runtime_remaining=0;
    std::string runtime_route_hints{};
    uint64_t runtime_samples=0;
    uint64_t tune_min_repeats=2;
    bool short_batch_tune=0;
    int groot_layout=-1;
    uint64_t stage1_batch=0;
    std::string stage1_csv{};
    std::string stage1_exponent{};
    uint64_t stage1_mhz=0;
    std::string stage1_profile{};
    double stage1_seconds=0;
    CurvePublication publication;
    std::string submission_aid{};
    uint64_t submission_exponent=0;
    std::vector<std::string> submission_known{};
    bool tune_child=0;
    std::string tune_ignore{};
    std::string tune_profile{};
    bool tune_runtime=0;
    bool fixed_queue_route=false;
    uint64_t queue_requested_b2=0;
    bool queue_pipeline=false;
    uint64_t queue_curve=0; // One-based position in the task, independent of child/process slot.
    std::string queue_phase_gate;
    unsigned queue_workspace_lease=0;
    int worker=0;
    // Called synchronously at the original selection-output position, before
    // logging that line; domain validation/cache policy belongs to the caller.
    std::function<void(const std::string &)> observe_selection;
    std::function<void(const std::string &)> commit_outcome;
};
// Returns 0 (success), 3 (verified factor excluded from timing), or 1 (failure).
// Resource/I/O/stop failures throw; Job/pipe/process handles close on unwinding.
int run_curve_process(const CurveProcessRequest &request);
}
