#pragma once
// Private normalized invocation/state shared by the Stage2 executable modules.
// TOML Settings stays in the application; computation modules borrow no parser.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "../../platform/process/stage2_curve_process.h"
#include <gmp.h>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../platform/generated/configuration_defaults.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "../../backend/cuda/stage2/ecm_cuda_stage2.h"
#include "../planning/legacy/ecm_stage2_portable_model.h"
namespace s2prod {
namespace fs=std::filesystem;
struct TaskInputError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Big {
    mpz_t z;
    Big() { mpz_init(z); }
    ~Big() { mpz_clear(z); }
    Big(const Big &) = delete;
    Big &operator=(const Big &) = delete;
};
struct Record {
    std::string n, x, a24;
    uint64_t sigma = 0, b1 = 0, offset = 0, hash = 0, index = 0;
    bool checksum = false, edwards = false;
};
struct Options {
    std::string ini, save, worktodo, results, log, debug_file, receipt,publication_task,submission,submission_ini,submission_aid;
    uint64_t submission_exponent=0;
    std::vector<std::string> submission_known;
    bool debug_log = false;
    uint64_t b2 = 0, d = 0, skip = 0, curves = 0, batch = 0, arena = 0;
    unsigned carrier_exponent=0;
    bool has_carrier=false;
    bool auto_carrier=ecm_config::defaults::stage2_stage2_auto_carrier;
    int device = -1, worker = 1;
    int log_level = -1;
    bool ntt_counts=ecm_config::defaults::stage2_stage2_ntt_counts;
    bool relocation=ecm_config::defaults::stage2_stage2_relocation;
    bool dry = false, once = false, selection = false, help = false;
    bool has_d = false, has_batch = false, has_arena = false;
    bool child = false;
    bool gui_events=false;
    uint64_t stop_handle=0;
    uint64_t offset = 0, hash = 0, index = 0;
    bool plan_only = false, tune_options = false;
    std::string tune, tune_file;
    int tune_first = 16, tune_last = 27, tune_repeats = 5;
    std::vector<uint64_t> tune_slices{1};bool has_tune_slices=false;
    int tune_level = 0;
    bool has_tune_lengths = false, has_tune_repeats = false,has_tune_memory=false;
    bool tune_child=false;
    std::string tune_save;
    std::string tune_worktodo;
    bool tune_warmup=ecm_config::defaults::stage2_stage2_tune_warmup;
    std::vector<std::string> tune_merge;
    std::vector<std::string> tune_ntt_profiles;
    std::vector<uint64_t> tune_ds,tune_b2s,tune_exponents;
    unsigned tune_carrier=0;
    uint64_t tune_max_batches=0;bool has_tune_max_batches=false;
    unsigned tune_tail_samples=0;bool has_tune_tail_samples=false;
    uint64_t tune_memory_mb = 1024;
    bool factorize_hits = false;
    bool factor_only = false;
    bool has_gp = false, has_factor_timeout = false;
    std::string gp = ecm_config::defaults::stage2_stage2_gp;
    unsigned factor_timeout = static_cast<unsigned>(ecm_config::defaults::stage2_stage2_factor_timeout);
    bool auto_b2=false,cost_device_info=false;
    std::string cost_profile,tune_profile;
    uint64_t auto_min=0,auto_max=0,owner_mb=ecm_config::defaults::stage2_stage2_fold_mb,stage1_batch=ecm_config::defaults::stage2_stage1_batch;
    std::string stage1_profile,stage1_exponent="lcm",check_stage1_profile;
    bool has_stage1_exponent=false;
    double stage1_seconds=ecm_config::defaults::stage2_stage1_seconds_per_curve,ratio_adjust=ecm_config::defaults::stage2_stage2_ratio_adjust;
    std::string tune_ignore="gpu,driver,cuda,backend,environment",stage1_csv,condition_tag;
    uint64_t tune_budget=1800,short_budget=10,stage1_mhz=0,factor_bits=0;
    unsigned inferred_carrier=0;
    bool short_calibration=true;
    bool short_batch_tune=false;
    bool pack_zero=ecm_config::defaults::stage2_stage2_pack_zero;
    bool tune_runtime=ecm_config::defaults::stage2_stage2_tune_runtime;
    uint64_t runtime_samples=ecm_config::defaults::stage2_stage2_tune_runtime_samples;
    bool runtime_probe=false;
    uint64_t runtime_remaining=0; // Parent queue hint; not part of a measurement identity.
    bool runtime_prepare_cache=true; // Diagnostic A/B switch; not an arithmetic or timing-data identity.
    std::string runtime_prepare_hints; // Parent-memory cost-search pauses, never a checkpoint.
    std::string runtime_route_hints; // Sealed, successful geometry plans for this parent only.
    std::string prediction_policy="model",tune_arithmetic="both";
    double tune_error=.08;
    std::set<std::string> portable_cli;
    bool has_owner=false,has_stage1_seconds=false,has_stage1_batch=false,has_ratio=false;
};
struct QueueSelection {
    EcmStage2Task task;
    uint64_t b2 = 0, skip = 0, count = 0;
};

struct RuntimeTuneObservation {
    ecm_stage2::tune::portable::Shape shape;
    ecm_stage2::tune::Fields condition,sample;
    uint64_t plan_cache_hits=0;
};
// These are the only Settings fields consumed by planning/process output.
// Preserve the original configured console budgets, not a tuned Options budget.
struct TaskDefaults {uint64_t b2,arena,owner_mb;};
extern volatile LONG stop_requests;
}
