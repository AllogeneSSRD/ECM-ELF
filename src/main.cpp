#include "io/normalize.h"
#include "ecm/application/cli/group_order.h"
#include "io/cli.h"
#include "sync/files.h"
#include "sync/p95.h"
#include "ecm/platform/log_file.h"
#include "ecm/storage/save/publication.h"
using ecm_stage1_app::get_exe_dir_local;
using ecm_stage1_app::get_gp_executable;
using ecm_stage1_app::normalize_path_local;
using ecm_stage1_app::print_group_order_factor;
using ecm_stage1_app::resolve_gp_path;
using ecm_stage1_app::resolve_rel_local;
using ecm_stage1_app::stage1_result_settings;
using ecm_stage1_app::stage1_save_construction;
using ecm_stage1_app::stage1_selection_error;
using ecm_stage1_app::stage1_task_error;
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#else
#include <unistd.h>
#endif

#include <gmp.h>

#include "ecm/service/execution/legacy_gpu.h"      /* GPU backend seam (OpenCL or CUDA glue) */
#include "ecm/storage/checkpoint/ecm_checkpoint.h" /* opencl_ecm_set_work_dir */
#include "ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "ecm/core/stage1/selection.h"
#include "ecm/service/stage1/execution_contract.h"
#include "ecm/application/configuration/runtime.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "io/config.h"
#include "io/worktodo.h"
#include "ecm/storage/transfer/ecm_stage2_transfer.h"
#include "ecm/storage/queue/ecm_worker_list.h"
#include "ecm/application/configuration/ecm_expr.h" /* shared exact integer expression parser */
#include "ecm/storage/transfer/p95_transfer.h" /* finished task -> Prime95 worktodo.add (D4/§13) */
#include "ecm/storage/cache/ecm_stage1_exp_cache.h" /* validated on-disk cache for that value */
#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
// Application assembly still selects the CPU default NAF window.
#include "ecm/backend/cpu/stage1/edwards/ecm_edwards_cpu.h"

#include "ecm/service/stage1/execution.h"
#include "ecm/service/stage1/preparation.h"
using ecm_stage1::stage1_prefixed_save;
using ecm_stage1::stage1_save_prefix;

using ecm_stage1::FailureSource;
using ecm_stage1::run_stage1_task;
using ecm_stage1::Stage1Options;
using ecm_stage1::TaskStatus;

#include "ecm/storage/queue/owner_lease.h"
#include "ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.h"

static void trim(std::string& s) {
    while (!s.empty() && isspace((unsigned char)s.back()))
        s.pop_back();
    while (!s.empty() && isspace((unsigned char)s.front()))
        s.erase(s.begin());
}

static const std::string& stage1_build_identity() {
    static const std::string identity = [] {
        std::ifstream in(std::filesystem::path(get_exe_dir_local()) /
                         "ecm_stage1_build_identity.txt");
        std::string value;
        std::getline(in, value);
        return value.size() <= 128 && !value.empty() &&
                       value.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz.-") ==
                           value.npos
                   ? value
                   : std::string(ECM_BUILD_ID);
    }();
    return identity;
}

// Selection is application composition; task invokes the selected entry directly.
static ecm_stage1::TaskDependencies stage1_tasks(const Stage1Options& options) {
    return {ecm_stage1::select_stage1_entry(options.execution.selection), stage1_build_identity,
            edwards_get_naf_w, ecm_stage1::defer_gpu_checkpoint_validation};
}

// Rebuild the original N expression for the save-file N= field.

/* A value for the machine-readable `p95_add:` notice lines. The GUI parses them, so a
   free-text reason (or a path with spaces) is quoted; the only escape is `\"` for a quote
   that belongs to the text, plus newlines become spaces -- so a Windows path stays
   readable in the log pane. A value must never end with a lone backslash (it would make
   the closing quote ambiguous), hence the final doubling. Documented in
   src/gui/log_parse.{h,cpp}, which unescapes exactly this way. */
/* Directory part / join for the Prime95 paths. The driver has its own resolve helper for
   ini values, but these two are about Prime95's own directory (prime.log, results.txt). */

/* ── D4: --gpu-info ───────────────────────────────────────────────────────────────
   Reports what the linked backend would do, as machine-readable `key=value` text:

     gpu_info=1               (or gpu_info=not_applicable on the OpenCL build)
     backend=<name>  device=<i>  name=<gpu>  sm_count=<n>  cc=<major>.<minor>
     curve_model=<mont|ed>  gpu_param=<0|2|3|-1>  fold=<0|1>
     carry_bits=<n>  picked=<0|1>  tier_count=<n>
     ini=<path|->  worker=<n>
     tier bits=<n> tpb=<n> tpi=<n> ipb=<n> blocks_per_sm=<n> blocks_min=<n>
          curves_min=<n> blocks_wave=<n> curves_wave=<n>
          [fold_blocks_min=<n> fold_curves_min=<n>]

   The leading `tier` token (the only token without an '=') is what marks a tier line,
   and `curves_wave` is the number the run path itself names when it suggests a
   `gpucurves` that keeps whole waves: blocks_per_sm * sm_count * (tpb / tpi).

   Zero side effects: no kernel launch, no checkpoint, no log file, no ini creation --
   it is safe to ask while another worker is computing. Exit code 0 for the
   "not applicable" answer too (asking is not an error); 1 only for a real failure
   (unknown -d, --bits too large for any tier, bad --gpu-param). */
static int print_gpu_info(const char* ini_used, int worker, int device_index, int gpu_param,
                          bool edwards, uint32_t want_bits) {
    ecm_backend_gpu_info info;
    const char* err = nullptr;
    const int rc =
        ecm_backend_query_gpu(device_index, gpu_param, edwards ? 1 : 0, want_bits, &info, &err);

    if (rc == ECM_BACKEND_QUERY_NOT_APPLICABLE) {
        fprintf(stdout, "gpu_info=not_applicable\nbackend=%s\nreason=%s\n", ecm_backend_name(),
                (err != nullptr) ? err : "not applicable");
        fflush(stdout);
        return 0;
    }
    if (rc != ECM_BACKEND_QUERY_OK) {
        fprintf(stderr, "gpu_info=error\ngpu_info_error=%s\n",
                (err != nullptr) ? err : "unknown error");
        fflush(stderr);
        return 1;
    }

    fprintf(stdout, "gpu_info=1\n");
    fprintf(stdout, "backend=%s\n", ecm_backend_name());
    fprintf(stdout, "device=%d\n", info.device_index);
    fprintf(stdout, "name=%s\n", info.name);
    fprintf(stdout, "sm_count=%d\n", info.sm_count);
    fprintf(stdout, "cc=%d.%d\n", info.cc_major, info.cc_minor);
    fprintf(stdout, "curve_model=%s\n", info.edwards ? "ed" : "mont");
    fprintf(stdout, "gpu_param=%d\n", info.gpu_param);
    fprintf(stdout, "fold=%d\n", info.fold);
    fprintf(stdout, "carry_bits=%u\n", info.carry_bits);
    fprintf(stdout, "picked=%d\n", info.picked);
    fprintf(stdout, "tier_count=%d\n", info.tier_count);
    fprintf(stdout, "ini=%s\n", (ini_used != nullptr && ini_used[0] != '\0') ? ini_used : "-");
    fprintf(stdout, "worker=%d\n", worker);

    for (int i = 0; i < info.tier_count; ++i) {
        const ecm_backend_tier& t = info.tiers[i];
        fprintf(stdout,
                "tier bits=%u tpb=%u tpi=%u ipb=%u blocks_per_sm=%u blocks_min=%u "
                "curves_min=%llu blocks_wave=%u curves_wave=%llu",
                t.bits, t.tpb, t.tpi, t.ipb, t.blocks_per_sm, t.blocks_min,
                (unsigned long long)t.curves_min, t.blocks_wave, (unsigned long long)t.curves_wave);
        if (info.fold != 0) {
            fprintf(stdout, " fold_blocks_min=%u fold_curves_min=%llu", t.fold_blocks_min,
                    (unsigned long long)t.fold_curves_min);
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
    return 0;
}

#include "ecm/platform/cuda_loading.h"

namespace ecm_stage1_app {
using namespace ecm_stage1;

struct QueuePaths {
    std::string worktodo, finished, exe_dir, sync1, sync2;
    int worker = 1;
    bool full_sync = false;
};

struct QueueItem : WorktodoStage1Task {
    long long marker = 0;
    std::string auto_stage2_line;
    const P95TransferConfig* p95 = nullptr;
    const Stage2TransferConfig* stage2 = nullptr;
};
// MarkError advances only the original ticket; Error preserves it and stops.
enum class QueueReadStatus { Ready, Empty, Error, MarkError };

static bool finish_record(const QueuePaths& paths) {
    const auto receipt = paths.worktodo + ".stage1_completed_worker" + std::to_string(paths.worker);
    if (!std::filesystem::exists(receipt))
        return true;
    CompletedWorktodo completed;
    std::string error;
    if (!ecm_worktodo_commit_completion(receipt, paths.worktodo, paths.worker, completed, error)) {
        ecm_ts_fprintf(stderr, "FATAL: %s\n", error.c_str());
        return false;
    }
    if (!completed.checkpoint.empty())
        ecm_stage1_remove_checkpoints(completed.checkpoint);
    std::error_code ec;
    std::filesystem::remove(receipt, ec);
    if (ec) {
        ecm_ts_fprintf(stderr, "FATAL: cannot acknowledge completion: %s\n", ec.message().c_str());
        return false;
    }
    return true;
}

static bool run_one(const QueuePaths& paths, const QueueItem& item, int* processed) {
    const auto N = item.input.modulus;
    const double B1 = item.input.b1, B2 = item.input.b2;
    const auto curves = item.input.curves;
    const auto &save_name = item.input.save_file, &n_expr = item.input.expression,
               &aid = item.input.options.identity.result_aid, &line = item.ticket;
    const auto& opt = item.input.options;
    auto published_line = item.published_line, stage2_line = item.stage2_line,
         auto_stage2_line = item.auto_stage2_line;
    const auto* p95 = item.p95;
    const auto* stage2 = item.stage2;
    const auto &worktodo_path = paths.worktodo, &finished_path = paths.finished;
    const auto &exe_dir = paths.exe_dir, &sync1 = paths.sync1, &sync2 = paths.sync2;
    const auto worker = paths.worker;
    const auto full_sync = paths.full_sync;
    const auto marker = item.marker;
    auto result = run_stage1_task(item.input.request(), stage1_tasks(opt));

    const bool has_factors = result.has_factors();

    if (result.failure_source() == FailureSource::Preparation ||
        result.failure_source() == FailureSource::Publication ||
        result.failure_source() == FailureSource::Persistence ||
        result.failure_source() == FailureSource::Device) {
        ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged.\n",
                       result.failure_source() == FailureSource::Device
                           ? "Stage1 device execution failed"
                       : result.failure_source() == FailureSource::Persistence
                           ? "Stage1 checkpoint commit failed"
                       : result.failure_source() == FailureSource::Publication
                           ? "Stage1 save/result publication failed"
                           : "backend prepare failed");
        return false;
    }

    bool found_factor = false;
    /* D3 (docs/usage/GUI.md): the hit line carries everything a consumer needs to
       build a results record, because each back-end prints its own, mutually
       incompatible hit line ("curve i sigma=M -> factor found" from the CPU paths,
       "GPU: factor ... with curve c (sigma p:s)" from the GPU ones). Fields: the
       factor, the curve index inside this batch, the EXACT per-curve sigma
       (result.sigma(), when the path knows it), the parametrization, which engine ran,
       and the save-file name (generated for GPU ECM=/ECM2= tasks). */
    const char* curve_token = ecm_stage1::curve_model_name(opt.execution.selection.curve);
    const char* target_token = ecm_stage1::execution_target_name(opt.execution.selection.target);
    if (has_factors) {
        for (uint32_t i = 0; i < result.curves(); ++i) {
            if (result.found(i) != ECM_NO_FACTOR_FOUND) {
                char* fs = mpz_get_str(nullptr, 10, result.factor(i));
                std::cout << "factor[" << i << "]=" << (fs ? fs : "?") << " curve=" << i;
                if (result.has_sigmas()) {
                    std::cout << " sigma=" << result.sigma(i);
                }
                if (opt.execution.selection.curve == ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param="
                              << (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                                      ? opt.execution.selection.gpu_param
                                      : 0);
                std::cout << " curve_model=" << curve_token << " execution_target=" << target_token
                          << " save=" << save_name << "\n";
                free(fs);
                found_factor = true;
            }
        }
    }
    if (found_factor) {
        ecm_ts_fprintf(stdout, "FACTOR FOUND aid=%s task=%s\n", aid.empty() ? "N/A" : aid.c_str(),
                       line.c_str());
    }

    if (result.status() == TaskStatus::Stopped) {
        ecm_ts_fprintf(stdout, "STOP: Stage1 interrupted; worktodo unchanged: %s\n", line.c_str());
        return false;
    }

    if (result.status() == TaskStatus::Failed) {
        ecm_ts_fprintf(stderr, "ERROR: stage1 failed for task: %s\n", line.c_str());
        // Marking the line is what stops the queue from re-running it: if that write
        // fails, abort instead of spinning on the same task forever.
        if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot mark the failed line in %s; aborting\n",
                           worktodo_path.c_str());
            return false;
        }
        return true;
    }

    try {
        const auto& range = result.save_range();
        uint64_t total = 0;
        (void)ecm_save::verified_snapshot(
            range, &total); // Force a fresh disk check after save publication.
        if (range.records.size() > UINT32_MAX)
            throw std::runtime_error("published curve count exceeds uint32");
        const auto count = static_cast<uint32_t>(range.records.size());
        std::string error, rewritten;
        if (!count) {
            stage2 = nullptr;
            p95 = nullptr;
            stage2_line.clear();
            auto_stage2_line.clear();
            ecm_ts_fprintf(stdout, "Stage1 completed without saved points; no Stage2 handoff\n");
        }
        if (published_line.empty()) {
            published_line = stage2_line;
            published_line.replace(0, 10, "ECMSTAGE1=");
        }
        if (published_line.compare(0, 10, "ECMSTAGE1=") == 0) {
            if (!ecm_rewrite_stage2_range(published_line, range.skip, count, rewritten, error))
                throw std::runtime_error(error);
            published_line = rewritten;
        }
        for (auto* handoff : {&stage2_line, &auto_stage2_line})
            if (!handoff->empty()) {
                if (!ecm_rewrite_stage2_range(*handoff, range.skip, count, rewritten, error))
                    throw std::runtime_error(error);
                *handoff = rewritten;
            }
        if (p95 && !p95->worktodo_path.empty() && !found_factor) {
            if (stage2_line.empty())
                throw std::runtime_error("P95 handoff lacks a normalized ECMSTAGE2 assignment");
            if (!ecm_sync::park_p95(*p95, stage2_line, range, error))
                throw std::runtime_error(error);
        }
        ecm_ts_fprintf(stdout, "Stage1 save range: skip=%llu curves=%u total=%llu\n",
                       static_cast<unsigned long long>(range.skip), count,
                       static_cast<unsigned long long>(total));
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "FATAL: cannot finalize Stage1 save: %s; worktodo retained\n",
                       e.what());
        return false;
    }
    const auto& completed_line = published_line.empty() ? line : published_line;
    int stage2_receiver = stage2 ? stage2->worker : worker;
    if (stage2 != nullptr && !found_factor) {
        std::string transfer_error;
        if (!ecm_stage2_transfer_park(*stage2, stage2_line, transfer_error, &stage2_receiver)) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", transfer_error.c_str());
            return false;
        }
    }
    const auto receipt = paths.worktodo + ".stage1_completed_worker" + std::to_string(worker);
    std::string completion_error;
    if (!ecm_worktodo_store_completion(receipt,
                                       {line, completed_line,
                                        (!found_factor ? auto_stage2_line : std::string{}),
                                        finished_path, result.completed_checkpoint()},
                                       completion_error)) {
        ecm_ts_fprintf(stderr, "FATAL: %s; worktodo retained\n", completion_error.c_str());
        return false;
    }
    if (!finish_record(paths))
        return false;

    ecm_sync::backup_saves(exe_dir, sync1, sync2, full_sync, marker);
    if (!opt.persistence.tmp_dir.empty() && opt.persistence.tmp_dir != exe_dir)
        ecm_sync::backup_saves(opt.persistence.tmp_dir, sync1, sync2, full_sync, marker);

    if (stage2 != nullptr && !found_factor) {
        size_t remaining = 0;
        std::string transfer_error;
        const bool delivered =
            ecm_stage2_transfer_flush(stage2->pending, remaining, transfer_error);
        if (delivered)
            ecm_ts_fprintf(stdout, "stage2_transfer: ok worker=%d file=\"%s\"\n", stage2_receiver,
                           stage2->worktodo.c_str());
        else
            ecm_ts_fprintf(
                stdout, "stage2_transfer: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                stage2_receiver, remaining, stage2->worktodo.c_str(),
                p95_transfer_quote(transfer_error).c_str());
    }

    if (p95 && !p95->worktodo_path.empty() && !found_factor) {
        std::string error;
        if (!ecm_sync::flush_p95(*p95, error))
            ecm_ts_fprintf(stderr,
                           "p95_add: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                           worker, ecm_sync::pending_p95_count(*p95),
                           p95_transfer_quote(p95_transfer_add_path(p95->worktodo_path)).c_str(),
                           p95_transfer_quote(error).c_str());
    }

    if (processed) {
        (*processed)++;
    }
    return true;
}

static long long current_epoch_seconds() {
    return static_cast<long long>(std::time(nullptr));
}

static std::string p95_dir_of(const std::string& path) {
    const size_t s = path.find_last_of("\\/");
    if (s == std::string::npos)
        return std::string(".");
    if (s == 0)
        return path.substr(0, 1);
    return path.substr(0, s);
}

static std::string p95_join_path(const std::string& dir, const std::string& name) {
    if (dir.empty())
        return name;
    const char last = dir[dir.size() - 1];
    return (last == '\\' || last == '/') ? (dir + name) : (dir + "\\" + name);
}

static int run_worktodo(const CliInput& input) {
    const auto& ini_path = input.ini_path;
    const int worker = input.worker_index;
    const std::string raw_exe_dir = get_exe_dir_local();
    const std::string exe_dir = raw_exe_dir.empty() ? "." : raw_exe_dir;
    /* Default home of the stage-1 exponent cache (overridable by --exp-cache / exp_cache). */
    ecm_exp_cache_set_dir(exe_dir);

    // Resolve ini path (default: exe_dir/ecm.toml).
    std::string ini = ini_path;
    if (ini.empty()) {
        ini = exe_dir + "/ecm.toml";
    } else {
        ini = resolve_rel_local(exe_dir, ini);
    }

    // Load config; on first run (missing ini), write a default template.
    // `worker` selects the TOML worker projection over shared/stage defaults.
    EcmQueueConfig cfg;
    ParsedConfig parsed_config;
    parsed_config.path = ini;
    if (!read_config(ini, parsed_config)) {
        if (access(ini.c_str(), 0) == 0) {
            ecm_ts_fprintf(stderr, "FATAL: invalid or unreadable config %s\n", ini.c_str());
            return 1;
        }
        if (!ecm_queue_config_write_default(ini)) {
            ecm_ts_fprintf(stderr, "FATAL: cannot create default config %s\n", ini.c_str());
            return 1;
        }
        ecm_ts_fprintf(stdout, "Created default config: %s\n", ini.c_str());
    }
    std::string normalization_error;
    if (!normalize_config(parsed_config, input, cfg, normalization_error)) {
        ecm_ts_fprintf(stderr, "%s\n", normalization_error.c_str());
        return 1;
    }
    // Several workers appending to one screen.log interleave their lines, so workers
    // 2..N get their own default log unless the ini set log_file explicitly.
    if (worker > 1 && !cfg.log_file_explicit &&
        cfg.log_file == ecm_config::defaults::stage1_log_file) {
        cfg.log_file = ecm_config::worker_file(ecm_config::defaults::stage1_log_file, worker);
        ecm_ts_fprintf(stdout,
                       "note: log_file not set in %s; worker %d logs to %s instead of "
                       "screen.log (per-worker default)\n",
                       ini.c_str(), worker, cfg.log_file.c_str());
    }

    opencl_ecm_set_work_dir(exe_dir.c_str());

    ecm_log_set_progress_color(cfg.progress_color.c_str());
    /* Progress cadence for the log FILE only (ini progress_log_seconds): a task running for
       hours would otherwise write a progress line into screen.log every 200 ms. The pipe
       keeps its cadence, and the 100% line always lands in the file. */
    ecm_log_set_progress_log_seconds(cfg.progress_log_seconds);

    // Open the mirror log (screen.log by default).
    EcmLogMirror log;
    if (!cfg.log_file.empty()) {
        const std::string logpath = resolve_rel_local(exe_dir, cfg.log_file);
        if (!log.open(logpath)) {
            ecm_ts_fprintf(stderr, "FATAL: cannot open log file %s\n", logpath.c_str());
            return 1;
        }
    }

    const std::string ini_dir = std::filesystem::path(ini).parent_path().string();
    const std::string worktodo_path = resolve_rel_local(ini_dir, cfg.worktodo);
    ecm_queue::OwnerLease owner;
    try {
        owner.acquire(worktodo_path, worker);
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    const std::string finished_path = resolve_rel_local(exe_dir, cfg.finished);
    const std::string sync1 = resolve_rel_local(exe_dir, cfg.save_sync_dir_1);
    const std::string sync2 = resolve_rel_local(exe_dir, cfg.save_sync_dir_2);

    const std::string stage2_pending =
        worktodo_path + ".stage2_pending_worker" + std::to_string(worker);
    const auto same_path = [](const std::string& a, const std::string& b) {
        std::error_code ec;
        if (std::filesystem::equivalent(a, b, ec) && !ec)
            return true;
#ifdef _WIN32
        return ecm_stage1::lower_ascii(a) == ecm_stage1::lower_ascii(b);
#else
        return a == b;
#endif
    };
    const auto auto_worktodo = resolve_rel_local(
        ini_dir, cfg.auto_stage2_worktodo.empty() ? cfg.worktodo : cfg.auto_stage2_worktodo);
    const bool auto_stage2_routed =
        cfg.auto_stage2 && (!ecm_workers::trim(cfg.auto_stage2_workers).empty() ||
                            !same_path(auto_worktodo, worktodo_path));
    Stage2TransferConfig stage2cfg;
    if (auto_stage2_routed || (!cfg.auto_stage2 && cfg.edwards_stage2_transfer)) {
        const int receiver = cfg.auto_stage2                  ? worker
                             : cfg.edwards_stage2_worker == 0 ? worker
                                                              : cfg.edwards_stage2_worker;
        if (receiver < 1 || receiver > 1000000) {
            ecm_ts_fprintf(stderr, "FATAL: Stage2 receiver worker must be 1..1000000\n");
            return 1;
        }
        std::vector<ecm_config::TomlLine> lines;
        if (!ecm_config::read_toml(ini, lines, ecm_config::TomlReadMode::Computation)) {
            ecm_ts_fprintf(stderr, "FATAL: cannot read Stage2 receiver configuration\n");
            return 1;
        }
        const auto receiver_cfg =
            ecm_config::read_stage2(ecm_config::cli_entries(lines, receiver, true));
        stage2cfg.worktodo =
            cfg.auto_stage2 ? auto_worktodo : resolve_rel_local(ini_dir, receiver_cfg.worktodo);
        stage2cfg.finished = resolve_rel_local(ini_dir, receiver_cfg.finished);
        stage2cfg.pending = stage2_pending;
        stage2cfg.worker = receiver;
        if (cfg.auto_stage2) {
            const auto spec = ecm_workers::trim(cfg.auto_stage2_workers);
            stage2cfg.auto_workers = ecm_stage1::lower_ascii(spec) == "auto";
            int ignored = 0;
            if (!stage2cfg.auto_workers && !spec.empty() &&
                (!ecm_workers::parse_list(spec, 1000000, stage2cfg.workers, ignored) || ignored ||
                 stage2cfg.workers.empty())) {
                ecm_ts_fprintf(stderr, "FATAL: auto_stage2_workers must be a worker list/range in "
                                       "1..1000000 or auto\n");
                return 1;
            }
            stage2cfg.configured_workers =
                ecm_config::read_gui(ecm_config::section_entries(lines, "gui")).NumWorkers;
            const auto global = ecm_config::read_stage2(ecm_config::cli_entries(lines, 0, true));
            stage2cfg.finished = resolve_rel_local(ini_dir, global.finished);
            for (const auto& entry : lines)
                if (entry.worker_scope > 0 &&
                    !stage2cfg.finished_by_worker.count(entry.worker_scope)) {
                    const auto selected = ecm_config::read_stage2(
                        ecm_config::cli_entries(lines, entry.worker_scope, true));
                    stage2cfg.finished_by_worker[entry.worker_scope] =
                        resolve_rel_local(ini_dir, selected.finished);
                }
        }
        const auto valid_finished = [&](const std::string& finished) {
            return !finished.empty() && !same_path(stage2cfg.worktodo, finished) &&
                   !same_path(finished, worktodo_path) && !same_path(finished, finished_path) &&
                   !same_path(finished, stage2_pending);
        };
        bool valid = !stage2cfg.worktodo.empty() && !same_path(stage2cfg.worktodo, finished_path) &&
                     !same_path(stage2cfg.worktodo, stage2_pending) &&
                     valid_finished(stage2cfg.finished);
        for (const auto& item : stage2cfg.finished_by_worker)
            valid = valid && valid_finished(item.second);
        if (!valid) {
            ecm_ts_fprintf(
                stderr,
                "FATAL: Stage2 transfer queue/finished paths conflict with Stage1 output\n");
            return 1;
        }
        ecm_ts_fprintf(
            stdout, "stage2_transfer: ready source_worker=%d receiver=%s file=\"%s\"\n", worker,
            cfg.auto_stage2 && !cfg.auto_stage2_workers.empty() ? cfg.auto_stage2_workers.c_str()
                                                                : std::to_string(receiver).c_str(),
            stage2cfg.worktodo.c_str());
    }
    {
        size_t remaining = 0;
        std::string transfer_error;
        if (!ecm_stage2_transfer_flush(stage2_pending, remaining, transfer_error))
            ecm_ts_fprintf(stdout, "stage2_transfer: pending worker=%d lines=%zu error=\"%s\"\n",
                           worker, remaining, p95_transfer_quote(transfer_error).c_str());
    }

    // Initialize handoff only for an eligible actual task. A typed save may override
    // the TOML curve, and unsupported constructions must not touch transfer paths.
    P95TransferConfig p95cfg;
    bool p95_initialized = false, p95_ignore_reported = false;
    auto p95_for_task = [&](const Stage1Options& actual) -> const P95TransferConfig* {
        const bool ignore =
            actual.execution.selection.curve == ecm_stage1::CurveModel::Edwards ||
            (actual.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu &&
             actual.execution.selection.gpu_param == 2);
        if (ignore) {
            if (!cfg.p95_worktodo_path.empty() && !p95_ignore_reported) {
                ecm_ts_fprintf(stdout, "Prime95 transfer path ignored for this curve construction; "
                                       "Stage1 saves remain local\n");
                p95_ignore_reported = true;
            }
            return nullptr;
        }
        if (p95_initialized)
            return &p95cfg;
        if (!cfg.p95_worktodo_path.empty() && cfg.p95_sync_dir.empty())
            throw std::runtime_error(
                "INPUT[config] " + ini + " worker=" + std::to_string(worker) +
                " key=p95_sync_dir value=''; expected save destination directory for supported "
                "Prime95 handoff; configure p95_sync_dir (backup paths are independent)");
        p95_initialized = true;
        p95cfg.worktodo_path = resolve_rel_local(exe_dir, cfg.p95_worktodo_path);
        p95cfg.sync_dir = resolve_rel_local(exe_dir, cfg.p95_sync_dir);
        p95cfg.add_workers = cfg.p95_add_workers;
        p95cfg.pending_path = p95_transfer_pending_path(exe_dir);
        p95cfg.keep_aid = cfg.p95_keep_aid;
        p95cfg.recover_lost = cfg.p95_recover_lost;
        p95cfg.sent_ledger_path = p95_transfer_ledger_path(exe_dir);
        if (!p95cfg.worktodo_path.empty()) {
            const std::string add_path = p95_transfer_add_path(p95cfg.worktodo_path);
            const std::string p95_dir = p95_dir_of(p95cfg.worktodo_path);
            /* Prime95's own logs tell us which assignment keys PrimeNet has rejected: a line
               delivered with such a key is DELETED by Prime95 (commonc.c:6567), so those keys
               must not be attached any more and the work has to be re-delivered without them. */
            p95cfg.prime_log_path = p95_join_path(p95_dir, "prime.log");
            p95cfg.results_txt_path = p95_join_path(p95_dir, "results.txt");
            const size_t parked = p95_transfer_pending_count(p95cfg.pending_path);
            const std::vector<std::string> rejected =
                p95_rejected_aids(p95cfg.prime_log_path, p95cfg.results_txt_path);
            ecm_ts_fprintf(stdout,
                           "p95_add: ready workers=\"%s\" file=\"%s\" pending=%zu keep_aid=%d "
                           "rejected_aids=%zu\n",
                           p95_transfer_quote(cfg.p95_add_workers).c_str(),
                           p95_transfer_quote(add_path).c_str(), parked, cfg.p95_keep_aid ? 1 : 0,
                           rejected.size());
            for (const std::string& aid : rejected) {
                /* One line per rejected key: the GUI shows them in the log pane, and the
                   driver's own recovery (below) uses the same set. */
                ecm_ts_fprintf(stdout, "p95_add: aid_rejected aid=\"%s\"\n", aid.c_str());
            }
            if (parked > 0) {
                /* Lines from a previous run are still waiting: the GUI must show red
                   until a delivery succeeds and clears them. */
                ecm_ts_fprintf(stdout,
                               "p95_add: pending worker=0 lines=%zu file=\"%s\" "
                               "error=\"%zu line(s) from an earlier run are still waiting\"\n",
                               parked, p95_transfer_quote(add_path).c_str(), parked);
            }
            /* Work Prime95 threw away because of a rejected key is re-delivered right away
               (without the key), so a restart is enough to recover it. */
            if (cfg.p95_recover_lost) {
                size_t rejected_count = 0;
                const std::vector<std::string> lost =
                    p95_transfer_recover_lines(p95cfg, &rejected_count);
                if (!lost.empty()) {
                    ecm_ts_fprintf(stdout,
                                   "p95_add: recovered worker=0 lines=%zu file=\"%s\" "
                                   "note=\"%zu line(s) Prime95 dropped (rejected AID) will be "
                                   "re-delivered without the AID\"\n",
                                   lost.size(), p95_transfer_quote(add_path).c_str(), lost.size());
                }
            }
        }
        std::string retry_error;
        if (!ecm_sync::flush_p95(p95cfg, retry_error))
            ecm_ts_fprintf(stderr, "p95_add: pending worker=0 lines=%zu file=\"%s\" error=\"%s\"\n",
                           ecm_sync::pending_p95_count(p95cfg),
                           p95_transfer_quote(p95_transfer_add_path(p95cfg.worktodo_path)).c_str(),
                           p95_transfer_quote(retry_error).c_str());
        return &p95cfg;
    };

    auto opt = normalize_options(cfg, exe_dir, ini_dir, worker);
    opt.persistence.retain_completed_checkpoint = true;
    ecm_exp_cache_set_dir(cfg.exp_cache.empty() ? exe_dir : cfg.exp_cache);
    ecm_runtime_config().device_index = cfg.device;
    ecm_runtime_config().tpi = cfg.tpi;
    ecm_runtime_config().wg_size = cfg.wg_size;
    // Validate semantic combinations per task, after applying any save prefix.
    /* ---- [handoff] ----------------------------------------------------------- */
    if (!cfg.p95_dir.empty()) {
        ecm_ts_fprintf(stdout,
                       "note: p95_dir=%s is informational here -- ecm.exe only writes local saves\n"
                       "      to %s; the stage-2 handoff program ecm_p95feeder reads its own\n"
                       "      feeder.ini (same key name)\n",
                       cfg.p95_dir.c_str(), opt.persistence.tmp_dir.c_str());
    }

    ecm_ts_fprintf(stdout, "%s %s (Stage1)\n", ecm_program::name, ecm_program::version);
    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worker : %d (TOML [workers.%d.*]; worktodo [Worker #%d])\n", worker,
                   worker, worker);
    ecm_ts_fprintf(stdout, "worktodo : %s\n", worktodo_path.c_str());
    ecm_ts_fprintf(stdout, "finished : %s\n", finished_path.c_str());
    ecm_ts_fprintf(stdout, "log_file : %s\n", cfg.log_file.c_str());
    ecm_ts_fprintf(stdout, "saves : %s\n", opt.persistence.tmp_dir.c_str());
    ecm_ts_fprintf(stdout, "curve model : %s\n",
                   ecm_stage1::curve_model_name(opt.execution.selection.curve));
    ecm_ts_fprintf(stdout, "execution target : %s\n",
                   ecm_stage1::execution_target_name(opt.execution.selection.target));
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu) {
        /* The GPU implementation is chosen at LINK time (ecm.exe = OpenCL glue,
           ecm_cuda.exe = CUDA backend), so ask the backend instead of hardcoding
           "OpenCL" -- the CUDA build used to print "gpu (OpenCL)" here. */
        if (opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
            ecm_ts_fprintf(stdout, "gpu backend : %s, Edwards Atkin-Morain, device %d\n",
                           ecm_backend_name(), opt.execution.device_index);
        else
            ecm_ts_fprintf(stdout, "gpu backend : %s, param%d, device %d\n", ecm_backend_name(),
                           opt.execution.selection.gpu_param, opt.execution.device_index);
    }

    // Startup full sync (matches the old work_manager.ps1 behaviour).
    ecm_sync::backup_saves(exe_dir, sync1, sync2, /*full=*/true, 0);
    if (!opt.persistence.tmp_dir.empty() && opt.persistence.tmp_dir != exe_dir)
        ecm_sync::backup_saves(opt.persistence.tmp_dir, sync1, sync2, /*full=*/true, 0);

    if (!finish_record(
            {worktodo_path, finished_path, exe_dir, sync1, sync2, worker, cfg.sync_mode == "full"}))
        return 1;
    try {
        std::vector<std::string> rows;
        std::string error;
        if (!ecm_worktodo_stage1_lines(worktodo_path, worker, rows, error))
            throw std::runtime_error(error);
        std::map<std::string, uint64_t> counts;
        for (const auto& row : rows) {
            EcmStage2Task task;
            if (!ecm_parse_stage1_line(row, task, error) || task.skip_curves != 0)
                continue;
            auto actual = opt;
            if (!stage1_save_construction(task.save_name, actual, error))
                continue;
            const auto typed = stage1_prefixed_save(task.save_name, stage1_save_prefix(actual));
            auto file = resolve_rel_local(exe_dir, typed);
            const auto legacy = resolve_rel_local(exe_dir, task.save_name);
            if ((!std::filesystem::exists(file) || std::filesystem::file_size(file) == 0) &&
                std::filesystem::exists(legacy))
                file = legacy;
            auto count = counts.find(file);
            if (count == counts.end())
                count = counts.emplace(file, ecm_save::record_count(file)).first;
            if (!count->second)
                continue;
            std::string repaired;
            if (!ecm_rewrite_stage2_range(row, count->second, task.curves_to_run, repaired,
                                          error) ||
                !ecm_worktodo_advance(worktodo_path, worker, row, WorktodoAction::Replace,
                                      repaired))
                throw std::runtime_error("cannot repair current worker Stage1 skip: " + error);
            ecm_ts_fprintf(stdout, "worktodo: repaired skip=%llu save=\"%s\"\n",
                           static_cast<unsigned long long>(count->second), file.c_str());
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }

    bool saw_task = false;
    const bool full_sync = (cfg.sync_mode == "full");
    const QueuePaths paths{worktodo_path, finished_path, exe_dir, sync1, sync2, worker, full_sync};
    const auto read_task = [&](QueueItem& item) -> QueueReadStatus {
        std::string line;
        std::string read_error;
        size_t source_line = 0;
        const WorktodoReadResult read = ecm_worktodo_read_next_line(
            worktodo_path, worker, {cfg.worktodo_priority, cfg.worktodo_skip},
            WorktodoStage::Stage1, line, read_error, &source_line);
        if (read == WorktodoReadResult::Error) {
            ecm_ts_fprintf(stderr, "ERROR: %s\n", read_error.c_str());
            return QueueReadStatus::Error;
        }
        if (read == WorktodoReadResult::Empty) {
            if (!saw_task)
                p95_for_task(opt); // Preserve idle-start recovery for an eligible TOML.
            return QueueReadStatus::Empty;
        }
        saw_task = true;
        item.ticket = line;
        item.origin =
            worktodo_path + ":" + std::to_string(source_line) + " worker=" + std::to_string(worker);

        ecm_ts_fprintf(stdout, "START: %s\n", line.c_str());
        const auto input_error = [&](const std::string& error) {
            ecm_ts_fprintf(stderr, "INPUT[worktodo] %s:%zu worker=%d %s; value=\"%s\"\n",
                           worktodo_path.c_str(), source_line, worker, error.c_str(), line.c_str());
        };

        std::string error;
        if (!normalize_worktodo(line, opt, exe_dir, item, error)) {
            input_error(error);
            return QueueReadStatus::MarkError;
        }
        // Resolve optional delivery only after the actual typed construction.
        item.p95 = p95_for_task(item.input.options);
        item.marker = current_epoch_seconds();
        if (item.save_assignment) {
            const auto& legacy = item.input.options.persistence.legacy_save_source;
            const auto& save = item.input.save_file;
            const auto source =
                ((!std::filesystem::exists(save) || std::filesystem::file_size(save) == 0) &&
                 !legacy.empty() && std::filesystem::exists(legacy))
                    ? legacy
                    : save;
            const auto baseline = ecm_save::record_count(source);
            std::string repaired;
            if (!ecm_rewrite_stage2_range(line, baseline, item.input.curves, repaired, error))
                throw std::runtime_error(error);
            if (repaired != line) {
                if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::Replace,
                                          repaired))
                    throw std::runtime_error("task row changed before claim");
                item.ticket = repaired;
            }
            if (!ecm_rewrite_stage2_range(item.published_line, baseline, item.input.curves,
                                          repaired, error))
                throw std::runtime_error(error);
            item.published_line = repaired;
        }
        item.stage2 = auto_stage2_routed || (!cfg.auto_stage2 && cfg.edwards_stage2_transfer &&
                                             item.input.options.execution.selection.curve ==
                                                 ecm_stage1::CurveModel::Edwards)
                          ? &stage2cfg
                          : nullptr;
        if (cfg.auto_stage2 && !auto_stage2_routed)
            item.auto_stage2_line = item.stage2_line;
        return QueueReadStatus::Ready;
    };
    int processed = 0;
    for (;;) {
        if (ecm_stage1_stop_requested())
            return 1;
        QueueItem item;
        QueueReadStatus read;
        try {
            read = read_task(item);
        } catch (const std::exception& e) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo retained\n", e.what());
            return 1;
        }
        if (read == QueueReadStatus::Error)
            return 1;
        if (read == QueueReadStatus::Empty)
            break;
        if (read == QueueReadStatus::MarkError) {
            if (!ecm_worktodo_advance(worktodo_path, worker, item.ticket,
                                      WorktodoAction::MarkError))
                return 1;
            continue;
        }
        if (!run_one(paths, item, &processed))
            return 1;
    }
    ecm_ts_fprintf(stdout, "===== queue done, %d task(s) processed =====\n", processed);
    return 0;
}
} // namespace ecm_stage1_app

int main(int argc, char** argv) {
#ifdef ECM_HAS_CUDA_STAGE1
    ecm_cuda_default_module_loading();
#endif
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui-queue-protocol") {
            std::puts("ECM_QUEUE_OWNER=1");
            return 0;
        }
    if (ecm_stage1_app::wants_usage(argc, argv)) {
        ecm_stage1_app::print_usage(argv[0]);
        return 0;
    }

    // Timestamped-stream install is deferred until after argv parsing / config fill,
    // so --no-log-timestamp takes effect.
    ecm_stage1_app::CliInput input;
    if (ecm_stage1_app::parse_cli(argc, argv, input))
        return 1;
    if (input.version) {
        std::cout << ecm_program::name << ' ' << ecm_program::version << std::endl;
        return 0;
    }
    try {
        ecm_stage1_app::normalize_cli(input);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    if (input.pos.empty() && !input.show_gpu_info && !input.show_kernels) {
        std::string error;
        if (!ecm_stage1_app::validate_cli_mode(input, error)) {
            std::cerr << error << "\n";
            return 1;
        }
    }
    EcmQueueConfig single_config;
    const auto single_ini = std::filesystem::path(
        input.ini_path.empty() ? get_exe_dir_local() + "/ecm.toml"
                               : resolve_rel_local(get_exe_dir_local(), input.ini_path));
    if (!input.pos.empty()) {
        ParsedConfig parsed;
        parsed.path = single_ini.string();
        if (!read_config(single_ini.string(), parsed) &&
            (!input.ini_path.empty() || std::filesystem::exists(single_ini)))
            return 1;
        std::string error;
        if (!ecm_stage1_app::normalize_config(parsed, input, single_config, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        ecm_stage1_app::normalize_cli(input, single_config);
    }
    auto& verbose = input.verbose;
    auto& selection = input.selection;
    auto& stage1_threads = input.stage1_threads;
    auto& naf_w = input.naf_w;
    auto& affinity_spec = input.affinity_spec;
    auto& gpucurves = input.gpucurves;
    auto& ckpt_seconds = input.ckpt_seconds;
    auto& ckpt_set = input.ckpt_set;
    auto& sigma_fixed = input.sigma_fixed;
    auto& fixed_sigma64 = input.fixed_sigma64;
    auto& gpu_device_index = input.gpu_device_index;
    auto& device_set = input.device_set;
    auto& gpu_param_cli = input.gpu_param_cli;
    auto& gpu_param_set = input.gpu_param_set;
    auto& sigma_param_claim = input.sigma_param_claim;
    auto& sigma_param_set = input.sigma_param_set;
    auto& print_group_order = input.print_group_order;
    auto& go_settings = input.go_settings;
    auto& savefilename = input.savefilename;
    auto& saveappend = input.saveappend;
    auto& gp_bin_path = input.gp_bin_path;
    auto& gpu_mul_path = input.gpu_mul_path;
    auto& gpu_sqr_path = input.gpu_sqr_path;
    auto& gpu_add_path = input.gpu_add_path;
    auto& gpu_sub_path = input.gpu_sub_path;
    auto& gpu_special_mult_path = input.gpu_special_mult_path;
    auto& show_kernels = input.show_kernels;
    auto& show_gpu_info = input.show_gpu_info;
    auto& curve_set = input.curve_set;
    auto& gpu_info_bits = input.gpu_info_bits;
    auto& ini_path = input.ini_path;
    auto& worker_index = input.worker_index;
    auto& tmp_dir = input.tmp_dir;
    auto& p95_dir_ignored = input.p95_dir_ignored;
    auto& stop_handle = input.stop_handle;
    auto& pos = input.pos;
    auto& exp_cache = input.exp_cache;
    auto& version = input.version;
    // Argv parsed: fold driver-level args into the runtime config (single source),
    // then install timestamped streams.
    ecm_runtime_config() = input.runtime;
    ecm_runtime_config().device_index = gpu_device_index;
    if (!input.exp_cache.empty())
        ecm_exp_cache_set_dir(input.exp_cache);
    if (!gp_bin_path.empty()) {
        ecm_runtime_config().gp_bin = gp_bin_path;
    }
    ecm_install_timestamped_iostreams();
    ecm_enable_console_ansi();

    if (show_kernels) {
        ecm_backend_print_kernels(stdout);
        return 0;
    }

    if (show_gpu_info) {
        /* Effective settings resolve CLI > ini [Worker #N] > built-in, the same order the
           queue path uses, so `-ini ecm.toml --gpu-info` reports what the workers will
           actually run. The ini is read, never created (a missing one leaves the CLI
           defaults alone and creates nothing). */
        const std::string raw_exe_dir = get_exe_dir_local();
        const std::string exe_dir = raw_exe_dir.empty() ? "." : raw_exe_dir;
        std::string ini =
            ini_path.empty() ? (exe_dir + "/ecm.toml") : resolve_rel_local(exe_dir, ini_path);
        int eff_device = gpu_device_index;
        int eff_param = gpu_param_set ? gpu_param_cli : ecm_config::defaults::stage1_gpu_param;
        ecm_stage1::CurveModel eff_curve = selection.curve;
        const char* ini_used = nullptr;
        EcmQueueConfig info_cfg;
        if (ecm_queue_config_load(ini, worker_index, info_cfg)) {
            ini_used = ini.c_str();
            if (!device_set)
                eff_device = info_cfg.device;
            if (!gpu_param_set)
                eff_param = info_cfg.gpu_param;
            if (!curve_set && !ecm_stage1::parse_curve_model(info_cfg.curve_model, eff_curve)) {
                std::cerr << "Invalid curve in config: " << info_cfg.curve_model << std::endl;
                return 1;
            }
        } else if (access(ini.c_str(), 0) == 0) {
            std::cerr << "Invalid or unreadable config: " << ini << std::endl;
            return 1;
        }
        return print_gpu_info(ini_used, worker_index, eff_device, eff_param,
                              eff_curve == ecm_stage1::CurveModel::Edwards, gpu_info_bits);
    }

    if (stop_handle && !ecm_stage1_set_stop_handle(stop_handle)) {
        std::cerr << "Invalid inherited Stage1 stop handle" << std::endl;
        return 1;
    }
    ecm_stage1_install_stop_handler();

    if (pos.empty()) {
        // No positional B1/B2 → queue-manager mode (reads ecm.toml + worktodo).
        return ecm_stage1_app::run_worktodo(input);
    }

    ecm_ts_fprintf(stdout, "%s %s (Stage1)\n", ecm_program::name, ecm_program::version);

    // Diagnosis follows durable task publication; GP availability is not admission.
    go_settings.gp = resolve_gp_path(get_gp_executable(gp_bin_path));
    ecm_diagnostics::Session go_session(go_settings);

    std::cout << "ecm driver starting" << std::endl;
    std::cout << "  curve_model: " << ecm_stage1::curve_model_name(selection.curve)
              << ", execution_target: " << ecm_stage1::execution_target_name(selection.target)
              << ", gpucurves=" << gpucurves << ", ckpt="
              << (ckpt_set ? ckpt_seconds : ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS / 1000.0) << "s"
              << ", device=" << gpu_device_index
              << ", group_order=" << (print_group_order ? "on" : "off");
    if (!gpu_mul_path.empty()) {
        std::cout << ", mul=" << gpu_mul_path;
    }
    if (!gpu_sqr_path.empty()) {
        std::cout << ", sqr=" << gpu_sqr_path;
    }
    if (!gpu_add_path.empty()) {
        std::cout << ", add=" << gpu_add_path;
    }
    if (!gpu_sub_path.empty()) {
        std::cout << ", sub=" << gpu_sub_path;
    }
    if (!gpu_special_mult_path.empty()) {
        std::cout << ", special_mult=" << gpu_special_mult_path;
    }
    if (!gp_bin_path.empty()) {
        std::cout << ", gp=" << gp_bin_path;
    }
    if (ecm_runtime_config().gpu_local) {
        std::cout << ", local";
    }
    if (ecm_runtime_config().wg_size > 0) {
        std::cout << ", wg=" << ecm_runtime_config().wg_size;
    }
    std::cout << std::endl;
    // if(!pos.empty()){
    //     std::cout << "  B1=" << pos[0];
    //     if(pos.size() >= 2){
    //         std::cout << ", B2=" << pos[1];
    //     }
    //     std::cout << std::endl;
    // }

    // read N from stdin
    std::string nline;
    {
        std::ostringstream oss;
        std::string line;
        while (std::getline(std::cin, line)) {
            oss << line;
        }
        nline = oss.str();
        trim(nline);
    }
    if (nline.empty()) {
        std::cerr << "No input number on stdin" << std::endl;
        return 1;
    }

    ecm_stage1_app::NormalizedStage1Task task;
    std::string input_error;
    if (!ecm_stage1_app::normalize_direct(input, single_config, nline, get_exe_dir_local(),
                                          single_ini.parent_path().string(), task, input_error)) {
        std::cerr << input_error << std::endl;
        return 1;
    }
    const auto& opt = task.options;
    const auto B1 = task.b1;
    if (ecm_exp_cache_get_dir().empty())
        ecm_exp_cache_set_dir(get_exe_dir_local().empty() ? "." : get_exe_dir_local());
    if (!p95_dir_ignored.empty())
        std::cout
            << "Prime95 transfer path ignored: --p95-dir is obsolete; use TOML p95_worktodo_path."
            << std::endl;
    auto result = run_stage1_task(task.request(), stage1_tasks(opt));
    const int rc1 = result.task_return();

    // Failed/stopped tasks may have no result arrays or only a partial owned
    // payload. Inspect factors only after successful execution; the owner
    // releases every allocation on all exits, including group-order errors.
    const bool cli_has_factors =
        (result.legacy_return() != ECM_ERROR && result.has_factors() && result.has_found());

    // Numeric Montgomery parameter used by the group-order helper. CPU Suyama
    // uses PARAM0; GPU Montgomery uses its selected family. Edwards hit metadata
    // identifies the curve model and carries no Montgomery PARAM.
    const int go_param = opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                             ? opt.execution.selection.gpu_param
                             : 0;

    if (cli_has_factors) {
        /* Same D3 fields as the queue path (docs/usage/GUI.md): a consumer must not
           have to guess which of the three back-end hit lines it is looking at. */
        const char* curve_token = ecm_stage1::curve_model_name(opt.execution.selection.curve);
        const char* target_token =
            ecm_stage1::execution_target_name(opt.execution.selection.target);
        for (uint32_t i = 0; i < result.curves(); i++) {
            if (result.found(i) != ECM_NO_FACTOR_FOUND) {
                char* s = mpz_get_str(NULL, 10, result.factor(i));
                std::cout << "factor[" << i << "]=" << (s ? s : "?") << " curve=" << i;
                if (result.has_sigmas()) {
                    std::cout << " sigma=" << result.sigma(i);
                }
                if (opt.execution.selection.curve == ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << go_param;
                std::cout << " curve_model=" << curve_token << " execution_target=" << target_token
                          << " save=" << savefilename << "\n";
                free(s);
                if (print_group_order)
                    print_group_order_factor(
                        result, i, go_param, go_session, B1,
                        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards ? 48
                        : opt.execution.selection.scalar_construction ==
                                ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12
                            ? 12
                            : 1,
                        opt.execution.selection.curve == ecm_stage1::CurveModel::Montgomery
                            ? ecm_group_order::Family::Montgomery
                            : ecm_group_order::Family::Edwards);
            }
        }
    }
    /* 让调用方 (脚本/Prime95 前端) 能看到失败, 而不是把 0 当成"跑完了没因子"。 */
    return (rc1 == ECM_ERROR || result.legacy_return() == ECM_ERROR) ? 1 : 0;
}
