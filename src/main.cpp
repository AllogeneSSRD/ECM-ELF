#include "ecm/platform/gpu_worker.h"
#include "ecm/group_order/backends.h"
#include "ecm/io/save.h"
#include "ecm/platform/log_file.h"
#include "io/cli.h"
#include "io/files.h"
#include "io/normalize.h"
#include "sync/backup.h"
#include "sync/p95.h"
using ecm_stage1_app::get_exe_dir_local;
using ecm_stage1_app::normalize_path_local;
using ecm_stage1_app::resolve_rel_local;
using ecm_stage1_app::stage1_result_settings;
using ecm_stage1_app::stage1_save_construction;
using ecm_stage1_app::stage1_selection_error;
using ecm_stage1_app::stage1_task_error;
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#define access _access
#else
#include <unistd.h>
#endif

#include <gmp.h>

#include "ecm/application/configuration/runtime.h"
#include "ecm/stage1/selection.h"
#include "ecm/stage1/progress.h"
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "ecm/stage1/gpu_backend.h" /* GPU backend seam (OpenCL or CUDA glue) */
#include "ecm/stage1/execution.h"
#include "ecm/stage1/cpu_runtime.h"
#include "ecm/stage1/sigma.h"
#include "ecm/stage2/geometry.h"
#include "io/config.h"
#include "io/stage1_checkpoint.h"
#include "io/worktodo.h"
#include "sync/p95.h" /* finished task -> Prime95 worktodo.add (D4/§13) */
#include "sync/stage2.h"
#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
#if ECM_HAS_CUDA_STAGE1
#include "ecm/stage1/montgomery/cuda/ntt/tune.h"
#endif
// Application assembly still selects the CPU default NAF window.
#include "ecm/stage1/edwards/cpu/ecm_edwards_cpu.h"

#include "ecm/stage1/execution.h"
using ecm_save::prefixed_name;
using ecm_stage1::save_naming;

using ecm_stage1::FailureSource;
using ecm_stage1::run_stage1_task;
using ecm_stage1::Stage1Options;
using ecm_stage1::TaskStatus;

#include "io/files.h"
#include "io/stage1_checkpoint.h"

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
            edwards_get_naf_w, ecm_stage1::defer_gpu_checkpoint_validation,
            ecm_cpu_stage1::freeze_cpu_implementation_policy};
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

static void print_parts(const std::vector<ecm_group_order::PrimePower>&parts,double bound=-1) {
    std::cout<<"[ ";bool first=true;
    for(const auto &part:parts){
        if(bound>=0 && mpz_cmp_d(part.prime.v,bound)>0)continue;
        if(!first)std::cout<<", ";first=false;
        std::cout<<'<'<<part.prime.str()<<", "<<part.exponent<<'>';
    }
    std::cout<<" ]";
}
static void print_group_order_factor(const ecm_stage1::TaskResult &result,uint32_t i,int param,
                              ecm_diagnostics::Session &session,double B1,uint32_t multiplier,ecm_group_order::Family family) {
    try {
        ecm_group_order::CurveRequest request;
        mpz_set(request.prime.v,result.factor(i));request.family=family;
        request.parameter=family==ecm_group_order::Family::Montgomery?param:-1;
        request.sigma=result.has_sigmas()?result.sigma(i):uint64_t(result.first_sigma())+i;
        auto diagnosis=session.group(request);const auto &value=diagnosis.value;
        std::cout<<"  go_status["<<i<<"]: backend="<<diagnosis.backend<<" status="<<ecm_group_order::name(value.status)
            <<" all_proven="<<(value.all_proven?1:0)<<" reason="<<ecm_diagnostics::quote(value.reason)
            <<" pari_failure="<<ecm_diagnostics::quote(diagnosis.pari_failure)<<" gp_failure="<<ecm_diagnostics::quote(diagnosis.gp_failure)<<'\n';
        if(value.group_order_computed){
            std::cout<<"  go["<<i<<"]="<<value.group_order.str()<<'\n';
            std::cout<<"  go_factor["<<i<<"]=";print_parts(value.multiple_factorization.factors,B1);std::cout<<'\n';
            if(value.multiple_factorization.status==ecm_group_order::Status::Complete)
            {
                const auto bounds=ecm_group_order::group_bounds(value.multiple_factorization.factors);
                std::cout<<"  go_min_bounds["<<i<<"]: param="<<request.parameter
                    <<" largest_prime="<<bounds.prime.str()<<" min_b1_stage1="<<bounds.stage1.str()
                    <<" min_b1_stage2="<<bounds.stage2_b1.str()<<" min_b2_stage2="<<bounds.prime.str()
                    <<" largest_exponent="<<bounds.exponent<<'\n';
            }
        }
        if(value.status==ecm_group_order::Status::Complete && value.point_minimality_checked){
            std::cout<<"  go_point_order["<<i<<"]="<<value.point_order.str()<<'\n';
            std::cout<<"  go_point_factor["<<i<<"]=";print_parts(value.point_factorization);std::cout<<'\n';
            const auto bounds=ecm_group_order::point_bounds(value.point_factorization,multiplier);
            std::cout<<"  go_point_bounds["<<i<<"]: scalar="<<multiplier<<"LCM min_b1_stage1="
                     <<bounds.stage1.str()<<" frontier=[ ";
            bool first=true;
            for(const auto &[b1,b2]:bounds.frontier) {
                if(!first)std::cout<<", ";
                first=false;
                std::cout<<'<'<<b1.str()<<", "<<b2.str()<<'>';
            }
            std::cout<<" ]\n";
        }
    }catch(const std::exception &error){
        std::cerr<<"  go_status["<<i<<"]: status=unresolved reason="<<ecm_diagnostics::quote(error.what())<<'\n';
    }
}

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

static int handoff_param(const Selection& selection) {
    if (selection.curve == CurveModel::Edwards)
        return -1;
    return selection.target == ExecutionTarget::Cpu ? 0 : selection.gpu_param;
}

static const char* prime95_unsupported(const NormalizedStage1Task& task) {
    const int param = handoff_param(task.options.execution.selection);
    if (param != 0 && param != 3)
        return "Prime95 supports Montgomery PARAM0/3 saves";
    const auto& identity = task.options.identity;
    if (identity.sigma_fixed &&
        !sigma_batch_fits(identity.fixed_sigma64, task.curves, 1, INT64_MAX))
        return "Prime95 requires sigma in 1..2^63-1";
    return nullptr;
}

static const char* local_stage2_unsupported(const QueueItem& item) {
    const auto& task = item.input;
    const int param = handoff_param(task.options.execution.selection);
    if (param != -1 && param != 0)
        return "ECM-ELF Stage2 supports Edwards and Montgomery PARAM0 saves";
    if (task.b1 < 2 || mpz_cmp_ui(task.modulus, 3) <= 0 || !mpz_odd_p(task.modulus) ||
        mpz_sizeinbase(task.modulus, 2) > ecm_stage2::max_input_bits)
        return "modulus or B1 is outside the ECM-ELF Stage2 input range";
    const auto& identity = task.options.identity;
    if (param == 0 && identity.sigma_fixed && identity.fixed_sigma64 < 6)
        return "ECM-ELF Stage2 requires Montgomery PARAM0 sigma >= 6";
    EcmStage2Task line;
    std::string error;
    if (!ecm_parse_stage2_line(item.stage2_line, line, error))
        throw std::runtime_error("invalid normalized ECMSTAGE2 assignment: " + error);
    if (line.n > ecm_stage2::max_input_bits)
        return "assignment exponent is outside the ECM-ELF Stage2 input range";
    return nullptr;
}

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
    if (!completed.checkpoint.empty()) {
        ecm_stage1_checkpoint::Context cleanup;
        cleanup.remove(completed.checkpoint);
    }
    std::error_code ec;
    std::filesystem::remove(receipt, ec);
    if (ec) {
        ecm_ts_fprintf(stderr, "FATAL: cannot acknowledge completion: %s\n", ec.message().c_str());
        return false;
    }
    return true;
}

// Error reporting must not throw again while counting unreadable responsibility
// files. Keep the original flush failure and report an unknown count explicitly.
static void report_p95_pending(const P95TransferConfig& cfg, int worker, const std::string& error) {
    std::string count = "unknown";
    try {
        count = std::to_string(ecm_sync::pending_p95_count(cfg));
    } catch (const std::exception&) {
    }
    const auto path = cfg.worktodo_path.empty() ? cfg.pending_path + ".handoff"
                                               : p95_transfer_add_path(cfg.worktodo_path);
    ecm_ts_fprintf(stderr, "p95_add: pending worker=%d lines=%s file=\"%s\" error=\"%s\"\n", worker,
                   count.c_str(), p95_transfer_quote(path).c_str(), p95_transfer_quote(error).c_str());
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
            report_p95_pending(*p95, worker, error);
    }

    if (processed) {
        (*processed)++;
    }
    return true;
}

static long long current_epoch_seconds() { return static_cast<long long>(std::time(nullptr)); }

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

static int run_worktodo(const ParsedCli& input, Stage1Settings& settings,
                        const ParsedConfig& parsed_config, const InputPaths& input_paths,
                        const std::string& ini) {
    const int worker = input.worker_index;
    const auto& exe_dir = input_paths.exe;
    auto& cfg = settings.config;
    const bool shared_output = cfg.config_version >= 3;
    const auto& ini_dir = input_paths.config_directory;
    const std::string worktodo_path = resolve_rel_local(ini_dir, cfg.worktodo);
    const std::string finished_path = shared_output ? resolve_rel_local(ini_dir, cfg.shared_finished_file)
                                                   : resolve_rel_local(exe_dir, cfg.finished);
    // Several workers appending to one screen.log interleave their lines, so workers
    // 2..N get their own default log unless the ini set log_file explicitly.
    if (!shared_output && worker > 1 && !cfg.log_file_explicit &&
        cfg.log_file == ecm_config::defaults::stage1_log_file) {
        cfg.log_file = ecm_config::worker_file(ecm_config::defaults::stage1_log_file, worker);
        ecm_ts_fprintf(stdout,
                       "note: log_file not set in %s; worker %d logs to %s instead of "
                       "screen.log (per-worker default)\n",
                       ini.c_str(), worker, cfg.log_file.c_str());
    }

    ecm_io::set_work_dir(exe_dir.c_str());

    ecm_log_set_progress_color(cfg.progress_color.c_str());
    /* Progress cadence for the log FILE only (ini progress_log_seconds): a task running for
       hours would otherwise write a progress line into screen.log every 200 ms. The pipe
       keeps its cadence, and the 100% line always lands in the file. */
    ecm_log_set_progress_log_seconds(cfg.progress_log_seconds);
    ecm_stage1_gpu_progress::set_average_seconds(cfg.progress_average_seconds);

    const auto& log_name = shared_output ? cfg.shared_log_file : cfg.log_file;
    const auto logpath = resolve_rel_local(shared_output ? ini_dir : exe_dir, log_name);
    if (shared_output) {
        try {
            const auto same = ecm_io::same_file_resource;
            const auto& output = settings.options.persistence;
            const std::string private_paths[]{ini, worktodo_path,
                worktodo_path + ".stage2_pending_worker" + std::to_string(worker)};
            for (const auto& path : private_paths)
                if (same(logpath, path) || same(finished_path, path))
                    throw std::runtime_error("shared log or finished file conflicts with a configuration or queue path");
            if (finished_path.empty() || same(logpath, finished_path) ||
                same(logpath, output.results_file) || same(logpath, output.submission_file) ||
                same(finished_path, output.results_file) || same(finished_path, output.submission_file) ||
                same(logpath, settings.options.execution.ntt.tune_file) ||
                same(finished_path, settings.options.execution.ntt.tune_file))
                throw std::runtime_error("shared log, finished file and result paths must be distinct");
        } catch (const std::exception& e) {
            ecm_ts_fprintf(stderr, "FATAL: %s\n", e.what());
            return 1;
        }
    }
    EcmLogMirror log;
    if (!log_name.empty()) {
        if (!log.open(logpath, worker, shared_output ? cfg.log_file_max_kb : 0)) {
            ecm_ts_fprintf(stderr, "FATAL: cannot open log file %s\n", logpath.c_str());
            return 1;
        }
    }

    ecm_queue::OwnerLease owner;
    try {
        owner.acquire(worktodo_path, worker);
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    const std::string sync1 = resolve_rel_local(exe_dir, cfg.save_sync_dir_1);
    const std::string sync2 = resolve_rel_local(exe_dir, cfg.save_sync_dir_2);

    const std::string stage2_pending =
        worktodo_path + ".stage2_pending_worker" + std::to_string(worker);
    const auto same_path = ecm_io::same_file_resource;
    const bool legacy_handoff = cfg.handoff_mode.empty();
    const auto& persistence = settings.options.persistence;
    const std::string reserved_handoff_paths[]{
        ini, logpath, finished_path, stage2_pending,
        worktodo_path + ".stage1_completed_worker" + std::to_string(worker),
        persistence.results_file, persistence.submission_file};
    const auto conflicts_with_output = [&](const std::string& path) {
        return std::any_of(std::begin(reserved_handoff_paths), std::end(reserved_handoff_paths),
                           [&](const auto& reserved) { return same_path(path, reserved); });
    };
    const bool can_fallback = cfg.handoff_mode == "prime95" &&
                              cfg.handoff_unsupported == "ecm-elf";
    const auto auto_worktodo = resolve_rel_local(
        ini_dir, cfg.auto_stage2_worktodo.empty() ? cfg.worktodo : cfg.auto_stage2_worktodo);
    const auto auto_workers = ecm_workers::trim(cfg.auto_stage2_workers);
    Stage2TransferConfig stage2cfg;
    const auto local_route_is_external = [&] {
        return !auto_workers.empty() || !same_path(auto_worktodo, worktodo_path);
    };
    const auto stage2_receiver_config = [&](bool all_curves) -> const Stage2TransferConfig* {
        if (!stage2cfg.worktodo.empty())
            return &stage2cfg;
        const int receiver = all_curves || cfg.edwards_stage2_worker == 0
                                 ? worker : cfg.edwards_stage2_worker;
        if (receiver < 1 || receiver > 1000000)
            throw std::runtime_error("Stage2 receiver worker must be 1..1000000");
        const auto& lines = parsed_config.lines;
        const auto selected_stage2 = [&](int who) {
            return ecm_config::read_stage2(ecm_config::selected_entries(
                lines, who, {"worktodo", "stage2_finished"}, ini));
        };
        stage2cfg.worktodo = all_curves
                                 ? auto_worktodo
                                 : resolve_rel_local(ini_dir, selected_stage2(receiver).worktodo);
        stage2cfg.finished =
            shared_output ? finished_path : resolve_rel_local(ini_dir, selected_stage2(receiver).finished);
        stage2cfg.pending = stage2_pending;
        stage2cfg.worker = receiver;
        if (all_curves) {
            stage2cfg.auto_workers = ecm_stage1::lower_ascii(auto_workers) == "auto";
            int ignored = 0;
            if (!stage2cfg.auto_workers && !auto_workers.empty() &&
                (!ecm_workers::parse_list(auto_workers, 1000000, stage2cfg.workers, ignored) ||
                 ignored || stage2cfg.workers.empty()))
                throw std::runtime_error("handoff.workers must be a worker list/range in "
                                         "1..1000000 or auto");
            stage2cfg.configured_workers =
                ecm_config::read_gui(ecm_config::section_entries(lines, "gui")).NumWorkers;
            if (!shared_output) {
                stage2cfg.finished = resolve_rel_local(ini_dir, selected_stage2(0).finished);
                for (const auto& entry : lines)
                    if (entry.worker_scope > 0 &&
                        !stage2cfg.finished_by_worker.count(entry.worker_scope))
                        stage2cfg.finished_by_worker[entry.worker_scope] = resolve_rel_local(
                            ini_dir, selected_stage2(entry.worker_scope).finished);
            }
        }
        const auto valid_finished = [&](const std::string& finished) {
            return !finished.empty() && !same_path(stage2cfg.worktodo, finished) &&
                   !same_path(finished, worktodo_path) &&
                   (shared_output || !same_path(finished, finished_path)) &&
                   !same_path(finished, stage2_pending);
        };
        bool valid = !stage2cfg.worktodo.empty() && !same_path(stage2cfg.worktodo, finished_path) &&
                     !same_path(stage2cfg.worktodo, stage2_pending) &&
                     (!shared_output || !conflicts_with_output(stage2cfg.worktodo)) &&
                     valid_finished(stage2cfg.finished);
        for (const auto& item : stage2cfg.finished_by_worker)
            valid = valid && valid_finished(item.second);
        if (!valid)
            throw std::runtime_error("Stage2 transfer queue/finished paths conflict with Stage1 output");
        ecm_ts_fprintf(
            stdout, "stage2_transfer: ready source_worker=%d receiver=%s file=\"%s\"\n", worker,
            all_curves && !auto_workers.empty() ? auto_workers.c_str()
                                                : std::to_string(receiver).c_str(),
            stage2cfg.worktodo.c_str());
        return &stage2cfg;
    };
    const bool legacy_auto_routed = legacy_handoff && cfg.auto_stage2 && local_route_is_external();
    try {
        if (legacy_auto_routed || (legacy_handoff && !cfg.auto_stage2 && cfg.edwards_stage2_transfer))
            stage2_receiver_config(cfg.auto_stage2);
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
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
    auto prime95_receiver_config = [&]() -> const P95TransferConfig* {
        if (p95_initialized)
            return &p95cfg;
        if (!legacy_handoff && cfg.p95_worktodo_path.empty())
            throw std::runtime_error(ecm_config::config_diagnostic(
                ini, 0, worker, "prime95.worktodo_file", "\"\"", "missing Prime95 queue",
                "configure prime95.worktodo_file for Prime95 handoff"));
        if (!cfg.p95_worktodo_path.empty() && cfg.p95_sync_dir.empty())
            throw std::runtime_error(ecm_config::config_diagnostic(
                ini, 0, worker, "prime95.save_dir (v1: p95_sync_dir)", "\"\"", "missing Prime95 save destination",
                "save destination directory for supported Prime95 handoff; configure "
                "prime95.save_dir (v1: p95_sync_dir)"));
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
            if (!legacy_handoff &&
                (conflicts_with_output(add_path) || conflicts_with_output(p95cfg.worktodo_path) ||
                 same_path(add_path, p95cfg.worktodo_path) ||
                 same_path(add_path, worktodo_path) || same_path(p95cfg.worktodo_path, worktodo_path)))
                throw std::runtime_error("Prime95 worktodo/add paths conflict with Stage1 input/output");
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
        if (!p95cfg.worktodo_path.empty() && !ecm_sync::flush_p95(p95cfg, retry_error))
            report_p95_pending(p95cfg, 0, retry_error);
        return &p95cfg;
    };
    const auto legacy_prime95_for = [&](const Stage1Options& actual) -> const P95TransferConfig* {
        const int param = handoff_param(actual.execution.selection);
        if (param == 0 || param == 3)
            return prime95_receiver_config();
        if (!cfg.p95_worktodo_path.empty() && !p95_ignore_reported) {
            ecm_ts_fprintf(stdout, "Prime95 transfer path ignored for this curve construction; "
                                   "Stage1 saves remain local\n");
            p95_ignore_reported = true;
        }
        return nullptr;
    };

    // Modern P95 intents freeze their destination, so old work can be recovered
    // before selecting any route under the current v3 settings.
    if (!legacy_handoff) {
        P95TransferConfig recovery;
        recovery.pending_path = p95_transfer_pending_path(exe_dir);
        std::string error;
        if (!ecm_sync::flush_p95(recovery, error))
            report_p95_pending(recovery, worker, error);
    }

    auto opt = settings.options;
    opt.persistence.retain_completed_checkpoint = true;
    // Validate semantic combinations per task, after applying any save prefix.
    /* ---- [handoff] ----------------------------------------------------------- */
    if (!cfg.p95_dir.empty()) {
        ecm_ts_fprintf(stdout,
                       "note: legacy Prime95 directory %s is informational; configure "
                       "prime95.worktodo_file and prime95.save_dir for Prime95 handoff\n",
                       cfg.p95_dir.c_str());
    }

    ecm_ts_fprintf(stdout, "%s %s (Stage1)\n", ecm_program::name, ecm_program::version);
    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worker : %d (TOML v%d; worktodo [Worker #%d])\n",
                   worker, parsed_config.config_version, worker);
    ecm_ts_fprintf(stdout, "worktodo : %s\n", worktodo_path.c_str());
    ecm_ts_fprintf(stdout, "finished : %s\n", finished_path.c_str());
    ecm_ts_fprintf(stdout, "log_file : %s\n", logpath.c_str());
    ecm_ts_fprintf(stdout, "saves : %s\n", opt.persistence.tmp_dir.c_str());
    ecm_ts_fprintf(stdout, "curve model : %s\n",
                   ecm_stage1::curve_model_name(opt.execution.selection.curve));
    ecm_ts_fprintf(stdout, "execution target : %s\n",
                   ecm_stage1::execution_target_name(opt.execution.selection.target));
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu) {
        /* The GPU implementation is chosen at LINK time (ecm.exe = OpenCL glue,
           ecm_cuda.exe = CUDA backend), so ask the backend instead of hardcoding
           "OpenCL" -- the CUDA build used to print "gpu (OpenCL)" here. */
        if (opt.execution.selection.gpu_algorithm == ecm_stage1::GpuAlgorithm::Ntt)
            ecm_ts_fprintf(stdout, "gpu backend : CUDA/NTT, %s, device %d\n",
                           opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards
                               ? "Edwards Atkin-Morain" : "Montgomery param0",
                           opt.execution.device_index);
        else if (opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
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
            const auto typed = prefixed_name(task.save_name, save_naming(actual).curve);
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
            if (!saw_task && legacy_handoff)
                legacy_prime95_for(opt);
            if (!saw_task && !legacy_handoff && cfg.handoff_mode == "prime95" &&
                !cfg.p95_worktodo_path.empty() && !cfg.p95_sync_dir.empty())
                prime95_receiver_config(); // Legacy pending and rejected-AID recovery.
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
        if (shared_output && (same_path(logpath, item.input.save_file) ||
                              same_path(finished_path, item.input.save_file))) {
            input_error("Stage1 save conflicts with the shared log or finished file");
            return QueueReadStatus::Error;
        }
        // Resolve delivery only after the actual typed construction.
        if (legacy_handoff) {
            item.p95 = legacy_prime95_for(item.input.options);
            if (legacy_auto_routed ||
                (!cfg.auto_stage2 && cfg.edwards_stage2_transfer &&
                 item.input.options.execution.selection.curve == CurveModel::Edwards))
                item.stage2 = &stage2cfg;
            else if (cfg.auto_stage2)
                item.auto_stage2_line = item.stage2_line;
        } else {
            bool use_local = cfg.handoff_mode == "ecm-elf";
            if (cfg.handoff_mode == "prime95") {
                if (const char* reason = prime95_unsupported(item.input)) {
                    use_local = can_fallback;
                    ecm_ts_fprintf(stdout, "Stage2 handoff: %s; %s\n", reason,
                                   use_local ? "checking ECM-ELF Stage2" : "retaining Stage1 save");
                } else {
                    item.p95 = prime95_receiver_config();
                }
            }
            if (use_local) {
                if (const char* reason = local_stage2_unsupported(item))
                    ecm_ts_fprintf(stdout, "Stage2 handoff: %s; retaining Stage1 save\n", reason);
                else if (local_route_is_external())
                    item.stage2 = stage2_receiver_config(true);
                else
                    item.auto_stage2_line = item.stage2_line;
            }
            if (item.stage2 && same_path(item.stage2->worktodo, item.input.save_file))
                throw std::runtime_error("Stage2 receiver queue conflicts with the Stage1 save");
            if (item.p95) {
                const auto target = p95_join_path(item.p95->sync_dir,
                    std::filesystem::path(item.input.save_file).filename().string());
                const auto add = p95_transfer_add_path(item.p95->worktodo_path);
                if (conflicts_with_output(target) || same_path(target, worktodo_path) ||
                    same_path(target, item.p95->worktodo_path) ||
                    same_path(target, add) || same_path(item.input.save_file, add) ||
                    same_path(item.input.save_file, item.p95->worktodo_path))
                    throw std::runtime_error("Prime95 save destination conflicts with a queue or Stage1 output");
            }
        }
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
    ecm_stage1_app::ParsedCli input;
    if (ecm_stage1_app::parse_cli(argc, argv, input))
        return 1;
    if (input.version) {
        std::cout << ecm_program::name << ' ' << ecm_program::version << std::endl;
        return 0;
    }
    std::string error;
    if (!ecm_stage1_app::validate_cli_mode(input, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    // Kernel listing needs no configuration or task. Install its explicit runtime once.
    if (input.show_kernels) {
        ecm_runtime_config() = input.runtime;
        ecm_install_timestamped_iostreams();
        ecm_enable_console_ansi();
        ecm_backend_print_kernels(stdout);
        return 0;
    }
    const auto raw_exe = get_exe_dir_local();
    const auto exe = raw_exe.empty() ? std::string(".") : raw_exe;
    const auto ini = input.ini_path.empty() ? resolve_rel_local(exe, "ecm.toml")
                                            : resolve_rel_local(exe, input.ini_path);
    const ecm_stage1_app::InputPaths paths{exe, std::filesystem::path(ini).parent_path().string(),
                                           std::filesystem::current_path().string()};
    const bool queue = input.pos.empty() && !input.show_gpu_info && !input.tune_ntt;
    ParsedConfig parsed;
    parsed.path = ini;
    bool config_used = std::filesystem::exists(ini);
    if (!config_used && queue) {
        if (!ecm_queue_config_write_default(ini)) {
            std::cerr << "FATAL: cannot create default config " << ini << "\n";
            return 1;
        }
        std::cout << "Created default config: " << ini << "\n";
        config_used = true;
    }
    if (config_used) {
        if (!read_config(ini, parsed))
            return 1;
    } else if (!input.ini_path.empty() && !input.show_gpu_info) {
        std::cerr << "INPUT[config] cannot read explicit configuration " << ini << "\n";
        return 1;
    }
    ecm_stage1_app::Stage1Settings settings;
    if (!ecm_stage1_app::normalize_settings(parsed, input, paths, settings, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (input.tune_ntt || settings.options.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu) {
        try {
            auto backend = input.gpu_backend.empty() ? settings.config.gpu_backend : input.gpu_backend;
            if (input.tune_ntt || settings.options.execution.selection.gpu_algorithm == ecm_stage1::GpuAlgorithm::Ntt) {
                if (backend == "opencl") throw std::invalid_argument("Stage1 NTT requires CUDA");
                backend = "cuda";
            }
            const auto routed = ecm_gpu::route(exe, false, ECM_HAS_CUDA_STAGE1 != 0, backend,
                settings.options.execution.device_index);
            if (routed) return *routed;
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    }
    ecm_runtime_config() = settings.runtime;
    ecm_install_timestamped_iostreams();
    ecm_enable_console_ansi();
    if (input.show_gpu_info) {
        const auto& selection = settings.options.execution.selection;
        return print_gpu_info(config_used ? ini.c_str() : nullptr, input.worker_index,
                              settings.options.execution.device_index, selection.gpu_param,
                              selection.curve == ecm_stage1::CurveModel::Edwards,
                              input.gpu_info_bits);
    }
    const auto stop_handle = input.stop_handle;
    if (stop_handle && !ecm_stage1_set_stop_handle(stop_handle)) {
        std::cerr << "Invalid inherited Stage1 stop handle" << std::endl;
        return 1;
    }
    ecm_stage1_install_stop_handler();

    EcmLogMirror direct_log;
    const bool shared_output = settings.config.config_version >= 3;
    const auto& log_name = shared_output ? settings.config.shared_log_file : settings.config.log_file;
    const auto log_path = resolve_rel_local(shared_output ? paths.config_directory : exe, log_name);
    const auto open_direct_log = [&](const std::vector<std::string>& protected_paths) {
        if (log_path.empty()) return true;
        try {
            if (shared_output && (ecm_io::same_file_resource(log_path, ini) ||
                std::any_of(protected_paths.begin(), protected_paths.end(), [&](const auto& path) {
                    return ecm_io::same_file_resource(log_path, path);
                }))) throw std::runtime_error("shared log conflicts with a Stage1 input or output path");
            if (direct_log.open(log_path, input.worker_index, shared_output ? settings.config.log_file_max_kb : 0))
                return true;
        } catch (const std::exception& error) {
            ecm_ts_fprintf(stderr, "FATAL: %s\n", error.what());
            return false;
        }
        ecm_ts_fprintf(stderr, "FATAL: cannot open Stage1 log file %s\n", log_path.c_str());
        return false;
    };
    if (input.tune_ntt) {
        if (!open_direct_log({settings.options.execution.ntt.tune_file})) return 1;
#if ECM_HAS_CUDA_STAGE1
        return ecm_mersenne_ntt::tune_command(settings.options.execution.device_index,
            settings.options.execution.ntt, input.gpu_info_bits, input.tune_exhaustive);
#else
        std::cerr << "Stage1 NTT tune requires the CUDA worker\n";
        return 1;
#endif
    }
    if (queue) {
        // No positional B1/B2 → queue-manager mode (reads ecm.toml + worktodo).
        return ecm_stage1_app::run_worktodo(input, settings, parsed, paths, ini);
    }

    ecm_ts_fprintf(stdout, "%s %s (Stage1)\n", ecm_program::name, ecm_program::version);

    // Diagnosis follows durable task publication; GP availability is not admission.
    auto go_settings = input.go_settings;
    go_settings.gp = ecm_diagnostics::resolve_gp_path(go_settings.gp);
    ecm_diagnostics::Session go_session(go_settings);
    const auto& execution = settings.options.execution;
    std::cout << "ecm driver starting\n  curve_model: "
              << ecm_stage1::curve_model_name(execution.selection.curve) << ", execution_target: "
              << ecm_stage1::execution_target_name(execution.selection.target)
              << ", gpucurves=" << input.gpucurves
              << ", ckpt=" << settings.options.persistence.ckpt_ms / 1000.0
              << "s, device=" << execution.device_index
              << ", group_order=" << (input.print_group_order ? "on" : "off") << std::endl;

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
    if (!ecm_stage1_app::normalize_direct(input, settings.options, nline, task, input_error)) {
        std::cerr << input_error << std::endl;
        return 1;
    }
    const auto& opt = task.options;
    if ((shared_output || opt.execution.selection.gpu_algorithm == ecm_stage1::GpuAlgorithm::Ntt) &&
        !open_direct_log({task.save_file, opt.persistence.results_file, opt.persistence.submission_file,
                          opt.execution.ntt.tune_file}))
        return 1;
    const auto B1 = task.b1;
    if (!input.p95_dir_ignored.empty())
        std::cout
            << "Prime95 transfer path ignored: --p95-dir is obsolete; use TOML prime95.worktodo_file."
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
                          << " save=" << task.save_file << "\n";
                free(s);
                if (input.print_group_order)
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
