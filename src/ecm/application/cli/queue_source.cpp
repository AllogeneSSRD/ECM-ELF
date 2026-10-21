#include "queue_source.h"
#include "stage1_input.h"
#include "ecm/service/work/queue_workflow.h"
#include "../../service/stage1/execution_contract.h"
#include "../../service/stage1/preparation.h"
#include "../configuration/ecm_queue_config.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "../../storage/queue/ecm_worker_list.h"
#include "../../storage/transfer/ecm_stage2_transfer.h"
#include "../../storage/transfer/p95_transfer.h"
#include "../../storage/checkpoint/ecm_checkpoint.h"
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../storage/cache/ecm_stage1_exp_cache.h"
#include "../../backend/cpu/stage1/edwards/ecm_edwards_cpu.h"
#include "ecm/service/execution/legacy_gpu.h"
#include "ecm/platform/program_version.h"
#include "ecm/platform/logging.h"
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <io.h>
#define access _access
#else
#include <unistd.h>
#endif
namespace ecm_stage1_app {
using ecm_stage1::Stage1Options;
using ecm_stage1::QueueReadStatus;
using ecm_stage1::ensure_dir;
using ecm_stage1::mont_format_save_name;
using ecm_stage1::stage1_save_prefix;
using ecm_stage1::stage1_prefixed_save;
static long long current_epoch_seconds() {
    return static_cast<long long>(std::time(nullptr));
}

static std::string build_n_expr(const std::string &k, const std::string &b,
                                unsigned long n, const std::string &c,
                                const std::vector<std::string> &factors) {
    std::string cstr = c;
    std::string sign = "+";
    if (!cstr.empty() && cstr[0] == '-') {
        sign = "-";
        cstr = cstr.substr(1);
    }
    std::string e = "(" + k + "*" + b + "^" + std::to_string(n) + sign + cstr + ")";
    if (!factors.empty()) {
        e += "/(";
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (i != 0) e += "*";
            e += factors[i];
        }
        e += ")";
    }
    return e;
}

static std::string p95_dir_of(const std::string &path) {
    const size_t s = path.find_last_of("\\/");
    if (s == std::string::npos) return std::string(".");
    if (s == 0) return path.substr(0, 1);
    return path.substr(0, s);
}

static std::string p95_join_path(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    const char last = dir[dir.size() - 1];
    return (last == '\\' || last == '/') ? (dir + name) : (dir + "\\" + name);
}

static std::string edwards_stage2_line(const Ecm2Task &task, const std::string &save,
                                       uint64_t skip) {
    std::ostringstream out;
    std::string quoted_save = "\"";
    for (char c : save) {
        quoted_save += c;
        if (c == '"') quoted_save += '"';
    }
    quoted_save += '"';
    out << "ECMSTAGE2=";
    if (!task.aid.empty()) out << task.aid << ',';
    out << task.k << ',' << task.b << ',' << task.n << ',' << task.c << ','
        << quoted_save << ',' << task.b2_text << ',' << skip << ','
        << task.curves_to_run;
    if (!task.factors.empty()) {
        out << ",\"";
        for (size_t i = 0; i < task.factors.size(); ++i) {
            if (i) out << ',';
            out << task.factors[i];
        }
        out << '"';
    }
    return out.str();
}

static uint64_t stage2_save_record_count(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    uint64_t count = 0;
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first != std::string::npos && line[first] != '#' && line[first] != ';') ++count;
    }
    return count;
}
static QueueReadStatus finalize_queue_input(ecm_stage1::QueueItem &item,uint64_t sigma,bool sigma_fixed) {
    auto &opt=item.options;
    opt.identity.result_aid=item.aid;
    if(sigma_fixed){opt.identity.sigma_fixed=true;opt.identity.fixed_sigma64=sigma;}
    std::string input_error;
    if(stage1_selection_error(opt.execution.selection,input_error) || stage1_task_error(opt,item.curves,input_error)) {
        ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",input_error.c_str());
        return QueueReadStatus::Error;
    }
    return QueueReadStatus::Ready;
}
int run_queue_manager(const std::string &ini_path, int worker,const ecm_stage1::TaskDependencies &tasks) {
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
    if (!ecm_queue_config_load(ini, worker, cfg)) {
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
    // Several workers appending to one screen.log interleave their lines, so workers
    // 2..N get their own default log unless the ini set log_file explicitly.
    if (worker > 1 && !cfg.log_file_explicit && cfg.log_file == ecm_config::defaults::stage1_log_file) {
        cfg.log_file = ecm_config::worker_file(ecm_config::defaults::stage1_log_file,worker);
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
    FILE *logf = nullptr;
    if (!cfg.log_file.empty()) {
        const std::string logpath = resolve_rel_local(exe_dir, cfg.log_file);
        logf = fopen(logpath.c_str(), "a");
        if (logf == nullptr) {
            ecm_ts_fprintf(stderr, "FATAL: cannot open log file %s\n", logpath.c_str());
            return 1;
        }
        ecm_log_set_mirror(logf);
    }

    const std::string ini_dir = std::filesystem::path(ini).parent_path().string();
    const std::string worktodo_path = resolve_rel_local(ini_dir, cfg.worktodo);
    const std::string finished_path = resolve_rel_local(exe_dir, cfg.finished);
    const std::string sync1 = resolve_rel_local(exe_dir, cfg.save_sync_dir_1);
    const std::string sync2 = resolve_rel_local(exe_dir, cfg.save_sync_dir_2);

    const std::string stage2_pending = worktodo_path + ".stage2_pending_worker" + std::to_string(worker);
    const auto same_path=[](const std::string &a,const std::string &b) {
        std::error_code ec;
        if(std::filesystem::equivalent(a,b,ec)&&!ec)return true;
#ifdef _WIN32
        return ecm_stage1::lower_ascii(a)==ecm_stage1::lower_ascii(b);
#else
        return a==b;
#endif
    };
    const auto auto_worktodo=resolve_rel_local(ini_dir,cfg.auto_stage2_worktodo.empty()?cfg.worktodo:cfg.auto_stage2_worktodo);
    const bool auto_stage2_routed=cfg.auto_stage2 &&
        (!ecm_workers::trim(cfg.auto_stage2_workers).empty() || !same_path(auto_worktodo,worktodo_path));
    Stage2TransferConfig stage2cfg;
    if(auto_stage2_routed || (!cfg.auto_stage2 && cfg.edwards_stage2_transfer)) {
        const int receiver=cfg.auto_stage2?worker:cfg.edwards_stage2_worker==0?worker:cfg.edwards_stage2_worker;
        if(receiver<1 || receiver>1000000) {
            ecm_ts_fprintf(stderr,"FATAL: Stage2 receiver worker must be 1..1000000\n");return 1;
        }
        std::vector<ecm_config::TomlLine> lines;
        if(!ecm_config::read_toml(ini,lines)) {
            ecm_ts_fprintf(stderr,"FATAL: cannot read Stage2 receiver configuration\n");return 1;
        }
        const auto receiver_cfg=ecm_config::read_stage2(ecm_config::cli_entries(lines,receiver,true));
        stage2cfg.worktodo=cfg.auto_stage2?auto_worktodo:resolve_rel_local(ini_dir,receiver_cfg.worktodo);
        stage2cfg.finished=resolve_rel_local(ini_dir,receiver_cfg.finished);
        stage2cfg.pending=stage2_pending;stage2cfg.worker=receiver;
        if(cfg.auto_stage2) {
            const auto spec=ecm_workers::trim(cfg.auto_stage2_workers);
            stage2cfg.auto_workers=ecm_stage1::lower_ascii(spec)=="auto";
            int ignored=0;
            if(!stage2cfg.auto_workers && !spec.empty() &&
               (!ecm_workers::parse_list(spec,1000000,stage2cfg.workers,ignored) ||
                ignored || stage2cfg.workers.empty())) {
                ecm_ts_fprintf(stderr,"FATAL: auto_stage2_workers must be a worker list/range in 1..1000000 or auto\n");return 1;
            }
            stage2cfg.configured_workers=ecm_config::read_gui(ecm_config::section_entries(lines,"gui")).NumWorkers;
            const auto global=ecm_config::read_stage2(ecm_config::cli_entries(lines,0,true));
            stage2cfg.finished=resolve_rel_local(ini_dir,global.finished);
            for(const auto &entry:lines)if(entry.worker_scope>0 &&
                !stage2cfg.finished_by_worker.count(entry.worker_scope)) {
                const auto selected=ecm_config::read_stage2(ecm_config::cli_entries(lines,entry.worker_scope,true));
                stage2cfg.finished_by_worker[entry.worker_scope]=resolve_rel_local(ini_dir,selected.finished);
            }
        }
        const auto valid_finished=[&](const std::string &finished) {
            return !finished.empty() && !same_path(stage2cfg.worktodo,finished) &&
                !same_path(finished,worktodo_path) && !same_path(finished,finished_path) &&
                !same_path(finished,stage2_pending);
        };
        bool valid= !stage2cfg.worktodo.empty() &&
            !same_path(stage2cfg.worktodo,finished_path) && !same_path(stage2cfg.worktodo,stage2_pending) &&
            valid_finished(stage2cfg.finished);
        for(const auto &item:stage2cfg.finished_by_worker)valid=valid&&valid_finished(item.second);
        if(!valid) {
            ecm_ts_fprintf(stderr,"FATAL: Stage2 transfer queue/finished paths conflict with Stage1 output\n");return 1;
        }
        ecm_ts_fprintf(stdout,"stage2_transfer: ready source_worker=%d receiver=%s file=\"%s\"\n",
            worker,cfg.auto_stage2 && !cfg.auto_stage2_workers.empty()?cfg.auto_stage2_workers.c_str():std::to_string(receiver).c_str(),stage2cfg.worktodo.c_str());
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
    bool p95_initialized=false,p95_ignore_reported=false;
    auto p95_for_task=[&](const Stage1Options &actual)->const P95TransferConfig* {
        const bool ignore=actual.execution.selection.curve==ecm_stage1::CurveModel::Edwards ||
            (actual.execution.selection.target==ecm_stage1::ExecutionTarget::Gpu && actual.execution.selection.gpu_param==2);
        if(ignore) {
            if(!cfg.p95_worktodo_path.empty() && !p95_ignore_reported) {
                ecm_ts_fprintf(stdout,"Prime95 transfer path ignored for this curve construction; Stage1 saves remain local\n");
                p95_ignore_reported=true;
            }
            return nullptr;
        }
        if(p95_initialized)return &p95cfg;
        p95_initialized=true;
        p95cfg.worktodo_path = resolve_rel_local(exe_dir, cfg.p95_worktodo_path);
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
                           p95_transfer_quote(add_path).c_str(), parked,
                           cfg.p95_keep_aid ? 1 : 0, rejected.size());
            for (const std::string &aid : rejected) {
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
        return &p95cfg;
    };

    Stage1Options opt;
    if (!ecm_stage1::parse_curve_model(cfg.curve_model, opt.execution.selection.curve) ||
        !ecm_stage1::parse_execution_target(cfg.execution_target, opt.execution.selection.target)) {
        ecm_ts_fprintf(stderr, "ERROR: invalid curve or target in ecm.toml\n");
        return 1;
    }
    opt.execution.affinity_cpus = parse_affinity_spec(cfg.affinity);
    /* ---- [cpu] shared by edwards and mont ----------------------------------- */
    /* CPU arithmetic: auto | simd | scalar */
    {
        if (!ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation, opt.execution.selection.cpu_implementation)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid impl in ecm.toml\n");
            return 1;
        }
    }
    /* SIMD reduction: auto | montgomery | mersenne-fold. */
    {
        if (!ecm_stage1::parse_simd_reduction(cfg.simd_reduction, opt.execution.selection.simd_reduction)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid reduction in ecm.toml\n");
            return 1;
        }
    }
    stage1_result_settings(opt,cfg,std::filesystem::path(ini).parent_path().string(),worker);
    opt.execution.stage1_threads = cfg.stage1_threads;
    if (!cfg.save_name_pattern.empty()) opt.persistence.save_name_pattern = cfg.save_name_pattern;
    if (!cfg.exp_cache.empty()) ecm_exp_cache_set_dir(cfg.exp_cache);
    /* ---- [edwards] only ------------------------------------------------------ */
    if (cfg.naf_w >= 3 && cfg.naf_w <= 12) edwards_set_naf_w(cfg.naf_w);
    opt.execution.naf_w = cfg.naf_w;
    /* Montgomery scalar construction, shared by CPU and GPU. */
    {
        if (!ecm_stage1::parse_scalar_construction(cfg.scalar_construction, opt.execution.selection.scalar_construction)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid lcm in ecm.toml (expected 1 or 12)\n");
            return 1;
        }
    }
    opt.execution.verbose = cfg.verbose ? 1 : 0;
    /* ---- [gpu] only ---------------------------------------------------------- */
    opt.execution.device_index = cfg.device;
    opt.execution.selection.gpu_param = cfg.gpu_param;
    opt.persistence.ckpt_ms = (cfg.ckpt_seconds > 0.0)
                         ? (unsigned long)(cfg.ckpt_seconds * 1000.0)
                         : 0UL;
    opt.persistence.ckpt_backups = (unsigned)std::clamp(cfg.ckpt_backups, 0,
                                           (int)ECM_STAGE1_MAX_BACKUPS);
    opt.execution.gpu_mul_path = cfg.kernel_mul;
    opt.execution.gpu_sqr_path = cfg.kernel_sqr;
    opt.execution.gpu_add_path = cfg.kernel_add;
    opt.execution.gpu_sub_path = cfg.kernel_sub;
    opt.execution.gpu_special_mult_path = cfg.kernel_special_mult;
    /* ---- [task] -------------------------------------------------------------- */
    opt.identity.sigma_fixed = (cfg.sigma != 0);
    opt.identity.fixed_sigma64 = cfg.sigma;
    opt.persistence.tmp_dir = resolve_rel_local(exe_dir, cfg.tmp_dir);
    // Validate semantic combinations per task, after applying any save prefix.
    /* ---- [handoff] ----------------------------------------------------------- */
    if (!cfg.p95_dir.empty()) {
        ecm_ts_fprintf(stdout,
            "note: p95_dir=%s is informational here -- ecm.exe only writes local saves\n"
            "      to %s; the stage-2 handoff program ecm_p95feeder reads its own\n"
            "      feeder.ini (same key name)\n",
            cfg.p95_dir.c_str(), opt.persistence.tmp_dir.c_str());
    }

    ecm_ts_fprintf(stdout,"%s %s (Stage1)\n",ecm_program::name,ecm_program::version);
    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worker : %d (TOML [workers.%d.*]; worktodo [Worker #%d])\n", worker, worker, worker);
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
        if(opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards)
            ecm_ts_fprintf(stdout,"gpu backend : %s, Edwards Atkin-Morain, device %d\n",
                           ecm_backend_name(),opt.execution.device_index);
        else ecm_ts_fprintf(stdout, "gpu backend : %s, param%d, device %d\n",
                            ecm_backend_name(), opt.execution.selection.gpu_param, opt.execution.device_index);
    }

    // Startup full sync (matches the old work_manager.ps1 behaviour).
    ecm_sync_save_files(exe_dir, sync1, sync2, /*full=*/true, 0);
    if (!opt.persistence.tmp_dir.empty() && opt.persistence.tmp_dir != exe_dir)
        ecm_sync_save_files(opt.persistence.tmp_dir, sync1, sync2, /*full=*/true, 0);

    bool saw_task=false;
    const bool full_sync=(cfg.sync_mode=="full");
    const ecm_stage1::QueuePaths paths{worktodo_path,finished_path,exe_dir,sync1,sync2,worker,full_sync};
    const auto next_task=[&](ecm_stage1::QueueItem &item)->QueueReadStatus {
        std::string line;
        std::string read_error;
        const WorktodoReadResult read = ecm_worktodo_read_next_line(worktodo_path, worker,
            {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage1, line, read_error);
        if (read == WorktodoReadResult::Error) {
            ecm_ts_fprintf(stderr, "ERROR: %s\n", read_error.c_str());
            return QueueReadStatus::Error;
        }
        if (read == WorktodoReadResult::Empty) {
            if(!saw_task)p95_for_task(opt); // Preserve idle-start recovery for an eligible TOML.
            return QueueReadStatus::Empty;
        }
        saw_task=true;item.ticket=line;

        ecm_ts_fprintf(stdout, "START: %s\n", line.c_str());

        const long long marker = current_epoch_seconds();
        std::string err;

        if (line.compare(0, 5, "ECM2=") == 0 || line.compare(0, 4, "ECM=") == 0) {
            // Prime95 ECM= / ECM2= (等价) 格式: k,b,n,c,B1[,B2][,curves][,sigma][,"factors"].
            Ecm2Task task;
            if (!ecm_parse_ecm2_line(line, task, err)) {
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                return QueueReadStatus::MarkError;
            }
            mpz_ptr N=item.modulus;
            if (!ecm_compute_ecm2_n(task, N, err)) {
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                return QueueReadStatus::MarkError;
            }
            const std::string n_expr = build_n_expr(task.k, task.b, task.n, task.c, task.factors);
            // 交接字段 (从 ECM= 任务读取)
            {
                opt.identity.target_kbnc = true;
                opt.identity.handoff_k_str = task.k;
                opt.identity.handoff_k = std::strtod(task.k.c_str(), nullptr);
                opt.identity.handoff_b = (uint32_t)std::strtoul(task.b.c_str(), nullptr, 10);
                opt.identity.handoff_n = (uint32_t)task.n;
                opt.identity.handoff_c = (int32_t)std::strtol(task.c.c_str(), nullptr, 10);
                opt.identity.handoff_factors.clear();
                for (size_t fi = 0; fi < task.factors.size(); ++fi) {
                    if (fi) opt.identity.handoff_factors += ",";
                    opt.identity.handoff_factors += task.factors[fi];
                }
            }
            // ECM=/ECM2= has no save filename in the assignment. Give the GPU path a
            // deterministic destination so a completed queue task publishes its Stage1
            // point for Stage2 and save synchronization. CPU Montgomery already writes
            // its own shared save under tmp_dir; both Edwards targets use text saves.
            std::string queue_save_name;
            if (cfg.auto_stage2 || opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu ||
                opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards) {
                if (opt.persistence.tmp_dir.empty() || !ensure_dir(opt.persistence.tmp_dir)) {
                    ecm_ts_fprintf(stderr, "FATAL: cannot create tmp_dir for GPU save: %s\n",
                                   opt.persistence.tmp_dir.c_str());
                            return QueueReadStatus::Error;
                }
                queue_save_name = opt.persistence.tmp_dir + "/" +
                    mont_format_save_name(opt.persistence.save_name_pattern,N,task.B1,&opt);
            }
            const Stage2TransferConfig *stage2 =
                auto_stage2_routed || (!cfg.auto_stage2 && cfg.edwards_stage2_transfer && opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
                    ? &stage2cfg : nullptr;
            std::string stage2_line;
            if (stage2 != nullptr || cfg.auto_stage2) {
                try { (void)ecm_config::unsigned_integer(task.b2_text, "Stage2 handoff B2"); }
                catch (const std::exception &e) {
                    ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", e.what());
                    return QueueReadStatus::Error;
                }
                const auto save = normalize_path_local(std::filesystem::absolute(resolve_rel_local(exe_dir, queue_save_name)));
                const auto skip = stage2_save_record_count(save);
                stage2_line = edwards_stage2_line(task, save, skip);
            }
            item.b1=task.B1;
        item.b2=task.B2;
        item.curves=task.curves_to_run;
        item.save_file=queue_save_name;
        item.expression=n_expr;
        item.aid=task.aid;
        item.ticket=line;
        item.options=opt;
        item.marker=marker;
        item.p95=p95_for_task(opt);
        item.published_line={};
        item.stage2=stage2;
        item.stage2_line=stage2 != nullptr ? stage2_line : std::string{};
        item.auto_stage2_line=cfg.auto_stage2 && !auto_stage2_routed ? stage2_line : std::string{};
        return finalize_queue_input(item,task.has_sigma ? task.sigma : 0,task.has_sigma);
        }

        // ECMSTAGE1= uses the save-based task format. ECMSTAGE2 belongs to Stage2.
        EcmStage2Task task;
        if (!ecm_parse_stage1_line(line, task, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            return QueueReadStatus::MarkError;
        }

        double B1 = 0.0;
        if (!ecm_extract_b1_from_save_name(task.save_name, &B1, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            return QueueReadStatus::MarkError;
        }

        mpz_ptr N=item.modulus;
        if (!ecm_compute_stage2_n(task, N, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            return QueueReadStatus::MarkError;
        }

        const std::string n_expr = build_n_expr(task.k, task.b, task.n, task.c, task.factors);
        // 交接字段 (从 ECMSTAGE2 任务读取)
        {
            opt.identity.target_kbnc = true;
                opt.identity.handoff_k_str = task.k;
            opt.identity.handoff_k = std::strtod(task.k.c_str(), nullptr);
            opt.identity.handoff_b = (uint32_t)std::strtoul(task.b.c_str(), nullptr, 10);
            opt.identity.handoff_n = (uint32_t)task.n;
            opt.identity.handoff_c = (int32_t)std::strtol(task.c.c_str(), nullptr, 10);
            opt.identity.handoff_factors.clear();
            for (size_t fi = 0; fi < task.factors.size(); ++fi) {
                if (fi) opt.identity.handoff_factors += ",";
                opt.identity.handoff_factors += task.factors[fi];
            }
        }
        // The typed save owns curve/param and B1; untyped legacy names use the TOML.
        // Clone options so the next task always starts from the configured defaults.
        auto task_opt=opt;
        if(!stage1_save_construction(task.save_name,task_opt,err) ||
           stage1_selection_error(task_opt.execution.selection,err) || stage1_task_error(task_opt,task.curves_to_run,err)) {ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());
            return QueueReadStatus::Error;
        }
        ecm_ts_fprintf(stdout,"queue_save: curve=%s param=%s B1=%.0f\n",
            ecm_stage1::curve_model_name(task_opt.execution.selection.curve),
            task_opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards?"-":stage1_save_prefix(task_opt).substr(1).c_str(),B1);
        const auto queue_save_name=stage1_prefixed_save(task.save_name,stage1_save_prefix(task_opt));
        std::string published_line=line;
        if(queue_save_name!=task.save_name && !ecm_rewrite_stage2_save(line,queue_save_name,published_line,err)) {ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());return QueueReadStatus::Error;
        }
        if(queue_save_name!=task.save_name)task_opt.persistence.legacy_save_source=task.save_name;
        const Stage2TransferConfig *stage2 =
            auto_stage2_routed || (!cfg.auto_stage2 && cfg.edwards_stage2_transfer && task_opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards)
                ? &stage2cfg : nullptr;
        std::string stage2_line;
        if (stage2 != nullptr || cfg.auto_stage2) {
            // The explicit queue save is published relative to the process work
            // directory. Handoff must name that same file, including when the
            // config and executable live in different directories.
            const auto save = normalize_path_local(std::filesystem::absolute(queue_save_name));
            if (!ecm_rewrite_stage2_save(published_line, save, stage2_line, err)) {
                ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", err.c_str());
                return QueueReadStatus::Error;
            }
            stage2_line.replace(0, 10, "ECMSTAGE2=");
        }
        item.b1=B1;
        item.b2=/*B2=*/0.0;
        item.curves=task.curves_to_run;
        item.save_file=queue_save_name;
        item.expression=n_expr;
        item.aid=task.aid;
        item.ticket=line;
        item.options=task_opt;
        item.marker=marker;
        item.p95=p95_for_task(task_opt);
        item.published_line=published_line;
        item.stage2=stage2;
        item.stage2_line=stage2 != nullptr ? stage2_line : std::string{};
        item.auto_stage2_line=cfg.auto_stage2 && !auto_stage2_routed ? stage2_line : std::string{};
        return finalize_queue_input(item,/*sigma=*/0,/*sigma_fixed=*/false);
        };
    const int status=ecm_stage1::run_queue_workflow(paths,next_task,tasks);
    if(logf){ecm_log_set_mirror(nullptr);fclose(logf);}
    return status;
}
} // namespace ecm_stage1_app
