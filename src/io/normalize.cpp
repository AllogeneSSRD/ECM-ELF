#include "io/normalize.h"
#include "ecm/application/configuration/ecm_expr.h"
#include "ecm/service/execution/legacy_gpu.h"
#include "ecm/service/stage1/preparation.h"
#include "io/cli.h"
#include "io/config.h"
#include "io/stage1_checkpoint.h"
#include "io/worktodo.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
namespace ecm_stage1_app {

// Resolve optional cache policy once. Empty input means the existing
// executable-directory default; an explicit disable token remains empty and is
// never replaced again during execution.
static std::string normalized_cache_dir(const std::string& value, const std::string& exe,
                                        const std::string& cwd) {
    if (value == "off" || value == "none" || value == "0" || value == "-")
        return {};
    const auto path = value.empty() ? (exe.empty() ? std::string(".") : exe) : value;
    // absolute also resolves Windows drive-relative spellings such as D:cache.
    return std::filesystem::absolute(std::filesystem::u8path(cwd) / std::filesystem::u8path(path))
        .lexically_normal()
        .u8string();
}

void normalize_assignment(Ecm2Task& task) {
    if (!task.has_curves)
        task.curves_to_run = 100; // Prime95 ECM/ECM2 grammar default.
    if (!task.has_b2) {
        task.B2 = 0;
        task.b2_text = "0";
    }
}

template <class Task> static void target_identity(const Task& task, NormalizedStage1Task& out) {
    auto& id = out.options.identity;
    id.target_kbnc = true;
    id.handoff_k_str = task.k;
    id.handoff_k = std::strtod(task.k.c_str(), nullptr);
    id.handoff_b = static_cast<uint32_t>(ecm_config::bounded_integer(task.b, "b", 2, UINT32_MAX));
    id.handoff_n = static_cast<uint32_t>(task.n);
    id.handoff_c = ecm_config::signed_integer(task.c, "c", INT32_MIN, INT32_MAX);
    id.handoff_factors.clear();
    for (const auto& factor : task.factors) {
        if (!id.handoff_factors.empty())
            id.handoff_factors += ',';
        id.handoff_factors += factor;
    }
    out.expression = "(" + task.k + "*" + task.b + "^" + std::to_string(task.n) +
                     (task.c.empty() || task.c[0] != '-' ? "+" : "") + task.c + ")";
    if (!task.factors.empty()) {
        out.expression += "/(";
        for (size_t i = 0; i < task.factors.size(); ++i) {
            if (i)
                out.expression += '*';
            out.expression += task.factors[i];
        }
        out.expression += ')';
    }
    id.result_aid = task.aid;
    out.curves = task.curves_to_run;
}

template <class Task>
static std::string saved_stage2_line(const Task& task, const std::string& save) {
    std::string quoted = "\"";
    for (char c : save) {
        quoted += c;
        if (c == '"')
            quoted += '"';
    }
    quoted += '"';
    std::ostringstream out;
    out << "ECMSTAGE2=";
    if (!task.aid.empty())
        out << task.aid << ',';
    out << task.k << ',' << task.b << ',' << task.n << ',' << task.c << ',' << quoted << ','
        << task.b2_text << ",0," << task.curves_to_run;
    if (!task.factors.empty()) {
        out << ",\"";
        for (size_t i = 0; i < task.factors.size(); ++i) {
            if (i)
                out << ',';
            out << task.factors[i];
        }
        out << '"';
    }
    return out.str();
}

bool normalize_worktodo(const std::string& line, const ecm_stage1::Stage1Options& defaults,
                        const std::string& exe, WorktodoStage1Task& queue, std::string& error) {
    auto& out = queue.input;
    try {
        out.options = defaults;
        out.append_save = true;
        queue.ticket = line;
        if (line.compare(0, 4, "ECM=") == 0 || line.compare(0, 5, "ECM2=") == 0) {
            Ecm2Task task;
            if (!ecm_parse_ecm2_line(line, task, error))
                return false;
            normalize_assignment(task);
            if (!ecm_compute_ecm2_n(task, out.modulus, error))
                return false;
            target_identity(task, out);
            out.b1 = task.B1;
            out.b2 = task.B2;
            if (task.has_sigma) {
                out.options.identity.sigma_fixed = true;
                out.options.identity.fixed_sigma64 = task.sigma;
            }
            (void)ecm_config::unsigned_integer(task.b2_text,
                                               "B2 (integer >=0 for saved Stage2 handoff)");
            out.save_file = resolve_rel_local(
                out.options.persistence.tmp_dir,
                ecm_stage1::mont_format_save_name(out.options.persistence.save_name_pattern,
                                                  out.modulus, out.b1, &out.options));
            queue.stage2_line = saved_stage2_line(task, out.save_file);
            queue.published_line = queue.stage2_line;
            queue.published_line.replace(0, 10, "ECMSTAGE1=");
        } else {
            EcmStage2Task task;
            if (!ecm_parse_stage1_line(line, task, error))
                return false;
            if (!ecm_extract_b1_from_save_name(task.save_name, &out.b1, error) ||
                !ecm_compute_stage2_n(task, out.modulus, error))
                return false;
            target_identity(task, out);
            queue.save_assignment = true;
            if (!stage1_save_construction(task.save_name, out.options, error))
                return false;
            out.save_file = resolve_rel_local(
                exe, ecm_stage1::stage1_prefixed_save(task.save_name,
                                                      ecm_stage1::stage1_save_prefix(out.options)));
            const auto original = resolve_rel_local(exe, task.save_name);
            if (out.save_file != original)
                out.options.persistence.legacy_save_source = original;
            if (!ecm_rewrite_stage2_save(line, out.save_file, queue.published_line, error))
                return false;
            queue.stage2_line = queue.published_line;
            queue.stage2_line.replace(0, 10, "ECMSTAGE2=");
        }
        if (stage1_selection_error(out.options.execution.selection, error) ||
            stage1_task_error(out.options, out.curves, error))
            return false;
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

bool normalize_direct(const ParsedCli& input, const ecm_stage1::Stage1Options& defaults,
                      const std::string& expression, NormalizedStage1Task& out,
                      std::string& error) {
    try {
        if (input.pos.empty())
            throw std::runtime_error("missing B1");
        out.b1 = input.b1;
        out.b2 = input.b2;
        out.curves = input.gpucurves;
        out.expression = expression;
        if (expression.empty())
            throw std::runtime_error("No input number on stdin");
        if (!ecm_parse_expression(expression, out.modulus, &error))
            throw std::runtime_error("key=N value='" + expression + "'; " + error);
        out.options = defaults;
        auto& opt = out.options;
        const auto& selection = opt.execution.selection;
        out.save_file = input.savefilename;
        out.append_save = input.saveappend;
        // Keep exact k*b^n+c metadata when the direct expression supplies it.
        std::smatch match;
        const std::regex form(
            R"(^\s*\(?\s*(?:(\d+)\s*\*\s*)?(\d+)\s*\^\s*(\d+)\s*([+-])\s*(\d+)\s*\)?\s*$)");
        if (std::regex_match(expression, match, form)) {
            const auto b = std::stoull(match[2]), n = std::stoull(match[3]);
            const auto c = std::stoll(match[5]) * (match[4] == "-" ? -1 : 1);
            if (b <= UINT32_MAX && n <= UINT32_MAX && c >= INT32_MIN && c <= INT32_MAX) {
                auto& id = opt.identity;
                id.target_kbnc = true;
                id.handoff_k_str = match[1].matched ? match[1].str() : "1";
                id.handoff_b = uint32_t(b);
                id.handoff_n = uint32_t(n);
                id.handoff_c = int32_t(c);
            }
        }
        if (stage1_selection_error(selection, error) || stage1_task_error(opt, out.curves, error))
            return false;
        return true;
    } catch (const std::exception& e) {
        error = "INPUT[direct] " + std::string(e.what());
        return false;
    }
}

static ecm_stage1::Stage1Options normalize_options(const EcmQueueConfig& cfg,
                                                   const std::string& exe, const std::string& base,
                                                   const std::string& cwd, int worker) {
    ecm_stage1::Stage1Options opt;
    opt.execution.selection = cfg.selection;
    opt.execution.affinity_cpus = cfg.affinity_cpus;
    opt.execution.stage1_threads = cfg.stage1_threads;
    opt.execution.naf_w = cfg.naf_w;
    opt.execution.verbose = cfg.verbose;
    opt.execution.device_index = cfg.device;
    opt.execution.gpu_mul_path = cfg.kernel_mul;
    opt.execution.gpu_sqr_path = cfg.kernel_sqr;
    opt.execution.gpu_add_path = cfg.kernel_add;
    opt.execution.gpu_sub_path = cfg.kernel_sub;
    opt.execution.gpu_special_mult_path = cfg.kernel_special_mult;
    opt.persistence.save_name_pattern = cfg.save_name_pattern;
    opt.persistence.exponent_cache_dir = normalized_cache_dir(cfg.exp_cache, exe, cwd);
    const auto ms = cfg.ckpt_seconds * 1000.0;
    opt.persistence.ckpt_ms = ms <= 0 ? 0
                              : ms >= double(ULONG_MAX)
                                  ? ULONG_MAX
                                  : static_cast<unsigned long>(std::llround(ms));
    opt.persistence.ckpt_backups = static_cast<unsigned>(cfg.ckpt_backups);
    opt.persistence.tmp_dir = resolve_rel_local(exe, cfg.tmp_dir);
    opt.identity.sigma_fixed = cfg.sigma != 0;
    opt.identity.fixed_sigma64 = cfg.sigma;
    stage1_result_settings(opt, cfg, base, worker);
    return opt;
}

bool validate_cli_mode(const ParsedCli& input, std::string& error) {
    const bool queue = input.pos.empty() && !input.show_gpu_info && !input.show_kernels;
    const std::vector<std::string> forbidden = {
        "--curve", "--gpu-param", "--lcm",        "-gpucurves",      "-save",
        "-savea",  "--go",        "--go-backend", "--go-gp-timeout", "--bits"};
    for (const auto& key : forbidden)
        if (queue && input.explicit_options.count(key)) {
            const auto& at = input.explicit_options.at(key);
            error = "INPUT[conflict] argv[" + std::to_string(at.argument) + "] key=" + key +
                    " value='" + at.value +
                    "'; task fields belong to worktodo in queue mode. Remove this "
                    "CLI option and "
                    "edit the task/config instead.";
            return false;
        }
    if (queue && input.sigma_param_claim >= 0) {
        error = "INPUT[conflict] key=sigma; queue mode accepts an unprefixed sigma "
                "only; "
                "curve/param belong to the task";
        return false;
    }
    return true;
}

// Cross-field execution preferences have one rule shared by startup and task
// admission.
static bool execution_preferences_error(const ecm_stage1::Selection& s, std::string& error) {
    if (s.target == ecm_stage1::ExecutionTarget::Gpu &&
        (s.cpu_implementation != ecm_stage1::CpuImplementation::Auto ||
         s.simd_reduction != ecm_stage1::SimdReduction::Auto)) {
        error = "keys=target,impl,reduction; --impl and --reduction require "
                "--target cpu";
        return true;
    }
    if (s.target == ecm_stage1::ExecutionTarget::Cpu &&
        s.simd_reduction != ecm_stage1::SimdReduction::Auto &&
        s.cpu_implementation != ecm_stage1::CpuImplementation::IfmaSimd) {
        error = "keys=impl,reduction; explicit --reduction requires --impl simd";
        return true;
    }
    return false;
}

bool normalize_settings(const ParsedConfig& parsed, const ParsedCli& input, const InputPaths& paths,
                        Stage1Settings& out, std::string& error) {
    if (!validate_cli_mode(input, error))
        return false;
    try {
        auto& cfg = out.config;
        project_stage1_config(parsed, input.worker_index, cfg);
        const bool direct = !input.pos.empty() && !input.show_gpu_info;
        // Typed CLI values overlay configuration once. Presence belongs to the
        // source; no effective value is written back into ParsedCli.
        if (input.has("-v"))
            cfg.verbose = input.verbose;
        else if (direct && !cfg.config_keys.count("verbose"))
            cfg.verbose = false;
        if (input.has("-d"))
            cfg.device = input.gpu_device_index;
        if (input.has("--stage1-threads"))
            cfg.stage1_threads = input.stage1_threads;
        if (input.has("--naf-w"))
            cfg.naf_w = input.naf_w;
        if (input.has("--affinity"))
            cfg.affinity_cpus = input.affinity_cpus;
        if (input.has("--ckpt"))
            cfg.ckpt_seconds = input.ckpt_seconds;
        if (input.has("--exp-cache"))
            cfg.exp_cache = input.exp_cache;
        if (input.has("--mul"))
            cfg.kernel_mul = input.gpu_mul_path;
        if (input.has("--sqr"))
            cfg.kernel_sqr = input.gpu_sqr_path;
        if (input.has("--add"))
            cfg.kernel_add = input.gpu_add_path;
        if (input.has("--sub"))
            cfg.kernel_sub = input.gpu_sub_path;
        if (input.has("--special-mult"))
            cfg.kernel_special_mult = input.gpu_special_mult_path;
        if (input.has("--tpi"))
            cfg.tpi = input.runtime.tpi;
        if (input.has("--wg"))
            cfg.wg_size = input.runtime.wg_size;
        if (input.has("-sigma"))
            cfg.sigma = input.fixed_sigma64;
        auto& s = cfg.selection;
        if (input.has("--curve"))
            s.curve = input.selection.curve;
        if (input.has("--target"))
            s.target = input.selection.target;
        if (input.has("--impl"))
            s.cpu_implementation = input.selection.cpu_implementation;
        if (input.has("--reduction"))
            s.simd_reduction = input.selection.simd_reduction;
        if (input.has("--lcm"))
            s.scalar_construction = input.selection.scalar_construction;
        if (input.has("--gpu-param"))
            s.gpu_param = input.gpu_param_cli;
        if (direct) {
            const bool prefixed = input.sigma_param_claim >= 0;
            if (s.curve == ecm_stage1::CurveModel::Edwards &&
                s.target == ecm_stage1::ExecutionTarget::Gpu &&
                (input.has("--gpu-param") || prefixed || input.print_group_order))
                throw std::runtime_error("GPU Edwards uses an unprefixed sigma; "
                                         "--gpu-param and --go do not apply");
            if (s.target == ecm_stage1::ExecutionTarget::Cpu && input.has("--gpu-param"))
                throw std::runtime_error("--gpu-param requires --target gpu");
            if (s.target == ecm_stage1::ExecutionTarget::Cpu && prefixed &&
                (s.curve != ecm_stage1::CurveModel::Montgomery || input.sigma_param_claim != 0))
                throw std::runtime_error(
                    "CPU accepts a parameterized sigma only for Montgomery PARAM0");
            if (prefixed && !ecm_stage1::resolve_gpu_param(s.gpu_param, input.has("--gpu-param"),
                                                           input.sigma_param_claim, s.gpu_param))
                throw std::runtime_error("conflict between -sigma parametrization and --gpu-param");
        }
        // A query shares projection and overrides, but does not invent a task or
        // impose computation-only capability/combination checks.
        if (!input.show_gpu_info && !input.show_kernels && execution_preferences_error(s, error))
            throw std::runtime_error(error);
        out.options = normalize_options(cfg, paths.exe, paths.config_directory, paths.cwd,
                                        input.worker_index);
        if (input.has("--tmp-dir"))
            out.options.persistence.tmp_dir =
                resolve_rel_local(direct ? paths.cwd : paths.exe,
                                  direct && input.tmp_dir.empty() ? "." : input.tmp_dir);
        else if (direct && !cfg.config_keys.count("tmp_dir"))
            out.options.persistence.tmp_dir = paths.cwd;
        if (direct && out.options.persistence.tmp_dir.empty())
            out.options.persistence.tmp_dir = paths.cwd;
        if (!input.show_gpu_info && s.target == ecm_stage1::ExecutionTarget::Cpu &&
            stage1_task_error(out.options, 0, error))
            throw std::runtime_error(error);
        out.runtime = input.runtime;
        out.runtime.device_index = out.options.execution.device_index;
        out.runtime.tpi = cfg.tpi;
        out.runtime.wg_size = cfg.wg_size;
        if (!input.gp_bin_path.empty())
            out.runtime.gp_bin = input.gp_bin_path;
        return true;
    } catch (const std::exception& e) {
        error = "INPUT[normalize] " + parsed.path +
                " worker=" + std::to_string(input.worker_index) + " " + e.what();
        return false;
    }
}

using Stage1Options = ecm_stage1::Stage1Options;

static bool is_absolute_path_local(const std::string& p) {
#ifdef _WIN32
    if (p.size() >= 2 && std::isalpha((unsigned char)p[0]) && p[1] == ':')
        return true;
    if (p.size() >= 1 && (p[0] == '\\' || p[0] == '/'))
        return true;
    return false;
#else
    return !p.empty() && p[0] == '/';
#endif
}

std::string normalize_path_local(std::filesystem::path path) {
    if (path.empty())
        return {};
    path = path.lexically_normal();
    // A directory's trailing separator is unnecessary, except at a root.
    if (path.filename().empty() && path != path.root_path())
        path = path.parent_path();
    return path.make_preferred().string();
}

std::string resolve_rel_local(const std::string& base, const std::string& p) {
    if (p.empty())
        return {};
    return normalize_path_local(is_absolute_path_local(p) ? std::filesystem::path(p)
                                                          : std::filesystem::path(base) / p);
}

std::string get_exe_dir_local() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::string p(buf, n);
        const std::size_t slash = p.find_last_of("\\/");
        if (slash != std::string::npos)
            return p.substr(0, slash);
    }
    return "";
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p(buf);
        const std::size_t slash = p.find_last_of('/');
        if (slash != std::string::npos)
            return p.substr(0, slash);
    }
    return "";
#endif
}

bool stage1_save_construction(const std::string& name, Stage1Options& opt, std::string& error) {
    const auto leaf = ecm_stage1::lower_ascii(std::filesystem::path(name).filename().string());
    const auto prefix = leaf.substr(0, leaf.find('_'));
    if (prefix == "e" || prefix == "ed") {
        opt.execution.selection.curve = ecm_stage1::CurveModel::Edwards;
        opt.execution.selection.scalar_construction = ecm_stage1::ScalarConstruction::PrimePowerLcm;
    } else if (prefix == "m0" || prefix == "m2" || prefix == "m3") {
        opt.execution.selection.curve = ecm_stage1::CurveModel::Montgomery;
        opt.execution.selection.gpu_param = static_cast<unsigned>(prefix[1] - '0');
        if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
            opt.execution.selection.gpu_param != 0) {
            error = "save prefix selects Montgomery PARAM" + prefix.substr(1) +
                    ", which requires target=gpu";
            return false;
        }
    }
    return true;
}

bool stage1_selection_error(const ecm_stage1::Selection& selection, std::string& error) {
    if (!ecm_stage1::valid_route(selection, ECM_HAS_CUDA_STAGE1 != 0)) {
        error = "GPU Edwards requires the CUDA backend";
        return true;
    }
    if (selection.curve == ecm_stage1::CurveModel::Edwards &&
        selection.scalar_construction != ecm_stage1::ScalarConstruction::PrimePowerLcm) {
        error = "Edwards uses its fixed 48*lcm(1..B1) scalar; --lcm 12 is unsupported";
        return true;
    }
    if (execution_preferences_error(selection, error))
        return true;
    if (selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        selection.curve == ecm_stage1::CurveModel::Edwards)
        return false;
    if (!ecm_stage1::valid_gpu_param(selection.gpu_param)) {
        error = "invalid gpu_param (expected 0, 2 or 3)";
        return true;
    }
    if (const char* reason = ecm_backend_stage1_param_error(selection.gpu_param)) {
        error = std::string(ecm_backend_name()) + ": " + reason;
        return true;
    }
    return false;
}

bool stage1_task_error(const Stage1Options& opt, uint32_t curves, std::string& error) {
    if (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        opt.execution.naf_w != 0 && (opt.execution.naf_w < 3 || opt.execution.naf_w > 12)) {
        error = "key=naf_w value='" + std::to_string(opt.execution.naf_w) +
                "'; CPU requires 0 (auto) or [3,12]";
        return true;
    }
    if (opt.identity.sigma_fixed && curves != 0 &&
        static_cast<uint64_t>(curves - 1) > UINT64_MAX - opt.identity.fixed_sigma64) {
        error = "sigma + curve index exceeds 2^64-1";
        return true;
    }
    if (opt.execution.selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards ||
        opt.execution.selection.gpu_param == 0 || !opt.identity.sigma_fixed)
        return false;
    if (opt.identity.fixed_sigma64 > 0xFFFFFFFFull) {
        error = "GPU batch sigma exceeds 2^32-1";
        return true;
    }
    if (curves != 0 && curves > 0x100000000ull - opt.identity.fixed_sigma64) {
        error = "GPU batch sigma + curves must be <= 2^32";
        return true;
    }
    return false;
}

void stage1_result_settings(Stage1Options& opt, const EcmQueueConfig& cfg, const std::string& base,
                            int worker) {
    opt.identity.result_user = cfg.result_user;
    opt.identity.result_computer = cfg.result_computer;
    opt.persistence.results_file =
        (std::filesystem::path(base) /
         (cfg.has_results ? cfg.results : ecm_config::worker_file(cfg.results, worker)))
            .string();
    opt.persistence.submission_file =
        (std::filesystem::path(base) /
         (cfg.has_submission ? cfg.submission : ecm_config::worker_file(cfg.submission, worker)))
            .string();
}
} // namespace ecm_stage1_app
