#include "stage1_input.h"
#include "../configuration/ecm_queue_config.h"
#include "ecm/service/execution/legacy_gpu.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
namespace ecm_stage1_app {
using Stage1Options=ecm_stage1::Stage1Options;
static bool is_absolute_path_local(const std::string &p) {
#ifdef _WIN32
    if (p.size() >= 2 && std::isalpha((unsigned char)p[0]) && p[1] == ':') return true;
    if (p.size() >= 1 && (p[0] == '\\' || p[0] == '/')) return true;
    return false;
#else
    return !p.empty() && p[0] == '/';
#endif
}

std::string normalize_path_local(std::filesystem::path path) {
    if(path.empty())return {};
    path=path.lexically_normal();
    // A directory's trailing separator is unnecessary, except at a root.
    if(path.filename().empty() && path!=path.root_path())path=path.parent_path();
    return path.make_preferred().string();
}

std::string resolve_rel_local(const std::string &base, const std::string &p) {
    if(p.empty())return {};
    return normalize_path_local(is_absolute_path_local(p)?std::filesystem::path(p):
                                std::filesystem::path(base)/p);
}

std::string get_exe_dir_local() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::string p(buf, n);
        const std::size_t slash = p.find_last_of("\\/");
        if (slash != std::string::npos) return p.substr(0, slash);
    }
    return "";
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p(buf);
        const std::size_t slash = p.find_last_of('/');
        if (slash != std::string::npos) return p.substr(0, slash);
    }
    return "";
#endif
}

bool stage1_save_construction(const std::string &name,Stage1Options &opt,std::string &error) {
    const auto leaf=ecm_stage1::lower_ascii(std::filesystem::path(name).filename().string());
    const auto prefix=leaf.substr(0,leaf.find('_'));
    if(prefix=="e" || prefix=="ed") {
        opt.execution.selection.curve=ecm_stage1::CurveModel::Edwards;
        opt.execution.selection.scalar_construction=ecm_stage1::ScalarConstruction::PrimePowerLcm;
    } else if(prefix=="m0" || prefix=="m2" || prefix=="m3") {
        opt.execution.selection.curve=ecm_stage1::CurveModel::Montgomery;
        opt.execution.selection.gpu_param=static_cast<unsigned>(prefix[1]-'0');
        if(opt.execution.selection.target==ecm_stage1::ExecutionTarget::Cpu && opt.execution.selection.gpu_param!=0) {
            error="save prefix selects Montgomery PARAM"+prefix.substr(1)+", which requires target=gpu";
            return false;
        }
    }
    return true;
}

bool stage1_selection_error(const ecm_stage1::Selection &selection,
                                   std::string &error) {
    if (!ecm_stage1::valid_route(selection, ECM_HAS_CUDA_STAGE1 != 0)) {
        error = "GPU Edwards requires the CUDA backend";
        return true;
    }
    if (selection.curve == ecm_stage1::CurveModel::Edwards &&
        selection.scalar_construction != ecm_stage1::ScalarConstruction::PrimePowerLcm) {
        error = "Edwards uses its fixed 48*lcm(1..B1) scalar; --lcm 12 is unsupported";
        return true;
    }
    if (selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        (selection.cpu_implementation != ecm_stage1::CpuImplementation::Auto ||
         selection.simd_reduction != ecm_stage1::SimdReduction::Auto)) {
        error = "--impl and --reduction require --target cpu";
        return true;
    }
    if (selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        selection.simd_reduction != ecm_stage1::SimdReduction::Auto &&
        selection.cpu_implementation != ecm_stage1::CpuImplementation::IfmaSimd) {
        error = "explicit --reduction requires --impl simd";
        return true;
    }
    if (selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        selection.curve == ecm_stage1::CurveModel::Edwards) return false;
    if (!ecm_stage1::valid_gpu_param(selection.gpu_param)) {
        error = "invalid gpu_param (expected 0, 2 or 3)";
        return true;
    }
    if (const char *reason = ecm_backend_stage1_param_error(selection.gpu_param)) {
        error = std::string(ecm_backend_name()) + ": " + reason;
        return true;
    }
    return false;
}

bool stage1_task_error(const Stage1Options &opt, uint32_t curves,
                              std::string &error) {
    if (opt.identity.sigma_fixed && curves != 0 &&
        static_cast<uint64_t>(curves - 1) > UINT64_MAX - opt.identity.fixed_sigma64) {
        error = "sigma + curve index exceeds 2^64-1";
        return true;
    }
    if (opt.execution.selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        opt.execution.selection.curve == ecm_stage1::CurveModel::Edwards ||
        opt.execution.selection.gpu_param == 0 || !opt.identity.sigma_fixed) return false;
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

std::vector<unsigned> parse_affinity_spec(const std::string &spec) {
    std::vector<unsigned> out;
    std::string s;
    for (char ch : spec) if (ch != ' ' && ch != '\t' && ch != '"') s.push_back(ch);
    if (s.empty() || s == "none" || s == "auto") return out;

    /* One index, strictly: "12" ok, "0-7" / "1x" / "" rejected.  The old version
       used plain std::stol(), which happily parses "0-7" as 0 and silently pinned
       every worker to CPU 0 -- a 5.5x slowdown that looks like a hardware problem. */
    auto one = [](const std::string &t, long *v) -> bool {
        if (t.empty()) return false;
        size_t used = 0;
        try { *v = std::stol(t, &used); } catch (...) { return false; }
        return used == t.size();
    };

    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            const size_t dash = tok.find('-');
            long lo = 0, hi = 0;
            bool ok;
            if (dash == std::string::npos) {                 /* single index */
                ok = one(tok, &lo);
                hi = lo;
            } else {                                          /* inclusive range a-b */
                ok = one(tok.substr(0, dash), &lo) && one(tok.substr(dash + 1), &hi) && hi >= lo;
            }
            if (!ok || lo < 0 || hi > 1023) {
                std::cerr << "Affinity: ignoring invalid entry '" << tok
                          << "' (expected <cpu>, <lo>-<hi>, or a comma separated list)"
                          << std::endl;
            } else {
                for (long v = lo; v <= hi; v++) out.push_back((unsigned)v);
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (out.size() > 1024) {
        std::cerr << "Affinity: list too long, ignoring it" << std::endl;
        out.clear();
    }
    return out;
}

void stage1_result_settings(Stage1Options &opt,const EcmQueueConfig &cfg,
                                   const std::string &base,int worker) {
    opt.identity.result_user=cfg.result_user;opt.identity.result_computer=cfg.result_computer;
    opt.persistence.results_file=(std::filesystem::path(base)/(cfg.has_results?cfg.results:ecm_config::worker_file(cfg.results,worker))).string();
    opt.persistence.submission_file=(std::filesystem::path(base)/(cfg.has_submission?cfg.submission:ecm_config::worker_file(cfg.submission,worker))).string();
}
} // namespace ecm_stage1_app
