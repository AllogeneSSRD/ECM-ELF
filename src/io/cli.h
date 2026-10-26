#pragma once
#include "ecm/application/configuration/runtime.h"
#include "ecm/core/stage1/selection.h"
#include "ecm/service/diagnostics/order_analysis.h"
#include <map>
#include <vector>

namespace ecm_stage1_app {
struct ParsedCli {
    // Value storage is meaningful only with explicit_options presence until
    // normalization. Worker/mode are the minimal bootstrap policy.
    bool verbose = false;
    ecm_stage1::Selection selection{};
    /* Both CPU curve models share the implementation choice and thread count. */
    uint32_t stage1_threads = 0;         // 0 = auto
    int naf_w = 0;                       // Edwards NAF window; 0 = built-in default
    std::vector<unsigned> affinity_cpus; // Parsed --affinity indices.
    uint32_t gpucurves = 0;
    double ckpt_seconds = -1.0;
    bool sigma_fixed = false;
    uint64_t fixed_sigma64 = 0;
    int gpu_device_index = 0;
    int gpu_param_cli = 0;      /* --gpu-param 0|2|3 */
    int sigma_param_claim = -1; /* "-sigma i:s" -> i (gmp-ecm: selects -param i) */
    bool print_group_order = false;
    ecm_diagnostics::Settings go_settings;
    std::string savefilename;
    bool saveappend = false;
    std::string gp_bin_path;
    std::string gpu_mul_path;
    std::string gpu_sqr_path;
    std::string gpu_add_path;
    std::string gpu_sub_path;
    std::string gpu_special_mult_path;
    bool show_kernels = false;
    bool show_gpu_info = false; // --gpu-info: report tiers/occupancy and exit
    uint32_t gpu_info_bits = 0; // --bits N: only the tier that N would pick (0 = all)
    std::string ini_path;
    int worker_index = 1;        // --worker N: ini + worktodo section (see the flag's parse site)
    std::string tmp_dir;         // 本地 stage-1 落盘目录
    std::string p95_dir_ignored; // 兼容旧脚本: 现在由 ecm_p95feeder 处理
    uintptr_t stop_handle = 0;
    // Positional text is retained for diagnostics; bounds are converted once.
    std::vector<std::string> pos;
    double b1 = 0, b2 = 0;
    EcmRuntimeConfig runtime;
    std::string exp_cache;
    bool version = false;
    struct Source {
        int argument;
        std::string spelling, value;
    };
    std::map<std::string, Source> explicit_options;

    bool has(const std::string& key) const { return explicit_options.count(key) != 0; }
};

int parse_cli(int, char**, ParsedCli&);
bool wants_usage(int, char**);
void print_usage(const char*);
} // namespace ecm_stage1_app
