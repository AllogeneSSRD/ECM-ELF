#include "stage1_input.h"
#include "group_order.h"
#include "queue_source.h"
using ecm_stage1_app::normalize_path_local;
using ecm_stage1_app::resolve_rel_local;
using ecm_stage1_app::get_exe_dir_local;
using ecm_stage1_app::stage1_save_construction;
using ecm_stage1_app::stage1_selection_error;
using ecm_stage1_app::stage1_task_error;
using ecm_stage1_app::parse_affinity_spec;
using ecm_stage1_app::stage1_result_settings;
using ecm_stage1_app::get_gp_executable;
using ecm_stage1_app::resolve_gp_path;
using ecm_stage1_app::gp_executable_exists;
using ecm_stage1_app::build_primes_up_to_B1;
using ecm_stage1_app::print_group_order_factor;
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>     /* _mkdir for the local save directory */
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>   /* __rdtsc() for the Prime95-style sigma entropy mix */
#endif

#include <gmp.h>

#include "ecm/service/execution/legacy_gpu.h"           /* GPU backend seam (OpenCL or CUDA glue) */
#include "../../storage/save/ecm_save.h"
#include "../../storage/checkpoint/ecm_checkpoint.h"         /* opencl_ecm_set_work_dir */
#include "../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "../../storage/checkpoint/ecm_stage1_legacy_gpu.h"
#include "ecm/core/stage1/selection.h"
#include "../../service/stage1/execution_contract.h"
#include "ecm/application/configuration/runtime.h"
#include "../../service/execution/ecm.h"
#include "../../service/execution/cgbn_stage1.h"            /* gpu_pick_random_sigma / gpu_compute_batch_d */
#include "ecm/platform/logging.h"
#include "ecm/platform/program_version.h"
#include "../configuration/ecm_queue_config.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "../../storage/transfer/ecm_stage2_transfer.h"
#include "../../storage/queue/ecm_worker_list.h"
#include "../configuration/ecm_expr.h"              /* shared exact integer expression parser */
#include "../../storage/transfer/p95_transfer.h"           /* finished task -> Prime95 worktodo.add (D4/§13) */
#include "ecm/core/stage1/scalar.h"         /* s = torsion * lcm(1..B1), shared product tree */
#include "../../storage/cache/ecm_stage1_exp_cache.h"   /* validated on-disk cache for that value */
#include <random>
#include <algorithm>
#include "../../storage/save/ecm_edwards_io.h"
#include <regex>
#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_HAS_CUDA_STAGE1
#define ECM_HAS_CUDA_STAGE1 0
#endif
// Application assembly still selects the CPU default NAF window.
#include "../../backend/cpu/stage1/edwards/ecm_edwards_cpu.h"

#include "../../service/stage1/execution.h"
#include "../../service/stage1/preparation.h"
using ecm_stage1::fixed_sigma_batch_exceeds_prime95_range;
using ecm_stage1::ensure_dir;
using ecm_stage1::mont_format_save_name;
using ecm_stage1::stage1_save_prefix;
using ecm_stage1::stage1_prefixed_save;

using ecm_stage1::Stage1Options;
using ecm_stage1::ExecutionOptions;
using ecm_stage1::ExecutionResult;
using ecm_stage1::TaskStatus;
using ecm_stage1::FailureSource;
using ecm_stage1::run_stage1_task;
using ecm_stage1::stage1_publish_factor;

static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}





// Parse a 64-bit sigma (for the Edwards Atkin-Morain path). Optional "param:"
// prefix is accepted and ignored for compatibility with -sigma parsing.
// 64-bit sigma parser.
//
// gmp-ecm accepts two forms:
//   -sigma s        s is the parameter used to compute the curve coefficients
//   -sigma i:s      the same, AND i selects the parametrization (-param i)
// The prefix is therefore a parametrization claim, not decoration: "-sigma 3:s" together with
// "--gpu-param 0" is a contradiction and gmp-ecm rejects it with
//   "Error, conflict between -sigma and -param arguments".
// We keep the prefix so that the CLI can apply the same rule (see the caller), instead of
// silently letting --gpu-param win.
static bool parse_sigma64_arg(const std::string &arg, uint64_t *sigma_out, int *param_out) {
    std::string s = arg;
    if (param_out != nullptr) { *param_out = -1; }
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
        if (param_out != nullptr) {
            const std::string prefix = s.substr(0, colon);
            try {
                *param_out = std::stoi(prefix);
            } catch (...) {
                return false;             /* "-sigma x:..." with a non-numeric prefix */
            }
        }
        s = s.substr(colon + 1);
    }
    try {
        unsigned long long v = std::stoull(s);
        if (v == 0) {
            return false;
        }
        *sigma_out = (uint64_t)v;
        return true;
    } catch (...) {
        return false;
    }
}

static void print_ecm_usage(const char *prog) {
    const char *name = prog;
    if (name != nullptr) {
        const char *slash = std::strrchr(name, '\\');
        const char *slash2 = std::strrchr(name, '/');
        if (slash2 != nullptr && (slash == nullptr || slash2 > slash)) {
            slash = slash2;
        }
        if (slash != nullptr && slash[1] != '\0') {
            name = slash + 1;
        }
    } else {
        name = "ecm";
    }

    std::cout << ecm_backend_name() << " ECM stage-1 driver\n\n"
              << "Usage:\n"
              << "  echo '<N>' | " << name << " [options] B1 [B2]\n\n"
              << "Input:\n"
              << "  N is read from stdin as a decimal integer or expression\n"
              << "  (e.g. '(2^991-1)', '0xdeadbeef'). Whitespace is ignored.\n\n"
              << "Positional:\n"
              << "  B1              Stage-1 bound (required for meaningful runs)\n"
              << "  B2              Stage-2 bound (optional; 0 disables stage 2)\n\n"
              << "Options:\n"
              << "  --curve <m>          mont|ed (Montgomery or Edwards; default mont)\n"
              << "  --target <t>         cpu|gpu (default gpu; GPU Edwards requires CUDA)\n"
              << "  --impl <m>           auto|simd|scalar (CPU only). The SIMD path\n"
              << "                       requires AVX512-IFMA; auto selects by batch size.\n"
              << "  --reduction <m>      auto|montgomery|mersenne-fold (CPU SIMD). auto\n"
              << "                       uses the Mersenne fold when N = 2^k-1 (half the madds\n"
              << "                       per multiply); explicit values require --impl simd.\n"
              << "  --stage1-threads <n> CPU stage-1 worker threads (0=auto, 1=serial). One task\n"
              << "                       is an 8-curve SIMD batch (simd backend) or one curve\n"
              << "  --exp-cache <dir>    cache s = torsion*lcm(1..B1) on disk (default: the exe\n"
              << "                       directory; \"off\" disables).  B1 = 260e6 costs ~10 s to\n"
              << "                       build and ~0.3 s to load from a validated cache.\n"
              << "  --lcm <1|12>         Stage1 scalar multiplier for lcm(1..B1). 1 is the\n"
              << "                       default; 12 is Prime95-style (use when Prime95 runs\n"
              << "                       stage 2 on our point, see doc section 16.7)\n"
              << "  --naf-w <w>          Edwards NAF window (0 = CPU 12 / GPU 6); dictionary = 2^(w-2)\n"
              << "  --affinity <list>    Pin worker t to logical CPU list[t %% len], e.g. 0,1,2,3\n"
              << "                       or 0-7.  On hybrid CPUs (Zen5 + Zen5c, P+E cores, ...)\n"
              << "                       this can hurt: measure first (doc section 13.4)\n"
              << "  --tmp-dir <dir>      Local directory for text saves and one batch .ckpt\n"
              << "                       (default: current dir; ecm.exe never writes to p95)\n"
              << "  -gpucurves <n>       Number of ECM curves per launch (Edwards: total curves)\n"
              << "  --ckpt <sec>         Mid-stage-1 checkpoint interval in seconds, for every\n"
              << "                       implementation that has one (GPU, CPU Edwards, CPU Montgomery;\n"
              << "                       default: 600, 0 = no autosave -- Ctrl+C still saves)\n"
              << "  -d <index>           " << ecm_backend_name()
              << " device index (default: 0)\n"
              << "  --gpu-param <0|2|3>  GPU curve parametrization (ini: gpu_param):\n"
              << "                       0 = Suyama param0 (same curves as CPU Montgomery,\n"
              << "                           Z/12 torsion, param0 save file) -- CUDA/CGBN only;\n"
              << "                       2 = gmp-ecm batch 2 / PARAM=2 -- CUDA/CGBN only;\n"
              << "                       3 = gmp-ecm batch/PARAM=3 (default, historical GPU path)\n"
              << "  -sigma <value>       Fixed curve sigma (1..2^64-1; optional param:sigma prefix).\n"
              << "                       GPU PARAM2/3 need sigma+curves <= 2^32.\n"
              << "  -v                   Verbose output\n"
              << "  -save <file>         Append factorization lines to file\n"
              << "  -savea <file>        Same as -save (append mode)\n"
              << "  --go                 Print optional group order diagnostics (GP preferred, CPU fallback)\n"
              << "  --go-backend <mode>  auto (default), gp or native; diagnosis never changes task status\n"
              << "  --go-gp-timeout <s>  GP diagnostic deadline in seconds (1..600; default 30)\n"
              << "  --gp <path>          Path to gp executable (default: gp on PATH)\n"
              << "  --mul <path>         Montgomery mul kernel path (OpenCL only)\n"
              << "  --sqr <path>         Montgomery sqr kernel path (OpenCL only)\n"
              << "  --add <path>         Modular add kernel path (OpenCL only)\n"
              << "  --sub <path>         Modular sub kernel path (OpenCL only)\n"
              << "  --special-mult <path>  special_mult (R=2^32) kernel path (OpenCL only)\n"
              << "  --showkernel         List available " << ecm_backend_name()
              << " kernel paths and exit\n"
              << "  --gpu-info           Report the device this build would use and, per kernel\n"
              << "                       tier, the curves a full GPU needs; no curve is run and\n"
              << "                       no file is written. Honours -d, --curve and\n"
              << "                       --gpu-param (Montgomery only)\n"
              << "  --bits <n>           With --gpu-info: only the tier the kernel would pick\n"
              << "                       for an N of n bits (default: every tier of this build)\n"
              << "  -h, --help           Show this help and exit\n"
              << "\nRuntime tuning (kebab-case; replaces former environment variables):\n"
              << " Device / operators:\n"
              << "  --tpi <1..32>              Threads per instance (default 8)\n"
              << "  --force-normalize <0|1>    Stage1 force-normalize path\n"
              << "  --addsub-fused-unroll <1|2>  add/sub fused-unroll mode\n"
              << "  --local                    Use LDS-based kernel (reduce scratch spill at large bits)\n"
              << "  --wg <N>                   Explicit work-group size (0=auto; 1,4,8,16,32...)\n"
              << " Kernel source / cache:\n"
              << "  --kernel-root <dir>        Kernel source root\n"
              << "  --kernel-cache-dir <dir>   OpenCL binary cache directory (OpenCL only)\n"
              << "  --no-kernel-cache          Disable kernel binary cache (OpenCL only)\n"
              << "  --kernel-cache-verbose     Verbose cache hit/miss logging (OpenCL only)\n"
              << "  --compile-verbose          Verbose compile timing (OpenCL only)\n"
              << " Logging / debug / verify:\n"
              << "  --no-log-timestamp         Disable log timestamps (default on)\n"
              << "  --gpu-dump [--gpu-dump-file <f>]      Dump GPU state to CSV\n"
              << "  --profile-ops [--profile-ops-file <f>]  Operator-count profiling\n"
              << "  --sync-each-batch          Synchronize after each batch\n"
              << "  --verify-gpu [--verify-gpu-strict]   CPU cross-check GPU results\n\n"
              << "Queue manager (run with no positional B1/B2):\n"
              << "  --config <path>                Config file (default: <exe dir>/ecm.toml)\n"
              << "  --worker <n>               Which worker this process is (1-based, default 1).\n"
              << "                             Selects [workers.N.*] TOML overrides and the\n"
              << "                             [Worker #N] worktodo queue section.\n\n"
              << "Examples:\n"
              << "  echo '(2^991-1)' | " << name << " -v --go --curve mont --target gpu -gpucurves 384 1e6 0\n"
              << "  echo '(2^421-1)' | " << name << " --curve mont --target gpu -gpucurves 256 -d 1 1e5 0\n"
              << "  echo '(2^4003-1)' | " << name << " --curve mont --target gpu -gpucurves 384 --add asm_b32 1e6 0\n"
              << "  " << name << " --showkernel\n"
              << "  " << name << " --config ecm.toml --worker 2      (queue manager, worker #2)\n\n"
              << "Add/sub path names (for --add / --sub): default, fused, fused_unroll,\n"
              << "  asm/unroll_128b, asm/unroll_192b, asm/unroll_256b, asm/unroll_384b,\n"
              << "  asm/unroll_512b, asm/unroll_4096b (legacy: asm_b16, asm_b32, fused_unroll_b16).\n"
              << "Run with --showkernel for full Montgomery and add/sub path lists.\n";
}

static bool ecm_wants_usage(int argc, char **argv) {
    // No explicit -h/--help: let main() dispatch (no args → queue manager mode,
    // positional B1 → single-run CLI mode). Only explicit help flags print usage.
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help" || a == "/?") {
            return true;
        }
    }
    return false;
}

static const std::string &stage1_build_identity() {
    static const std::string identity=[] {
        std::ifstream in(std::filesystem::path(get_exe_dir_local())/"ecm_stage1_build_identity.txt");
        std::string value;std::getline(in,value);
        return value.size()<=128 && !value.empty() && value.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz.-")==value.npos?value:std::string(ECM_BUILD_ID);
    }();return identity;
}

// Application assembly of the existing execution Adapter; no registry/select layer.
static const ecm_stage1::TaskDependencies stage1_tasks{
    ecm_stage1::execute_stage1,stage1_build_identity,edwards_get_naf_w,ecm_stage1::defer_gpu_checkpoint_validation};

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
static int print_gpu_info(const char *ini_used, int worker, int device_index,
                          int gpu_param, bool edwards, uint32_t want_bits) {
    ecm_backend_gpu_info info;
    const char *err = nullptr;
    const int rc = ecm_backend_query_gpu(device_index, gpu_param, edwards ? 1 : 0,
                                         want_bits, &info, &err);

    if (rc == ECM_BACKEND_QUERY_NOT_APPLICABLE) {
        fprintf(stdout, "gpu_info=not_applicable\nbackend=%s\nreason=%s\n",
                ecm_backend_name(), (err != nullptr) ? err : "not applicable");
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
        const ecm_backend_tier &t = info.tiers[i];
        fprintf(stdout,
                "tier bits=%u tpb=%u tpi=%u ipb=%u blocks_per_sm=%u blocks_min=%u "
                "curves_min=%llu blocks_wave=%u curves_wave=%llu",
                t.bits, t.tpb, t.tpi, t.ipb, t.blocks_per_sm, t.blocks_min,
                (unsigned long long)t.curves_min, t.blocks_wave,
                (unsigned long long)t.curves_wave);
        if (info.fold != 0) {
            fprintf(stdout, " fold_blocks_min=%u fold_curves_min=%llu",
                    t.fold_blocks_min, (unsigned long long)t.fold_curves_min);
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv){
    for (int i=1;i<argc;++i) if (std::string(argv[i])=="--gui-queue-protocol") {
        std::puts("ECM_QUEUE_OWNER=1"); return 0;
    }
    if (ecm_wants_usage(argc, argv)) {
        print_ecm_usage(argv[0]);
        return 0;
    }

    // Timestamped-stream install is deferred until after argv parsing / config fill,
    // so --no-log-timestamp takes effect.
    bool verbose = false;
    ecm_stage1::Selection selection{};
    /* Both CPU curve models share the implementation choice and thread count. */
    uint32_t stage1_threads = ecm_config::defaults::stage1_stage1_threads;      // 0 = auto
    int  naf_w = ecm_config::defaults::stage1_naf_w;                   // Edwards NAF window; 0 = built-in default
    std::string affinity_spec;        // --affinity, same syntax as the ini key
    uint32_t gpucurves = 0;
    double ckpt_seconds = -1.0;
    bool ckpt_set = false;
    bool sigma_fixed = false;
    uint64_t fixed_sigma64 = 0;
    int gpu_device_index = ecm_config::defaults::stage1_device;
    bool device_set = false;        /* -d given: wins over the ini's `device` */
    int gpu_param_cli = ecm_config::defaults::stage1_gpu_param;          /* --gpu-param 0|2|3 */
    bool gpu_param_set = false;
    int sigma_param_claim = -1;     /* "-sigma i:s" -> i (gmp-ecm: selects -param i) */
    bool sigma_param_set = false;
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
    bool show_gpu_info = false;           // --gpu-info: report tiers/occupancy and exit
    bool curve_set = false;
    uint32_t gpu_info_bits = 0;           // --bits N: only the tier that N would pick (0 = all)
    std::string ini_path;
    int worker_index = 1;      // --worker N: ini + worktodo section (see the flag's parse site)
    std::string tmp_dir;                  // 本地 stage-1 落盘目录
    std::string p95_dir_ignored;          // 兼容旧脚本: 现在由 ecm_p95feeder 处理
    uintptr_t stop_handle = 0;
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a=="--version"){std::cout<<ecm_program::name<<' '<<ecm_program::version<<std::endl;return 0;}
        if(a == "-v") { verbose = true; continue; }
        if(a == "--affinity" && i+1<argc){
            affinity_spec = argv[++i];
            continue;
        }
        /* Each option selects one mathematical or execution axis. */
        if(a == "--stage1-threads" && i+1<argc){
            try { stage1_threads = (uint32_t)std::stoul(argv[++i]); }
            catch (...) { std::cerr << "Invalid --stage1-threads value, expected >= 0" << std::endl; return 1; }
            continue;
        }
        if(a == "--impl" && i+1<argc){
            if (!ecm_stage1::parse_cpu_implementation(argv[++i], selection.cpu_implementation)) {
                std::cerr << "Invalid --impl, expected auto|simd|scalar" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "--reduction" && i+1<argc){
            if (!ecm_stage1::parse_simd_reduction(argv[++i], selection.simd_reduction)) {
                std::cerr << "Invalid --reduction, expected auto|montgomery|mersenne-fold" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "--naf-w" && i+1<argc){
            try { naf_w = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Invalid --naf-w value, expected integer" << std::endl; return 1; }
            continue;
        }
        if(a == "--lcm" && i+1<argc){
            if (!ecm_stage1::parse_scalar_construction(argv[++i], selection.scalar_construction)) {
                std::cerr << "Invalid --lcm, expected 1|12" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "--curve" && i+1<argc){
            if (!ecm_stage1::parse_curve_model(argv[++i], selection.curve)) {
                std::cerr << "Invalid --curve, expected mont|ed" << std::endl;
                return 1;
            }
            curve_set = true;
            continue;
        }
        if(a == "--target" && i+1<argc){
            if (!ecm_stage1::parse_execution_target(argv[++i], selection.target)) {
                std::cerr << "Invalid --target, expected cpu|gpu" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "-gpucurves" && i+1<argc){ gpucurves = (uint32_t)std::stoul(argv[++i]); continue; }
        if((a == "--ckpt" || a == "-gpuckpt") && i+1<argc){
            try {
                ckpt_seconds = std::stod(argv[++i]);
                ckpt_set = true;
            } catch (...) {
                std::cerr << "Invalid --ckpt value, expected number of seconds" << std::endl;
                return 1;
            }
            continue;
        }
        // Inherited manual-reset event created by ecm_gui for immediate graceful stop.
        if(a == "--stage1-stop-handle" && i+1<argc){
            try { stop_handle = (uintptr_t)std::stoull(argv[++i]); }
            catch (...) { std::cerr << "Invalid Stage1 stop handle" << std::endl; return 1; }
            continue;
        }
        if(a == "--exp-cache" && i+1<argc){
            ecm_exp_cache_set_dir(argv[++i]);
            continue;
        }
        if(a == "--gpu-param" && i+1<argc){
            try {
                gpu_param_cli = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --gpu-param value, expected 0, 2 or 3" << std::endl;
                return 1;
            }
            if (!ecm_stage1::valid_gpu_param(gpu_param_cli)) {
                std::cerr << "Invalid --gpu-param value, expected 0 (Suyama param0), "
                             "2 (gmp-ecm batch 2 / 6-torsion) or 3 (gmp-ecm batch)" << std::endl;
                return 1;
            }
            gpu_param_set = true;
            continue;
        }
        if(a == "-d" && i+1<argc){
            try {
                gpu_device_index = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid -d value, expected integer device index" << std::endl;
                return 1;
            }
            if (gpu_device_index < 0) {
                std::cerr << "Invalid -d value, expected >= 0" << std::endl;
                return 1;
            }
            device_set = true;
            continue;
        }
        if((a == "-sigma" || a == "--sigma") && i+1<argc){
            int sigma_prefix = -1;                 /* "-sigma i:s" claims -param i */
            if(!parse_sigma64_arg(argv[++i], &fixed_sigma64, &sigma_prefix)){
                std::cerr << "Invalid -sigma value (need 1..2^64-1, optional param: prefix)" << std::endl;
                return 1;
            }
            sigma_fixed = true;
            /* Like gmp-ecm, "-sigma i:s" also selects the parametrization.  An explicit
               --gpu-param that disagrees is a contradiction, not a precedence question. */
            if (sigma_prefix >= 0) {
                if (ecm_stage1::valid_gpu_param(sigma_prefix)) {
                    sigma_param_claim = sigma_prefix;
                    sigma_param_set = true;
                } else {
                    std::cerr << "Error, -sigma " << sigma_prefix
                              << ":... selects an unsupported parametrization "
                                 "(this build knows -param 0, 2 and 3)" << std::endl;
                    return 1;
                }
            }
            continue;
        }
        if(a == "-save" && i+1<argc) {
            savefilename = argv[++i];
            saveappend = false;
            continue;
        }
        if(a == "-savea" && i+1<argc) {
            savefilename = argv[++i];
            saveappend = true;
            continue;
        }
        if(a == "--go") {
            print_group_order = true;
            continue;
        }
        if(a == "--go-backend") {
            if(i+1>=argc){std::cerr<<"Error, --go-backend requires auto, gp or native\n";return 1;}
            try{go_settings.backend=ecm_diagnostics::parse_backend(argv[++i]);}
            catch(const std::exception &e){std::cerr<<"Error, "<<e.what()<<'\n';return 1;}
            continue;
        }
        if(a == "--go-gp-timeout") {
            if(i+1>=argc){std::cerr<<"Error, --go-gp-timeout requires seconds (1..600)\n";return 1;}
            const std::string value=argv[++i];
            if(value.empty() || value.size()>3 || value.find_first_not_of("0123456789")!=value.npos ||
               std::strtoul(value.c_str(),nullptr,10)<1 || std::strtoul(value.c_str(),nullptr,10)>600){
                std::cerr<<"Error, --go-gp-timeout must be 1..600 seconds\n";return 1;
            }
            go_settings.gp_timeout_ms=std::strtoul(value.c_str(),nullptr,10)*1000;
            continue;
        }
        if(a == "--gp" && i+1<argc) {
            gp_bin_path = argv[++i];
            continue;
        }
        if(a == "--mul" && i+1<argc) {
            gpu_mul_path = argv[++i];
            continue;
        }
        if(a == "--sqr" && i+1<argc) {
            gpu_sqr_path = argv[++i];
            continue;
        }
        if(a == "--add" && i+1<argc) {
            gpu_add_path = argv[++i];
            continue;
        }
        if(a == "--sub" && i+1<argc) {
            gpu_sub_path = argv[++i];
            continue;
        }
        if(a == "--special-mult" && i+1<argc) {
            gpu_special_mult_path = argv[++i];
            continue;
        }
        if(a == "--showkernel") {
            show_kernels = true;
            continue;
        }
        /* --gpu-info: pure query, handled after argv is fully parsed (it needs -d and
           --gpu-param, which may appear later on the command line). docs §D4. */
        if(a == "--gpu-info") {
            show_gpu_info = true;
            continue;
        }
        if(a == "--bits" && i+1<argc) {
            try {
                const long b = std::stol(argv[++i]);
                if (b < 0 || b > 100000000L) {
                    std::cerr << "Invalid --bits value, expected an integer >= 0" << std::endl;
                    return 1;
                }
                gpu_info_bits = (uint32_t)b;
            } catch (...) {
                std::cerr << "Invalid --bits value, expected an integer >= 0" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "--config" && i+1<argc) {
            ini_path = argv[++i];
            continue;
        }
        // Which worker this process is: selects the [Worker #N] section of BOTH the
        // ini and the worktodo file (docs/usage/GUI.md D1/D2). 1-based, default 1,
        // so a single-worker setup (no sections, no switch) behaves as before.
        if(a == "--worker" && i+1<argc) {
            try {
                const long w = std::stol(argv[++i]);
                if (w < 1 || w > 100000) {
                    std::cerr << "Invalid --worker value, expected 1..100000" << std::endl;
                    return 1;
                }
                worker_index = (int)w;
            } catch (...) {
                std::cerr << "Invalid --worker value, expected an integer >= 1" << std::endl;
                return 1;
            }
            continue;
        }
        if(a == "--tmp-dir" && i+1<argc) {
            tmp_dir = argv[++i];
            continue;
        }
        if(a == "--p95-dir" && i+1<argc) {
            // 兼容旧脚本: stage-1 结果现在只写本地, p95 交接由 ecm_p95feeder 负责.
            p95_dir_ignored = argv[++i];
            continue;
        }
        // ---- runtime tuning flags (replace former environment variables) ----
        // Naming convention (kebab-case):
        //   value flags:  --<group>-<noun> <value>   (e.g. --kernel-cache-dir)
        //   enable flags: --<feature>                (default-off feature on)
        //   disable flags:--no-<feature>             (turn a default-on feature off)
        {
            EcmRuntimeConfig &cfg = ecm_runtime_config();
            // device / launch
            if(a == "--tpi" && i+1<argc){ cfg.tpi = (uint32_t)std::stoul(argv[++i]); continue; }
            // operator tuning
            if(a == "--force-normalize" && i+1<argc){ cfg.stage1_force_normalize = std::stoi(argv[++i]); continue; }
            if(a == "--addsub-fused-unroll" && i+1<argc){ cfg.add_mod_fused_unroll = std::stoi(argv[++i]); continue; }
            if(a == "--sliced"){ cfg.gpu_sliced = true; continue; }
            if(a == "--sliced-t16"){ cfg.gpu_sliced_t16 = true; continue; }
            if(a == "--local"){ cfg.gpu_local = true; continue; }
            if(a == "--wg" && i+1<argc){
                int wg = std::stoi(argv[++i]);
                if (wg < 0) { std::cerr << "Invalid --wg value, expected >= 0" << std::endl; return 1; }
                cfg.wg_size = wg;
                continue;
            }
            // kernel source / cache group
            if(a == "--kernel-root" && i+1<argc){ cfg.kernel_root = argv[++i]; continue; }
            if(a == "--kernel-cache-dir" && i+1<argc){ cfg.cache_dir = argv[++i]; continue; }
            if(a == "--no-kernel-cache"){ cfg.cache_disable = true; continue; }
            if(a == "--kernel-cache-verbose"){ cfg.cache_verbose = true; continue; }
            if(a == "--compile-verbose"){ cfg.compile_verbose = true; continue; }
            // logging / debug / verification
            if(a == "--no-log-timestamp"){ cfg.log_timestamp = false; continue; }
            if(a == "--gpu-dump"){ cfg.gpu_dump = true; continue; }
            if(a == "--gpu-dump-file" && i+1<argc){ cfg.gpu_dump = true; cfg.gpu_dump_file = argv[++i]; continue; }
            if(a == "--profile-ops"){ cfg.profile_ops = true; continue; }
            if(a == "--profile-ops-file" && i+1<argc){ cfg.profile_ops = true; cfg.profile_ops_file = argv[++i]; continue; }
            if(a == "--sync-each-batch"){ cfg.sync_each_batch = true; continue; }
            if(a == "--verify-gpu"){ cfg.verify_gpu_results = true; continue; }
            if(a == "--verify-gpu-strict"){ cfg.verify_gpu_results = true; cfg.verify_gpu_strict = true; continue; }
        }
        if(a == "-h" || a == "--help" || a == "/?") {
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << std::endl;
            return 1;
        }
        pos.push_back(a);
    }

    // Argv parsed: fold driver-level args into the runtime config (single source),
    // then install timestamped streams.
    ecm_runtime_config().device_index = gpu_device_index;
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
        std::string ini = ini_path.empty() ? (exe_dir + "/ecm.toml")
                                           : resolve_rel_local(exe_dir, ini_path);
        int eff_device = gpu_device_index;
        int eff_param = gpu_param_set ? gpu_param_cli : ecm_config::defaults::stage1_gpu_param;
        ecm_stage1::CurveModel eff_curve = selection.curve;
        const char *ini_used = nullptr;
        EcmQueueConfig info_cfg;
        if (ecm_queue_config_load(ini, worker_index, info_cfg)) {
            ini_used = ini.c_str();
            if (!device_set) eff_device = info_cfg.device;
            if (!gpu_param_set) eff_param = info_cfg.gpu_param;
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
        return ecm_stage1_app::run_queue_manager(ini_path, worker_index,stage1_tasks);
    }

    ecm_ts_fprintf(stdout,"%s %s (Stage1)\n",ecm_program::name,ecm_program::version);

    unsigned long ckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    if (ckpt_set) {
        const double val_ms = ckpt_seconds * 1000.0;
        if (val_ms != val_ms) {
            std::cerr << "Error, invalid --ckpt value (NaN)" << std::endl;
            return 1;
        }
        if (val_ms <= 0.0) {
            ckpt_ms = 0;
        } else if (val_ms >= (double)ULONG_MAX) {
            ckpt_ms = ULONG_MAX;
        } else {
            ckpt_ms = (unsigned long)std::llround(val_ms);
        }
    }

    // Diagnosis follows durable task publication; GP availability is not admission.
    go_settings.gp=resolve_gp_path(get_gp_executable(gp_bin_path));
    ecm_diagnostics::Session go_session(go_settings);

    std::cout << "ecm driver starting" << std::endl;
    std::cout << "  curve_model: " << ecm_stage1::curve_model_name(selection.curve)
              << ", execution_target: " << ecm_stage1::execution_target_name(selection.target)
              << ", gpucurves=" << gpucurves
              << ", ckpt=" << (ckpt_ms == 0 ? 0.0 : ckpt_ms / 1000.0) << "s"
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
        while(std::getline(std::cin, line)){
            oss << line;
        }
        nline = oss.str();
        trim(nline);
    }
    if(nline.empty()){
        std::cerr << "No input number on stdin" << std::endl;
        return 1;
    }

    // positional B1 and B2
    double B1 = 0.0; double B2 = 0.0;
    if(pos.size() >= 1) {
        B1 = strtod(pos[0].c_str(), nullptr);
    }
    if(pos.size() >= 2) {
        B2 = strtod(pos[1].c_str(), nullptr);
    }

    mpz_t N; mpz_init(N);
    std::string parse_error;
    if(!ecm_parse_expression(nline, N, &parse_error)){
        std::cerr << "Failed to parse N: '"<< nline <<"'" << std::endl;
        std::cerr << "Parse error: " << parse_error << std::endl;
        mpz_clear(N);
        return 1;
    }

    // std::cout << "Parsed N bit-size: " << mpz_sizeinbase(N, 2) << std::endl;
    // if (verbose) {
    //     std::cout << "Parsed N = ";
    //     mpz_out_str(stdout, 10, N);
    //     std::cout << std::endl;
    // }

    if (selection.curve == ecm_stage1::CurveModel::Edwards &&
        selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        (gpu_param_set || sigma_param_set || print_group_order)) {
        std::cerr << "Error, GPU Edwards uses an unprefixed sigma; --gpu-param and --go do not apply" << std::endl;
        mpz_clear(N);return 1;
    }
    if (selection.target == ecm_stage1::ExecutionTarget::Cpu) {
        if (gpu_param_set) {
            std::cerr << "Error, --gpu-param requires --target gpu" << std::endl;
            mpz_clear(N);
            return 1;
        }
        if (sigma_param_set &&
            (selection.curve != ecm_stage1::CurveModel::Montgomery || sigma_param_claim != 0)) {
            std::cerr << "Error, CPU accepts a parameterized sigma only for Montgomery PARAM0"
                      << std::endl;
            mpz_clear(N);
            return 1;
        }
    }

    // gmp-ecm semantics for "-sigma i:s": the prefix selects the parametrization as well.
    // Adopt it when no --gpu-param was given, and reject a contradiction otherwise.  This
    // has to happen BEFORE the 32-bit sigma check below, which keys off gpu_param_cli.
    if (sigma_param_set) {
        int resolved_param = gpu_param_cli;
        if (!ecm_stage1::resolve_gpu_param(gpu_param_cli, gpu_param_set,
                                           sigma_param_claim, resolved_param)) {
            std::cerr << "Error, conflict between -sigma and -param arguments" << std::endl;
            std::cerr << "       -sigma " << sigma_param_claim << ":... selects -param "
                      << sigma_param_claim << ", but --gpu-param " << gpu_param_cli
                      << " was also given" << std::endl;
            mpz_clear(N);
            return 1;
        }
        gpu_param_cli = resolved_param;
        gpu_param_set = true;
    }

    Stage1Options opt;
    opt.execution.selection = selection;
    /* The two CPU curve models share the thread count. */
    opt.execution.stage1_threads = stage1_threads;
    opt.execution.naf_w = naf_w;                       /* [edwards] */
    if (!affinity_spec.empty()) opt.execution.affinity_cpus = parse_affinity_spec(affinity_spec);

    opt.execution.verbose = verbose ? 1 : 0;
    opt.execution.device_index = gpu_device_index;
    /* Default home of the stage-1 exponent cache for single runs -- run_queue_manager()
       sets the same default for queue mode, and an explicit --exp-cache / ecm.toml
       exp_cache always wins (the getter is only consulted when nothing was set). */
    if (ecm_exp_cache_get_dir().empty()) {
        const std::string d = get_exe_dir_local();
        ecm_exp_cache_set_dir(d.empty() ? "." : d);
    }
    opt.execution.selection.gpu_param = gpu_param_set ? gpu_param_cli
                                            : ecm_config::defaults::stage1_gpu_param;
    {
        std::string selection_error;
        if (stage1_selection_error(opt.execution.selection, selection_error)) {
            std::cerr << "ERROR: " << selection_error << std::endl;
            mpz_clear(N);
            return 1;
        }
    }
    opt.persistence.ckpt_ms = ckpt_ms;
    opt.execution.gpu_mul_path = gpu_mul_path;
    opt.execution.gpu_sqr_path = gpu_sqr_path;
    opt.execution.gpu_add_path = gpu_add_path;
    opt.execution.gpu_sub_path = gpu_sub_path;
    opt.execution.gpu_special_mult_path = gpu_special_mult_path;
    opt.identity.sigma_fixed = sigma_fixed;
    opt.identity.fixed_sigma64 = fixed_sigma64;
    {
        std::string input_error;
        if (stage1_task_error(opt, gpucurves, input_error)) {
            std::cerr << "ERROR: " << input_error << std::endl;
            mpz_clear(N);
            return 1;
        }
    }
    if (tmp_dir.empty()) tmp_dir = ".";      // 默认 = 当前目录 (本地)
    opt.persistence.tmp_dir = normalize_path_local(tmp_dir);
    {
        const auto ini=ini_path.empty()?std::filesystem::path(get_exe_dir_local())/"ecm.toml":std::filesystem::path(ini_path);
        EcmQueueConfig cfg;
        if(ecm_queue_config_load(ini.string(),worker_index,cfg))stage1_result_settings(opt,cfg,ini.parent_path().string(),worker_index);
        else if(!ini_path.empty()) {std::cerr<<"Cannot read Stage1 result settings from TOML"<<std::endl;mpz_clear(N);return 1;}
    }
    if (!p95_dir_ignored.empty()) {
        std::cout << "Prime95 transfer path ignored: --p95-dir is obsolete; Stage1 saves "
                     "remain local. Supported Montgomery handoff uses TOML p95_worktodo_path."
                  << std::endl;
    }
    {
        // 存档命名: 假设 N = 2^n - 1 (Mersenne), n = 位长
        opt.identity.handoff_k = 1.0;
        opt.identity.handoff_k_str = "1";
        opt.identity.handoff_b = 2;
        opt.identity.handoff_n = (uint32_t)mpz_sizeinbase(N, 2);
        opt.identity.handoff_c = -1;
    }

    auto result=run_stage1_task({N,B1,B2,gpucurves,savefilename,saveappend,nline,opt},stage1_tasks);
    const int rc1=result.task_return();


    // Failed/stopped tasks may have no result arrays or only a partial owned
    // payload. Inspect factors only after successful execution; the owner
    // releases every allocation on all exits, including group-order errors.
    const bool cli_has_factors = (result.legacy_return() != ECM_ERROR && result.has_factors() && result.has_found());

    // Numeric Montgomery parameter used by the group-order helper. CPU Suyama
    // uses PARAM0; GPU Montgomery uses its selected family. Edwards hit metadata
    // identifies the curve model and carries no Montgomery PARAM.
    const int go_param = opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                             ? opt.execution.selection.gpu_param : 0;

    if (cli_has_factors) {
        /* Same D3 fields as the queue path (docs/usage/GUI.md): a consumer must not
           have to guess which of the three back-end hit lines it is looking at. */
        const char *curve_token = ecm_stage1::curve_model_name(opt.execution.selection.curve);
        const char *target_token = ecm_stage1::execution_target_name(opt.execution.selection.target);
        for (uint32_t i = 0; i < result.curves(); i++) {
            if (result.found(i) != ECM_NO_FACTOR_FOUND) {
                char *s = mpz_get_str(NULL, 10, result.factor(i));
                std::cout << "factor[" << i << "]=" << (s ? s : "?") << " curve=" << i;
                if (result.has_sigmas()) {
                    std::cout << " sigma=" << result.sigma(i);
                }
                if(opt.execution.selection.curve==ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << go_param;
                std::cout << " curve_model=" << curve_token
                          << " execution_target=" << target_token
                          << " save=" << savefilename << "\n";
                free(s);
                if (print_group_order)print_group_order_factor(result,i,go_param,go_session,B1,
                    opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards?48:
                        opt.execution.selection.scalar_construction==ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12?12:1,
                    opt.execution.selection.curve==ecm_stage1::CurveModel::Montgomery?ecm_group_order::Family::Montgomery:ecm_group_order::Family::Edwards);
            }
        }

    }
    mpz_clear(N);
    /* 让调用方 (脚本/Prime95 前端) 能看到失败, 而不是把 0 当成"跑完了没因子"。 */
    return (rc1 == ECM_ERROR || result.legacy_return() == ECM_ERROR) ? 1 : 0;
}
