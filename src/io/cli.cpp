#include "cli.h"
#include "config_format.h"
#include "ecm/platform/program_version.h"
#include "ecm/service/execution/legacy_gpu.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>

namespace ecm_stage1_app {
void print_usage(const char* prog) {
    const char* name = prog;
    if (name != nullptr) {
        const char* slash = std::strrchr(name, '\\');
        const char* slash2 = std::strrchr(name, '/');
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
              << "  --curve <m>          mont|ed (Montgomery or Edwards; default "
                 "mont)\n"
              << "  --target <t>         cpu|gpu (default gpu; GPU Edwards requires "
                 "CUDA)\n"
              << "  --impl <m>           auto|simd|scalar (CPU only). The SIMD path\n"
              << "                       requires AVX512-IFMA; auto selects by batch "
                 "size.\n"
              << "  --reduction <m>      auto|montgomery|mersenne-fold (CPU SIMD). "
                 "auto\n"
              << "                       uses the Mersenne fold when N = 2^k-1 (half "
                 "the madds\n"
              << "                       per multiply); explicit values require --impl "
                 "simd.\n"
              << "  --stage1-threads <n> CPU stage-1 worker threads (0=auto, "
                 "1=serial). One task\n"
              << "                       is an 8-curve SIMD batch (simd backend) or "
                 "one curve\n"
              << "  --exp-cache <dir>    cache s = torsion*lcm(1..B1) on disk "
                 "(default: the exe\n"
              << "                       directory; \"off\" disables).  B1 = 260e6 "
                 "costs ~10 s to\n"
              << "                       build and ~0.3 s to load from a validated "
                 "cache.\n"
              << "  --lcm <1|12>         Stage1 scalar multiplier for lcm(1..B1). 1 is "
                 "the\n"
              << "                       default; 12 is Prime95-style (use when "
                 "Prime95 runs\n"
              << "                       stage 2 on our point, see doc section 16.7)\n"
              << "  --naf-w <w>          Edwards NAF window (0 = CPU 12 / GPU 6); "
                 "dictionary = 2^(w-2)\n"
              << "  --affinity <list>    Pin worker t to logical CPU list[t %% len], "
                 "e.g. 0,1,2,3\n"
              << "                       or 0-7.  On hybrid CPUs (Zen5 + Zen5c, P+E "
                 "cores, ...)\n"
              << "                       this can hurt: measure first (doc section "
                 "13.4)\n"
              << "  --tmp-dir <dir>      Local directory for text saves and one batch "
                 ".ckpt\n"
              << "                       (default: current dir; ecm.exe never writes "
                 "to p95)\n"
              << "  -gpucurves <n>       Number of ECM curves per launch (Edwards: "
                 "total curves)\n"
              << "  --ckpt <sec>         Mid-stage-1 checkpoint interval in seconds, "
                 "for every\n"
              << "                       implementation that has one (GPU, CPU "
                 "Edwards, CPU Montgomery;\n"
              << "                       default: 600, 0 = no autosave -- Ctrl+C still "
                 "saves)\n"
              << "  -d <index>           " << ecm_backend_name() << " device index (default: 0)\n"
              << "  --gpu-param <0|2|3>  GPU curve parametrization (ini: gpu_param):\n"
              << "                       0 = Suyama param0 (same curves as CPU "
                 "Montgomery,\n"
              << "                           Z/12 torsion, param0 save file) -- "
                 "CUDA/CGBN only;\n"
              << "                       2 = gmp-ecm batch 2 / PARAM=2 -- CUDA/CGBN "
                 "only;\n"
              << "                       3 = gmp-ecm batch/PARAM=3 (default, "
                 "historical GPU path)\n"
              << "  -sigma <value>       Fixed curve sigma (1..2^64-1; optional "
                 "param:sigma prefix).\n"
              << "                       GPU PARAM2/3 need sigma+curves <= 2^32.\n"
              << "  -v                   Verbose output\n"
              << "  -save <file>         Append factorization lines to file\n"
              << "  -savea <file>        Same as -save (append mode)\n"
              << "  --go                 Print optional group order diagnostics (GP "
                 "preferred, CPU "
                 "fallback)\n"
              << "  --go-backend <mode>  auto (PARI DLL then GP), pari/native, gp or "
                 "gmp; diagnosis "
                 "never changes task status\n"
              << "  --go-gp-timeout <s>  GP diagnostic deadline in seconds (1..600; "
                 "default 30)\n"
              << "  --gp <path>          Path to gp executable (default: gp on PATH)\n"
              << "  --mul <path>         Montgomery mul kernel path (OpenCL only)\n"
              << "  --sqr <path>         Montgomery sqr kernel path (OpenCL only)\n"
              << "  --add <path>         Modular add kernel path (OpenCL only)\n"
              << "  --sub <path>         Modular sub kernel path (OpenCL only)\n"
              << "  --special-mult <path>  special_mult (R=2^32) kernel path (OpenCL "
                 "only)\n"
              << "  --showkernel         List available " << ecm_backend_name()
              << " kernel paths and exit\n"
              << "  --gpu-info           Report the device this build would use and, "
                 "per kernel\n"
              << "                       tier, the curves a full GPU needs; no curve "
                 "is run and\n"
              << "                       no file is written. Honours -d, --curve and\n"
              << "                       --gpu-param (Montgomery only)\n"
              << "  --bits <n>           With --gpu-info: only the tier the kernel "
                 "would pick\n"
              << "                       for an N of n bits (default: every tier of "
                 "this build)\n"
              << "  -h, --help           Show this help and exit\n"
              << "\nRuntime tuning (kebab-case; replaces former environment "
                 "variables):\n"
              << " Device / operators:\n"
              << "  --tpi <1..32>              Threads per instance (default 8)\n"
              << "  --force-normalize <0|1>    Stage1 force-normalize path\n"
              << "  --addsub-fused-unroll <1|2>  add/sub fused-unroll mode\n"
              << "  --local                    Use LDS-based kernel (reduce scratch "
                 "spill at large "
                 "bits)\n"
              << "  --wg <N>                   Explicit work-group size (0=auto; "
                 "1,4,8,16,32...)\n"
              << " Kernel source / cache:\n"
              << "  --kernel-root <dir>        Kernel source root\n"
              << "  --kernel-cache-dir <dir>   OpenCL binary cache directory (OpenCL "
                 "only)\n"
              << "  --no-kernel-cache          Disable kernel binary cache (OpenCL "
                 "only)\n"
              << "  --kernel-cache-verbose     Verbose cache hit/miss logging (OpenCL "
                 "only)\n"
              << "  --compile-verbose          Verbose compile timing (OpenCL only)\n"
              << " Logging / debug / verify:\n"
              << "  --no-log-timestamp         Disable log timestamps (default on)\n"
              << "  --gpu-dump [--gpu-dump-file <f>]      Dump GPU state to CSV\n"
              << "  --profile-ops [--profile-ops-file <f>]  Operator-count profiling\n"
              << "  --sync-each-batch          Synchronize after each batch\n"
              << "  --verify-gpu [--verify-gpu-strict]   CPU cross-check GPU "
                 "results\n\n"
              << "Queue manager (run with no positional B1/B2):\n"
              << "  --config <path>                Config file (default: <exe "
                 "dir>/ecm.toml)\n"
              << "  --worker <n>               Which worker this process is (1-based, "
                 "default 1).\n"
              << "                             Selects [workers.N.*] TOML overrides "
                 "and the\n"
              << "                             [Worker #N] worktodo queue section.\n\n"
              << "Examples:\n"
              << "  echo '(2^991-1)' | " << name
              << " -v --go --curve mont --target gpu -gpucurves 384 1e6 0\n"
              << "  echo '(2^421-1)' | " << name
              << " --curve mont --target gpu -gpucurves 256 -d 1 1e5 0\n"
              << "  echo '(2^4003-1)' | " << name
              << " --curve mont --target gpu -gpucurves 384 --add asm_b32 1e6 0\n"
              << "  " << name << " --showkernel\n"
              << "  " << name << " --config ecm.toml --worker 2      (queue manager, worker #2)\n\n"
              << "Add/sub path names (for --add / --sub): default, fused, "
                 "fused_unroll,\n"
              << "  asm/unroll_128b, asm/unroll_192b, asm/unroll_256b, "
                 "asm/unroll_384b,\n"
              << "  asm/unroll_512b, asm/unroll_4096b (legacy: asm_b16, asm_b32, "
                 "fused_unroll_b16).\n"
              << "Run with --showkernel for full Montgomery and add/sub path lists.\n";
}

bool wants_usage(int argc, char** argv) {
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

// Every occurrence is converted and checked before precedence is applied.
// Parsing does not install runtime, logging or filesystem state.
int parse_cli(int argc, char** argv, ParsedCli& input) {
    static const std::set<std::string> switches = {"--version",
                                                   "-v",
                                                   "--go",
                                                   "--showkernel",
                                                   "--gpu-info",
                                                   "--sliced",
                                                   "--sliced-t16",
                                                   "--local",
                                                   "--no-kernel-cache",
                                                   "--kernel-cache-verbose",
                                                   "--compile-verbose",
                                                   "--no-log-timestamp",
                                                   "--gpu-dump",
                                                   "--profile-ops",
                                                   "--sync-each-batch",
                                                   "--verify-gpu",
                                                   "--verify-gpu-strict",
                                                   "-h",
                                                   "--help",
                                                   "/?"};
    static const std::set<std::string> values = {"--affinity",
                                                 "--stage1-threads",
                                                 "--impl",
                                                 "--reduction",
                                                 "--naf-w",
                                                 "--lcm",
                                                 "--curve",
                                                 "--target",
                                                 "-gpucurves",
                                                 "--ckpt",
                                                 "-gpuckpt",
                                                 "--stage1-stop-handle",
                                                 "--exp-cache",
                                                 "--gpu-param",
                                                 "-d",
                                                 "-sigma",
                                                 "--sigma",
                                                 "-save",
                                                 "-savea",
                                                 "--go-backend",
                                                 "--go-gp-timeout",
                                                 "--gp",
                                                 "--mul",
                                                 "--sqr",
                                                 "--add",
                                                 "--sub",
                                                 "--special-mult",
                                                 "--bits",
                                                 "--config",
                                                 "--worker",
                                                 "--tmp-dir",
                                                 "--p95-dir",
                                                 "--tpi",
                                                 "--force-normalize",
                                                 "--addsub-fused-unroll",
                                                 "--wg",
                                                 "--kernel-root",
                                                 "--kernel-cache-dir",
                                                 "--gpu-dump-file",
                                                 "--profile-ops-file"};
    for (int i = 1; i < argc; ++i) {
        const int argument = i;
        const std::string key = argv[i];
        std::string value, expected = "known option (use --help)";
        try {
            const auto reject = [] { throw std::runtime_error("invalid value"); };
            if (values.count(key)) {
                expected = "one value after " + key;
                if (i + 1 == argc)
                    reject();
                value = argv[i + 1];
                if (values.count(value) || switches.count(value)) {
                    expected = "a value, not another option (prefix a filename beginning "
                               "with '-' with './')";
                    reject();
                }
                ++i;
            } else if (!switches.count(key)) {
                if (!key.empty() && (key[0] == '-' || key == "/?")) {
                    value = i + 1 < argc ? argv[i + 1] : "<missing>";
                    reject();
                }
                value = key;
                const auto at = input.pos.size();
                expected = at < 2 ? (at == 0 ? "finite positive B1" : "finite nonnegative B2")
                                  : "B1 [B2], without extra positional arguments";
                if (at >= 2)
                    reject();
                size_t used = 0;
                const auto bound = std::stod(value, &used);
                if (used != value.size() || !std::isfinite(bound) ||
                    (at == 0 ? bound <= 0 : bound < 0))
                    reject();
                (at == 0 ? input.b1 : input.b2) = bound;
                input.pos.push_back(value);
                continue;
            }
            const auto integer = [&](uint64_t low, uint64_t high) {
                expected = "integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
                if (value.empty() || value.find_first_not_of("0123456789") != value.npos)
                    reject();
                const auto n = std::stoull(value);
                if (n < low || n > high)
                    reject();
                return n;
            };
            auto& runtime = input.runtime;
            auto& selection = input.selection;
            if (key == "--version")
                input.version = true;
            else if (key == "-v")
                input.verbose = true;
            else if (key == "--go")
                input.print_group_order = true;
            else if (key == "--showkernel")
                input.show_kernels = true;
            else if (key == "--gpu-info")
                input.show_gpu_info = true;
            else if (key == "--affinity") {
                expected = "comma-separated CPU indices/ranges in [0,1023], at most "
                           "1024 entries";
                input.affinity_cpus = ecm_config::affinity_indices(value);
            } else if (key == "--stage1-threads")
                input.stage1_threads = uint32_t(integer(0, UINT32_MAX));
            else if (key == "--impl") {
                expected = "auto|simd|scalar";
                if (!ecm_stage1::parse_cpu_implementation(value, selection.cpu_implementation))
                    reject();
            } else if (key == "--reduction") {
                expected = "auto|montgomery|mersenne-fold";
                if (!ecm_stage1::parse_simd_reduction(value, selection.simd_reduction))
                    reject();
            } else if (key == "--curve") {
                expected = "mont|ed";
                if (!ecm_stage1::parse_curve_model(value, selection.curve))
                    reject();
            } else if (key == "--target") {
                expected = "cpu|gpu";
                if (!ecm_stage1::parse_execution_target(value, selection.target))
                    reject();
            } else if (key == "--lcm") {
                expected = "1|12";
                if (!ecm_stage1::parse_scalar_construction(value, selection.scalar_construction))
                    reject();
            } else if (key == "--naf-w") {
                const auto n = integer(0, 12);
                expected = "0 (auto) or integer in [2,12] (CPU requires [3,12])";
                if (n == 1)
                    reject();
                input.naf_w = int(n);
            } else if (key == "-gpucurves")
                input.gpucurves = uint32_t(integer(1, UINT32_MAX));
            else if (key == "--ckpt" || key == "-gpuckpt") {
                expected = "finite seconds (<=0 disables autosave)";
                size_t used = 0;
                const auto seconds = std::stod(value, &used);
                if (used != value.size() || !std::isfinite(seconds))
                    reject();
                input.ckpt_seconds = seconds;
            } else if (key == "--stage1-stop-handle")
                input.stop_handle = uintptr_t(integer(0, UINTPTR_MAX));
            else if (key == "--exp-cache")
                input.exp_cache = value;
            else if (key == "--gpu-param") {
                expected = "0|2|3";
                if (value != "0" && value != "2" && value != "3")
                    reject();
                input.gpu_param_cli = value[0] - '0';
            } else if (key == "-d")
                input.gpu_device_index = int(integer(0, INT32_MAX));
            else if (key == "-sigma" || key == "--sigma") {
                expected = "1..18446744073709551615, optionally 0: / 2: / 3: prefix";
                const auto colon = value.find(':');
                int claim = -1;
                auto digits = value;
                if (colon != value.npos) {
                    const auto prefix = value.substr(0, colon);
                    if (prefix != "0" && prefix != "2" && prefix != "3")
                        reject();
                    claim = prefix[0] - '0';
                    digits = value.substr(colon + 1);
                }
                if (digits.empty() || digits.find_first_not_of("0123456789") != digits.npos)
                    reject();
                const auto sigma = std::stoull(digits);
                if (!sigma)
                    reject();
                input.fixed_sigma64 = sigma;
                input.sigma_fixed = true;
                input.sigma_param_claim = claim; // Replace both value and optional parameter claim.
            } else if (key == "-save" || key == "-savea") {
                input.savefilename = value;
                input.saveappend = key == "-savea";
            } else if (key == "--go-backend") {
                expected = "auto|pari|native|gp|gmp";
                input.go_settings.backend = ecm_diagnostics::parse_backend(value);
            } else if (key == "--go-gp-timeout")
                input.go_settings.gp_timeout_ms = unsigned(integer(1, 600)) * 1000;
            else if (key == "--gp")
                input.gp_bin_path = value;
            else if (key == "--mul")
                input.gpu_mul_path = value;
            else if (key == "--sqr")
                input.gpu_sqr_path = value;
            else if (key == "--add")
                input.gpu_add_path = value;
            else if (key == "--sub")
                input.gpu_sub_path = value;
            else if (key == "--special-mult")
                input.gpu_special_mult_path = value;
            else if (key == "--bits")
                input.gpu_info_bits = uint32_t(integer(0, 100000000));
            else if (key == "--config")
                input.ini_path = value;
            else if (key == "--worker")
                input.worker_index = int(integer(1, 1000000));
            else if (key == "--tmp-dir")
                input.tmp_dir = value;
            else if (key == "--p95-dir")
                input.p95_dir_ignored = value;
            else if (key == "--tpi") {
                const auto n = integer(1, 32);
                expected = "power of two in {1,2,4,8,16,32}";
                if (n & (n - 1))
                    reject();
                runtime.tpi = uint32_t(n);
            } else if (key == "--force-normalize")
                runtime.stage1_force_normalize = int(integer(0, 1));
            else if (key == "--addsub-fused-unroll")
                runtime.add_mod_fused_unroll = int(integer(1, 2));
            else if (key == "--sliced")
                runtime.gpu_sliced = true;
            else if (key == "--sliced-t16")
                runtime.gpu_sliced_t16 = true;
            else if (key == "--local")
                runtime.gpu_local = true;
            else if (key == "--wg")
                runtime.wg_size = int(integer(0, INT32_MAX));
            else if (key == "--kernel-root")
                runtime.kernel_root = value;
            else if (key == "--kernel-cache-dir")
                runtime.cache_dir = value;
            else if (key == "--no-kernel-cache")
                runtime.cache_disable = true;
            else if (key == "--kernel-cache-verbose")
                runtime.cache_verbose = true;
            else if (key == "--compile-verbose")
                runtime.compile_verbose = true;
            else if (key == "--no-log-timestamp")
                runtime.log_timestamp = false;
            else if (key == "--gpu-dump")
                runtime.gpu_dump = true;
            else if (key == "--gpu-dump-file") {
                runtime.gpu_dump = true;
                runtime.gpu_dump_file = value;
            } else if (key == "--profile-ops")
                runtime.profile_ops = true;
            else if (key == "--profile-ops-file") {
                runtime.profile_ops = true;
                runtime.profile_ops_file = value;
            } else if (key == "--sync-each-batch")
                runtime.sync_each_batch = true;
            else if (key == "--verify-gpu")
                runtime.verify_gpu_results = true;
            else if (key == "--verify-gpu-strict") {
                runtime.verify_gpu_results = true;
                runtime.verify_gpu_strict = true;
            }
            // Alias spelling and source position remain available for diagnostics.
            const auto canonical = key == "--sigma" ? "-sigma" : key == "-gpuckpt" ? "--ckpt" : key;
            input.explicit_options[canonical] = {argument, key, value};
        } catch (const std::exception& e) {
            std::cerr << "INPUT[cli] argv[" << argument << "] key=" << key << " value='"
                      << (value.empty() && argument + 1 == argc ? "<missing>" : value)
                      << "'; expected " << expected << "; " << e.what()
                      << ". Correct this argument and retry.\n";
            return 1;
        }
    }
    return 0;
}
} // namespace ecm_stage1_app
