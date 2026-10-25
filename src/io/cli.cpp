#include "cli.h"
#include "config_format.h"
#include "ecm/service/execution/legacy_gpu.h"
#include "ecm/platform/program_version.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <cmath>
#include <set>

namespace ecm_stage1_app {
static bool parse_sigma64_arg(const std::string& arg, uint64_t* sigma_out, int* param_out) {
    std::string s = arg;
    if (param_out != nullptr) {
        *param_out = -1;
    }
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
        if (param_out != nullptr) {
            const std::string prefix = s.substr(0, colon);
            try {
                *param_out = std::stoi(prefix);
            } catch (...) {
                return false; /* "-sigma x:..." with a non-numeric prefix */
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

    std::cout
        << ecm_backend_name() << " ECM stage-1 driver\n\n"
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
        << "  -d <index>           " << ecm_backend_name() << " device index (default: 0)\n"
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
        << "  --go                 Print optional group order diagnostics (GP preferred, CPU "
           "fallback)\n"
        << "  --go-backend <mode>  auto (PARI DLL then GP), pari/native, gp or gmp; diagnosis "
           "never changes task status\n"
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
        << "  --local                    Use LDS-based kernel (reduce scratch spill at large "
           "bits)\n"
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
        << "  echo '(2^991-1)' | " << name
        << " -v --go --curve mont --target gpu -gpucurves 384 1e6 0\n"
        << "  echo '(2^421-1)' | " << name
        << " --curve mont --target gpu -gpucurves 256 -d 1 1e5 0\n"
        << "  echo '(2^4003-1)' | " << name
        << " --curve mont --target gpu -gpucurves 384 --add asm_b32 1e6 0\n"
        << "  " << name << " --showkernel\n"
        << "  " << name << " --config ecm.toml --worker 2      (queue manager, worker #2)\n\n"
        << "Add/sub path names (for --add / --sub): default, fused, fused_unroll,\n"
        << "  asm/unroll_128b, asm/unroll_192b, asm/unroll_256b, asm/unroll_384b,\n"
        << "  asm/unroll_512b, asm/unroll_4096b (legacy: asm_b16, asm_b32, fused_unroll_b16).\n"
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

// Validate the complete token before the typed conversion below. No runtime
// singleton, filesystem, cache or logging state is changed while parsing.
static bool validate_cli_token(int argc, char** argv, int i) {
    const std::string key = argv[i];
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
    if (switches.count(key) || key.empty() || (key[0] != '-' && key != "/?"))
        return true;
    const std::string value = i + 1 < argc ? argv[i + 1] : "<missing>";
    std::string expected = "known option (use --help)";
    const auto fail = [&]() {
        std::cerr << "INPUT[cli] argv[" << i << "] key=" << key << " value='" << value
                  << "'; expected " << expected << ". Correct this argument and retry.\n";
        return false;
    };
    if (!values.count(key))
        return fail();
    expected = "one value after " + key;
    if (i + 1 >= argc)
        return fail();
    if (values.count(value) || switches.count(value)) {
        expected = "a value, not another option (prefix a filename beginning with '-' with './')";
        return fail();
    }
    const auto integer = [&](uint64_t low, uint64_t high) {
        expected = "integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
        if (value.empty() || value.find_first_not_of("0123456789") != value.npos)
            return false;
        try {
            size_t used = 0;
            const auto n = std::stoull(value, &used);
            return used == value.size() && n >= low && n <= high;
        } catch (...) {
            return false;
        }
    };
    bool ok = true;
    if (key == "--stage1-threads")
        ok = integer(0, UINT32_MAX);
    else if (key == "-gpucurves")
        ok = integer(1, UINT32_MAX);
    else if (key == "--worker")
        ok = integer(1, 1000000);
    else if (key == "-d" || key == "--wg")
        ok = integer(0, INT32_MAX);
    else if (key == "--bits")
        ok = integer(0, 100000000);
    else if (key == "--stage1-stop-handle")
        ok = integer(0, UINTPTR_MAX);
    else if (key == "--tpi") {
        ok = integer(1, 32);
        expected = "power of two in {1,2,4,8,16,32}";
        if (ok) {
            const auto n = std::stoul(value);
            ok = (n & (n - 1)) == 0;
        }
    } else if (key == "--force-normalize")
        ok = integer(0, 1);
    else if (key == "--addsub-fused-unroll")
        ok = integer(1, 2);
    else if (key == "--naf-w") {
        ok = integer(0, 12);
        expected = "0 (auto) or integer in [2,12] (CPU requires [3,12])";
        ok = ok && std::stoul(value) != 1;
    } else if (key == "--go-gp-timeout")
        ok = integer(1, 600);
    else if (key == "--gpu-param") {
        expected = "0|2|3";
        ok = value == "0" || value == "2" || value == "3";
    } else if (key == "--ckpt" || key == "-gpuckpt") {
        expected = "finite seconds (<=0 disables autosave)";
        try {
            size_t used = 0;
            const auto n = std::stod(value, &used);
            ok = used == value.size() && std::isfinite(n);
        } catch (...) {
            ok = false;
        }
    } else if (key == "-sigma" || key == "--sigma") {
        expected = "1..18446744073709551615, optionally 0: / 2: / 3: prefix";
        auto token = value;
        const auto colon = token.find(':');
        if (colon != token.npos) {
            const auto prefix = token.substr(0, colon);
            ok = prefix == "0" || prefix == "2" || prefix == "3";
            token.erase(0, colon + 1);
        }
        ok = ok && !token.empty() && token.find_first_not_of("0123456789") == token.npos;
        if (ok)
            try {
                ok = std::stoull(token) > 0;
            } catch (...) {
                ok = false;
            }
    } else if (key == "--curve") {
        ecm_stage1::CurveModel v;
        expected = "mont|ed";
        ok = ecm_stage1::parse_curve_model(value, v);
    } else if (key == "--target") {
        ecm_stage1::ExecutionTarget v;
        expected = "cpu|gpu";
        ok = ecm_stage1::parse_execution_target(value, v);
    } else if (key == "--impl") {
        ecm_stage1::CpuImplementation v;
        expected = "auto|simd|scalar";
        ok = ecm_stage1::parse_cpu_implementation(value, v);
    } else if (key == "--reduction") {
        ecm_stage1::SimdReduction v;
        expected = "auto|montgomery|mersenne-fold";
        ok = ecm_stage1::parse_simd_reduction(value, v);
    } else if (key == "--lcm") {
        ecm_stage1::ScalarConstruction v;
        expected = "1|12";
        ok = ecm_stage1::parse_scalar_construction(value, v);
    } else if (key == "--go-backend") {
        expected = "auto|pari|native|gp|gmp";
        try {
            (void)ecm_diagnostics::parse_backend(value);
        } catch (...) {
            ok = false;
        }
    } else if (key == "--affinity") {
        expected = "CPU indices/ranges in [0,1023], at most 1024 entries, e.g. 0-7,10 "
                   "(empty|auto|none unpins)";
        try {
            (void)ecm_config::affinity_indices(value);
        } catch (...) {
            ok = false;
        }
    }
    return ok || fail();
}

int parse_cli(int argc, char** argv, CliInput& input) {
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
    try {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            const int argument = i;
            if (!validate_cli_token(argc, argv, i))
                return 1;
            input.explicit_options[a] = {argument, i + 1 < argc ? argv[i + 1] : ""};
            if (a == "--version") {
                input.version = true;
                continue;
            }
            if (a == "-v") {
                verbose = true;
                continue;
            }
            if (a == "--affinity" && i + 1 < argc) {
                affinity_spec = argv[++i];
                continue;
            }
            /* Each option selects one mathematical or execution axis. */
            if (a == "--stage1-threads" && i + 1 < argc) {
                try {
                    stage1_threads = (uint32_t)std::stoul(argv[++i]);
                } catch (...) {
                    std::cerr << "Invalid --stage1-threads value, expected >= 0" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--impl" && i + 1 < argc) {
                if (!ecm_stage1::parse_cpu_implementation(argv[++i],
                                                          selection.cpu_implementation)) {
                    std::cerr << "Invalid --impl, expected auto|simd|scalar" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--reduction" && i + 1 < argc) {
                if (!ecm_stage1::parse_simd_reduction(argv[++i], selection.simd_reduction)) {
                    std::cerr << "Invalid --reduction, expected auto|montgomery|mersenne-fold"
                              << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--naf-w" && i + 1 < argc) {
                try {
                    naf_w = std::stoi(argv[++i]);
                } catch (...) {
                    std::cerr << "Invalid --naf-w value, expected integer" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--lcm" && i + 1 < argc) {
                if (!ecm_stage1::parse_scalar_construction(argv[++i],
                                                           selection.scalar_construction)) {
                    std::cerr << "Invalid --lcm, expected 1|12" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--curve" && i + 1 < argc) {
                if (!ecm_stage1::parse_curve_model(argv[++i], selection.curve)) {
                    std::cerr << "Invalid --curve, expected mont|ed" << std::endl;
                    return 1;
                }
                curve_set = true;
                continue;
            }
            if (a == "--target" && i + 1 < argc) {
                if (!ecm_stage1::parse_execution_target(argv[++i], selection.target)) {
                    std::cerr << "Invalid --target, expected cpu|gpu" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "-gpucurves" && i + 1 < argc) {
                gpucurves = (uint32_t)std::stoul(argv[++i]);
                continue;
            }
            if ((a == "--ckpt" || a == "-gpuckpt") && i + 1 < argc) {
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
            if (a == "--stage1-stop-handle" && i + 1 < argc) {
                try {
                    stop_handle = (uintptr_t)std::stoull(argv[++i]);
                } catch (...) {
                    std::cerr << "Invalid Stage1 stop handle" << std::endl;
                    return 1;
                }
                continue;
            }
            if (a == "--exp-cache" && i + 1 < argc) {
                input.exp_cache = argv[++i];
                continue;
            }
            if (a == "--gpu-param" && i + 1 < argc) {
                try {
                    gpu_param_cli = std::stoi(argv[++i]);
                } catch (...) {
                    std::cerr << "Invalid --gpu-param value, expected 0, 2 or 3" << std::endl;
                    return 1;
                }
                if (!ecm_stage1::valid_gpu_param(gpu_param_cli)) {
                    std::cerr << "Invalid --gpu-param value, expected 0 (Suyama param0), "
                                 "2 (gmp-ecm batch 2 / 6-torsion) or 3 (gmp-ecm batch)"
                              << std::endl;
                    return 1;
                }
                gpu_param_set = true;
                continue;
            }
            if (a == "-d" && i + 1 < argc) {
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
            if ((a == "-sigma" || a == "--sigma") && i + 1 < argc) {
                int sigma_prefix = -1; /* "-sigma i:s" claims -param i */
                if (!parse_sigma64_arg(argv[++i], &fixed_sigma64, &sigma_prefix)) {
                    std::cerr << "Invalid -sigma value (need 1..2^64-1, optional param: prefix)"
                              << std::endl;
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
                                     "(this build knows -param 0, 2 and 3)"
                                  << std::endl;
                        return 1;
                    }
                }
                continue;
            }
            if (a == "-save" && i + 1 < argc) {
                savefilename = argv[++i];
                saveappend = false;
                continue;
            }
            if (a == "-savea" && i + 1 < argc) {
                savefilename = argv[++i];
                saveappend = true;
                continue;
            }
            if (a == "--go") {
                print_group_order = true;
                continue;
            }
            if (a == "--go-backend") {
                if (i + 1 >= argc) {
                    std::cerr << "Error, --go-backend requires auto, pari/native, gp or gmp\n";
                    return 1;
                }
                try {
                    go_settings.backend = ecm_diagnostics::parse_backend(argv[++i]);
                } catch (const std::exception& e) {
                    std::cerr << "Error, " << e.what() << '\n';
                    return 1;
                }
                continue;
            }
            if (a == "--go-gp-timeout") {
                if (i + 1 >= argc) {
                    std::cerr << "Error, --go-gp-timeout requires seconds (1..600)\n";
                    return 1;
                }
                const std::string value = argv[++i];
                if (value.empty() || value.size() > 3 ||
                    value.find_first_not_of("0123456789") != value.npos ||
                    std::strtoul(value.c_str(), nullptr, 10) < 1 ||
                    std::strtoul(value.c_str(), nullptr, 10) > 600) {
                    std::cerr << "Error, --go-gp-timeout must be 1..600 seconds\n";
                    return 1;
                }
                go_settings.gp_timeout_ms = std::strtoul(value.c_str(), nullptr, 10) * 1000;
                continue;
            }
            if (a == "--gp" && i + 1 < argc) {
                gp_bin_path = argv[++i];
                continue;
            }
            if (a == "--mul" && i + 1 < argc) {
                gpu_mul_path = argv[++i];
                continue;
            }
            if (a == "--sqr" && i + 1 < argc) {
                gpu_sqr_path = argv[++i];
                continue;
            }
            if (a == "--add" && i + 1 < argc) {
                gpu_add_path = argv[++i];
                continue;
            }
            if (a == "--sub" && i + 1 < argc) {
                gpu_sub_path = argv[++i];
                continue;
            }
            if (a == "--special-mult" && i + 1 < argc) {
                gpu_special_mult_path = argv[++i];
                continue;
            }
            if (a == "--showkernel") {
                show_kernels = true;
                continue;
            }
            /* --gpu-info: pure query, handled after argv is fully parsed (it needs -d and
           --gpu-param, which may appear later on the command line). docs §D4. */
            if (a == "--gpu-info") {
                show_gpu_info = true;
                continue;
            }
            if (a == "--bits" && i + 1 < argc) {
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
            if (a == "--config" && i + 1 < argc) {
                ini_path = argv[++i];
                continue;
            }
            // Which worker this process is: selects the [Worker #N] section of BOTH the
            // ini and the worktodo file (docs/usage/GUI.md D1/D2). 1-based, default 1,
            // so a single-worker setup (no sections, no switch) behaves as before.
            if (a == "--worker" && i + 1 < argc) {
                try {
                    const long w = std::stol(argv[++i]);
                    if (w < 1 || w > 1000000) {
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
            if (a == "--tmp-dir" && i + 1 < argc) {
                tmp_dir = argv[++i];
                continue;
            }
            if (a == "--p95-dir" && i + 1 < argc) {
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
                EcmRuntimeConfig& cfg = input.runtime;
                // device / launch
                if (a == "--tpi" && i + 1 < argc) {
                    cfg.tpi = (uint32_t)std::stoul(argv[++i]);
                    continue;
                }
                // operator tuning
                if (a == "--force-normalize" && i + 1 < argc) {
                    cfg.stage1_force_normalize = std::stoi(argv[++i]);
                    continue;
                }
                if (a == "--addsub-fused-unroll" && i + 1 < argc) {
                    cfg.add_mod_fused_unroll = std::stoi(argv[++i]);
                    continue;
                }
                if (a == "--sliced") {
                    cfg.gpu_sliced = true;
                    continue;
                }
                if (a == "--sliced-t16") {
                    cfg.gpu_sliced_t16 = true;
                    continue;
                }
                if (a == "--local") {
                    cfg.gpu_local = true;
                    continue;
                }
                if (a == "--wg" && i + 1 < argc) {
                    int wg = std::stoi(argv[++i]);
                    if (wg < 0) {
                        std::cerr << "Invalid --wg value, expected >= 0" << std::endl;
                        return 1;
                    }
                    cfg.wg_size = wg;
                    continue;
                }
                // kernel source / cache group
                if (a == "--kernel-root" && i + 1 < argc) {
                    cfg.kernel_root = argv[++i];
                    continue;
                }
                if (a == "--kernel-cache-dir" && i + 1 < argc) {
                    cfg.cache_dir = argv[++i];
                    continue;
                }
                if (a == "--no-kernel-cache") {
                    cfg.cache_disable = true;
                    continue;
                }
                if (a == "--kernel-cache-verbose") {
                    cfg.cache_verbose = true;
                    continue;
                }
                if (a == "--compile-verbose") {
                    cfg.compile_verbose = true;
                    continue;
                }
                // logging / debug / verification
                if (a == "--no-log-timestamp") {
                    cfg.log_timestamp = false;
                    continue;
                }
                if (a == "--gpu-dump") {
                    cfg.gpu_dump = true;
                    continue;
                }
                if (a == "--gpu-dump-file" && i + 1 < argc) {
                    cfg.gpu_dump = true;
                    cfg.gpu_dump_file = argv[++i];
                    continue;
                }
                if (a == "--profile-ops") {
                    cfg.profile_ops = true;
                    continue;
                }
                if (a == "--profile-ops-file" && i + 1 < argc) {
                    cfg.profile_ops = true;
                    cfg.profile_ops_file = argv[++i];
                    continue;
                }
                if (a == "--sync-each-batch") {
                    cfg.sync_each_batch = true;
                    continue;
                }
                if (a == "--verify-gpu") {
                    cfg.verify_gpu_results = true;
                    continue;
                }
                if (a == "--verify-gpu-strict") {
                    cfg.verify_gpu_results = true;
                    cfg.verify_gpu_strict = true;
                    continue;
                }
            }
            if (a == "-h" || a == "--help" || a == "/?") {
                continue;
            }
            if (!a.empty() && a[0] == '-') {
                std::cerr << "Unknown option: " << a << std::endl;
                return 1;
            }
            pos.push_back(a);
        }

    } catch (const std::exception& e) {
        std::cerr << "INPUT[cli]: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
} // namespace ecm_stage1_app
