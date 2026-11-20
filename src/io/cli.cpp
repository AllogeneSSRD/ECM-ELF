#include "cli.h"
#include "config_format.h"
#include "ecm/platform/program_version.h"
#include "ecm/stage1/gpu_backend.h"
#include "ecm/stage1/ntt_length.h"
#include <cmath>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>

namespace ecm_stage1_app {
namespace {
void parse_tune(std::string spec, ecm_stage1::NttTuneGrid& grid) {
    using namespace ecm_stage1;
    if (spec.empty() || spec == "ntt") return;
    if (spec.rfind("ntt,", 0) == 0) spec.erase(0, 4);
    else if (spec.front() == ',') spec.erase(0, 1);
    std::set<std::string> keys;
    size_t pos = 0;
    do {
        const auto end = spec.find(',', pos);
        const auto field = spec.substr(pos, end - pos);
        const auto eq = field.find('=');
        const auto key = field.substr(0, eq);
        if (eq == field.npos || !keys.insert(key).second)
            throw std::invalid_argument("expected unique key=value in tune");
        const auto value = field.substr(eq + 1);
        const auto number = [&](const std::string& s, unsigned lo, unsigned hi) {
            if (s.empty() || s.find_first_not_of("0123456789") != s.npos)
                throw std::invalid_argument("invalid tune " + key);
            const auto n = std::stoull(s);
            if (n < lo || n > hi) throw std::invalid_argument("out of range: " + key);
            return unsigned(n);
        };
        auto range = [&](unsigned lo, unsigned hi) {
            const auto dash = value.find('-');
            const unsigned first = number(value.substr(0, dash), lo, hi);
            const unsigned last = dash == value.npos ? first : number(value.substr(dash + 1), lo, hi);
            if (first > last) throw std::invalid_argument("reversed tune range: " + key);
            return TuneRange{first, last};
        };
        if (value.empty()) {
            if (key != "exp" && key != "length" && key != "naf" && key != "batch" && key != "time" && key != "repeat")
                throw std::invalid_argument("unknown tune field: " + key);
        } else if (key == "exp") grid.exp = range(stage1_ntt_min_exponent, stage1_ntt_max_exponent);
        else if (key == "length") grid.length = range(2, stage1_ntt_max_length);
        else if (key == "naf") grid.naf = value == "0" ? TuneRange{} : range(2, 12);
        else if (key == "batch") {
            grid.batch = range(1, stage1_ntt_max_curves);
            grid.batch_explicit = true;
        }
        else if (key == "repeat") {
            const auto colon = value.find(':');
            if (colon == value.npos) throw std::invalid_argument("expected repeat=coarse:fine");
            grid.coarse_repeats = number(value.substr(0, colon), 1, ntt_max_repeats);
            grid.fine_repeats = number(value.substr(colon + 1), 1, ntt_max_repeats);
        } else if (key == "time") {
            size_t used = 0;
            const double ms = std::round(100 * std::stod(value, &used));
            if (value.find_first_not_of("0123456789.") != value.npos || used != value.size() ||
                !std::isfinite(ms) || ms < 1 || ms > UINT32_MAX)
                throw std::invalid_argument("time must give a positive representable duration in ms");
            grid.group_ms = unsigned(ms);
        } else throw std::invalid_argument("unknown tune field: " + key);
        if (end == spec.npos) break;
        pos = end + 1;
    } while (true);
    if (keys.count("exp") && keys.count("length")) throw std::invalid_argument("tune conflict: exp and length");
}
} // namespace

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

    std::cout << ecm_program::name << ' ' << ecm_program::version
              << " Stage1 (" << ecm_backend_name() << ")\n\n"
              << "Usage:\n"
              << "  " << name << " [--config FILE] [--worker N]          Run the Stage1 queue\n"
              << "  echo '<N>' | " << name << " [options] B1 [B2]    Run Stage1 on stdin\n"
              << "  " << name << " -tune [ntt][,exp=P|,length=L][,naf=W][,batch=B][,time=N][,repeat=A:B]\n"
              << R"(
Curve and task:
  --param <e|m0|m2|m3>  Curve construction: Edwards or Montgomery PARAM0/2/3.
                       Queue default: Edwards. Direct defaults: CPU m0, GPU m3.
                       A typed ECMSTAGE1 save name selects that task's construction.
  --curve <ed|mont>     Select only the curve model; use --param for parametrization.
                       Choose either --param or --curve.
  -sigma, --sigma <S>   Fix the starting sigma (1..2^64-1); curve i uses S+i.
                       Optional 0:S / 2:S / 3:S prefixes select Montgomery parameters.
                       Edwards uses bare S. PARAM2/3 require S+curves <= 2^32.
  -gpucurves <N>       Direct-run curve count. Queue counts come from each work line.
                       CUDA NTT: batch up to ntt_max_batch (default 512); split to fit memory.
                       NTT defaults to one curve when this option is omitted.
  --target <cpu|gpu>   Execution target (default gpu).
  B1 [B2]             B1 is a finite positive bound. B2 is optional, nonnegative.
                       This worker computes Stage1 and saves points for Stage2.
  N                    stdin accepts integers or expressions, e.g. (2^4423-1).

Queue input and Stage2 handoff:
  --config <FILE>     Load TOML (default: ecm.toml beside the executable).
                       Explicit command-line options override configuration.
  --worker <N>        Select [worker.N] settings and [Worker #N] in worktodo.
                       N is 1-based; default 1.
  Queue curve/param/count are configured in TOML or the work line. --param,
  --curve, --lcm, -gpucurves and -save/-savea are direct-run options.
  Stage1 worktodo formats:
    ECM|ECM2=[AID,]k,b,n,c,B1,B2,curves[,"known-factors"]
    ECMSTAGE1=[AID,]k,b,n,c,filename,B2-or-zero,skip_curves,num_curves[,"known-factors"]
  ECM|ECM2 means either prefix. AID is optional: omit it, use N/A, or an empty field.
  A nonempty AID has 32 hexadecimal characters. ECM/ECM2 may also specify a numeric
  sigma after curves; omitted B2 and curves default to 0 and 100.
  Quote comma-separated known factors. N is (k*b^n+c) divided by those factors.
  ECMSTAGE1 gets B1 from the filename's final _B1.save field.
  Its e_, m0_, m2_, m3_ prefix overrides the configured construction.

  Completed tasks can produce these lines for a separate Stage2 worker:
    ECMSTAGE2=[AID,]k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"]
    ECMSTAGE2N=[AID,]filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"]
  ECMSTAGE2 checks the expected modulus; ECMSTAGE2N reads the full N from the save.
  Both read B1 from the save; omitted/zero count means all records after skip.
  TOML [stage1] handoff_mode = "ecm-elf" (default), "off", or "prime95".
  ECM-ELF handoff appends an eligible ECMSTAGE2 line to the original worker.
  handoff.workers / handoff.worktodo_file select another destination.
  Published counts use the actual saved range, including any extra NTT tail curves.

GPU execution:
  --gpu-backend <MODE> auto|cuda|opencl (backend routing on Windows).
  -d <INDEX>          GPU device index (default 0).
  --gpu-algorithm <A> 1|2|3|cgbn|ntt|thread or a quoted slice list.
                       Default: "ntt[16384:], cgbn[]"; ranges use actual modulus bits.
                       CUDA NTT: full 2^p-1 or a known-factor Mersenne assignment;
                       )" << ecm_stage1::stage1_ntt_min_exponent << " <= p <= "
              << ecm_stage1::stage1_ntt_max_exponent << R"(; Edwards or m0.
                       CUDA thread: Edwards through 1024 bits.
                       OpenCL: Edwards through 1024 bits or Montgomery PARAM3.
  --gpu-info          Report device and compiled tiers, then exit.
                       Honors -d, --param and --curve.
  --bits <P>          With --gpu-info, select one operand width (0 = all tiers).
                       With --tune ntt, select one exponent width (0 = full grid).
  -tune, --tune       Optional ntt,exp=P or length=L,naf=W,batch=B,time=N,repeat=A:B.
                       P,L,W,B accept inclusive min-max. Groups target 100*N ms. A:B sets coarse:fine groups (default 2:2; each 1..20).
                       Optional positional B1 defaults to 128. Use --param e --naf-w W for Edwards NAF.
                       Store and reuse measured configurations in the shared TOML tune_file.
  --tune-exhaustive   With --tune ntt, measure every eligible batch size.
  --tune-seconds <N>  Stop starting new measurements after N seconds. Save completed rows.
  --showkernel        List available backend kernel paths, then exit.

Edwards and CPU arithmetic:
  --naf-w <W>         Edwards window: 0 or 2..12 (CPU: 0 or 3..12).
                       0 starts at CPU w12 or CGBN/OpenCL/thread w6.
                       CUDA NTT: 0 picks the measured NAF window; 2..12 fixes the window.
                       Fresh CPU/CGBN/OpenCL tasks may lower W to fit memory.
  --lcm <1|12>        Montgomery scalar multiplier; default 1.
                       Edwards uses its fixed 48*lcm(1..B1) scalar.
  --impl <MODE>       CPU auto|simd|scalar|gwnum. SIMD requires AVX-512 IFMA.
                       GWNUM requires gwnum.dll beside the worker and param m0.
                       Above 10240 bits auto needs SIMD; Edwards supports explicit scalar.
  --reduction <MODE>  CPU SIMD auto|montgomery|mersenne-fold.
                       An explicit reduction requires --impl simd.
  --cpu-mont-chain <C> CPU IFMA Montgomery ladder|prac.
  --cpu-threads <N>   CPU Stage1 threads (0 = auto).
  --gpu-init-threads <N> CPU threads for supported GPU Edwards curve setup (0 = auto).
  --stage1-threads <N> Legacy fallback for both thread counts.
                       0 selects automatically; 1 selects serial construction.
  --affinity <LIST>   Logical CPUs, e.g. 0,2,4 or 0-7.

Saves and checkpoints:
  -save <FILE>        Publish completed Stage1 points/factors to this text save.
  -savea <FILE>       Publish completed Stage1 points/factors in append mode.
  --tmp-dir <DIR>     Stage1 output/checkpoint directory; direct-run default is cwd.
  --ckpt <SECONDS>   Periodic checkpoint interval (default 600).
                       <=0 disables periodic saves; normal stop still saves.
                       Restart the original task to resume a compatible checkpoint.
  --exp-cache <DIR>  Stage1 scalar and PRAC cache; "off" disables it.
                       Default: cache/ at the checkout root or beside the executable.

Additional TOML controls:
  Set config_version = 4 at the root. In [worker.N], use the same flat keys
  as in their global tables.
  [shared] tune_file selects the common Stage1/Stage2 tuning database.
  [stage1] ntt_pad_batch = 0 runs the requested count; 1 rounds the tail batch
           up with new sigmas and includes those curves in saves and handoff.
           ntt_allow_known_factors = 1 permits full-Mersenne arithmetic for
           cofactor assignments; checks and saves use the cofactor. 0 disables it.
           ntt_convolution = "auto"|"linear"|"weighted" selects convolution.
           ntt_length, ntt_bits_per_word, ntt_tile, ntt_outer_radix_limit,
           ntt_window_bits, ntt_warp_tail and ntt_memory_mib control NTT plans.
           cuda_montgomery_chain = "ladder"|"prac" selects CGBN PARAM0's chain.
           cuda_prac_search, cpu_prac_search, naf_memory_mb and edwards_doubling
           control PRAC search, NAF memory and Edwards doubling.
  Full keys, ranges and defaults: docs/ECM_TOML_REFERENCE.en.md.

OpenCL operators and diagnostics:
  --tpi <1|2|4|8|16|32>     OpenCL threads per integer.
  --wg <N>                  OpenCL work-group size (0=auto); CUDA thread block
                            size 32|64|128|256 (0=128).
  --mul/--sqr/--add/--sub/--special-mult <PATH>
                            OpenCL Montgomery operator paths.
  --force-normalize <0|1>    OpenCL Montgomery normalization mode.
  --addsub-fused-unroll <1|2> OpenCL Montgomery add/sub unrolling.
  --sliced / --sliced-t16    OpenCL Montgomery sliced kernels.
  --local                   OpenCL Montgomery local-memory kernel.
  --kernel-root <DIR>       OpenCL kernel source root.
  --kernel-cache-dir <DIR>  OpenCL binary cache.
  --no-kernel-cache / --kernel-cache-verbose / --compile-verbose
                            OpenCL cache and compilation diagnostics.
  --gpu-dump [--gpu-dump-file FILE]
                            Dump supported GPU state to CSV.
  --profile-ops [--profile-ops-file FILE]
                            OpenCL Montgomery operator counts.
  --sync-each-batch         OpenCL Montgomery batch synchronization.
  --verify-gpu [--verify-gpu-strict]
                            OpenCL Montgomery results checked on CPU.

Logging and group-order diagnostics:
  -v                        Verbose output.
  --no-log-timestamp         Disable console timestamps.
  --go                      Diagnose group orders after a factor is found.
                            GPU Edwards uses the separate PARI diagnostic tool.
  --go-backend <MODE>        auto|pari|native|gp|gmp.
                            auto tries local PARI, then external GP.
  --go-gp-timeout <SECONDS>  External GP deadline, 1..600 (default 30).
  --gp <PATH>               GP executable path.
  --version                 Print program name and version, then exit.
  -h, -help, --help, /?      Show this help, then exit.

Examples:
)"
              << "  " << name << " --config ecm.toml --worker 2\n"
              << "  echo '(2^421-1)' | " << name
              << " --param m0 -sigma 26 --target gpu -gpucurves 32 -d 1 1e5 0\n"
              << "  echo '(2^8191-1)' | " << name
              << " --param e --target gpu --gpu-algorithm ntt --naf-w 8 -gpucurves 8 1e4 0\n"
              << "  echo '(2^991-1)' | " << name
              << " --param e --target cpu --impl simd -gpucurves 8 1e4 0\n"
              << "  " << name << " --tune ntt --bits 16319 -d 1\n\n"
              << "Worktodo examples:\n"
              << "  ECM2=1,2,3001,-1,100000,1000000,64\n"
              << "  ECMSTAGE1=1,2,3001,-1,\"e_m3001_1e5.save\",1000000,0,64\n"
              << "  ECMSTAGE2=1,2,3001,-1,\"e_m3001_1e5.save\",1000000,0,64\n"
              << "  ECMSTAGE2N=\"e_bit4096_1e5.save\",1000000,0,64\n";
}

bool wants_usage(int argc, char** argv) {
    // No explicit -h/--help: let main() dispatch (no args → queue manager mode,
    // positional B1 → single-run CLI mode). Only explicit help flags print usage.
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "-help" || a == "--help" || a == "/?") {
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
                                                   "-tune", "--tune", "--tune-exhaustive",
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
                                                   "-help",
                                                   "--help",
                                                   "/?"};
    static const std::set<std::string> values = {"--affinity",
                                                 "--gpu-backend",
                                                 "--stage1-threads",
                                                 "--cpu-threads", "--gpu-init-threads",
                                                 "--impl",
                                                 "--reduction",
                                                 "--cpu-mont-chain",
                                                 "--naf-w",
                                                 "--lcm",
                                                 "--curve", "--param",
                                                 "--target",
                                                 "-gpucurves",
                                                 "--ckpt",
                                                 "-gpuckpt",
                                                 "--stage1-stop-handle",
                                                 "--exp-cache",
                                                 "--gpu-algorithm",
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
                                                 "--tune-seconds",
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
            if (key == "-tune" || key == "--tune") {
                expected = "[ntt][,exp=P|,length=L][,naf=W][,batch=B][,time=N][,repeat=A:B]";
                if (input.tune_ntt) throw std::invalid_argument("duplicate tune option");
                if (i + 1 < argc) {
                    const std::string next = argv[i + 1];
                    if (next.empty() || (next.front() != '-' &&
                        (std::isalpha(static_cast<unsigned char>(next.front())) || next.front() == ',')))
                        value = argv[++i];
                }
                parse_tune(ecm_stage1::lower_ascii(value), input.tune_grid);
            } else if (values.count(key)) {
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
            else if (key == "--cpu-threads")
                input.cpu_threads = uint32_t(integer(0, UINT32_MAX));
            else if (key == "--gpu-init-threads")
                input.gpu_init_threads = uint32_t(integer(0, UINT32_MAX));
            else if (key == "--impl") {
                expected = "auto|simd|scalar|gwnum";
                if (!ecm_stage1::parse_cpu_implementation(value, selection.cpu_implementation))
                    reject();
            } else if (key == "--reduction") {
                expected = "auto|montgomery|mersenne-fold";
                if (!ecm_stage1::parse_simd_reduction(value, selection.simd_reduction))
                    reject();
            } else if (key == "--cpu-mont-chain") {
                expected = "ladder|prac";
                if (!ecm_stage1::parse_cpu_montgomery_chain(value, selection.cpu_montgomery_chain))
                    reject();
            } else if (key == "--param") {
                expected = "e|m0|m2|m3";
                input.param = ecm_stage1::lower_ascii(value);
                if (input.param != "e" && input.param != "m0" && input.param != "m2" && input.param != "m3")
                    reject();
            } else if (key == "--curve") {
                expected = "mont|ed";
                if (!ecm_stage1::parse_curve_model(value, selection.curve))
                    reject();
            } else if (key == "--gpu-algorithm") {
                expected = "1|2|3|cgbn|ntt|thread or slice list";
                if (!ecm_stage1::parse_gpu_algorithm_policy(value, selection))
                    reject();
            } else if (key == "--gpu-backend") {
                expected = "auto|cuda|opencl";
                if (value != "auto" && value != "cuda" && value != "opencl") reject();
                input.gpu_backend = value;
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
                input.go_settings.gp = value;
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
            else if (key == "-tune" || key == "--tune") {
                input.tune_ntt = true;
            } else if (key == "--tune-exhaustive")
                input.tune_exhaustive = true;
            else if (key == "--tune-seconds")
                input.tune_seconds = unsigned(integer(1, UINT32_MAX));
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
