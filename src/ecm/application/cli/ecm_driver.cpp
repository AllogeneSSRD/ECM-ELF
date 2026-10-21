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
#include "../configuration/ecm_stage1_selection.h"
#include "../../service/stage1/legacy_execution.h"
#include "ecm/application/configuration/runtime.h"
#include "../../service/execution/legacy/ecm.h"
#include "../../service/execution/legacy/cgbn_stage1.h"            /* gpu_pick_random_sigma / gpu_compute_batch_d */
#include "ecm/platform/logging.h"
#include "../configuration/ecm_queue_config.h"
#include "../../storage/queue/ecm_worktodo.h"
#include "../../storage/transfer/ecm_stage2_transfer.h"
#include "../../storage/queue/ecm_worker_list.h"
#include "../configuration/legacy/ecm_expr.h"              /* shared exact integer expression parser */
#include "../../storage/transfer/p95_transfer.h"           /* finished task -> Prime95 worktodo.add (D4/§13) */
#include "ecm/core/stage1/scalar.h"         /* s = torsion * lcm(1..B1), shared product tree */
#include "../../storage/cache/ecm_stage1_exp_cache.h"   /* validated on-disk cache for that value */
#include <random>
#include <algorithm>
#include "../../storage/save/ecm_edwards_io.h"
#include "../../storage/results/ecm_internal_result.h"
#include "../../storage/results/ecm_submission_writer.h"
#include <regex>
#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_HAS_CUDA_EDWARDS
#define ECM_HAS_CUDA_EDWARDS 0
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

using ecm_stage1::Stage1RunOptions;
using ecm_stage1::Stage1ExecutionOptions;
using ecm_stage1::LegacyStage1Result;
using ecm_stage1::Stage1TaskStatus;
using ecm_stage1::Stage1FailureSource;
using ecm_stage1::run_stage1_task;
using ecm_stage1::stage1_publish_factor;
using ecm_stage1::stage1_results_complete;



static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}

// Compute batch product s = torsion * lcm(1..B1).
// The product tree itself lives in src/core/ecm_stage1_exp.cpp, shared with the
// CPU Montgomery path (which used to have its own quadratic builder and spent
// 29 s at B1 = 1e7 before the ladder started); this wrapper only keeps the
// range/lifetime guards and the double -> integer rounding of B1.


static bool parse_sigma_arg(const std::string &arg, uint32_t *sigma_out) {
    std::string s = arg;
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
        s = s.substr(colon + 1);
    }
    try {
        unsigned long long v = std::stoull(s);
        if (v == 0 || v > 0xFFFFFFFFull) {
            return false;
        }
        *sigma_out = (uint32_t)v;
        return true;
    } catch (...) {
        return false;
    }
}

static std::string mpz_to_dec_string(const mpz_t v) {
    char *s = mpz_get_str(nullptr, 10, v);
    std::string out = s ? s : "";
    free(s);
    return out;
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

// 64-bit random sigma for the Edwards (Atkin-Morain) path. Direct port of
// Prime95 ecm.cpp stage-1 init (choose curve, ~line 7397):
//   sigma  = ((uint64_t)(rand() & 0x1F)) << 48;
//   sigma += ((uint64_t)(rand() & 0xFFFF)) << 32;
//   sigma += lo ^ hi ^ ((uint32_t)rand() << 16);   // rdtsc for extra entropy
//   reject sigma <= 5.
// `rand()` is seeded once (srand(time)), and __rdtsc() supplies per-call entropy
// so consecutive curves differ even within the same second.


struct PrimePowerBound {
    uint32_t p;
    uint32_t exp;
};

static bool build_primes_up_to_B1(double B1, std::vector<uint32_t> &primes) {
    primes.clear();
    if (B1 < 2.0) {
        return true;
    }
    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }
    const uint32_t limit = (uint32_t)limit64;
    std::vector<char> sieve((size_t)limit + 1u, 1);
    sieve[0] = sieve[1] = 0;
    for (uint32_t p = 2; (uint64_t)p * (uint64_t)p <= limit; ++p) {
        if (!sieve[p]) continue;
        for (uint64_t q = (uint64_t)p * (uint64_t)p; q <= limit; q += p) {
            sieve[(size_t)q] = 0;
        }
    }

    for (uint32_t p = 2; p <= limit; ++p) {
        if (!sieve[p]) continue;
        primes.push_back(p);
    }
    return true;
}

static std::vector<PrimePowerBound> factor_by_small_primes(
    const mpz_t n, const std::vector<uint32_t> &primes) {
    std::vector<PrimePowerBound> out;
    mpz_t work;
    mpz_init_set(work, n);
    for (uint32_t p : primes) {
        uint32_t exp = 0;
        while (mpz_divisible_ui_p(work, p) != 0) {
            mpz_fdiv_q_ui(work, work, p);
            ++exp;
        }
        if (exp > 0) {
            out.push_back({p, exp});
        }
    }
    mpz_clear(work);
    return out;
}

static std::string format_group_order_smooth(const std::vector<PrimePowerBound> &parts) {
    std::ostringstream oss;
    oss << "[ ";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << "<" << parts[i].p << ", " << parts[i].exp << ">";
    }
    oss << " ]";
    return oss.str();
}

static std::string normalize_gp_path(std::string s) {
    trim(s);
    // Remove any outer single/double quotes repeatedly.
    while (s.size() >= 2 &&
           ((s.front() == '"' && s.back() == '"') ||
            (s.front() == '\'' && s.back() == '\''))) {
        s = s.substr(1, s.size() - 2);
        trim(s);
    }
    while (!s.empty() && (s.front() == '"' || s.front() == '\'')) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == '"' || s.back() == '\'')) {
        s.pop_back();
    }
    s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\''), s.end());
    return s;
}

static bool gp_executable_exists(const std::string &exe_path) {
    const std::string cleaned = normalize_gp_path(exe_path);
    if (cleaned.empty()) return false;
#ifdef _WIN32
    char found[MAX_PATH];
    DWORD len = SearchPathA(nullptr, cleaned.c_str(), ".exe", MAX_PATH, found, nullptr);
    return len > 0;
#else
    if (cleaned.find('/') != std::string::npos) {
        return access(cleaned.c_str(), X_OK) == 0;
    }
    std::string cmd = "which \"" + cleaned + "\" > /dev/null 2>&1";
    return system(cmd.c_str()) == 0;
#endif
}

static std::string resolve_gp_path(const std::string &exe_path) {
    const std::string cleaned = normalize_gp_path(exe_path);
    if (cleaned.empty()) return cleaned;
#ifdef _WIN32
    // If the path contains a directory separator, use it as-is.
    if (cleaned.find('\\') != std::string::npos ||
        cleaned.find('/') != std::string::npos) {
        return cleaned;
    }
    // Bare name → resolve via SearchPath.
    char found[MAX_PATH];
    DWORD len = SearchPathA(nullptr, cleaned.c_str(), ".exe", MAX_PATH, found, nullptr);
    if (len > 0 && len < MAX_PATH) {
        return std::string(found);
    }
#endif
    return cleaned;
}

static std::string get_gp_executable(const std::string &explicit_path = "") {
    // 1. Explicit --gp argument takes highest priority.
    if (!explicit_path.empty()) {
        return normalize_gp_path(explicit_path);
    }
    // 2. Config (set from --gp). 3. Fall back to "gp" on PATH.
    if (!ecm_runtime_config().gp_bin.empty()) {
        return normalize_gp_path(ecm_runtime_config().gp_bin);
    }
    return "gp";
}

/* Group order of the ECM curve attached to `sigma` modulo the prime p, computed by gp/PARI.
 *
 * The four parametrizations follow the canonical FindGroupOrder() script (the same one gmp-ecm
 * ships): param 0 = Suyama sigma (CPU Montgomery and --gpu-param 0, i.e. our own stage
 * 1 default), 1 = "A = 4s^2/2^64-2", 2 = the 6-torsion family, 3 = the historical GPU path
 * (A = 4s/2^32-2).
 *
 * Besides the order we also return gp's exact FACTORIZATION of it (lines "<prime> <exponent>"),
 * because that is what determines the minimum B1/B2 needed to find this factor -- printed by
 * --go as go_min_bounds[].  The previous version hardcoded param 3 and returned only the order,
 * so for a param-0 curve (our default) the reported order was simply the wrong curve's. */
static bool compute_group_order_pari(mpz_t order_out, std::string *factor_lines_out,
                                     const mpz_t p, uint32_t sigma, int param,
                                     const std::string &gp_path, std::string *err) {
    char tmp_file[L_tmpnam];
    if (std::tmpnam(tmp_file) == nullptr) {
        if (err) *err = "failed to create temporary script path";
        return false;
    }
    // tmpnam on Windows can return paths that are awkward for cmd parsing.
    // Put the temporary script in the current workspace with a simple filename.
    std::string base(tmp_file);
    for (char &c : base) {
        if (c == '\\' || c == '/' || c == ':' || c == '.' || c == ' ') {
            c = '_';
        }
    }
    const std::string script_path = "ecm_go_tmp_" + base + ".gp";
    std::ofstream gpfile(script_path, std::ios::out | std::ios::trunc);
    if (!gpfile.is_open()) {
        if (err) *err = "failed to open temporary gp script";
        return false;
    }

    const std::string p_dec = mpz_to_dec_string(p);
    gpfile << "p = " << p_dec << ";\n";
    gpfile << "s = " << sigma << ";\n";
    switch (param) {
        case 0:   /* Suyama sigma: our own stage 1 */
            gpfile << "v = Mod(4*s, p);\n";
            gpfile << "u = Mod(s^2 - 5, p);\n";
            gpfile << "x = u^3;\n";
            gpfile << "A = (3*u + v)*(v - u)^3/(4*x*v) - 2;\n";
            gpfile << "x = x/v^3;\n";
            gpfile << "b = x*(x*(x + A) + 1);\n";
            break;
        case 1:
            gpfile << "A = Mod(4*s^2, p)/Mod(2^64, p) - 2;\n";
            gpfile << "b = 4*A + 10;\n";
            break;
        case 2:
            gpfile << "E0 = ellinit([0, Mod(36, p)]);\n";
            gpfile << "t = ellmul(E0, [-3, 3], s);\n";
            gpfile << "x3 = (3*t[1] + t[2] + 6)/(2*(t[2] - 3));\n";
            gpfile << "A = -(3*x3^4 + 6*x3^2 - 1)/(4*x3^3);\n";
            gpfile << "b = 1/(4*A + 10);\n";
            break;
        default:  /* param 3: the historical GPU path */
            gpfile << "A = Mod(4*s, p)/Mod(2^32, p) - 2;\n";
            gpfile << "b = 4*A + 10;\n";
            break;
    }
    gpfile << "E = ellinit([0, b*A, 0, b^2, 0]);\n";
    gpfile << "n = lift(ellcard(E));\n";
    gpfile << "print(n);\n";
    gpfile << "f = factor(n);\n";
    gpfile << "for(i = 1, #f~, print(f[i,1], \" \", f[i,2]));\n";
    gpfile << "quit();\n";
    gpfile.close();

    std::string output;
    const std::string gp_exe = get_gp_executable(gp_path);
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0)) {
        remove(script_path.c_str());
        if (err) *err = "CreatePipe failed for gp output";
        return false;
    }
    if (!SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(child_stdout_read);
        CloseHandle(child_stdout_write);
        remove(script_path.c_str());
        if (err) *err = "SetHandleInformation failed for gp output";
        return false;
    }

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = child_stdout_write;
    si.hStdError = child_stdout_write;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::string cmdline = "\"" + gp_exe + "\" -q -f \"" + script_path + "\"";
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    BOOL ok = CreateProcessA(
        gp_exe.c_str(),
        cmdline_buf.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi);
    CloseHandle(child_stdout_write);
    if (!ok) {
        DWORD code = GetLastError();
        CloseHandle(child_stdout_read);
        remove(script_path.c_str());
        if (err) *err = "CreateProcess(gp) failed, code=" + std::to_string((unsigned long)code) +
                        ", exe=" + gp_exe;
        return false;
    }

    char buffer[256];
    DWORD nread = 0;
    while (ReadFile(child_stdout_read, buffer, sizeof(buffer) - 1, &nread, nullptr) && nread > 0) {
        buffer[nread] = '\0';
        output += buffer;
    }
    CloseHandle(child_stdout_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 0;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    const std::string cmd = "\"" + gp_exe + "\" -q -f \"" + script_path + "\"";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        remove(script_path.c_str());
        if (err) *err = "failed to launch gp executable: " + gp_exe;
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int rc = pclose(pipe);
#endif
    remove(script_path.c_str());
    if ((long)rc != 0) {
        if (err) *err = "gp execution failed: " + gp_exe;
        return false;
    }

    std::istringstream iss(output);
    std::string line;
    std::string order_str;
    std::string factor_lines;
    while (std::getline(iss, line)) {
        trim(line);
        if (line.empty()) continue;
        bool ok = true;
        size_t start = (line[0] == '-') ? 1 : 0;
        if (start == line.size()) ok = false;
        for (size_t i = start; ok && i < line.size(); ++i) {
            if (!std::isdigit((unsigned char)line[i])) ok = false;
        }
        if (ok) {
            /* the order is printed before the factor lines */
            if (order_str.empty()) order_str = line;
            continue;
        }
        /* "<prime> <exponent>" pairs from factor(n) */
        std::istringstream ls(line);
        std::string a1, a2, extra;
        if ((ls >> a1 >> a2) && !(ls >> extra)) {
            bool ok1 = !a1.empty(), ok2 = !a2.empty();
            for (char c : a1) if (!std::isdigit((unsigned char)c)) ok1 = false;
            for (char c : a2) if (!std::isdigit((unsigned char)c)) ok2 = false;
            if (ok1 && ok2) factor_lines += a1 + " " + a2 + "\n";
        }
    }
    if (order_str.empty()) {
        if (err) *err = "gp returned no integer ellcard output";
        return false;
    }
    if (mpz_set_str(order_out, order_str.c_str(), 10) != 0) {
        if (err) *err = "failed to parse gp ellcard integer";
        return false;
    }
    if (factor_lines_out != nullptr) *factor_lines_out = factor_lines;
    return true;
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
              << "  --go                 Print group order diagnostics (requires gp/PARI)\n"
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

// ── Queue-manager helpers ─────────────────────────────────────────────────

static bool is_absolute_path_local(const std::string &p) {
#ifdef _WIN32
    if (p.size() >= 2 && std::isalpha((unsigned char)p[0]) && p[1] == ':') return true;
    if (p.size() >= 1 && (p[0] == '\\' || p[0] == '/')) return true;
    return false;
#else
    return !p.empty() && p[0] == '/';
#endif
}

static std::string normalize_path_local(std::filesystem::path path) {
    if(path.empty())return {};
    path=path.lexically_normal();
    // A directory's trailing separator is unnecessary, except at a root.
    if(path.filename().empty() && path!=path.root_path())path=path.parent_path();
    return path.make_preferred().string();
}

static std::string resolve_rel_local(const std::string &base, const std::string &p) {
    if(p.empty())return {};
    return normalize_path_local(is_absolute_path_local(p)?std::filesystem::path(p):
                                std::filesystem::path(base)/p);
}

static std::string get_exe_dir_local() {
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

static long long current_epoch_seconds() {
    return static_cast<long long>(std::time(nullptr));
}


static bool stage1_save_construction(const std::string &name,Stage1RunOptions &opt,std::string &error) {
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



static bool stage1_selection_error(const ecm_stage1::Selection &selection,
                                   std::string &error) {
    if (!ecm_stage1::valid_route(selection, ECM_HAS_CUDA_EDWARDS != 0)) {
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

static bool stage1_task_error(const Stage1RunOptions &opt, uint32_t curves,
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



// ---- 自我 checkpoint / 进度显示 (CPU stage-1: Edwards 与 Montgomery 共用) ----
//
// 两个 CPU 方法共用停止请求、输出锁、墙钟、进度条与速率窗口。
// OpenCL 使用自己的进度输出与存档格式，但共用停止请求。


// 多曲线并行时串行化输出 (避免 stdout 行交错).
static std::vector<unsigned> parse_affinity_spec(const std::string &spec) {
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



static void stage1_result_settings(Stage1RunOptions &opt,const EcmQueueConfig &cfg,
                                   const std::string &base,int worker) {
    opt.identity.result_user=cfg.result_user;opt.identity.result_computer=cfg.result_computer;
    opt.persistence.results_file=(std::filesystem::path(base)/(cfg.has_results?cfg.results:ecm_config::worker_file(cfg.results,worker))).string();
    opt.persistence.submission_file=(std::filesystem::path(base)/(cfg.has_submission?cfg.submission:ecm_config::worker_file(cfg.submission,worker))).string();
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
    ecm_stage1::execute_stage1,stage1_build_identity,edwards_get_naf_w};

// Rebuild the original N expression for the save-file N= field.
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

/* A value for the machine-readable `p95_add:` notice lines. The GUI parses them, so a
   free-text reason (or a path with spaces) is quoted; the only escape is `\"` for a quote
   that belongs to the text, plus newlines become spaces -- so a Windows path stays
   readable in the log pane. A value must never end with a lone backslash (it would make
   the closing quote ambiguous), hence the final doubling. Documented in
   src/gui/log_parse.{h,cpp}, which unescapes exactly this way. */
/* Directory part / join for the Prime95 paths. The driver has its own resolve helper for
   ini values, but these two are about Prime95's own directory (prime.log, results.txt). */
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
static std::string p95_quote(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"') { out += '\\'; out += c; }
        else if (c == '\r' || c == '\n') out += ' ';
        else out += c;
    }
    if (!out.empty() && out.back() == '\\') out += '\\';
    return out;
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

// Run one already-parsed queue task (ECM2= or ECMSTAGE2=), report factors, and
// advance the worktodo file. Returns true to keep processing, false to abort
// the queue (including a stop request, which leaves the task in worktodo).
static bool queue_run_one(const mpz_t N, double B1, double B2, uint32_t curves,
                          uint64_t sigma, bool sigma_fixed,
                          const std::string &save_name, const std::string &n_expr,
                          const std::string &aid, const std::string &line,
                          const std::string &worktodo_path, const std::string &finished_path,
                          int worker,
                          const Stage1RunOptions &base_opt,
                          const std::string &exe_dir, const std::string &sync1,
                          const std::string &sync2, bool full_sync, long long marker,
                          int *processed,
                          const P95TransferConfig *p95,const std::string &published_line={},
                          const Stage2TransferConfig *stage2=nullptr,
                          const std::string &stage2_line={},
                          const std::string &auto_stage2_line={}) {
    Stage1RunOptions opt = base_opt;
    opt.identity.result_aid=aid;
    if (sigma_fixed) {
        opt.identity.sigma_fixed = true;
        opt.identity.fixed_sigma64 = sigma;
    }
    {
        std::string input_error;
        if (stage1_selection_error(opt.execution.selection,input_error) || stage1_task_error(opt, curves, input_error)) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", input_error.c_str());
            return false;
        }
    }

    auto result=run_stage1_task({N,B1,B2,curves,save_name,/*append_save=*/true,n_expr,opt},stage1_tasks);

    const bool has_factors = result.has_factors();

    if (result.failure_source()==Stage1FailureSource::Preparation ||
        result.failure_source()==Stage1FailureSource::Publication) {
        ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged.\n",
                       result.failure_source()==Stage1FailureSource::Publication ? "Stage1 save/result publication failed" :
                                               "backend prepare failed");
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
    const char *curve_token = ecm_stage1::curve_model_name(opt.execution.selection.curve);
    const char *target_token = ecm_stage1::execution_target_name(opt.execution.selection.target);
    if (has_factors) {
        for (uint32_t i = 0; i < result.curves(); ++i) {
            if (result.found(i) != ECM_NO_FACTOR_FOUND) {
                char *fs = mpz_get_str(nullptr, 10, result.factor(i));
                std::cout << "factor[" << i << "]=" << (fs ? fs : "?") << " curve=" << i;
                if (result.has_sigmas()) {
                    std::cout << " sigma=" << result.sigma(i);
                }
                if(opt.execution.selection.curve==ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << (opt.execution.selection.target == ecm_stage1::ExecutionTarget::Gpu
                                                   ? opt.execution.selection.gpu_param : 0);
                std::cout << " curve_model=" << curve_token << " execution_target=" << target_token
                          << " save=" << save_name << "\n";
                free(fs);
                found_factor = true;
            }
        }
    }
    if (found_factor) {
        ecm_ts_fprintf(stdout, "FACTOR FOUND aid=%s task=%s\n",
                       aid.empty() ? "N/A" : aid.c_str(), line.c_str());
    }


    if (result.status()==Stage1TaskStatus::Stopped) {
        ecm_ts_fprintf(stdout, "STOP: Stage1 interrupted; worktodo unchanged: %s\n",
                       line.c_str());
        return false;
    }

    if (result.status()==Stage1TaskStatus::Failed) {
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

    const auto &completed_line=published_line.empty()?line:published_line;
    int stage2_receiver=stage2?stage2->worker:worker;
    if (stage2 != nullptr && !found_factor) {
        std::string transfer_error;
        if (!ecm_stage2_transfer_park(*stage2, stage2_line, transfer_error,&stage2_receiver)) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", transfer_error.c_str());
            return false;
        }
    }
    if (!ecm_append_text_line(finished_path, completed_line)) {
        ecm_ts_fprintf(stderr, "ERROR: cannot append to %s\n", finished_path.c_str());
        if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot mark the failed line in %s; aborting\n",
                           worktodo_path.c_str());
            return false;
        }
        return true;
    }
    // A Stage1 hit completes this factor search. Its saved point carries a
    // factor, so Stage2 cannot use it as a clean curve.
    const bool queue_stage2 = !found_factor && !auto_stage2_line.empty();
    if (!ecm_worktodo_advance(worktodo_path, worker, line,
            queue_stage2 ? WorktodoAction::Replace : WorktodoAction::Remove,
            queue_stage2 ? auto_stage2_line : std::string{})) {
        // The task itself succeeded (its .save is on disk), but the queue cannot drop
        // the line. Abort: continuing would re-run it and append a second time.
        ecm_ts_fprintf(stderr, "ERROR: cannot advance the finished line in %s; aborting "
                               "(the task's .save is already written)\n",
                       worktodo_path.c_str());
        return false;
    }

    ecm_sync_save_files(exe_dir, sync1, sync2, full_sync, marker);
    if (!opt.persistence.tmp_dir.empty() && opt.persistence.tmp_dir != exe_dir)
        ecm_sync_save_files(opt.persistence.tmp_dir, sync1, sync2, full_sync, marker);

    if (stage2 != nullptr && !found_factor) {
        size_t remaining = 0;
        std::string transfer_error;
        const bool delivered = ecm_stage2_transfer_flush(stage2->pending, remaining, transfer_error);
        if (delivered)
            ecm_ts_fprintf(stdout, "stage2_transfer: ok worker=%d file=\"%s\"\n",
                           stage2_receiver, stage2->worktodo.c_str());
        else
            ecm_ts_fprintf(stdout, "stage2_transfer: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                           stage2_receiver, remaining, stage2->worktodo.c_str(),
                           p95_quote(transfer_error).c_str());
    }

    /* ── Prime95 handoff (docs/usage/GUI.md) ──────────────────────────────────
       The task is finished and its .save is synchronized, so Prime95's stage 2 can take
       it from worktodo.add. AID, bounds and known factors are preserved; an ECMSTAGE2
       save token names the actual curve-prefixed file. Other fields remain verbatim.
       and a handoff failure NEVER fails the task: the line is parked and re-delivered
       with the next successful delivery, and the GUI shows a red notice meanwhile.
       Supported handoff remains available after local result publication. */
    if (p95 != nullptr && !p95->worktodo_path.empty()) {
        std::string p95_line=completed_line;
        if(p95_line.compare(0,10,"ECMSTAGE1=")==0)p95_line.replace(0,10,"ECMSTAGE2=");
        const P95TransferResult tr = p95_transfer_deliver(*p95, p95_line);
        if (!tr.ok) {
            ecm_ts_fprintf(stdout,
                           "p95_add: pending worker=%d lines=%zu file=\"%s\" error=\"%s\"\n",
                           tr.worker, tr.pending_left, tr.add_path.c_str(),
                           p95_quote(tr.error).c_str());
        } else if (!tr.note.empty()) {
            ecm_ts_fprintf(stdout,
                           "p95_add: warn worker=%d added=%zu pending_delivered=%zu recovered=%zu "
                           "aid=%s file=\"%s\" note=\"%s\"\n",
                           tr.worker, tr.added, tr.pending_delivered, tr.recovered,
                           tr.aid_dropped ? "dropped" : "kept", tr.add_path.c_str(),
                           p95_quote(tr.note).c_str());
        } else {
            ecm_ts_fprintf(stdout,
                           "p95_add: ok worker=%d added=%zu pending_delivered=%zu recovered=%zu "
                           "aid=%s file=\"%s\"\n",
                           tr.worker, tr.added, tr.pending_delivered, tr.recovered,
                           tr.aid_dropped ? "dropped" : "kept", tr.add_path.c_str());
        }
    }

    if (processed) {
        (*processed)++;
    }
    return true;
}

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

static int run_queue_manager(const std::string &ini_path, int worker) {
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
    // `worker` selects the [Worker #N] section; keys before the first section header
    // are the global defaults for every worker (see ecm_queue_config.h / D1).
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
                           worker, remaining, p95_quote(transfer_error).c_str());
    }

    // Initialize handoff only for an eligible actual task. A typed save may override
    // the TOML curve, and unsupported constructions must not touch transfer paths.
    P95TransferConfig p95cfg;
    bool p95_initialized=false,p95_ignore_reported=false;
    auto p95_for_task=[&](const Stage1RunOptions &actual)->const P95TransferConfig* {
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
                           p95_quote(cfg.p95_add_workers).c_str(),
                           p95_quote(add_path).c_str(), parked,
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
                               parked, p95_quote(add_path).c_str(), parked);
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
                                   lost.size(), p95_quote(add_path).c_str(), lost.size());
                }
            }
        }
        return &p95cfg;
    };

    Stage1RunOptions opt;
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

    int processed = 0;
    bool saw_task=false;
    const bool full_sync = (cfg.sync_mode == "full");
    while (true) {
        // Also cover a stop arriving after a successful task but before the next read.
        if (ecm_stage1_stop_requested()) {
            ecm_ts_fprintf(stdout, "STOP: queue interrupted; remaining worktodo unchanged\n");
            if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
            return 1;
        }
        std::string line;
        std::string read_error;
        const WorktodoReadResult read = ecm_worktodo_read_next_line(worktodo_path, worker,
            {cfg.worktodo_priority, cfg.worktodo_skip}, WorktodoStage::Stage1, line, read_error);
        if (read == WorktodoReadResult::Error) {
            ecm_ts_fprintf(stderr, "ERROR: %s\n", read_error.c_str());
            if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
            return 1;
        }
        if (read == WorktodoReadResult::Empty) {
            if(!saw_task)p95_for_task(opt); // Preserve idle-start recovery for an eligible TOML.
            break;
        }
        saw_task=true;

        ecm_ts_fprintf(stdout, "START: %s\n", line.c_str());

        const long long marker = current_epoch_seconds();
        std::string err;

        if (line.compare(0, 5, "ECM2=") == 0 || line.compare(0, 4, "ECM=") == 0) {
            // Prime95 ECM= / ECM2= (等价) 格式: k,b,n,c,B1[,B2][,curves][,sigma][,"factors"].
            Ecm2Task task;
            if (!ecm_parse_ecm2_line(line, task, err)) {
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError))
                    break;
                continue;
            }
            mpz_t N;
            mpz_init(N);
            if (!ecm_compute_ecm2_n(task, N, err)) {
                mpz_clear(N);
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError))
                    break;
                continue;
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
                    mpz_clear(N);
                    if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
                    return 1;
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
                    mpz_clear(N);
                    return 1;
                }
                const auto save = normalize_path_local(std::filesystem::absolute(resolve_rel_local(exe_dir, queue_save_name)));
                const auto skip = stage2_save_record_count(save);
                stage2_line = edwards_stage2_line(task, save, skip);
            }
            const bool cont = queue_run_one(
                N, task.B1, task.B2, task.curves_to_run,
                task.has_sigma ? task.sigma : 0, task.has_sigma,
                queue_save_name, n_expr, task.aid, line,
                worktodo_path, finished_path, worker, opt,
                exe_dir, sync1, sync2, full_sync, marker, &processed, p95_for_task(opt),
                {}, stage2, stage2 != nullptr ? stage2_line : std::string{},
                cfg.auto_stage2 && !auto_stage2_routed ? stage2_line : std::string{});
            mpz_clear(N);
            if (!cont) {
                if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
                return 1;
            }
            continue;
        }

        // ECMSTAGE1= uses the save-based task format. ECMSTAGE2 belongs to Stage2.
        EcmStage2Task task;
        if (!ecm_parse_stage1_line(line, task, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError))
                break;
            continue;
        }

        double B1 = 0.0;
        if (!ecm_extract_b1_from_save_name(task.save_name, &B1, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError))
                break;
            continue;
        }

        mpz_t N;
        mpz_init(N);
        if (!ecm_compute_stage2_n(task, N, err)) {
            mpz_clear(N);
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError))
                break;
            continue;
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
           stage1_selection_error(task_opt.execution.selection,err) || stage1_task_error(task_opt,task.curves_to_run,err)) {
            mpz_clear(N);ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());
            if(logf){ecm_log_set_mirror(nullptr);fclose(logf);}return 1;
        }
        ecm_ts_fprintf(stdout,"queue_save: curve=%s param=%s B1=%.0f\n",
            ecm_stage1::curve_model_name(task_opt.execution.selection.curve),
            task_opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards?"-":stage1_save_prefix(task_opt).substr(1).c_str(),B1);
        const auto queue_save_name=stage1_prefixed_save(task.save_name,stage1_save_prefix(task_opt));
        std::string published_line=line;
        if(queue_save_name!=task.save_name && !ecm_rewrite_stage2_save(line,queue_save_name,published_line,err)) {
            mpz_clear(N);ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());return 1;
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
                mpz_clear(N);
                ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", err.c_str());
                return 1;
            }
            stage2_line.replace(0, 10, "ECMSTAGE2=");
        }
        const bool cont = queue_run_one(
            N, B1, /*B2=*/0.0, task.curves_to_run,
            /*sigma=*/0, /*sigma_fixed=*/false,
            queue_save_name, n_expr, task.aid, line,
            worktodo_path, finished_path, worker, task_opt,
            exe_dir, sync1, sync2, full_sync, marker, &processed, p95_for_task(task_opt),published_line,
            stage2,stage2 != nullptr ? stage2_line : std::string{},
            cfg.auto_stage2 && !auto_stage2_routed ? stage2_line : std::string{});
        mpz_clear(N);
        if (!cont) {
            if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
            return 1;
        }
    }

    ecm_ts_fprintf(stdout, "===== queue done, %d task(s) processed =====\n", processed);
    if (logf) {
        ecm_log_set_mirror(nullptr);
        fclose(logf);
    }
    return 0;
}

int main(int argc, char **argv){
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
        return run_queue_manager(ini_path, worker_index);
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

    // Early check: when --go is requested, ensure gp/PARI is available before any GPU init.
    std::string go_gp_exe;
    if (print_group_order) {
        go_gp_exe = resolve_gp_path(get_gp_executable(gp_bin_path));
        if (!gp_executable_exists(go_gp_exe)) {
            std::cerr << "gp executable not found: " << go_gp_exe << "\n"
                      << "Please provide the gp path with: --gp <path/to/gp>\n"
                      << "(If gp/PARI is installed, ensure 'gp' is on PATH, "
                      << "or use --gp to specify the full path.)" << std::endl;
            return 1;
        }
    }

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

    Stage1RunOptions opt;
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

    std::vector<uint32_t> go_primes;
    if (print_group_order) {
        build_primes_up_to_B1(B1, go_primes);
    }

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
                if (print_group_order) {
                    uint32_t sigma_curve = result.first_sigma() + i;
                    if (mpz_probab_prime_p(result.factor(i), 25) <= 0) {
                        std::cout << "  go_factor[" << i << "]=[ ] (factor is not prime, skip #E(F_p))\n";
                        continue;
                    }
                    mpz_t go;
                    mpz_init(go);
                    std::string err;
                    std::string go_factor_lines;
                    if (!compute_group_order_pari(go, &go_factor_lines, result.factor(i),
                                                  sigma_curve, go_param, go_gp_exe, &err)) {
                        std::cerr << "go_factor[" << i << "]: gp error: " << err << "\n"
                                  << "Please verify gp is working, or provide path with: --gp <path/to/gp>"
                                  << std::endl;
                        mpz_clear(go);
                        return 1;
                    }
                    auto go_parts = factor_by_small_primes(go, go_primes);
                    std::cout << "  go[" << i << "]=" << mpz_to_dec_string(go) << "\n";
                    std::cout << "  go_factor[" << i << "]="
                              << format_group_order_smooth(go_parts) << "\n";
                    /* Minimum bounds that would find THIS factor, straight from the exact
                       factorization of #E(F_p) that gp just printed:
                         min_b1_stage1 : stage 1 alone must cover the largest prime power in the
                                         order (ECM removes each prime in the order once);
                         min_b1_stage2 / min_b2_stage2 : to let STAGE 2 finish the job, stage 1
                                         must cover the order divided by ONE copy of its largest
                                         prime factor, and B2 must reach that prime.
                       This is the deterministic way to build a test vector for a given factor
                       instead of sweeping sigmas. */
                    {
                        std::vector<std::pair<std::string, int>> facs;
                        std::istringstream fs(go_factor_lines);
                        std::string fl;
                        while (std::getline(fs, fl)) {
                            std::istringstream ls(fl);
                            std::string pr;
                            int ex = 0;
                            if (ls >> pr >> ex) facs.push_back(std::make_pair(pr, ex));
                        }
                        mpz_t pw, best, best2;
                        mpz_inits(pw, best, best2, nullptr);
                        mpz_set_ui(best, 0);
                        mpz_set_ui(best2, 1);
                        std::string best_prime, min_b1_stage1, min_b1_stage2 = "1";
                        int best_exp = 0;
                        for (const auto &fp : facs) {
                            mpz_set_str(pw, fp.first.c_str(), 10);
                            mpz_pow_ui(pw, pw, (unsigned long)fp.second);
                            if (mpz_cmp(pw, best) > 0) {
                                mpz_set(best, pw);
                                best_prime = fp.first;
                                best_exp = fp.second;
                                min_b1_stage1 = mpz_to_dec_string(pw);
                            }
                        }
                        for (const auto &fp : facs) {
                            const int e = fp.second - ((fp.first == best_prime) ? 1 : 0);
                            if (e <= 0) continue;
                            mpz_set_str(pw, fp.first.c_str(), 10);
                            mpz_pow_ui(pw, pw, (unsigned long)e);
                            if (mpz_cmp(pw, best2) > 0) {
                                mpz_set(best2, pw);
                                min_b1_stage2 = mpz_to_dec_string(pw);
                            }
                        }
                        std::cout << "  go_min_bounds[" << i << "]: param=" << go_param
                                  << " largest_prime=" << (best_prime.empty() ? "0" : best_prime)
                                  << " min_b1_stage1=" << (min_b1_stage1.empty() ? "0" : min_b1_stage1)
                                  << " min_b1_stage2=" << min_b1_stage2
                                  << " min_b2_stage2=" << (best_prime.empty() ? "0" : best_prime)
                                  << " largest_exponent=" << best_exp << "\n";
                        mpz_clears(pw, best, best2, nullptr);
                    }
                    mpz_clear(go);
                }
            }
        }

    }
    mpz_clear(N);
    /* 让调用方 (脚本/Prime95 前端) 能看到失败, 而不是把 0 当成"跑完了没因子"。 */
    return (rc1 == ECM_ERROR || result.legacy_return() == ECM_ERROR) ? 1 : 0;
}
