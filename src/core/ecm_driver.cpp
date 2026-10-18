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

#include "ecm_backend.h"           /* GPU backend seam (OpenCL or CUDA glue) */
#include "ecm_save.h"
#include "ecm_checkpoint.h"         /* opencl_ecm_set_work_dir */
#include "ecm_stage1_checkpoint_runtime.h"
#include "ecm_stage1_legacy_gpu.h"
#include "ecm_stage1_selection.h"
#include "opencl_ecm_runtime_config.h"
#include "ecm.h"
#include "cgbn_stage1.h"            /* gpu_pick_random_sigma / gpu_compute_batch_d */
#include "opencl_ecm_log.h"
#include "ecm_queue_config.h"
#include "ecm_worktodo.h"
#include "ecm_expr.h"              /* shared exact integer expression parser */
#include "p95_transfer.h"           /* finished task -> Prime95 worktodo.add (D4/§13) */
#include "ecm_edwards_cpu.h"        /* Edwards (Atkin-Morain) stage-1 CPU path */
#include "ecm_mont_cpu.h"           /* Suyama-sigma Montgomery stage-1 (scalar mpn) */
#include "simd_mont_curve.h"        /* Suyama-sigma Montgomery stage-1 (8-lane IFMA) */
#include "ecm_mont_ckpt.h"          /* mid-stage-1 checkpoints (Montgomery CPU) */
#include "ecm_stage1_exp.h"         /* s = torsion * lcm(1..B1), shared product tree */
#include "ecm_stage1_exp_cache.h"   /* validated on-disk cache for that value */
#include <random>
#include <algorithm>
#include "ecm_edwards_save.h"       /* Prime95 ECM 二进制存档读写 */
#include "ecm_edwards_io.h"
#include "ecm_internal_result.h"
#include "ecm_submission_writer.h"
#include <regex>
#ifndef ECM_BUILD_ID
#define ECM_BUILD_ID "development"
#endif
#ifndef ECM_HAS_CUDA_EDWARDS
#define ECM_HAS_CUDA_EDWARDS 0
#endif
/* IFMA_FIELD_* 常量 (SIMD 域选择)。头文件不含 AVX512 intrinsic, 基线 TU 可包含;
   只给 simd_*.cpp 加 /arch:AVX512, 见 §14.1。 */
#include "simd_mont_ifma.h"

static int to_ifma_field_mode(ecm_stage1::SimdReduction reduction) {
    switch (reduction) {
    case ecm_stage1::SimdReduction::Auto: return IFMA_FIELD_AUTO;
    case ecm_stage1::SimdReduction::Montgomery: return IFMA_FIELD_MONT;
    case ecm_stage1::SimdReduction::MersenneFold: return IFMA_FIELD_MERS;
    }
    return IFMA_FIELD_AUTO;
}

static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}

// Compute batch product s = torsion * lcm(1..B1).
// The product tree itself lives in src/core/ecm_stage1_exp.cpp, shared with the
// CPU Montgomery path (which used to have its own quadratic builder and spent
// 29 s at B1 = 1e7 before the ladder started); this wrapper only keeps the
// range/lifetime guards and the double -> integer rounding of B1.
static bool compute_batch_s(mpz_t s, double B1, unsigned torsion = 1){
    if(B1 < 2.0) {
        mpz_set_ui(s, torsion);
        return true;
    }
    const uint64_t limit64 = (uint64_t)std::floor(B1 + 0.0001);
    if (limit64 < 2 || limit64 > 5000000000ULL) {
        return false;
    }
    /* The exponent is cached on disk (validated on load, see ecm_stage1_exp_cache.h): at
       B1 = 260e6 building it costs ~10 s and every task in a worktodo would repeat that. */
    std::string status;
    if (!ecm_build_lcm_exponent_cached(s, limit64, torsion, ecm_exp_cache_get_dir(), &status)) {
        return false;
    }
    ecm_ts_fprintf(stdout, "stage1 exponent built: %s (torsion=%u)\n", status.c_str(), torsion);
    return true;
}

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
static uint64_t random_sigma_u64() {
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned)time(nullptr));
        seeded = true;
    }
    uint64_t sigma;
    do {
        uint32_t hi = 0, lo = 0;
        sigma = ((uint64_t)(rand() & 0x1F)) << 48;
        sigma += ((uint64_t)(rand() & 0xFFFF)) << 32;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        const unsigned __int64 tsc = __rdtsc();
        hi = (uint32_t)(tsc >> 32);
        lo = (uint32_t)tsc;
#endif
        sigma += (uint64_t)(lo ^ hi ^ ((uint32_t)rand() << 16));
    } while (sigma <= 5);
    return sigma;
}

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
              << "                       no file is written. Honours -d and --gpu-param\n"
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
              << "  -ini <path>                Config file (default: <exe dir>/ecm.ini)\n"
              << "  --worker <n>               Which worker this process is (1-based, default 1).\n"
              << "                             Selects the [Worker #n] section of the ini AND of\n"
              << "                             the worktodo file; keys before the first section\n"
              << "                             header are the global defaults for every worker.\n\n"
              << "Examples:\n"
              << "  echo '(2^991-1)' | " << name << " -v --go --curve mont --target gpu -gpucurves 384 1e6 0\n"
              << "  echo '(2^421-1)' | " << name << " --curve mont --target gpu -gpucurves 256 -d 1 1e5 0\n"
              << "  echo '(2^4003-1)' | " << name << " --curve mont --target gpu -gpucurves 384 --add asm_b32 1e6 0\n"
              << "  " << name << " --showkernel\n"
              << "  " << name << " -ini ecm.ini --worker 2      (queue manager, worker #2)\n\n"
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

struct Stage1RunOptions {
    /* CLI and INI resolve the same mathematical and execution choices. */
    ecm_stage1::Selection selection{};
    uint32_t stage1_threads = ecm_config::defaults::stage1_stage1_threads;     // 0 = auto (min(#tasks, #cores)); 1 = 顺序
    long long stage1_t0_ms = 0;      // 进度条计时起点
    uint32_t  stage1_curves = 0;     // 进度条总数
    std::vector<unsigned> affinity_cpus;   // 亲核性 (空 = 不绑定)
    int naf_w = ecm_config::defaults::stage1_naf_w;                   // Edwards 专属: 0 = ecm_edwards_cpu 的默认窗口
    /* Shared automatic save pattern. {curve} separates e/m0/m2/m3, {target}
       retains the Mersenne assignment or actual integer width, {b1} is compact.
       The Stage1 queue reader extracts B1 from the last '_' token. */
    std::string save_name_pattern = ecm_config::defaults::stage1_save_name_pattern;
    std::string legacy_save_source; // Queue-only source when a save gains its curve prefix.
    int verbose = 0;
    int device_index = ecm_config::defaults::stage1_device;
    unsigned long ckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    unsigned ckpt_backups = ecm_config::defaults::stage1_ckpt_backups;
    std::string gpu_mul_path, gpu_sqr_path, gpu_add_path, gpu_sub_path, gpu_special_mult_path;
    bool sigma_fixed = false;
    uint64_t fixed_sigma64 = 0;      // Full seed; GPU PARAM2/3 validate the 32-bit window
    // 本地 stage-1 落盘目录 (空 = 不落盘/不 checkpoint)。
    // One task owns one batch checkpoint; final text saves feed Stage2.
    std::string tmp_dir;
    double handoff_k = 1.0;
    std::string handoff_k_str = "1"; // 原始 k 字符串 (worktodo.add 用)
    uint32_t handoff_b = 2;
    uint32_t handoff_n = 0;
    int32_t handoff_c = -1;
    std::string handoff_factors;     // 透传给 worktodo.add 的已知因子 (逗号分隔)
    bool target_kbnc=false;
    std::string result_user,result_computer,result_aid;
    std::string results_file="stage1_results.jsonl",submission_file="results.json.txt";
};

static bool stage1_publish_factor(const Stage1RunOptions &,mpz_srcptr,uint64_t,uint64_t,mpz_srcptr);

struct Stage1RunResult {
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    bool publish_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    uint64_t firstsigma64 = 0;       // full 64-bit sigma
    mpz_t *factors = nullptr;
    int *array_found = nullptr;
    // Per-curve sigma of THIS batch (malloc'd, `curves` entries, caller frees; may be
    // null when a path does not know them). Needed because one batch does NOT always
    // share a sigma sequence: the Edwards path draws a random 64-bit sigma per curve,
    // so "firstsigma + i" would be wrong there. Consumers: the hit lines (D3, see
    // docs/usage/GUI.md) and therefore the GUI's results.txt sigma list.
    uint64_t *sigmas = nullptr;
};

static std::string mont_format_save_name(const std::string &pattern,const mpz_t N,double B1,
                                         const Stage1RunOptions *options=nullptr);
static std::string stage1_save_prefix(const Stage1RunOptions &opt) {
    if(opt.selection.curve==ecm_stage1::CurveModel::Edwards)return "e";
    return "m"+std::to_string(opt.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.selection.gpu_param:0);
}
static bool stage1_save_construction(const std::string &name,Stage1RunOptions &opt,std::string &error) {
    const auto leaf=ecm_stage1::lower_ascii(std::filesystem::path(name).filename().string());
    const auto prefix=leaf.substr(0,leaf.find('_'));
    if(prefix=="e" || prefix=="ed") {
        opt.selection.curve=ecm_stage1::CurveModel::Edwards;
        opt.selection.scalar_construction=ecm_stage1::ScalarConstruction::PrimePowerLcm;
    } else if(prefix=="m0" || prefix=="m2" || prefix=="m3") {
        opt.selection.curve=ecm_stage1::CurveModel::Montgomery;
        opt.selection.gpu_param=static_cast<unsigned>(prefix[1]-'0');
        if(opt.selection.target==ecm_stage1::ExecutionTarget::Cpu && opt.selection.gpu_param!=0) {
            error="save prefix selects Montgomery PARAM"+prefix.substr(1)+", which requires target=gpu";
            return false;
        }
    }
    return true;
}
static std::string stage1_prefixed_save(const std::string &name,const std::string &prefix) {
    const auto file=std::filesystem::path(name);auto leaf=file.filename().string();
    if(leaf.rfind(prefix+"_",0)==0)return name;
    const auto lower=ecm_stage1::lower_ascii(leaf);
    for(const auto *old:{"ed_","e_","m0_","m2_","m3_"})
        if(lower.rfind(old,0)==0){leaf.erase(0,std::strlen(old));break;}
    return (file.parent_path()/(prefix+"_"+leaf)).string();
}
static void stage1_copy_save_history(const Stage1RunOptions &opt,const std::string &destination) {
    if(opt.legacy_save_source.empty())return;
    const auto source=opencl_ecm_resolve_data_path_buf(opt.legacy_save_source.c_str());
    const auto target=opencl_ecm_resolve_data_path_buf(destination.c_str());
    if((std::filesystem::exists(target) && std::filesystem::file_size(target)) ||
       !std::filesystem::exists(source) || !std::filesystem::file_size(source))return;
    std::ifstream input(source,std::ios::binary);if(!input)throw std::runtime_error("cannot read legacy save history");
    std::string line;uint64_t records=0;
    const bool ed=opt.selection.curve==ecm_stage1::CurveModel::Edwards;
    const auto param=opt.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.selection.gpu_param:0;
    const auto trim=[](std::string s){const auto a=s.find_first_not_of(" \r\n\t"),b=s.find_last_not_of(" \r\n\t");return a==s.npos?std::string{}:s.substr(a,b-a+1);};
    while(std::getline(input,line)) {
        auto text=trim(line);
        if(text.compare(0,3,"\xef\xbb\xbf")==0)text.erase(0,3);
        if(text.empty() || text[0]=='#' || text[0]==';')continue;
        std::string method,kind="0";size_t at=0;
        while(at<text.size()) {
            const auto end=text.find(';',at);const auto field=trim(text.substr(at,end==text.npos?end:end-at));
            at=end==text.npos?text.size():end+1;const auto eq=field.find('=');if(eq==field.npos)continue;
            const auto key=trim(field.substr(0,eq)),value=trim(field.substr(eq+1));
            if(key=="METHOD")method=value;else if(key=="PARAM")kind=value;
        }
        if((ed && method!="ECM-EDWARDS-AM") || (!ed && (method!="ECM" || kind!=std::to_string(param))))
            throw std::runtime_error("legacy save belongs to another curve/parametrization; choose a separate curve-prefixed save name");
        ++records;
    }
    if(input.bad())throw std::runtime_error("cannot read complete legacy save history");
    if(!records)return;
    if(!opencl_ecm_ensure_parent_dir(target.c_str()))throw std::runtime_error("cannot create prefixed save directory");
    const auto temp=ecm_stage1_checkpoint_temp_path(target);
    std::filesystem::copy_file(source,temp,std::filesystem::copy_options::overwrite_existing);
    if(!ecm_stage1_commit_checkpoint(target,temp))throw std::runtime_error("cannot publish prefixed save history");
    ecm_ts_fprintf(stdout,"Stage1 save history copied: source=%s target=%s records=%llu original_retained=1\n",
                   source.c_str(),target.c_str(),static_cast<unsigned long long>(records));
}

// Retained legacy sources must not replay curves already durably published.
static bool stage1_results_complete(const Stage1RunOptions &opt,mpz_srcptr n,uint64_t b1,
                                     bool edwards,unsigned param,const std::vector<uint64_t> &sigmas) {
    if(!ecm_results::has_result_ledger(opt.results_file))return false;
    ecm_results::SubmissionLock lock(opt.results_file);
    const auto ledger=ecm_results::prepare_internal_ledger(opt.results_file);
    for(const auto sigma:sigmas) {
        const auto key=ecm_edwards::hex(n)+"|"+(edwards?"ed":"mont"+std::to_string(param))+"|"+
            std::to_string(sigma)+"|"+std::to_string(b1)+"|stage1";
        const auto entry=ledger/(std::to_string(ecm_edwards::hash(key))+".curve");
        if(!std::filesystem::exists(entry))return false;
        if(ecm_results::read_submission_entry(entry).key!=key)throw std::runtime_error("legacy result identity collision");
    }
    return !sigmas.empty();
}

static bool edwards_capture_save(std::string &line,const mpz_t n,uint64_t sigma,uint64_t b1,
                                  const mpz_t qx,const mpz_t qz,const mpz_t factor,bool hit) {
    ecm_edwards::Integer x,g,inv;
    if(hit) {
        if(mpz_cmp_ui(factor,1)<=0 || mpz_cmp(factor,n)>=0 || !mpz_divisible_p(n,factor))return false;
        mpz_set(x.v,factor);
    } else {
        mpz_gcd(g.v,qz,n);
        if(mpz_cmp(g.v,n)==0) {
            ecm_ts_fprintf(stdout,"Edwards identity: sigma=%llu gcd(Qz,N)=N, no Stage2 point\n",
                           static_cast<unsigned long long>(sigma));
            line.clear();return true;
        }
        if(mpz_cmp_ui(g.v,1)!=0 || !mpz_invert(inv.v,qz,n))return false;
        mpz_mul(x.v,qx,inv.v);mpz_mod(x.v,x.v,n);
    }
    line=ecm_edwards::save_line(n,x.v,sigma,b1,hit);return true;
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
    if (opt.sigma_fixed && curves != 0 &&
        static_cast<uint64_t>(curves - 1) > UINT64_MAX - opt.fixed_sigma64) {
        error = "sigma + curve index exceeds 2^64-1";
        return true;
    }
    if (opt.selection.target != ecm_stage1::ExecutionTarget::Gpu ||
        opt.selection.curve == ecm_stage1::CurveModel::Edwards ||
        opt.selection.gpu_param == 0 || !opt.sigma_fixed) return false;
    if (opt.fixed_sigma64 > 0xFFFFFFFFull) {
        error = "GPU batch sigma exceeds 2^32-1";
        return true;
    }
    if (curves != 0 && curves > 0x100000000ull - opt.fixed_sigma64) {
        error = "GPU batch sigma + curves must be <= 2^32";
        return true;
    }
    return false;
}

static bool fixed_sigma_batch_exceeds_prime95_range(uint64_t sigma, uint32_t curves) {
    return curves != 0 &&
           (sigma > static_cast<uint64_t>(INT64_MAX) ||
            static_cast<uint64_t>(curves - 1) > static_cast<uint64_t>(INT64_MAX) - sigma);
}

// ---- 自我 checkpoint / 进度显示 (CPU stage-1: Edwards 与 Montgomery 共用) ----
//
// 两个 CPU 方法共用停止请求、输出锁、墙钟、进度条与速率窗口。
// OpenCL 使用自己的进度输出与存档格式，但共用停止请求。


// 多曲线并行时串行化输出 (避免 stdout 行交错).
static std::mutex g_stage1_out_mutex;

static long long stage1_now_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Local Stage1 text-save publication and batch checkpoint records.

// 创建目录 (已存在视为成功).
static bool ensure_dir(const std::string &dir) {
    if (dir.empty()) return true;
#ifdef _WIN32
    if (_mkdir(dir.c_str()) == 0) return true;
#else
    if (mkdir(dir.c_str(), 0777) == 0) return true;
#endif
    return access(dir.c_str(), 0) == 0;
}

// 自我 checkpoint 上下文 + 进度回调 (写 STAGE1 存档)
struct EdwardsCheckpointCtx {
    const Stage1RunOptions *opt;
    uint64_t sigma;
    uint64_t B1, B2;
    std::string save_path;
    long long interval_ms;
    long long last_ms;
    mpz_srcptr N;
    mpz_srcptr s;
    bool write_failed = false;
};

static int edwards_checkpoint_progress(void *ctx, const edwards_checkpoint_t *cur) {
    EdwardsCheckpointCtx *c = (EdwardsCheckpointCtx *)ctx;
    const long long now = stage1_now_ms();
    if (c->save_path.empty()) return ecm_stage1_stop_requested()?0:1;
    const bool due = c->interval_ms>0 && (now - c->last_ms >= c->interval_ms);
    if (!due && !ecm_stage1_stop_requested()) return 1;   // 继续

    mpz_t d, Px, Py;
    mpz_inits(d, Px, Py, NULL);
    edwards_atkin_morain(d, Px, Py, c->sigma, c->N);

    ecm_save_common cm;
    cm.k = c->opt->handoff_k;
    cm.b = c->opt->handoff_b;
    cm.n = c->opt->handoff_n;
    cm.c = c->opt->handoff_c;
    cm.curve = 1;
    cm.B1 = c->B1;
    cm.B2 = c->B2;
    cm.sigma = c->sigma;
    const uint32_t expbuf = (uint32_t)mpz_sizeinbase(c->s, 2);
    const uint32_t dict_size = (uint32_t)1u << (edwards_get_naf_w() - 2);
    const bool written = ecm_edwards_write_stage1(c->save_path, cm, 2, expbuf, cur->bitnum, dict_size,
                                 Px, Py, cur->Rx, cur->Ry, cur->Rz);
    c->last_ms = now;
    mpz_clears(d, Px, Py, NULL);
    if (!written) {
        c->write_failed = true;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 checkpoint %s\n",
                       c->save_path.c_str());
        return 0;
    }

    if (ecm_stage1_stop_requested()) {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        ecm_ts_fprintf(stdout, "checkpoint written (stop): %s @ bitnum=%u\n",
                       c->save_path.c_str(), cur->bitnum);
        return 0;   // 中止
    }
    return 1;       // 继续
}

// ---------------------------------------------------------------------------
// 单条曲线的 stage-1 执行 (顺序/并行共用)
// ---------------------------------------------------------------------------
//
// 线程安全约定: N 与 s 只读共享 (GMP 允许同一 mpz_t 的并发只读访问);
// 输出 (factor_out / Qx,Qz / checkpoint 文件) 均按曲线索引独立, 无共享写.
// 返回: 1=因子, 0=无因子, -1=错误, 2=被 SIGINT 中止(已写 checkpoint).
/* 已完成曲线数 (文件作用域, 让"曲线内"的进度回调也能读到) */
static std::atomic<uint32_t> g_stage1_done(0);

/* 定义在后面 (需要 g_stage1_bar / stage1_progress_set), 这里先声明给调用点用 */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur);

static bool edwards_completed_read(uint32_t curve,uint64_t sigma,mpz_t factor,int &rc,std::string &line) {
    std::string bytes;if(!ecm_stage1_read_blob(ecm_stage1_curve_record(curve),bytes))return false;
    std::istringstream in(bytes);std::string tag,value,extra;uint64_t saved_sigma;
    if(!(in>>tag) || tag!="ECM_EDWARDS_DONE_V1" || !(in>>saved_sigma>>rc>>value>>std::quoted(line)) ||
       (in>>extra) || saved_sigma!=sigma || (rc!=0 && rc!=1) || mpz_set_str(factor,value.c_str(),16))return false;
    return true;
}
static bool edwards_completed_write(uint32_t curve,uint64_t sigma,const mpz_t factor,int rc,const std::string &line) {
    std::ostringstream out;out<<"ECM_EDWARDS_DONE_V1\n"<<sigma<<' '<<rc<<' '<<ecm_edwards::hex(factor)<<' '<<std::quoted(line)<<'\n';
    return ecm_stage1_write_blob(ecm_stage1_curve_record(curve),out.str());
}

static int edwards_run_one_curve(mpz_srcptr N, mpz_srcptr s,
                                 const Stage1RunOptions &opt, double B1, double B2,
                                 uint32_t idx, uint64_t sigma, uint32_t curves,
                                 mpz_t factor_out, const std::string &ckpt_path,std::string &save_line) {
    mpz_t Qx, Qz;
    mpz_inits(Qx, Qz, NULL);

    int rc;
    if(!ckpt_path.empty() && edwards_completed_read(idx,sigma,factor_out,rc,save_line)) {
        mpz_clears(Qx,Qz,NULL);return rc;
    }
    {
        EdwardsCheckpointCtx ck;
        ck.opt = &opt;
        ck.sigma = sigma;
        ck.B1 = (uint64_t)B1;
        ck.B2 = (uint64_t)B2;
        ck.save_path = ckpt_path;
        ck.interval_ms = (long long)opt.ckpt_ms;
        ck.last_ms = stage1_now_ms();
        ck.N = N;
        ck.s = s;

        // 检测并恢复 STAGE1 checkpoint
        edwards_checkpoint_t resume;
        edwards_checkpoint_init(&resume);
        {
            ecm_save_common rcm;
            uint64_t rsp; uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy, rex, rey, rez;
            mpz_inits(rdx, rdy, rex, rey, rez, NULL);
            if (ecm_edwards_read_stage1(ckpt_path, rcm, &rsp, &rebs, &rbn, &rds,
                                        rdx, rdy, rex, rey, rez)) {
                if (rcm.sigma == sigma && rcm.B1 == (uint64_t)B1 && rbn > 0) {
                    resume.bitnum = rbn;
                    mpz_set(resume.Rx, rex);
                    mpz_set(resume.Ry, rey);
                    mpz_set(resume.Rz, rez);
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stdout, "curve %u: resume STAGE1 checkpoint @ bitnum=%u\n",
                                   idx + 1, rbn);
                }
            }
            mpz_clears(rdx, rdy, rex, rey, rez, NULL);
        }

        const uint32_t chunk_bits = static_cast<uint32_t>(std::clamp<size_t>(mpz_sizeinbase(s,2)/512,64,4096));
        rc = edwards_stage1_curve_progress(factor_out, Qx, Qz, N, sigma, s, chunk_bits,
                                           resume.bitnum > 0 ? &resume : nullptr,
                                           edwards_checkpoint_progress_bar, &ck);
        edwards_checkpoint_clear(&resume);

        if (rc == 2) {
            mpz_clears(Qx, Qz, NULL);
            return ck.write_failed ? ECM_ERROR : 2;
        }
    }

    // Capture the canonical text-save record and retain the completed result.
    if (rc == 0 || rc == 1) {
        if(!edwards_capture_save(save_line,N,sigma,uint64_t(B1),Qx,Qz,factor_out,rc==1))rc=ECM_ERROR;
        if(rc>=0 && !ckpt_path.empty() && !edwards_completed_write(idx,sigma,factor_out,rc,save_line))rc=ECM_ERROR;
    }
    mpz_clears(Qx, Qz, NULL);
    return rc;
}

// ---------------------------------------------------------------------------
// 多曲线并行 (每线程一条曲线, 动态取号)
// ---------------------------------------------------------------------------

struct EdwardsWorkerCtx {
    mpz_srcptr N;
    mpz_srcptr s;
    const Stage1RunOptions *opt;
    double B1, B2;
    uint32_t curves;
    const uint64_t *sigmas;
    const std::string *ckpt_paths;
    mpz_t *factors;
    int *array_found;
    std::string *save_lines;
    std::atomic<uint32_t> *next;
    std::atomic<int> *aborted;
    /* SIMD 批量 (8 曲线/批) 相关 */
    int simd_ok = 0;
    std::atomic<uint32_t> *done_ctr = nullptr;
};

// ---------------------------------------------------------------------------
// SIMD 批量 stage-1 (8 曲线/批, AVX512-IFMA)
//
// 每批构造独立字典，逐 lane 生成统一文本记录，整个任务发布后清理检查点。
// SIMD 进度回调写入各 lane 检查点；同批位号一致时可恢复。
// 构造或字典逆元异常时逐曲线检查因子，不能继续使用失效的仿射字典。
// ---------------------------------------------------------------------------
#include "simd_edwards.h"   /* SoA 批量 Edwards 层 (AVX512-IFMA TU, 见 §13.8) */

/* ---------------------------------------------------------------------------
 * stage-1 进度条 (照 opencl_ecm_stage1.cpp 的 CUDA host 写法)
 *
 * 批量模式下一批 8 条曲线是不可分割的工作单元, 所以进度以"曲线"为粒度、每批跳一格;
 * 更新点都在 g_stage1_out_mutex 保护下, 避免多条 worker 线程同时动光标。
 * ------------------------------------------------------------------------- */
#include "indicators/indicators.hpp"

static indicators::ProgressBar *g_stage1_bar = nullptr;

/* ---------------------------------------------------------------------------
 * 速率显示 (对齐 kernels/cuda/cgbn_stage1.cu 的 print_progress)
 *
 * CUDA 那套的要点有两个, 这里都照做:
 *  1) 平均的是"速度样本"而不是时间: 每个采样点算 speed = delta_work/delta_t,
 *     放进环形窗口, avg = sum/count; 然后用 avg 反推 per_curve 与 remaining。
 *     这样并行批数/频率变化时显示不会乱跳 (直接平均时间就会跳)。
 *  2) TTY 与非 TTY 两种输出: 终端里原地 \r 更新; 重定向到日志时按衰减节奏打
 *     整行 (前 3 次、之后 10/100/1000/10000 的倍数), 否则日志会被刷爆。
 * work 单位是"曲线"(可为小数, 批内按 bit 比例折算), 与总曲线数同一量纲。
 * ------------------------------------------------------------------------- */
#define ED_SPEED_WINDOW 12
struct EdSpeedMeter {
    double ring[ED_SPEED_WINDOW];
    int count, idx;
    double sum;
    double t_last, w_last, t0;
    bool started;                    /* 第一个采样点只用来定基准, 不产生速度样本 */
    EdSpeedMeter() : count(0), idx(0), sum(0.0), t_last(0.0), w_last(0.0), t0(0.0),
                     started(false) {}
    void reset() { count = 0; idx = 0; sum = 0.0; t_last = 0.0; w_last = 0.0; t0 = 0.0;
                   started = false; }
    void sample(double now, double work) {
        if (!started) { started = true; t0 = now; t_last = now; w_last = work; return; }
        const double dt = now - t_last, dw = work - w_last;
        t_last = now;
        w_last = work;
        if (dt < 0.05 || dw <= 1e-12) return;       /* 极小 dt (收尾跳变) 会造出虚高速度样本 */
        const double speed = dw / dt;               /* 曲线/秒 */
        if (count < ED_SPEED_WINDOW) {
            ring[count++] = speed;
            sum += speed;
        } else {
            sum -= ring[idx];
            ring[idx] = speed;
            sum += speed;
            idx = (idx + 1) % ED_SPEED_WINDOW;
        }
    }
    double avg_speed() const { return count > 0 ? sum / (double)count : 0.0; }
};

static EdSpeedMeter g_speed;

static bool stdout_is_tty_local(void) {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

/* 重定向到日志时的打印节奏 (与 CUDA 的 emit_progress_line 一致) */
static bool emit_progress_line(unsigned n) {
    return (n < 3u) || (n < 30u && n % 10u == 0u) || (n < 500u && n % 100u == 0u) ||
           (n < 5000u && n % 1000u == 0u) || (n % 10000u == 0u);
}

/* ASCII 进度条 (日志模式下用, 与 CUDA print_progress 的条形一致) */
static const char *progress_bar_ascii(double pct) {
    static char bar[41];
    const int width = 40;
    int filled = (int)(width * (pct / 100.0));
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    for (int i = 0; i < width; i++) bar[i] = (i < filled) ? '=' : ' ';
    bar[width] = '\0';
    if (filled > 0 && filled < width) bar[filled - 1] = '>';
    return bar;
}

static void stage1_progress_set(double done, uint32_t total) {
    if (total == 0) return;
    /* 进度/速率估计必须单调: 多线程下各线程上报的 (已完成曲线 + 本曲线比例) 会互相穿插
       (A 报 3.5, B 接着报 5.2, A 下一次又报 3.6), 不单调会产生"大增量/小时间"的虚高
       速度样本, 把显示速率放大十几倍。取运行最大值即可。 */
    static double s_max_done = 0.0;
    if (done <= 0.0) { g_speed.reset(); s_max_done = 0.0; }
    if (done < s_max_done) done = s_max_done; else s_max_done = done;
    const double now = (double)stage1_now_ms() / 1000.0;
    g_speed.sample(now, done);

    double pct = 100.0 * done / (double)total;
    if (pct > 100.0) pct = 100.0;
    const double avg = g_speed.avg_speed();
    const double per_curve_s = avg > 0.0 ? 1.0 / avg : 0.0;              /* 由均值反推 */
    const double elapsed_s = (g_speed.t0 > 0.0) ? now - g_speed.t0 : 0.0;
    double remaining_s = 0.0;
    if (avg > 0.0 && done < (double)total) remaining_s = ((double)total - done) / avg;

    if (!stdout_is_tty_local()) {
        /* 日志模式: 不打 \r, 按衰减节奏打整行 (带时间戳, 会被 mirror 到 log 文件)。
           但"只看批号"是错的: 批号会从 checkpoint 恢复, 恢复后的运行只在 n % 10000 == 0
           时才打 —— 每批约 1 s 的话就是 ~1.7 小时一行, GUI 会一直显示"等待第一行进度"
           (2026-09-29 实测)。这里同样按 **时间** 兜底: 距上一行 ≥200 ms 就打一行,
           与交互式分支每批更新的节奏一致。 */
        static unsigned lines = 0;
        static std::atomic<uint32_t> last_emitted(0xFFFFFFFFu);
        static double last_line_s = -1.0e9;
        const uint32_t done_i = (uint32_t)done;
        const bool by_time = (now - last_line_s) >= 0.2;
        if (done >= (double)total || by_time || emit_progress_line(++lines)) {
            last_emitted.store(done_i, std::memory_order_relaxed);
            last_line_s = now;
            ecm_ts_fprintf(stdout,
                           "stage1: [%s] %.1f%%  %.1f/%u (~%.2f s/curve)  "
                           "elapsed %.1fs  ETA %.1fs\n",
                           progress_bar_ascii(pct), pct, done, total, per_curve_s,
                           elapsed_s, remaining_s);
        }
        return;
    }

    if (!g_stage1_bar) return;
    try {
        g_stage1_bar->set_progress((float)pct);
        char postfix[200];
        snprintf(postfix, sizeof(postfix),
                 "%.1f%%  %.1f/%u (~%.2f s/curve)  elapsed %.1fs  ETA %.1fs",
                 pct, done, total, per_curve_s, elapsed_s, remaining_s);
        g_stage1_bar->set_option(indicators::option::PostfixText{postfix});
        if (done >= (double)total) g_stage1_bar->mark_as_completed();
    } catch (...) {
        /* 进度条永远不能影响 stage-1 的正确性 */
    }
}

/* ---------------------------------------------------------------------------
 * 进度条对象: Edwards 与 Montgomery 各跑一个 stage-1 任务, 所以做成 create/destroy
 * 一对, 由 run_*_stage1 自己持有。g_stage1_bar 只是给 worker 回调看的"当前条",
 * destroy 里必须清空 —— 否则调用方返回后它还指着已析构的局部对象。
 *
 * 只对 TTY 建条: 重定向到日志时 stage1_progress_set() 走 ASCII 整行输出, 不需要
 * 终端控制序列 (建条本身也会往 stdout 写东西, 日志里会多出垃圾)。
 * ------------------------------------------------------------------------- */
static indicators::ProgressBar *stage1_bar_create() {
    if (!stdout_is_tty_local()) return nullptr;
    auto *bar = new indicators::ProgressBar{
        indicators::option::BarWidth{40},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PrefixText{"stage1: "},
        indicators::option::PostfixText{""},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::ForegroundColor{indicators::Color::cyan},
        indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};
    g_stage1_bar = bar;
    return bar;
}

static void stage1_bar_destroy(indicators::ProgressBar *bar) {
    g_stage1_bar = nullptr;
    if (!bar) return;
    try { bar->mark_as_completed(); } catch (...) {}
    delete bar;
}

/* 批内进度: bits 粒度, 把"已完成曲线 + 本批已跑比例 × 批大小"折算成总进度。
   同时承担 checkpoint: 每 ckpt_ms(或收到 SIGINT) 就按 lane 把当前点落盘, 存档格式
   与标量路径**完全相同**, 所以后续用标量或 SIMD 续跑都能读。 */
struct EdSimdProg {
    std::atomic<uint32_t> *done_ctr;
    uint32_t curves, batch_first, batch_count;
    size_t   s_bits;
    long long t0_ms;
    const Stage1RunOptions *opt;
    double   B1, B2;
    const uint64_t *sigmas;
    const ifma_ctx_t *mc;
    long long last_ms;
    long long interval_ms;
    bool write_failed = false;
};

static int edwards_simd_progress(void *p, size_t bits_done, size_t bits_total,
                                const uint64_t *Rx, const uint64_t *Ry, const uint64_t *Rz) {
    EdSimdProg *pr = (EdSimdProg *)p;
    const ifma_ctx_t *mc = pr->mc;

    /* --- checkpoint (与标量路径同一套写入器/路径/字段) --- */
    const long long now = stage1_now_ms();
    const bool due = !pr->opt->tmp_dir.empty() && ((pr->interval_ms>0 && now-pr->last_ms>=pr->interval_ms) || ecm_stage1_stop_requested());
    if (due && bits_done > 0) {
        if (mc) {
            mpz_t d, Px, Py, rx, ry, rz;
            mpz_inits(d, Px, Py, rx, ry, rz, NULL);
            size_t written = 0;
            for (uint32_t k = 0; k < pr->batch_count; k++) {
                const uint32_t i = pr->batch_first + k;
                const std::string path = ecm_stage1_curve_record(i);
                ecm_save_common cm;
                cm.k = pr->opt->handoff_k;
                cm.b = pr->opt->handoff_b;
                cm.n = pr->opt->handoff_n;
                cm.c = pr->opt->handoff_c;
                cm.curve = 1;
                cm.B1 = (uint64_t)pr->B1;
                cm.B2 = (uint64_t)pr->B2;
                cm.sigma = pr->sigmas[i];
                edwards_atkin_morain(d, Px, Py, pr->sigmas[i], mc->N);
                ifma_to_mpz_lane(rx, Rx, k, mc);
                ifma_to_mpz_lane(ry, Ry, k, mc);
                ifma_to_mpz_lane(rz, Rz, k, mc);
                const uint32_t expbuf = (uint32_t)pr->s_bits;
                const uint32_t dict_size = (uint32_t)1u << (edwards_get_naf_w() - 2);
                if (ecm_edwards_write_stage1(path, cm, 2, expbuf, (uint32_t)bits_done,
                                             dict_size, Px, Py, rx, ry, rz)) {
                    written++;
                } else {
                    pr->write_failed = true;
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 checkpoint %s\n",
                                   path.c_str());
                }
            }
            pr->last_ms = now;
            mpz_clears(d, Px, Py, rx, ry, rz, NULL);
            if (written && (ecm_stage1_stop_requested() || getenv("ED_SOA_DEBUG"))) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                ecm_ts_fprintf(stdout, "checkpoint written (simd batch %u): %zu lane(s) @ bitnum=%zu\n",
                               pr->batch_first, written, bits_done);
            }
        }
    }

    /* --- 进度条 --- */
    {
        const uint32_t base = pr->done_ctr ? pr->done_ctr->load(std::memory_order_relaxed) : 0;
        const double frac = (bits_total > 0) ? (double)bits_done / (double)bits_total : 0.0;
        const double done_f = (double)base + frac * (double)pr->batch_count;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done_f, pr->curves);
    }

    return (ecm_stage1_stop_requested() || pr->write_failed) ? 1 : 0;
}

/* 标量路径的"曲线内"进度: 包一层 checkpoint 回调, 按动态分块位数更新。
   没有它的话 B1 很大时一条曲线要跑几分钟, 进度条看着就是卡死的。
   (定义在这里而不是函数旁边, 因为要用到上面的 g_stage1_bar/stage1_progress_set。) */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur) {
    if (ctx) {
        EdwardsCheckpointCtx *ck = (EdwardsCheckpointCtx *)ctx;
        if (ck->opt && ck->s && ck->opt->stage1_curves > 0) {
            const size_t s_bits = (size_t)mpz_sizeinbase(ck->s, 2);
            if (s_bits > 0) {
                const uint32_t base = g_stage1_done.load(std::memory_order_relaxed);
                double frac = (double)cur->bitnum / (double)s_bits;
                if (frac > 1.0) frac = 1.0;
                const double done_f = (double)base + frac;
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                stage1_progress_set(done_f, ck->opt->stage1_curves);
            }
        }
    }
    return edwards_checkpoint_progress(ctx, cur);
}

static int edwards_run_batch_simd_impl(const Stage1RunOptions &opt, mpz_srcptr N, mpz_srcptr s,
                                       double B1, double B2, uint32_t first, uint32_t count,
                                       const uint64_t *sigmas, mpz_t *factors, int *array_found,
                                       std::atomic<uint32_t> *done_ctr,std::string *save_lines) {
    if(!opt.tmp_dir.empty()) {
        bool completed=true;std::vector<int> statuses(count);
        for(uint32_t k=0;k<count;++k)if(!edwards_completed_read(first+k,sigmas[first+k],factors[first+k],statuses[k],save_lines[first+k])){completed=false;break;}
        if(completed) {
            for(uint32_t k=0;k<count;++k)array_found[first+k]=statuses[k]?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
            if(done_ctr)done_ctr->fetch_add(count,std::memory_order_relaxed);
            return 0;
        }
    }
    // Exceptional setup lanes need scalar factor/error classification before
    // building an affine SIMD dictionary (which requires invertible inputs).
    bool exceptional=false;
    for(uint32_t k=0;k<count;++k) {
        ecm_edwards::Integer d,x,y,a24,f;
        if(ecm_edwards::construct(d.v,x.v,y.v,a24.v,f.v,sigmas[first+k],N)!=ecm_edwards::Setup::Ready)exceptional=true;
    }
    auto run_exceptional=[&]() {
        for(uint32_t k=0;k<count;++k) {
            const uint32_t i=first+k;
            const auto path=opt.tmp_dir.empty()?std::string():ecm_stage1_curve_record(i);
            const int rc=edwards_run_one_curve(N,s,opt,B1,B2,i,sigmas[i],opt.stage1_curves,factors[i],path,save_lines[i]);
            if(rc==2 || rc<0)return rc;
            array_found[i]=rc==1?ECM_FACTOR_FOUND_STEP1:ECM_NO_FACTOR_FOUND;
            if(rc==1 && !stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i]))return ECM_ERROR;
            if(done_ctr)done_ctr->fetch_add(1,std::memory_order_relaxed);
        }
        return 0;
    };
    if(exceptional)return run_exceptional();
    ed_soa_ctx_t ctx;
    /* 字典窗口跟随配置 (naf_w / ini), 不再硬编码 w=8:
       这样 SIMD 与标量用同一套 digit/字典, 存档逐字节可比, 也便于混合与续跑。
       代价是字典内存 = 3*2^(w-2)*8n 字节 (w=12/n=155 约 29 MB/批), 见启动信息。 */
    int w = edwards_get_naf_w();
    if (w < 3 || w > 12) w = 8;
    {
        const int rc = ed_soa_init_ex(&ctx, N, w, to_ifma_field_mode(opt.selection.simd_reduction));
        if (rc != 0) {
            /* 保持与原来完全相同的控制流 (先尝试建 ctx, 失败就返回 -1), 只把
               "SIMD stage-1 internal error" 这句含糊的日志换成能看出原因的话:
               reduction=mersenne-fold 要求 N = 2^k-1 (k>=64), 这是最常见的误用。 */
            bool want_mers = (opt.selection.simd_reduction == ecm_stage1::SimdReduction::MersenneFold);
            if (want_mers) {
                mpz_t t;
                mpz_init(t);
                mpz_add_ui(t, N, 1);
                const bool is_mersenne = (mpz_popcount(t) == 1 && mpz_scan1(t, 0) >= 64);
                mpz_clear(t);
                want_mers = !is_mersenne;
            }
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            if (want_mers)
                ecm_ts_fprintf(stderr,
                               "ERROR: --edwards-mersenne on was requested, but this N is not "
                               "2^k-1 with k >= 64 (N+1 is not a power of two), so the Mersenne "
                               "fold domain does not apply. Use --edwards-mersenne auto "
                               "(default) or off.\n");
            else
                ecm_ts_fprintf(stderr,
                               "ERROR: edwards SIMD context init failed (rc=%d) for a %zu-bit N "
                               "with dict w=%d.\n", rc, mpz_sizeinbase(N, 2), w);
            return -1;
        }
    }

    /* 第一次成功建 ctx 时把实际选中的域打印出来 (auto 会按 N 的形状决定) */
    {
        static std::atomic<int> field_logged(0);
        if (field_logged.fetch_add(1, std::memory_order_relaxed) == 0) {
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "field           : %s\n", ed_soa_field_name(&ctx));
        }
    }

    /* 不足 8 条时把最后一个 sigma 补满 (被补的 lane 结果丢弃) */
    uint64_t sg[8];
    for (uint32_t k = 0; k < 8; k++) sg[k] = sigmas[first + (k < count ? k : count - 1)];

    const int setup_rc=ed_soa_set_curves(&ctx,sg,8);
    if(setup_rc!=0){ed_soa_clear(&ctx);return setup_rc==-4?run_exceptional():-1;}

    EdSimdProg pr;
    pr.done_ctr = done_ctr;
    pr.curves = (done_ctr && opt.stage1_curves) ? opt.stage1_curves : count;
    pr.batch_first = first;
    pr.batch_count = count;
    pr.s_bits = mpz_sizeinbase(s, 2);
    pr.t0_ms = opt.stage1_t0_ms;
    pr.opt = &opt;
    pr.B1 = B1;
    pr.B2 = B2;
    pr.sigmas = sigmas;
    pr.last_ms = stage1_now_ms();
    pr.interval_ms = (long long)opt.ckpt_ms;
    pr.mc = &ctx.mc;
    /* ---- resume: 一批 8 条 lane 必须共用恢复点, 所以仅当**所有** lane 都有有效检查点
       且 bitnum 完全相同时才续跑 (这正是"整批一起被中止、一起落盘"的常态); 否则从头跑。 ---- */
    {
        std::vector<mpz_t> rxs(count), rys(count), rzs(count);
        for (uint32_t k = 0; k < count; k++) mpz_inits(rxs[k], rys[k], rzs[k], NULL);
        uint32_t common_bitnum = 0;
        int all_ok = (count > 0);
        for (uint32_t k = 0; k < count && all_ok; k++) {
            const uint32_t i = first + k;
            const std::string path = ecm_stage1_curve_record(i);
            ecm_save_common rcm;
            uint64_t rsp; uint32_t rebs, rbn, rds;
            mpz_t rdx, rdy;
            mpz_inits(rdx, rdy, NULL);
            const bool ok = ecm_edwards_read_stage1(path, rcm, &rsp, &rebs, &rbn, &rds,
                                                    rdx, rdy, rxs[k], rys[k], rzs[k]);
            mpz_clears(rdx, rdy, NULL);
            if (!ok || rbn == 0 || rcm.sigma != sigmas[i] || rcm.B1 != (uint64_t)B1) { all_ok = 0; break; }
            if (k == 0) common_bitnum = rbn;
            else if (rbn != common_bitnum) { all_ok = 0; break; }
        }
        if (all_ok && common_bitnum > 0) {
            const int rr = ed_soa_set_resume(&ctx, common_bitnum, (int)count,
                                             rxs.data(), rys.data(), rzs.data());
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "simd batch %u: resume from .ckpt @ bitnum=%u (%u lane(s))%s\n",
                           first, common_bitnum, count,
                           rr == 0 ? "" : " -- unusable, restarting from scratch");
            if (rr != 0) ctx.resume_digit = 0;
        }
        for (uint32_t k = 0; k < count; k++) mpz_clears(rxs[k], rys[k], rzs[k], NULL);
    }
    ed_soa_set_progress(&ctx, edwards_simd_progress, &pr);

    mpz_t Qx[8], Qz[8], fac[8];
    for (int k = 0; k < 8; k++) mpz_inits(Qx[k], Qz[k], fac[k], NULL);
    const int aborted = ed_soa_stage1(&ctx, s, 8, Qx, Qz, fac);
    if (aborted) {                                  /* 中途停: 不写任何存档 */
        for (int k = 0; k < 8; k++) mpz_clears(Qx[k], Qz[k], fac[k], NULL);
        ed_soa_clear(&ctx);
        return pr.write_failed ? ECM_ERROR : 2;
    }

    int rc = 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = first + k;
        if (mpz_cmp_ui(fac[k], 1) > 0) {
            mpz_set(factors[i], fac[k]);
            array_found[i] = ECM_FACTOR_FOUND_STEP1;
            if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i]))rc=ECM_ERROR;
            std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                           i, (unsigned long long)sigmas[i]);
        }
        if(!edwards_capture_save(save_lines[i],N,sigmas[i],uint64_t(B1),Qx[k],Qz[k],fac[k],array_found[i]==ECM_FACTOR_FOUND_STEP1))rc=ECM_ERROR;
        if(rc==0 && !opt.tmp_dir.empty() && !edwards_completed_write(i,sigmas[i],factors[i],array_found[i]==ECM_FACTOR_FOUND_STEP1?1:0,save_lines[i]))rc=ECM_ERROR;
    }
    for (int k = 0; k < 8; k++) mpz_clears(Qx[k], Qz[k], fac[k], NULL);
    ed_soa_clear(&ctx);

    if (done_ctr) {
        const uint32_t done = done_ctr->fetch_add(count, std::memory_order_relaxed) + count;
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done, opt.stage1_curves);
    }
    (void)s;
    return rc;
}

static void edwards_worker(EdwardsWorkerCtx *ctx) {
    const int use_simd = (ctx->opt->selection.cpu_implementation != ecm_stage1::CpuImplementation::GmpScalar) && ctx->simd_ok;
    for (;;) {
        if (ecm_stage1_stop_requested()) { ctx->aborted->store(1, std::memory_order_relaxed); break; }
        uint32_t i, n;
        if (use_simd) {                                   /* 静态分批: 一批 8 条 */
            i = ctx->next->fetch_add(8, std::memory_order_relaxed);
            if (i >= ctx->curves) break;
            n = ctx->curves - i; if (n > 8) n = 8;
        } else {
            i = ctx->next->fetch_add(1, std::memory_order_relaxed);
            if (i >= ctx->curves) break;
            n = 1;
        }
        if (use_simd && n >= 2) {
            /* 批内至少 2 条才值得批量 (1 条走标量, 见 §13.6 的 hybrid 规则) */
            const int rc = edwards_run_batch_simd_impl(*ctx->opt, ctx->N, ctx->s, ctx->B1, ctx->B2,
                                                       i, n, ctx->sigmas, ctx->factors,
                                                       ctx->array_found, ctx->done_ctr,ctx->save_lines);
            if(rc==2){ctx->aborted->store(1,std::memory_order_relaxed);break;}
            if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  batch " << i << " -> SIMD stage-1 internal error" << std::endl;
                ctx->aborted->store(1, std::memory_order_relaxed);
                break;
            }
            continue;
        }
        if (use_simd && n == 1) { /* fallthrough to scalar for the tail curve */ }
        {
            const uint64_t sigma = ctx->sigmas[i];
            const int rc = edwards_run_one_curve(ctx->N, ctx->s, *ctx->opt, ctx->B1, ctx->B2,
                                                 i, sigma, ctx->curves,
                                                 ctx->factors[i], ctx->ckpt_paths[i],ctx->save_lines[i]);
            if (rc == 1) {
                ctx->array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if(!stage1_publish_factor(*ctx->opt,ctx->N,uint64_t(ctx->B1),sigma,ctx->factors[i])){ctx->aborted->store(1);break;}
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                if (ctx->opt->verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigma
                              << " -> factor found" << std::endl;
                } else {
                    ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                                   i, (unsigned long long)sigma);
                }
            } else if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                std::cerr << "  curve " << i << " sigma=" << sigma
                          << " -> Edwards stage-1 internal error" << std::endl;
                ctx->aborted->store(1,std::memory_order_relaxed);break;
            } else if (rc == 2) {
                ctx->aborted->store(1, std::memory_order_relaxed);
                break;
            }
            if (ctx->done_ctr) {
                const uint32_t done = ctx->done_ctr->fetch_add(1, std::memory_order_relaxed) + 1;
                std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                stage1_progress_set(done, ctx->opt->stage1_curves);
            }
        }
    }
}

// 默认线程数: min(curves, 硬件并发数).
static uint32_t edwards_default_threads(uint32_t curves) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    uint32_t t = (uint32_t)hw;
    if (t > curves) t = curves;
    return t;
}

// Run one Edwards (Atkin-Morain) stage-1 batch on the CPU. One curve per sigma.
//
#if defined(_MSC_VER)
#include <intrin.h>
// AVX512-F + DQ + IFMA 且 OS 已启用 zmm/opmask 状态 (XCR0) 才允许进 simd_edwards。
// 必须在基线 TU 里探测: simd_*.cpp 是 /arch:AVX512 编的, 在它内部执行任何代码都可能
// 让编译器在探测前就发出 AVX512 指令。
static int driver_simd_isa_ok(void) {
    int r[4] = {0,0,0,0};
    __cpuid(r, 0);
    if (r[0] < 7) return 0;
    __cpuidex(r, 1, 0);
    if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1)) return 0;   /* OSXSAVE, AVX */
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6ULL) != 0xE6ULL) return 0;                  /* opmask+ZMM_Hi256+Hi16_ZMM */
    __cpuidex(r, 7, 0);
    return (((r[1] >> 16) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 21) & 1)) ? 1 : 0;
}
#else
static int driver_simd_isa_ok(void) { return 0; }   /* 非 MSVC 由 CMake 侧保证 */
#endif

/* ---------------------------------------------------------------------------
 * 亲核性 (Affinity): "1,3,5,7" -> {1,3,5,7}
 * 空 / "none" / "auto" -> 空列表 = 不绑定, 交给系统调度。
 * 第 t 个 worker 绑到 list[t % count]，因此列表长度通常取 = 线程数或物理核数。
 * ------------------------------------------------------------------------- */
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

#if defined(_WIN32)
#include <windows.h>
static void apply_thread_affinity(const std::vector<unsigned> &cpus, uint32_t t) {
    if (cpus.empty()) return;
    const unsigned cpu = cpus[t % cpus.size()];
    if (cpu >= 64) return;      /* 单处理器组只有 64 个逻辑 CPU 可用位图表达 */
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
}
#else
static void apply_thread_affinity(const std::vector<unsigned> &, uint32_t) {}
#endif

// curves == 1 or opt.stage1_threads == 1 -> 顺序执行 (与历史行为一致);
// 否则按曲线并行 (每线程独立曲线, 动态取号), 吞吐随核数线性提升.
static int run_edwards_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                              const std::string &savefilename, bool saveappend,
                              const std::string &n_expr,
                              const Stage1RunOptions &opt, Stage1RunResult *out) {
    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->publish_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->firstsigma64 = 0;
    out->factors = nullptr;
    out->array_found = nullptr;
    out->sigmas = nullptr;

    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    // NAF 窗口 (字典大小 2^(w-2)); 0 = 保持 ecm_edwards_cpu 的默认值.
    if (opt.naf_w >= 2) {
        edwards_set_naf_w(opt.naf_w);
    } else if (opt.naf_w != 0) {
        std::cerr << "naf_w must be 0 (default) or >= 2" << std::endl;
        return ECM_ERROR;
    }

    // s = 48 * lcm(1..B1). (GPU batch uses lcm(1..B1); Edwards adds the
    // lcm(12,16)=48 torsion factor, matching Prime95 ecm_calc_exp.)
    mpz_t s;
    mpz_init(s);
    if (!compute_batch_s(s, B1)) {
        std::cerr << "Failed to compute Edwards stage-1 exponent" << std::endl;
        mpz_clear(s);
        return ECM_ERROR;
    }
    mpz_mul_ui(s, s, 48);

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    std::vector<std::string> save_lines(curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    const bool use_ckpt = !opt.tmp_dir.empty();
    if (use_ckpt) {
        if (!ensure_dir(opt.tmp_dir)) {
            std::cerr << "ERROR: cannot create tmp_dir: " << opt.tmp_dir << std::endl;
            mpz_clear(s);
            for (uint32_t j = 0; j < curves; j++) mpz_clear(factors[j]);
            free(factors);
            free(array_found);
            return ECM_ERROR;
        }
    }

    // sigma 预生成: random_sigma_u64 使用全局 rand()/srand(), 非线程安全,
    // 因此全部在主线程生成后再分发给工作线程.
    //
    std::vector<uint64_t> sigmas(curves);
    for (uint32_t i = 0; i < curves; i++) {
        // Fixed sigma → batch start + i (matches -sigma/-gpucurves semantics);
        // otherwise a fresh Prime95-style random sigma per curve -- 但若该曲线已有合格的
        // .ckpt 就**采用存档里的 sigma**（存档覆盖参数）。否则每轮都随机出新的 sigma,
        // 与存档永久不匹配, 自动续跑永远不会发生（这正是"不会自动从 saves/ 续跑"的根因）。
        if (opt.sigma_fixed) {
            sigmas[i] = ecm_stage1_batch_sigma(i,opt.fixed_sigma64 + i);
        } else {
            sigmas[i] = ecm_stage1_batch_sigma(i,random_sigma_u64());
            if (!opt.tmp_dir.empty()) {
                const std::string cp = ecm_stage1_curve_record(i);
                ecm_save_common rcm;
                uint64_t rsp; uint32_t rebs, rbn, rds;
                mpz_t dx, dy, rx, ry, rz;
                mpz_inits(dx, dy, rx, ry, rz, NULL);
                if (ecm_edwards_read_stage1(cp, rcm, &rsp, &rebs, &rbn, &rds,
                                           dx, dy, rx, ry, rz) &&
                    rbn > 0 && rcm.sigma != 0 && rcm.B1 == (uint64_t)B1) {
                    sigmas[i] = (uint64_t)rcm.sigma;
                    std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
                    ecm_ts_fprintf(stdout,
                                   "  curve %u: found .ckpt -> adopting sigma=%llu (bitnum=%u)\n",
                                   i + 1, (unsigned long long)rcm.sigma, rbn);
                }
                mpz_clears(dx, dy, rx, ry, rz, NULL);
            }
        }
    }
    const uint64_t firstsigma = sigmas[0];

    // Curve codecs address records inside the single process-owned batch file.
    std::vector<std::string> ckpt_paths(curves);
    if (use_ckpt) {
        for (uint32_t i = 0; i < curves; i++) {
            ckpt_paths[i] = ecm_stage1_curve_record(i);
        }
    }

    uint32_t nthreads = opt.stage1_threads;
    if (nthreads == 0) nthreads = edwards_default_threads(curves);
    if (nthreads > curves) nthreads = curves;
    if (nthreads < 1) nthreads = 1;

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " Edwards curves, CPU";
    if (nthreads > 1) std::cout << ", " << nthreads << " threads";
    std::cout << ")";
    if (use_ckpt) std::cout << " [saves -> " << opt.tmp_dir << "]";
    std::cout << std::endl;

    auto *bar = stage1_bar_create();
    ((Stage1RunOptions &)opt).stage1_t0_ms = stage1_now_ms();
    ((Stage1RunOptions &)opt).stage1_curves = curves;
    g_stage1_done.store(0);stage1_progress_set(0,curves);
    bool aborted = false;
    if (nthreads <= 1 && curves == 1) {
        for (uint32_t i = 0; i < curves; i++) {
            const int rc = edwards_run_one_curve(N, s, opt, B1, B2, i, sigmas[i], curves,
                                                 factors[i], ckpt_paths[i],save_lines[i]);
            if (rc == 1) {
                array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[i],factors[i])){aborted=true;break;}
                if (opt.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigmas[i]
                              << " -> factor found" << std::endl;
                }
            } else if (rc < 0) {
                std::cerr << "  curve " << i << " sigma=" << sigmas[i]
                          << " -> Edwards stage-1 internal error" << std::endl;
                aborted=true;break;
            } else if (rc == 2) {
                aborted = true;
                break;
            }
        }
    } else {
        EdwardsWorkerCtx wctx;
        wctx.N = N;
        wctx.s = s;
        wctx.opt = &opt;
        /* ---- backend 选择 + 起始信息 (见 §13.6 第 7 条: simd 不允许静默降级) ---- */
        g_stage1_done.store(0, std::memory_order_relaxed);
        std::atomic<uint32_t> &done_ctr = g_stage1_done;
        {
            const int isa = driver_simd_isa_ok();
            if (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd && !isa) {
                ecm_ts_fprintf(stderr,
                               "ERROR: --impl simd requested but this CPU lacks "
                               "AVX512-F/DQ/IFMA. Refusing to fall back silently; use "
                               "--impl auto or scalar.\n");
                return -1;
            }
            wctx.simd_ok = (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd) ? 1
                         : (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::Auto) ? (isa && curves >= 8) : 0;
            wctx.done_ctr = &done_ctr;
            char sbuf[160];
            snprintf(sbuf, sizeof(sbuf),
                     "simd (AVX512-IFMA: 8 curves per batch, 1 thread/batch, dict w=%d)",
                     edwards_get_naf_w() >= 3 ? edwards_get_naf_w() : 8);
            ecm_ts_fprintf(stdout, "CPU implementation : %s\n",
                           wctx.simd_ok ? sbuf : "gmp (scalar mpn, 1 curve/thread)");
            ecm_ts_fprintf(stdout, "N / s           : %zu bit N, %zu bit s\n",
                           mpz_sizeinbase(N, 2), mpz_sizeinbase(s, 2));
            if (wctx.simd_ok) {
                /* 批量模式下真正的并行度是"批数"而不是线程数: 8 条曲线 = 1 批 = 1 个线程在干活,
                   其余线程空闲。这里如实显示, 免得用户以为 8 线程都忙。 */
                const uint32_t batches = (curves + 7u) / 8u;
                const uint32_t busy = batches < nthreads ? batches : nthreads;
                ecm_ts_fprintf(stdout,
                               "work split      : %u batch(es) of 8 curves -> %u thread(s) busy "
                               "(%u requested)%s\n",
                               batches, busy, nthreads,
                               busy < nthreads ? "  [raise --gpucurves for more parallelism]" : "");
            }
            if (!opt.affinity_cpus.empty()) {
                std::string a;
                for (size_t i = 0; i < opt.affinity_cpus.size(); i++) {
                    if (i) a += ",";
                    a += std::to_string(opt.affinity_cpus[i]);
                }
                ecm_ts_fprintf(stdout, "affinity        : %s (worker t -> cpu[%s][t %% %zu])\n",
                               a.c_str(), a.c_str(), opt.affinity_cpus.size());
            } else {
                ecm_ts_fprintf(stdout, "affinity        : none (OS scheduler)\n");
            }
            fflush(stdout);
        }
        /* 进度条: 照 CUDA stage-1 host 的样式, prefix 换成 stage1 */
        ((Stage1RunOptions &)opt).stage1_t0_ms = stage1_now_ms();
        ((Stage1RunOptions &)opt).stage1_curves = curves;
        stage1_progress_set(0, curves);
        wctx.B1 = B1;
        wctx.B2 = B2;
        wctx.curves = curves;
        wctx.sigmas = sigmas.data();
        wctx.ckpt_paths = ckpt_paths.data();
        wctx.factors = factors;
        wctx.array_found = array_found;
        wctx.save_lines = save_lines.data();
        std::atomic<uint32_t> next(0);
        std::atomic<int> aborted_flag(0);
        wctx.next = &next;
        wctx.aborted = &aborted_flag;

        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.emplace_back([&wctx, t]() {
                apply_thread_affinity(wctx.opt->affinity_cpus, t);
                edwards_worker(&wctx);
            });
        for (auto &th : pool) th.join();
        aborted = (aborted_flag.load(std::memory_order_relaxed) != 0);
    }

    if (!aborted) stage1_progress_set(curves,curves);
    stage1_bar_destroy(bar);
    if (aborted) {
        ecm_ts_fprintf(stdout, "stage-1 incomplete; checkpoints retained\n");
        mpz_clear(s);
        for (uint32_t j = 0; j < curves; j++) mpz_clear(factors[j]);
        free(factors);
        free(array_found);
        out->ret = ECM_ERROR;
        return ECM_ERROR;
    }

    const auto stem=mont_format_save_name(opt.save_name_pattern,N,B1,&opt);
    const auto save=savefilename.empty()?(opt.tmp_dir.empty()?std::string{}:opt.tmp_dir+"/")+stem:savefilename;
    std::string lines;for(const auto &line:save_lines)lines+=line;
    if(!ecm_edwards::publish_save(save,lines,saveappend)) {
        out->publish_failed=true;out->ret=ECM_ERROR;
        ecm_ts_fprintf(stderr,"ERROR: cannot publish Edwards save %s\n",save.c_str());
        mpz_clear(s);for(uint32_t i=0;i<curves;++i)mpz_clear(factors[i]);free(factors);free(array_found);return ECM_ERROR;
    }
    for(const auto &path:ckpt_paths)if(!path.empty())ecm_stage1_remove_checkpoints(path);
    ecm_ts_fprintf(stdout,"Edwards Stage1 save: %s (%u attempted curves)\n",save.c_str(),curves);
    (void)n_expr;

    mpz_clear(s);

    out->ret = ECM_NO_FACTOR_FOUND;
    out->firstsigma = (uint32_t)(firstsigma & 0xFFFFFFFFu);
    out->firstsigma64 = firstsigma;
    out->factors = factors;
    out->array_found = array_found;
    /* Edwards draws a random sigma per curve, so publish the exact per-curve values
       instead of letting consumers assume a sequence. */
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = sigmas[i];
    }
    return out->ret;
}

// Run one stage-1 batch. N and the caller's n_expr are borrowed (not cleared);
// factors / array_found are allocated here and returned via `out` (caller frees).
/* ---------------------------------------------------------------------------
 * Shared automatic Stage1 save names (ecm.ini: save_name_pattern).
 *
 *   {curve}: e / m0 / m2 / m3; {target}: m<original exponent> / bit<actual bits>;
 *   {n}  : original assignment exponent when available, else actual N bits;
 *   {b1} : compact bound -- 100000 -> "1e5", 110000000 -> "110e6", 12345 -> "12345".
 *
 * ecm_extract_b1_from_save_name() documents the same pattern on the reading side.
 * ------------------------------------------------------------------------- */
static std::string mont_format_save_name(const std::string &pattern, const mpz_t N, double B1,
                                         const Stage1RunOptions *options) {
    /* {n} */
    std::string n_str;
    {
        mpz_t t;
        mpz_init(t);
        mpz_add_ui(t, N, 1);
        if (mpz_popcount(t) == 1) n_str = std::to_string((unsigned long)mpz_scan1(t, 0));
        else                      n_str = std::to_string((unsigned long)mpz_sizeinbase(N, 2));
        mpz_clear(t);
    }
    bool mersenne=false;
    if(options && options->target_kbnc) {
        n_str=std::to_string(options->handoff_n);
        mersenne=options->handoff_k_str=="1" && options->handoff_b==2 && options->handoff_c==-1;
    } else {
        ecm_edwards::Integer plus;mpz_add_ui(plus.v,N,1);mersenne=mpz_popcount(plus.v)==1;
    }
    const auto curve=options?stage1_save_prefix(*options):std::string("m0");
    const auto target=mersenne?"m"+n_str:"bit"+std::to_string(mpz_sizeinbase(N,2));
    /* {b1}: mantissa + 'e' + number of trailing zeros when there are >= 3 of them */
    std::string b1_str;
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.0f", B1);
        std::string digits(buf);
        size_t zeros = 0;
        while (zeros < digits.size() && digits[digits.size() - 1 - zeros] == '0') zeros++;
        if (zeros >= 3 && zeros < digits.size())
            b1_str = digits.substr(0, digits.size() - zeros) + "e" + std::to_string(zeros);
        else
            b1_str = digits;
    }
    std::string out = pattern;
    for (size_t p = out.find("{n}"); p != std::string::npos; p = out.find("{n}", p))
        out.replace(p, 3, n_str);
    for (size_t p = out.find("{b1}"); p != std::string::npos; p = out.find("{b1}", p))
        out.replace(p, 4, b1_str);
    for(const auto &entry:{std::pair<std::string,std::string>{"{curve}",curve},{"{target}",target}})
        for(size_t p=out.find(entry.first);p!=std::string::npos;p=out.find(entry.first,p))
            out.replace(p,entry.first.size(),entry.second);
    if(out.find('{')!=std::string::npos || out.find('}')!=std::string::npos)
        throw std::runtime_error("unknown save_name_pattern token");
    // Old custom patterns still receive the mandatory curve/param prefix.
    out=stage1_prefixed_save(out,curve);
    return out;
}

/* Standalone codec fallback stem; the driver routes records into its batch. */
static std::string mont_save_stem(const std::string &pattern, const mpz_t N, double B1) {
    std::string f = mont_format_save_name(pattern, N, B1);
    const size_t dot = f.rfind('.');
    if (dot != std::string::npos) f.erase(dot);
    return f;
}

// ---------------------------------------------------------------------------
// Suyama-sigma Montgomery stage 1 (Prime95 sigma_type = 1 / gmp-ecm -param 0).
//
// Same result contract as run_edwards_stage1: fills out->factors / out->array_found
// and returns an ECM_* code, so the CLI caller needs no changes.
//
//   * s = torsion * lcm(1..B1)   (torsion 1 matches gmp-ecm, 12 matches Prime95;
//     see docs/architecture/STAGE1.md 4.1)
//   * 8 curves per AVX512-IFMA batch when available, else the scalar MPN path
//   * ONE shared save file per task (same N and B1, one self-contained line per
//     curve -- the reference reader parses METHOD/B1/N from every line, so the
//     lines must stay self-contained; docs 9.2)
// ---------------------------------------------------------------------------
/* Default worker count for the Montgomery stage-1 path.
 *
 * The unit of parallel work differs per backend: the SIMD path runs 8 curves per
 * batch, so its parallelism is capped by the number of batches (curves/8) and the
 * scalar path by the number of curves.  Extra threads beyond that would only
 * contend, so clamp -- 8 curves under the IFMA SIMD implementation is one thread of real
 * work no matter how many cores the machine has. */
static uint32_t mont_default_threads(uint32_t curves, bool use_simd) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    uint32_t tasks = use_simd ? ((curves + IFMA_LANES - 1) / IFMA_LANES) : curves;
    if (tasks == 0) tasks = 1;
    uint32_t t = (uint32_t)hw;
    if (t > tasks) t = tasks;
    return t;
}

/* ---------------------------------------------------------------------------
 * Montgomery stage-1 progress + checkpoint glue
 * ------------------------------------------------------------------------- */

/* Checkpoint / progress granularity: aim at ~512 callbacks per curve (a progress
   update every ~0.2% of the ladder) with a 4096-bit floor, rounded up to a power
   of two so the offsets in the checkpoint files stay readable.  It is derived
   from the exponent only, so a resumed run with the same command line computes
   the identical offsets and can match what the interrupted run wrote. */
static size_t mont_ckpt_chunk_bits(size_t nbits)
{
    size_t p = 4096;
    const size_t want = nbits / 512;
    while (p < want && p < ((size_t)1 << 30)) p <<= 1;
    return p;
}

/* Progress + pause decision for one SIMD batch.  Called from inside the ladder
   every `chunk` bits, with the batch holding the whole worker thread. */
struct MontProgCtx {
    std::atomic<uint64_t> *lane_bits;    /* [curves]: exponent bits done per curve */
    const uint32_t        *lane_curve;   /* [lanes] : curve index of each lane */
    uint32_t               lanes;
    uint32_t               total_curves;
    size_t                 nbits;
    std::atomic<long long> *last_ckpt_ms; /* shared by all workers: one timer */
    long long              interval_ms;   /* < 0 = no periodic autosave */
};

static int mont_progress_cb(void *p, size_t bitnum)
{
    MontProgCtx *pc = (MontProgCtx *)p;
    for (uint32_t k = 0; k < pc->lanes; k++)
        pc->lane_bits[pc->lane_curve[k]].store((uint64_t)bitnum, std::memory_order_relaxed);

    /* Work unit = one curve; the fractional part of a lane is bitnum/nbits.  A
       running sum over the per-curve counters is what keeps the number monotone
       when several workers report at different offsets, and it needs no lock. */
    uint64_t sum = 0;
    for (uint32_t i = 0; i < pc->total_curves; i++)
        sum += pc->lane_bits[i].load(std::memory_order_relaxed);
    const double done_f = pc->nbits ? (double)sum / (double)pc->nbits : 0.0;
    {
        std::lock_guard<std::mutex> lk(g_stage1_out_mutex);
        stage1_progress_set(done_f, pc->total_curves);
    }

    if (ecm_stage1_stop_requested()) return 1;                       /* Ctrl+C: stop at this chunk */
    if (pc->interval_ms > 0) {
        const long long now = stage1_now_ms();
        if (now - pc->last_ckpt_ms->load(std::memory_order_relaxed) >= pc->interval_ms) {
            pc->last_ckpt_ms->store(now, std::memory_order_relaxed);
            return 1;                                  /* autosave interval elapsed */
        }
    }
    return 0;
}

/* Scalar ladder wrapper: same policy, the offset arrives inside the state. */
static int mont_progress_cb_scalar(void *p, const mont_ladder_state_t *st)
{
    return mont_progress_cb(p, st->bitnum);
}

static int run_mont_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           const Stage1RunOptions &opt, Stage1RunResult *out) {
    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->firstsigma64 = 0;
    out->factors = nullptr;
    out->array_found = nullptr;
    out->sigmas = nullptr;
    (void)saveappend;
    (void)B2;

    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    mpz_t s;
    mpz_init(s);
    const int torsion = opt.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12 : 1;
    std::string s_status;
    const size_t s_bits = mont_build_s(s, (uint64_t)B1, (uint64_t)torsion, &s_status);
    if (s_bits == 0) {
        /* only reachable for B1 > 5e9 or an allocation failure; say so instead of
           running a ladder over the torsion-only exponent */
        ecm_ts_fprintf(stderr,
                       "ERROR: cannot build the stage-1 exponent for B1=%.0f "
                       "(bound must be <= 5e9 and the prime sieve must fit in memory)\n",
                       B1);
        mpz_clear(s);
        return ECM_ERROR;
    }

    const bool isa = driver_simd_isa_ok();
    if (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd && !isa) {
        ecm_ts_fprintf(stderr,
                       "ERROR: --impl simd requested but this CPU lacks "
                       "AVX512-F/DQ/IFMA. Use auto or scalar.\n");
        mpz_clear(s);
        return ECM_ERROR;
    }
    const bool use_simd = (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::IfmaSimd) ? true
                        : (opt.selection.cpu_implementation == ecm_stage1::CpuImplementation::GmpScalar) ? false
                        : (isa && curves >= 2);
    ecm_ts_fprintf(stdout, "curve model     : montgomery (Suyama sigma, %s, torsion=%d)\n",
                   use_simd ? "AVX512-IFMA 8-lane batch" : "scalar mpn", torsion);
    ecm_ts_fprintf(stdout, "stage1 exponent : s_bits=%zu (lcm(1..%.0f) x %d) [%s]\n",
                   s_bits, B1, torsion, s_status.c_str());

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    /* The exponent bit array depends only on (B1, torsion) -- not on N, not on sigma
       -- so it is built once here and read by every task below.  It is built before
       the sigmas because the checkpoint pre-pass needs s_bits to validate a resume. */
    size_t nbits = 0;
    uint8_t *bits = mont_expand_bits(s, &nbits);
    const size_t ckpt_chunk = mont_ckpt_chunk_bits(nbits);

    /* ---- mid-stage-1 checkpoints (docs/architecture/STAGE1.md) ----------
       One independently resumable record per curve in a shared batch file. The
       interval comes from ckpt_seconds / --ckpt; 0 = no autosave (Ctrl+C still
       checkpoints, which is what turns an interrupted run into a resumed one). */
    const bool use_ckpt = !opt.tmp_dir.empty();
    const std::string ckpt_stem = use_ckpt
        ? (opt.tmp_dir + "/" + mont_save_stem(opt.save_name_pattern, N, B1))
        : std::string();
    const long long ckpt_interval_ms =
        (opt.ckpt_ms > 0 && opt.ckpt_ms != ULONG_MAX) ? (long long)opt.ckpt_ms : -1;

    /* per-curve results: the normalised x for misses, the factor for hits */
    std::vector<mpz_t> xs(curves);
    int *hit = (int *)calloc(curves, sizeof(int));
    for (uint32_t i = 0; i < curves; i++) mpz_init(xs[i]);

    /* curve state recovered from checkpoints */
    std::vector<size_t> bit_off(curves, 0);      /* exponent bits already consumed */
    std::vector<char>   ckpt_done(curves, 0);    /* result restored: curve is finished */
    uint32_t n_done_ck = 0, n_inflight_ck = 0, n_seeded = 0;

    /* Sigmas: fixed base (from -sigma, then sigma+i) or random.
     *
     * Interoperability constraint (why this is not a full 64-bit random): Prime95's
     * ECMSTAGE2 path -- the one that runs stage 2 on a gmp-ecm/PARAM=0 save file --
     * parses SIGMA with  mpz_get_str() -> atoll()  (ecm.cpp:7375).  A sigma >= 2^63
     * overflows atoll, so Prime95 would rebuild a DIFFERENT curve from a mangled
     * sigma and quietly waste the stage-2 run (the bitmap/x-coordinate it loads came
     * from our curve).  Prime95 itself only ever generates sigmas below 2^53
     * (ecm.cpp:7416: (rand()&0x1F)<<48 + (rand()&0xFFFF)<<32 + rdtsc bits), so we use
     * the same generator the Edwards path already uses.  gmp-ecm's own reader is
     * mpz-based and has no such limit, but a sigma that works with every consumer is
     * strictly better -- and 2^53 still leaves ~9e15 curves to choose from. */
    std::vector<uint64_t> sigmas(curves);
    std::atomic<int> checkpoint_failed(0);
    for (uint32_t i = 0; i < curves; i++) {
        const uint64_t fixed_sigma = opt.fixed_sigma64 + i;
        bool adopted = false;
        if (use_ckpt) {
            /* A checkpoint owns its curve's sigma.  Adopting it is what makes the
               identical command line continue the identical curve set; without this
               a resumed run would draw fresh random sigmas and the checkpoints
               would never match.  (The Edwards path decides the same way.) */
            mont_ckpt_t ck;
            mont_ckpt_init(&ck);
            std::string why;
            if (mont_ckpt_read(mont_ckpt_path(ckpt_stem, i), N, B1, torsion, nbits, &ck, &why) &&
                ck.curve == i &&
                (!opt.sigma_fixed || ck.sigma == fixed_sigma)) {
                sigmas[i] = ck.sigma;
                adopted = true;
                if (ck.status == MONT_CKPT_DONE) {
                    ckpt_done[i] = 1;
                    n_done_ck++;
                    if (ck.hit) {
                        mpz_set(factors[i], ck.factor);
                        array_found[i] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[i], ck.factor);
                        hit[i] = 1;
                    } else {
                        mpz_set(xs[i], ck.xout);
                    }
                } else if (ck.bitnum > 0) {
                    bit_off[i] = ck.bitnum;
                    n_inflight_ck++;
                } else {
                    n_seeded++;
                }
            }
            mont_ckpt_clear(&ck);
        }
        if (!adopted) {
            sigmas[i] = ecm_stage1_batch_sigma(i,opt.sigma_fixed ? fixed_sigma : random_sigma_u64());
            if (use_ckpt) {
                /* Seed record: pins the sigma before any ladder work happens, so a
                   run that is killed while this curve is still queued resumes the
                   SAME curve later instead of substituting a new random one. */
                mont_ckpt_t ck;
                mont_ckpt_init(&ck);
                ck.status = MONT_CKPT_INFLIGHT;
                ck.bitnum = 0;
                mont_ckpt_ident(&ck, i, sigmas[i], B1, torsion, nbits,
                                use_simd ? "ifma" : "mpn");
                if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, i), N, &ck) != 0)
                    checkpoint_failed.store(1);
                mont_ckpt_clear(&ck);
                n_seeded++;
            }
        }
    }
    if (opt.sigma_fixed && fixed_sigma_batch_exceeds_prime95_range(opt.fixed_sigma64, curves)) {
        ecm_ts_fprintf(stderr,
            "WARNING: sigma >= 2^63; Prime95's ECMSTAGE2 reader (atoll) cannot read it.\n"
            "         Use gmp-ecm for stage 2 with these sigmas, or pass a smaller -sigma.\n");
    }
    out->firstsigma64 = sigmas[0];
    out->firstsigma = (uint32_t)(sigmas[0] & 0xFFFFFFFFu);
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = sigmas[i];
    }

    /* Work split.  A "task" is one SIMD batch (<= 8 curves, one thread) or, on the
       scalar backend, one curve.  Tasks are handed out by an atomic counter, so a
       worker that finishes a batch picks up the next one -- with uniform batch cost
       that is equivalent to a static split, but it degrades gracefully.
       
       Curves are grouped by "bits already consumed" (descending) before chunking,
       because the lanes of a batch share ONE exponent prefix: every lane has to
       start at the same offset.  On a fresh run every offset is 0, so this is
       exactly the old sequential split; after a resume it costs at most one partly
       empty batch per batch that the interrupted run was in the middle of. */
    std::vector<uint32_t> pending;
    for (uint32_t i = 0; i < curves; i++) if (!ckpt_done[i]) pending.push_back(i);
    std::stable_sort(pending.begin(), pending.end(),
                     [&](uint32_t a, uint32_t b) { return bit_off[a] > bit_off[b]; });

    struct MontTask { uint32_t first, count; size_t start_bit; };
    std::vector<MontTask> task_list;
    {
        const size_t per_task = use_simd ? (size_t)IFMA_LANES : (size_t)1;
        for (size_t p = 0; p < pending.size(); ) {
            size_t q = p;
            while (q < pending.size() && (q - p) < per_task &&
                   bit_off[pending[q]] == bit_off[pending[p]]) q++;
            MontTask tk = { (uint32_t)p, (uint32_t)(q - p), bit_off[pending[p]] };
            task_list.push_back(tk);
            p = q;
        }
    }
    const uint32_t tasks = (uint32_t)task_list.size();
    uint32_t nthreads = opt.stage1_threads ? opt.stage1_threads
                                         : mont_default_threads(curves, use_simd);
    if (nthreads > tasks) nthreads = tasks;
    if (nthreads < 1) nthreads = 1;
    ecm_ts_fprintf(stdout, "stage1 threads  : %u worker(s) x %u task(s) of %s\n",
                   nthreads, tasks, use_simd ? "8 curves" : "1 curve");
    if (use_ckpt) {
        char iv[48];
        if (ckpt_interval_ms > 0) snprintf(iv, sizeof(iv), "%.3g s", ckpt_interval_ms / 1000.0);
        else                      snprintf(iv, sizeof(iv), "off (Ctrl+C still saves)");
        ecm_ts_fprintf(stdout,
                       "checkpoint      : %s_c*.ckpt  [%u done, %u mid-ladder, %zu to run]"
                       "  autosave %s\n",
                       ckpt_stem.c_str(), n_done_ck, n_inflight_ck, pending.size(), iv);
    }
    if (!opt.affinity_cpus.empty()) {
        std::string a;
        for (size_t i = 0; i < opt.affinity_cpus.size(); i++) {
            if (i) a += ",";
            a += std::to_string(opt.affinity_cpus[i]);
        }
        ecm_ts_fprintf(stdout, "affinity        : %s (worker t -> cpu[%s][t %% %zu])\n",
                       a.c_str(), a.c_str(), opt.affinity_cpus.size());
    }
    fflush(stdout);

    std::atomic<uint32_t> next_task(0);
    std::atomic<uint32_t> found_atomic(0);
    std::atomic<int> init_failed(0);
    std::atomic<int> paused_flag(0);
    std::atomic<long long> last_ckpt_ms(stage1_now_ms());
    /* per-curve progress in exponent bits: restored curves start at their own
       offset (done ones at nbits), so the bar is right from the first sample */
    std::vector<std::atomic<uint64_t>> lane_bits(curves);
    for (uint32_t i = 0; i < curves; i++)
        lane_bits[i].store(ckpt_done[i] ? (uint64_t)nbits : (uint64_t)bit_off[i],
                           std::memory_order_relaxed);

    auto worker = [&]() {
        if (use_simd) {
            mont_soa_ctx_t ctx;                     /* per-thread scratch pool */
            if (mont_soa_init(&ctx, N, to_ifma_field_mode(opt.selection.simd_reduction)) != 0) {
                init_failed.store(1);
                return;
            }
            /* Report the field layer the context actually resolved to: with
               field = auto the choice depends on N (Mersenne fold for 2^k-1), so
               printing the request alone would be misleading. */
            {
                static std::atomic<int> shown(0);
                if (shown.fetch_add(1) == 0)
                    ecm_ts_fprintf(stdout, "field layer     : %s\n", ifma_field_name(&ctx.mc));
            }
            const size_t lw = 8 * ctx.n;
            std::vector<uint64_t> state(mont_soa_state_words(&ctx));
            std::vector<mpz_t> bx(IFMA_LANES), bg(IFMA_LANES);
            for (unsigned k = 0; k < IFMA_LANES; k++) { mpz_inits(bx[k], bg[k], NULL); }
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks) break;
                const MontTask &tk = task_list[t];
                uint32_t cur[IFMA_LANES];
                uint64_t sg[IFMA_LANES];
                /* a short batch repeats its last sigma: unread lanes only waste
                   SIMD slots, the used lanes stay exactly as requested */
                for (uint32_t k = 0; k < IFMA_LANES; k++) {
                    const uint32_t c = pending[tk.first + (k < tk.count ? k : tk.count - 1)];
                    cur[k] = c;
                    sg[k] = sigmas[c];
                }

                /* Resume: every lane of the task shares the offset, so each lane's
                   p0/p1 pair comes from its own checkpoint file.  If any lane's
                   state cannot be read (deleted, truncated, hand-edited) the whole
                   task restarts from bit 0 -- the lanes cannot be at different
                   offsets, the ladder walks one shared bit sequence. */
                size_t start = tk.start_bit;
                if (start > 0) {
                    for (uint32_t k = 0; k < tk.count; k++) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        std::string why;
                        const int ok = mont_ckpt_read(mont_ckpt_path(ckpt_stem, cur[k]), N, B1,
                                                      torsion, nbits, &ck, &why);
                        if (ok && ck.bitnum == start) {
                            ifma_from_mpz_lane(state.data() + 0 * lw, k, ck.X0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 1 * lw, k, ck.Z0, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 2 * lw, k, ck.X1, &ctx.mc);
                            ifma_from_mpz_lane(state.data() + 3 * lw, k, ck.Z1, &ctx.mc);
                        } else {
                            start = 0;
                        }
                        mont_ckpt_clear(&ck);
                        if (!start) break;
                    }
                }

                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = cur;
                pc.lanes = tk.count;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* The ladder stops at a checkpoint boundary for two different
                   reasons -- the autosave interval elapsed, or SIGINT arrived --
                   and they must NOT be confused: an autosave pause persists the
                   state and carries on with the same batch, while SIGINT ends the
                   run after persisting.  (Getting this wrong turns a 1 s autosave
                   interval into "the run stops after 1 s".) */
                int rc = MONT_SOA_DONE;
                for (;;) {
                    size_t bitnum = start;
                    rc = mont_soa_stage1_bits_ex(&ctx, bits, nbits, sg, start,
                                                 state.data(), &bitnum,
                                                 bx.data(), bg.data(),
                                                 mont_progress_cb, &pc, ckpt_chunk);
                    if (rc != MONT_SOA_PAUSED) break;
                    if (use_ckpt) {
                        for (uint32_t k = 0; k < tk.count; k++) {
                            const uint32_t c = cur[k];
                            mont_ckpt_t ck;
                            mont_ckpt_init(&ck);
                            ck.status = MONT_CKPT_INFLIGHT;
                            ck.bitnum = bitnum;
                            mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                            ifma_to_mpz_lane(ck.X0, state.data() + 0 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z0, state.data() + 1 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.X1, state.data() + 2 * lw, k, &ctx.mc);
                            ifma_to_mpz_lane(ck.Z1, state.data() + 3 * lw, k, &ctx.mc);
                            if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                                checkpoint_failed.store(1);
                            mont_ckpt_clear(&ck);
                        }
                    }
                    if (ecm_stage1_stop_requested()) break;      /* SIGINT: leave after saving */
                    start = bitnum;                /* autosave: continue from here */
                }
                if (rc == MONT_SOA_ERROR) { init_failed.store(1); break; }
                if (rc == MONT_SOA_PAUSED) { paused_flag.store(1); break; }

                uint32_t local = 0;
                for (uint32_t k = 0; k < tk.count; k++) {
                    const uint32_t c = cur[k];
                    lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                    if (mpz_cmp_ui(bg[k], 1) > 0 && mpz_cmp(bg[k], N) < 0) {
                        mpz_set(factors[c], bg[k]);
                        array_found[c] = ECM_FACTOR_FOUND_STEP1;
                        mpz_set(xs[c], bg[k]);
                        hit[c] = 1;
                        local++;
                        if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[c],factors[c]))checkpoint_failed.store(1);
                    } else {
                        mpz_set(xs[c], bx[k]);
                    }
                    if (use_ckpt) {
                        /* DONE record: if the process dies before the shared .save
                           is written, this is what keeps the curve from being
                           recomputed by the resumed run. */
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_DONE;
                        ck.hit = hit[c];
                        ck.bitnum = nbits;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "ifma");
                        if (ck.hit) mpz_set(ck.factor, factors[c]);
                        else        mpz_set(ck.xout, xs[c]);
                        if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                }
                if (local) found_atomic.fetch_add(local);
            }
            for (unsigned k = 0; k < IFMA_LANES; k++) { mpz_clears(bx[k], bg[k], NULL); }
            mont_soa_clear(&ctx);
        } else {
            for (;;) {
                const uint32_t t = next_task.fetch_add(1);
                if (t >= tasks) break;
                const MontTask &tk = task_list[t];
                const uint32_t c = pending[tk.first];
                mpz_t g, x, Qx, Qz, inv;
                mpz_inits(g, x, Qx, Qz, inv, NULL);

                mont_ladder_state_t st;
                mont_ladder_state_init(&st);
                size_t start = tk.start_bit;
                if (start > 0) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    std::string why;
                    if (mont_ckpt_read(mont_ckpt_path(ckpt_stem, c), N, B1, torsion, nbits,
                                       &ck, &why) && ck.bitnum == start) {
                        st.bitnum = ck.bitnum;
                        mpz_set(st.X0, ck.X0);
                        mpz_set(st.Z0, ck.Z0);
                        mpz_set(st.X1, ck.X1);
                        mpz_set(st.Z1, ck.Z1);
                    } else {
                        start = 0;
                    }
                    mont_ckpt_clear(&ck);
                }

                uint32_t lane[1] = { c };
                MontProgCtx pc;
                pc.lane_bits = lane_bits.data();
                pc.lane_curve = lane;
                pc.lanes = 1;
                pc.total_curves = curves;
                pc.nbits = nbits;
                pc.last_ckpt_ms = &last_ckpt_ms;
                pc.interval_ms = use_ckpt ? ckpt_interval_ms : -1;

                /* autosave pause vs SIGINT: see the SIMD branch above */
                int rc = MONT_LADDER_MISS;
                for (;;) {
                    rc = mont_stage1_curve_bits_ex(g, Qx, Qz, N, sigmas[c], bits, nbits,
                                                   start, &st, mont_progress_cb_scalar,
                                                   &pc, ckpt_chunk);
                    if (rc != MONT_LADDER_PAUSED) break;
                    if (use_ckpt) {
                        mont_ckpt_t ck;
                        mont_ckpt_init(&ck);
                        ck.status = MONT_CKPT_INFLIGHT;
                        ck.bitnum = st.bitnum;
                        mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                        mpz_set(ck.X0, st.X0);
                        mpz_set(ck.Z0, st.Z0);
                        mpz_set(ck.X1, st.X1);
                        mpz_set(ck.Z1, st.Z1);
                        if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                            checkpoint_failed.store(1);
                        mont_ckpt_clear(&ck);
                    }
                    if (ecm_stage1_stop_requested()) break;      /* SIGINT: leave after saving */
                    start = st.bitnum;             /* autosave: continue from here */
                }
                if (rc == MONT_LADDER_ERROR) {
                    init_failed.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }
                if (rc == MONT_LADDER_PAUSED) {
                    paused_flag.store(1);
                    mpz_clears(g, x, Qx, Qz, inv, NULL);
                    mont_ladder_state_clear(&st);
                    break;
                }

                lane_bits[c].store((uint64_t)nbits, std::memory_order_relaxed);
                /* same normalisation as mont_stage1_curve_bits_x() */
                if (mpz_sgn(Qz) != 0 && mpz_invert(inv, Qz, N)) {
                    mpz_mul(x, Qx, inv);
                    mpz_mod(x, x, N);
                } else {
                    mpz_set(x, Qx);
                }
                if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
                    mpz_set(factors[c], g);
                    array_found[c] = ECM_FACTOR_FOUND_STEP1;
                    mpz_set(xs[c], g);
                    hit[c] = 1;
                    found_atomic.fetch_add(1);
                    if(!stage1_publish_factor(opt,N,uint64_t(B1),sigmas[c],factors[c]))checkpoint_failed.store(1);
                } else {
                    mpz_set(xs[c], x);
                }
                if (use_ckpt) {
                    mont_ckpt_t ck;
                    mont_ckpt_init(&ck);
                    ck.status = MONT_CKPT_DONE;
                    ck.hit = hit[c];
                    ck.bitnum = nbits;
                    mont_ckpt_ident(&ck, c, sigmas[c], B1, torsion, nbits, "mpn");
                    if (ck.hit) mpz_set(ck.factor, factors[c]);
                    else        mpz_set(ck.xout, xs[c]);
                    if (mont_ckpt_write(mont_ckpt_path(ckpt_stem, c), N, &ck) != 0)
                        checkpoint_failed.store(1);
                    mont_ckpt_clear(&ck);
                }
                mpz_clears(g, x, Qx, Qz, inv, NULL);
                mont_ladder_state_clear(&st);
            }
        }
    };

    indicators::ProgressBar *bar = stage1_bar_create();
    ((Stage1RunOptions &)opt).stage1_t0_ms = stage1_now_ms();
    ((Stage1RunOptions &)opt).stage1_curves = curves;
    stage1_progress_set((double)(curves - pending.size()), curves);

    const auto t0 = std::chrono::steady_clock::now();
    if (nthreads <= 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t t = 0; t < nthreads; t++)
            pool.emplace_back([&worker, &opt, t]() {
                apply_thread_affinity(opt.affinity_cpus, t);
                worker();
            });
        for (auto &th : pool) th.join();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stage1_bar_destroy(bar);
    free(bits);

    if (init_failed.load() || checkpoint_failed.load()) {
        ecm_ts_fprintf(stderr, "ERROR: mont stage-1 %s\n",
                       checkpoint_failed.load() ? "checkpoint write failed" : "SIMD context init failed");
        mpz_clear(s);
        for (uint32_t i = 0; i < curves; i++) { mpz_clear(factors[i]); mpz_clear(xs[i]); }
        free(factors); free(array_found); free(hit);
        return ECM_ERROR;
    }

    /* Interrupted: every worker wrote its lane's mid-ladder state before leaving,
       and the curves it never reached keep their seed/older record, so rerunning
       the identical command line continues instead of restarting. */
    if (paused_flag.load()) {
        ecm_ts_fprintf(stdout,
                       "stage-1 paused (checkpoint saved under %s); rerun the same command "
                       "line to resume\n",
                       opt.tmp_dir.c_str());
        mpz_clear(s);
        for (uint32_t i = 0; i < curves; i++) { mpz_clear(factors[i]); mpz_clear(xs[i]); }
        free(factors); free(array_found); free(hit);
        out->ret = ECM_ERROR;
        out->factors = nullptr;
        out->array_found = nullptr;
        return ECM_ERROR;
    }

    stage1_progress_set((double)curves, curves);

    /* Hit lines are printed here, in curve order, rather than from inside the worker
       loop: with N workers the old placement interleaved lines at random. */
    const uint32_t found = found_atomic.load();
    for (uint32_t i = 0; i < curves; i++) {
        if (hit[i])
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                           i, (unsigned long long)sigmas[i]);
    }

    /* ONE shared save file for the whole task (same N, same B1), named with the
       CUDA/GPU convention: m{n}_{b1}.save */
    bool save_failed = false;
    if (use_ckpt || !savefilename.empty()) {
        const std::string fname = mont_format_save_name(opt.save_name_pattern, N, B1,&opt);
        const std::string path = savefilename.empty()?opt.tmp_dir+"/"+fname:
            opencl_ecm_resolve_data_path_buf(savefilename.c_str());
        if(!savefilename.empty() && !opencl_ecm_check_save_file_writable(path,saveappend))save_failed=true;
        if(!save_failed)save_failed = !ecm_append_save_lines_mont(
            path, N, B1, sigmas.data(), curves, xs.data(), hit,
            n_expr.empty() ? std::to_string(mpz_sizeinbase(N, 2)) : n_expr);
        if (save_failed) {
            ecm_ts_fprintf(stderr, "ERROR: could not publish Montgomery save %s\n", path.c_str());
        } else {
            ecm_ts_fprintf(stdout, "  save            : %s (%u curve lines, shared)\n",
                           path.c_str(), curves);
        }
        /* The .save above is now the durable artifact (and the interoperable one),
           so the resume scratch files are no longer needed: dropping them makes a
           repeated run a fresh run instead of an instant replay of this one. */
        if (!save_failed) {
            for (uint32_t i = 0; i < curves; i++)
                mont_ckpt_remove(mont_ckpt_path(ckpt_stem, i));
        }
    }

    if (n_done_ck || n_inflight_ck) {
        ecm_ts_fprintf(stdout, "  resumed from checkpoint: %u curve(s) already done, %u mid-ladder\n",
                       n_done_ck, n_inflight_ck);
    }
    ecm_ts_fprintf(stdout, "  curves=%u  hits=%u  wall=%.2fs  (%.3f s/curve)\n",
                   curves, found, elapsed, curves ? elapsed / curves : 0.0);

    for (uint32_t i = 0; i < curves; i++) mpz_clear(xs[i]);
    free(hit);
    mpz_clear(s);

    out->ret = save_failed ? ECM_ERROR :
               (found ? ECM_FACTOR_FOUND_STEP1 : ECM_NO_FACTOR_FOUND);
    out->publish_failed = save_failed;
    out->factors = factors;
    out->array_found = array_found;
    return out->ret;
}

#include "ecm_edwards_gpu_driver.inl"

static int run_stage1_impl(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           const Stage1RunOptions &opt, Stage1RunResult *out) {
    ecm_stage1_set_backup_limit(opt.ckpt_backups);
#if ECM_HAS_CUDA_EDWARDS
    if (opt.selection.target == ecm_stage1::ExecutionTarget::Gpu &&
        opt.selection.curve == ecm_stage1::CurveModel::Edwards)
        return run_edwards_gpu_stage1(N,B1,B2,curves,savefilename,saveappend,opt,out);
#endif
    if (opt.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        opt.selection.curve == ecm_stage1::CurveModel::Montgomery) {
        return run_mont_stage1(N, B1, B2, curves, savefilename, saveappend,
                               n_expr, opt, out);
    }
    if (opt.selection.target == ecm_stage1::ExecutionTarget::Cpu &&
        opt.selection.curve == ecm_stage1::CurveModel::Edwards) {
        return run_edwards_stage1(N, B1, B2, curves, savefilename, saveappend,
                                  n_expr, opt, out);
    }

    out->ret = ECM_ERROR;
    out->prepare_failed = false;
    out->curves = curves;
    out->firstsigma = 0;
    out->factors = nullptr;
    out->array_found = nullptr;

    if (curves == 0) {
        std::cerr << "gpucurves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    // Setup params.
    ecm_params params;
    ecm_init(params);
    params->gpu = 1;
    params->gpu_number_of_curves = curves;
    params->gpu_checkpoint_interval_ms = opt.ckpt_ms;
    params->gpu_param = opt.selection.gpu_param;
    if (!opt.gpu_mul_path.empty()) {
        strncpy(params->gpu_mul_path, opt.gpu_mul_path.c_str(), sizeof(params->gpu_mul_path) - 1u);
        params->gpu_mul_path[sizeof(params->gpu_mul_path) - 1u] = '\0';
    }
    if (!opt.gpu_sqr_path.empty()) {
        strncpy(params->gpu_sqr_path, opt.gpu_sqr_path.c_str(), sizeof(params->gpu_sqr_path) - 1u);
        params->gpu_sqr_path[sizeof(params->gpu_sqr_path) - 1u] = '\0';
    }
    if (!opt.gpu_add_path.empty()) {
        strncpy(params->gpu_add_path, opt.gpu_add_path.c_str(), sizeof(params->gpu_add_path) - 1u);
        params->gpu_add_path[sizeof(params->gpu_add_path) - 1u] = '\0';
    }
    if (!opt.gpu_sub_path.empty()) {
        strncpy(params->gpu_sub_path, opt.gpu_sub_path.c_str(), sizeof(params->gpu_sub_path) - 1u);
        params->gpu_sub_path[sizeof(params->gpu_sub_path) - 1u] = '\0';
    }
    if (!opt.gpu_special_mult_path.empty()) {
        strncpy(params->gpu_special_mult_path, opt.gpu_special_mult_path.c_str(),
                sizeof(params->gpu_special_mult_path) - 1u);
        params->gpu_special_mult_path[sizeof(params->gpu_special_mult_path) - 1u] = '\0';
    }
    params->verbose = opt.verbose ? 1 : 0;
    params->param = ECM_PARAM_BATCH_32BITS_D;

    mpz_t batch_s;
    mpz_init(batch_s);
    if (!compute_batch_s(batch_s, B1,
                         opt.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u)) {
        std::cerr << "Failed to compute batch_s" << std::endl;
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;
    {
        const char *algorithm=std::getenv("ECM_GPU_STAGE1_ALGO");
        if(std::string(ecm_backend_name())=="CUDA/CGBN" && (!algorithm || !*algorithm || std::string(algorithm)=="ladder"))
            ecm_stage1_import_legacy_gpu(N,batch_s,curves,uint32_t(params->gpu_param),
                opt.sigma_fixed,opt.fixed_sigma64,[&](uint64_t sigma,uint32_t count){
                    std::vector<uint64_t> sigmas(count);for(uint32_t i=0;i<count;++i)sigmas[i]=sigma+i;
                    return stage1_results_complete(opt,N,uint64_t(B1),false,unsigned(params->gpu_param),sigmas);
                });
    }

    {
        const int prep = ecm_backend_prepare((size_t)mpz_sizeinbase(N, 2), params->verbose,
                                             opt.device_index,
                                             params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                             params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
                                             params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
                                             params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
                                             params->gpu_special_mult_path[0] ? params->gpu_special_mult_path
                                                                             : nullptr);
        if (prep != 0) {
            std::cerr << "GPU: backend prepare failed" << std::endl;
            out->prepare_failed = true;
            mpz_clear(batch_s);
            ecm_clear(params);
            return ECM_ERROR;
        }
    }

    std::string resolved_save = savefilename;
    if (!resolved_save.empty()) {
        resolved_save = opencl_ecm_resolve_data_path(resolved_save.c_str());
        if (!opencl_ecm_check_save_file_writable(resolved_save, saveappend)) {
            out->publish_failed = true;
            mpz_clear(batch_s);
            ecm_clear(params);
            return ECM_ERROR;
        }
    }

    mpz_t *factors = (mpz_t *)malloc(sizeof(mpz_t) * curves);
    int *array_found = (int *)malloc(sizeof(int) * curves);
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    /* Curve index (sigma) of the first curve.
       gpu_param = 3 : the batch parametrization carries d = sigma/2^32 as a 32-bit
                       kernel parameter, so sigma must stay inside 2^32;
       gpu_param = 0 : Suyama sigma is a 64-bit value on the CPU path too, so use the
                       same 53-bit random generator there (Prime95's ECMSTAGE2 reader
                       only accepts sigma < 2^63, and random_sigma_u64() is < 2^53). */
    const bool gpu_param0 = (params->gpu_param == 0);
    uint64_t firstsigma64 = opt.sigma_fixed
        ? opt.fixed_sigma64
        : (gpu_param0 ? random_sigma_u64() : (uint64_t)gpu_pick_random_sigma(curves));
    firstsigma64=ecm_stage1_batch_sigma(0,firstsigma64);
    const uint32_t firstsigma = (uint32_t)(firstsigma64 & 0xFFFFFFFFull);
    if (!gpu_param0 &&
        (firstsigma64 > 0xFFFFFFFFull || curves > 0x100000000ull - firstsigma64)) {
        std::cerr << "sigma range overflows uint32 (sigma + curves > 2^32)" << std::endl;
        for (uint32_t i = 0; i < curves; i++) mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    for(uint32_t i=1;i<curves;++i)ecm_stage1_batch_sigma(i,firstsigma64+i);
    mpz_t batch_d;
    mpz_init(batch_d);
    /* batch_d is the batch parametrization's curve constant; only param3 uses it */
    gpu_compute_batch_d(batch_d, firstsigma, N);

    // B1/B2 are fixed for this run. The *actual* sigma is printed by the backend
    // after it applies any checkpoint resume (the checkpoint may override the
    // freshly-computed sigma), so it is not printed here.
    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " curves, " << ecm_backend_name() << ")" << std::endl;

    float gputime = 0.0f;
    int ret = ecm_backend_stage1(factors, array_found, N, params->batch_s, curves,
                                       &firstsigma64, params->gpu_checkpoint_interval_ms,
                                       &gputime, params->verbose, params->gpu_param,
                                       (uint64_t)std::floor(B1 + 0.0001),
                                       opt.selection.scalar_construction == ecm_stage1::ScalarConstruction::PrimePowerLcmTimes12 ? 12u : 1u,
                                       params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                       params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
                                       params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
                                       params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
                                       params->gpu_special_mult_path[0] ? params->gpu_special_mult_path
                                                                       : nullptr);

    std::cout << "GPU stage1 returned: " << ret << " gputime=" << gputime << " ms\n";
    if(ret!=ECM_ERROR)for(uint32_t i=0;i<curves;++i)if(array_found[i]!=ECM_NO_FACTOR_FOUND &&
        !stage1_publish_factor(opt,N,uint64_t(B1),firstsigma64+i,factors[i])){out->publish_failed=true;ret=ECM_ERROR;break;}

    if (ret != ECM_ERROR && !resolved_save.empty()) {
        /* The save must describe the SAME curve family the GPU just ran:
             gpu_param = 3 -> batch parametrization, written with PARAM=3, and with
                              the batch path's historical N rewriting (N divided by
                              the factors found so far);
             gpu_param = 0 -> Suyama param0, written in the param0 form (no PARAM=),
                              which is what gmp-ecm -param 0 and Prime95 sigma_type=1
                              read back for stage 2.  It must carry the ORIGINAL N:
                              gmp-ecm checks the line checksum against N, so the
                              factor-stripped expression makes it reject every line
                              with "bad checksum". */
        const std::string n_expr_stripped =
            opencl_ecm_build_saved_n_expr(n_expr, N, curves, factors, array_found);
        const std::string n_expr_param0 =
            n_expr.empty() ? std::to_string(mpz_sizeinbase(N, 2)) : n_expr;
        bool wrote = false;
        if (params->gpu_param == 0) {
            std::vector<uint64_t> sigmas(curves);
            for (uint32_t i = 0; i < curves; i++) sigmas[i] = firstsigma64 + i;
            std::vector<int> hit(curves, 0);
            /* factors[i] holds the normalized x for a miss and the factor for a hit,
               exactly like the CPU path's xs[] -- and findfactor()/array_found tells
               us which is which. */
            for (uint32_t i = 0; i < curves; i++)
                hit[i] = (array_found[i] != ECM_NO_FACTOR_FOUND) ? 1 : 0;
            wrote = ecm_append_save_lines_mont(resolved_save, N, B1, sigmas.data(), curves,
                                               factors, hit.data(), n_expr_param0);
            if (!wrote)
                std::cerr << "Failed to append param0 save lines into " << resolved_save << std::endl;
        } else if (params->gpu_param == 2) {
            /* param2 ("batch 2", 6-torsion): PARAM=2 with the ORIGINAL N for the same
               checksum reason as param0, and SIGMA = the scalar multiplier sigma0+i the
               GPU used.  gmp-ecm reads this back with -param 2; Prime95 cannot read it
               (sigma_type only accepts 0/1/3).  See
               docs/performance/STAGE1.md */
            wrote = opencl_ecm_append_save_lines(resolved_save, N, B1, firstsigma, curves, factors,
                                                 n_expr_param0, 2);
            if (!wrote)
                std::cerr << "Failed to append param2 save lines into " << resolved_save << std::endl;
        } else {
            wrote = opencl_ecm_append_save_lines(resolved_save, N, B1, firstsigma, curves, factors,
                                                 n_expr_stripped);
            if (!wrote)
                std::cerr << "Failed to append GPU save lines into " << resolved_save << std::endl;
        }
        if (!wrote) {
            out->publish_failed = true;
            ret = ECM_ERROR;
        }
    }

    mpz_clear(batch_d);
    mpz_clear(batch_s);
    ecm_clear(params);

    out->ret = ret;
    out->firstsigma = (uint32_t)(firstsigma64 & 0xFFFFFFFFull);
    out->firstsigma64 = firstsigma64;
    out->factors = factors;
    out->array_found = array_found;
    /* GPU batches use the contiguous convention: curve i has sigma = firstsigma + i
       (the same rule as -sigma / -gpucurves). */
    out->sigmas = (uint64_t *)malloc(sizeof(uint64_t) * (curves ? curves : 1));
    if (out->sigmas != nullptr) {
        for (uint32_t i = 0; i < curves; i++) out->sigmas[i] = firstsigma64 + i;
    }
    return ret;
}

static void stage1_result_settings(Stage1RunOptions &opt,const EcmQueueConfig &cfg,
                                   const std::string &base,int worker) {
    opt.result_user=cfg.result_user;opt.result_computer=cfg.result_computer;
    opt.results_file=(std::filesystem::path(base)/(cfg.has_results?cfg.results:ecm_config::worker_file(cfg.results,worker))).string();
    opt.submission_file=(std::filesystem::path(base)/(cfg.has_submission?cfg.submission:ecm_config::worker_file(cfg.submission,worker))).string();
}
static const std::string &stage1_build_identity() {
    static const std::string identity=[] {
        std::ifstream in(std::filesystem::path(get_exe_dir_local())/"ecm_stage1_build_identity.txt");
        std::string value;std::getline(in,value);
        return value.size()<=128 && !value.empty() && value.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz.-")==value.npos?value:std::string(ECM_BUILD_ID);
    }();return identity;
}

static ecm_results::Submission stage1_submission(const Stage1RunOptions &opt,const mpz_t N,uint64_t b1) {
    ecm_results::Submission r;r.b1=b1;r.b2=b1;r.stage1_only=true;
    r.edwards=opt.selection.curve==ecm_stage1::CurveModel::Edwards;
    r.param=opt.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.selection.gpu_param:0;
    r.user=opt.result_user;r.computer=opt.result_computer;r.aid=opt.result_aid;
    if(opt.target_kbnc) {
        r.kbnc=true;r.k=opt.handoff_k_str;r.b=opt.handoff_b;r.exponent=opt.handoff_n;r.c=opt.handoff_c;
        std::istringstream known(opt.handoff_factors);std::string f;
        while(std::getline(known,f,','))if(!f.empty())r.known.push_back(f);
    }
    // Validate complete metadata against the actual modulus before running.
    (void)ecm_results::target_exponent(r,N);return r;
}

static bool stage1_publish_factor(const Stage1RunOptions &opt,mpz_srcptr n,uint64_t b1,uint64_t sigma,mpz_srcptr factor) {
    static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);
    try {
        auto r=stage1_submission(opt,n,b1);r.stage=1;r.sigma=sigma;r.when=ecm_results::utc_timestamp();
        r.factors.push_back(ecm_results::decimal(factor));ecm_results::publish_submission(opt.submission_file,r,n);return true;
    } catch(const std::exception &e){ecm_ts_fprintf(stderr,"ERROR: immediate Stage1 factor publication: %s\n",e.what());return false;}
}

static int run_stage1_once(const mpz_t N,double B1,double B2,uint32_t curves,
                           const std::string &savefilename,bool saveappend,const std::string &n_expr,
                           const Stage1RunOptions &options,Stage1RunResult *out) {
    Stage1RunOptions opt=options;*out=Stage1RunResult{};
    try {
        if(!opt.target_kbnc) {
            // Direct CLI expressions retain k*b^n+c for naming and reporting.
            std::smatch m;
            const std::regex form(R"(^\s*\(?\s*(?:(\d+)\s*\*\s*)?(\d+)\s*\^\s*(\d+)\s*([+-])\s*(\d+)\s*\)?\s*$)");
            if(std::regex_match(n_expr,m,form)) {
                const auto b=std::stoull(m[2]),n=std::stoull(m[3]);const auto c=std::stoll(m[5])*(m[4]=="-"?-1:1);
                if(b<=UINT32_MAX && n<=UINT32_MAX && c>=INT32_MIN && c<=INT32_MAX) {
                    opt.target_kbnc=true;opt.handoff_k_str=m[1].matched?m[1].str():"1";
                    opt.handoff_b=static_cast<uint32_t>(b);opt.handoff_n=static_cast<uint32_t>(n);opt.handoff_c=static_cast<int32_t>(c);
                }
            }
        }
        const auto base=stage1_submission(opt,N,uint64_t(B1));
        (void)ecm_results::submission_json(base,N);
        const auto same=[](const std::string &a,const std::string &b){if(a.empty() || b.empty())return false;std::error_code ec;
            return std::filesystem::absolute(a).lexically_normal()==std::filesystem::absolute(b).lexically_normal() || std::filesystem::equivalent(a,b,ec);};
        if(same(opt.results_file,opt.submission_file) || same(savefilename,opt.results_file) || same(savefilename,opt.submission_file))
            throw std::runtime_error("Stage1 save, internal results and submission paths must differ");
        for(const auto &file:{opt.results_file,opt.submission_file}) {
            if(file.empty())throw std::runtime_error("empty Stage1 result output path");
            if(std::filesystem::exists(file) && !ecm_results::has_result_ledger(file))
                throw std::runtime_error("result output has no owned ledger; select a fresh Stage1 result path");
        }
        ecm_edwards::Integer plus;mpz_add_ui(plus.v,N,1);
        const auto exponent=opt.target_kbnc?opt.handoff_n:(mpz_popcount(plus.v)==1?mpz_scan1(plus.v,0):0);
        const auto name=std::string(base.edwards?".e":".m"+std::to_string(base.param))+"_s1_"+
            (exponent?"m"+std::to_string(exponent):"bit"+std::to_string(mpz_sizeinbase(N,2)))+"_"+std::to_string(uint64_t(B1))+".ckpt";
        const auto path=opt.tmp_dir.empty()?std::string{}:normalize_path_local(std::filesystem::path(opt.tmp_dir)/name);
        if(!path.empty())std::filesystem::create_directories(opt.tmp_dir);
        ecm_stage1_set_backup_limit(opt.ckpt_backups);ecm_stage1_set_checkpoint_path(path);
        {
            auto identity=ecm_edwards::hex(N)+"|"+(base.edwards?"ed":"mont"+std::to_string(base.param))+"|"+
                std::to_string(uint64_t(B1))+"|"+std::to_string(int(opt.selection.scalar_construction))+"|"+
                ecm_stage1::execution_target_name(opt.selection.target);
            if(opt.selection.target==ecm_stage1::ExecutionTarget::Cpu)
                identity+="|"+std::to_string(opt.naf_w?opt.naf_w:edwards_get_naf_w())+"|"+std::to_string(int(opt.selection.cpu_implementation));
            else if(!base.edwards) {
                const char *algorithm=std::getenv("ECM_GPU_STAGE1_ALGO");
                identity+="|"+std::string(algorithm && *algorithm?algorithm:"ladder");
            }
            ecm_stage1_begin_batch(identity,curves,opt.sigma_fixed,opt.fixed_sigma64);
        }
        stage1_copy_save_history(opt,savefilename);
        const auto begin=std::chrono::steady_clock::now();
        const int rc=run_stage1_impl(N,B1,B2,curves,savefilename,saveappend,n_expr,opt,out);
        if(rc==ECM_ERROR){if(!ecm_stage1_stop_requested() && !out->publish_failed)out->prepare_failed=true;ecm_stage1_end_batch(false);return rc;}
        if(!out->sigmas || !out->factors || !out->array_found)throw std::runtime_error("Stage1 completed without per-curve result identity");
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::vector<std::pair<std::string,std::string>> internal_records;
        std::string task_identity=ecm_edwards::hex(N)+"|"+std::to_string(base.b1)+(base.edwards?"ed":"mont"+std::to_string(base.param));
        for(uint32_t i=0;i<out->curves;++i)task_identity+="|"+std::to_string(out->sigmas[i]);
        const auto publication_task=std::to_string(ecm_edwards::hash(task_identity));
        for(uint32_t i=0;i<out->curves;++i) {
            auto r=base;r.publication_task=publication_task;r.sigma=out->sigmas[i];r.when=ecm_results::utc_timestamp();
            if(out->array_found[i]!=ECM_NO_FACTOR_FOUND){r.stage=1;r.factors.push_back(ecm_results::decimal(out->factors[i]));}
            (void)ecm_results::submission_json(r,N);
            const auto key=ecm_edwards::hex(N)+"|"+(r.edwards?"ed":"mont"+std::to_string(r.param))+"|"+std::to_string(r.sigma)+"|"+std::to_string(r.b1)+"|stage1";
            std::ostringstream extra;extra<<",\"stage1-engine\":{\"curve\":"<<i<<",\"target\":"<<ecm_results::quoted(ecm_stage1::execution_target_name(opt.selection.target))
                <<",\"batch-seconds\":"<<seconds<<",\"coverage\":\"B1\"}";
            internal_records.emplace_back(key,ecm_results::internal_json(r,N,0,stage1_build_identity(),extra.str()));
            ecm_results::publish_submission(opt.submission_file,r,N,false);
        }
        ecm_results::publish_internal_batch(opt.results_file,internal_records);
        ecm_results::seal_submissions(opt.submission_file,1,publication_task);
        ecm_stage1_end_batch(true);
        ecm_ts_fprintf(stdout,"Stage1 results: internal=%s submission=%s\n",opt.results_file.c_str(),opt.submission_file.c_str());
        return out->ret;
    } catch(const std::exception &e) {
        out->publish_failed=out->factors!=nullptr;out->prepare_failed=!out->publish_failed;out->ret=ECM_ERROR;
        ecm_stage1_end_batch(false);ecm_ts_fprintf(stderr,"ERROR: Stage1: %s\n",e.what());return ECM_ERROR;
    }
}

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
                          const P95TransferConfig *p95,const std::string &published_line={}) {
    Stage1RunOptions opt = base_opt;
    opt.result_aid=aid;
    if (sigma_fixed) {
        opt.sigma_fixed = true;
        opt.fixed_sigma64 = sigma;
    }
    {
        std::string input_error;
        if (stage1_selection_error(opt.selection,input_error) || stage1_task_error(opt, curves, input_error)) {
            ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged\n", input_error.c_str());
            return false;
        }
    }

    Stage1RunResult result;
    run_stage1_once(N, B1, B2, curves, save_name, /*saveappend=*/true, n_expr, opt, &result);

    const bool has_factors = (result.factors != nullptr);

    if (result.prepare_failed || result.publish_failed) {
        if (has_factors) {
            for (uint32_t i = 0; i < result.curves; ++i) mpz_clear(result.factors[i]);
            free(result.factors);
            free(result.array_found);
            free(result.sigmas);
        }
        ecm_ts_fprintf(stderr, "FATAL: %s; worktodo unchanged.\n",
                       result.publish_failed ? "Stage1 save/result publication failed" :
                                               "backend prepare failed");
        return false;
    }

    bool found_factor = false;
    /* D3 (docs/usage/GUI.md): the hit line carries everything a consumer needs to
       build a results record, because each back-end prints its own, mutually
       incompatible hit line ("curve i sigma=M -> factor found" from the CPU paths,
       "GPU: factor ... with curve c (sigma p:s)" from the GPU ones). Fields: the
       factor, the curve index inside this batch, the EXACT per-curve sigma
       (result.sigmas, when the path knows it), the parametrization, which engine ran,
       and the save-file name (generated for GPU ECM=/ECM2= tasks). */
    const char *curve_token = ecm_stage1::curve_model_name(opt.selection.curve);
    const char *target_token = ecm_stage1::execution_target_name(opt.selection.target);
    if (has_factors) {
        for (uint32_t i = 0; i < result.curves; ++i) {
            if (result.array_found[i] != ECM_NO_FACTOR_FOUND) {
                char *fs = mpz_get_str(nullptr, 10, result.factors[i]);
                std::cout << "factor[" << i << "]=" << (fs ? fs : "?") << " curve=" << i;
                if (result.sigmas != nullptr) {
                    std::cout << " sigma=" << result.sigmas[i];
                }
                if(opt.selection.curve==ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << (opt.selection.target == ecm_stage1::ExecutionTarget::Gpu
                                                   ? opt.selection.gpu_param : 0);
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

    if (has_factors) {
        for (uint32_t i = 0; i < result.curves; ++i) mpz_clear(result.factors[i]);
        free(result.factors);
        free(result.array_found);
        free(result.sigmas);
    }

    if (result.ret == ECM_ERROR && ecm_stage1_stop_requested()) {
        ecm_ts_fprintf(stdout, "STOP: Stage1 interrupted; worktodo unchanged: %s\n",
                       line.c_str());
        return false;
    }

    if (result.ret == ECM_ERROR) {
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
    if (!ecm_append_text_line(finished_path, completed_line)) {
        ecm_ts_fprintf(stderr, "ERROR: cannot append to %s\n", finished_path.c_str());
        if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::MarkError)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot mark the failed line in %s; aborting\n",
                           worktodo_path.c_str());
            return false;
        }
        return true;
    }
    if (!ecm_worktodo_advance(worktodo_path, worker, line, WorktodoAction::Remove)) {
        // The task itself succeeded (its .save is on disk), but the queue cannot drop
        // the line. Abort: continuing would re-run it and append a second time.
        ecm_ts_fprintf(stderr, "ERROR: cannot remove the finished line from %s; aborting "
                               "(the task's .save is already written)\n",
                       worktodo_path.c_str());
        return false;
    }

    ecm_sync_save_files(exe_dir, sync1, sync2, full_sync, marker);
    if (!opt.tmp_dir.empty() && opt.tmp_dir != exe_dir)
        ecm_sync_save_files(opt.tmp_dir, sync1, sync2, full_sync, marker);

    /* ── Prime95 handoff (docs/usage/GUI.md) ──────────────────────────────────
       The task is finished and its .save is synchronized, so Prime95's stage 2 can take
       it from worktodo.add. AID, bounds and known factors are preserved; an ECMSTAGE2
       save token names the actual curve-prefixed file. Other fields remain verbatim.
       and a handoff failure NEVER fails the task: the line is parked and re-delivered
       with the next successful delivery, and the GUI shows a red notice meanwhile.
       Supported handoff remains available after local result publication. */
    if (p95 != nullptr && !p95->worktodo_path.empty()) {
        const P95TransferResult tr = p95_transfer_deliver(*p95, completed_line);
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
     gpu_param=<0|2|3>  fold=<0|1>  carry_bits=<n>  picked=<0|1>  tier_count=<n>
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
                          int gpu_param, uint32_t want_bits) {
    ecm_backend_gpu_info info;
    const char *err = nullptr;
    const int rc = ecm_backend_query_gpu(device_index, gpu_param, want_bits, &info, &err);

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

    // Resolve ini path (default: exe_dir/ecm.ini).
    std::string ini = ini_path;
    if (ini.empty()) {
        ini = exe_dir + "/ecm.ini";
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

    const std::string worktodo_path = resolve_rel_local(exe_dir, cfg.worktodo);
    const std::string finished_path = resolve_rel_local(exe_dir, cfg.finished);
    const std::string sync1 = resolve_rel_local(exe_dir, cfg.save_sync_dir_1);
    const std::string sync2 = resolve_rel_local(exe_dir, cfg.save_sync_dir_2);

    // Initialize handoff only for an eligible actual task. A typed save may override
    // the INI curve, and unsupported constructions must not touch transfer paths.
    P95TransferConfig p95cfg;
    bool p95_initialized=false,p95_ignore_reported=false;
    auto p95_for_task=[&](const Stage1RunOptions &actual)->const P95TransferConfig* {
        const bool ignore=actual.selection.curve==ecm_stage1::CurveModel::Edwards ||
            (actual.selection.target==ecm_stage1::ExecutionTarget::Gpu && actual.selection.gpu_param==2);
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
    if (!ecm_stage1::parse_curve_model(cfg.curve_model, opt.selection.curve) ||
        !ecm_stage1::parse_execution_target(cfg.execution_target, opt.selection.target)) {
        ecm_ts_fprintf(stderr, "ERROR: invalid curve or target in ecm.ini\n");
        return 1;
    }
    opt.affinity_cpus = parse_affinity_spec(cfg.affinity);
    /* ---- [cpu] shared by edwards and mont ----------------------------------- */
    /* CPU arithmetic: auto | simd | scalar */
    {
        if (!ecm_stage1::parse_cpu_implementation(cfg.cpu_implementation, opt.selection.cpu_implementation)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid impl in ecm.ini\n");
            return 1;
        }
    }
    /* SIMD reduction: auto | montgomery | mersenne-fold. */
    {
        if (!ecm_stage1::parse_simd_reduction(cfg.simd_reduction, opt.selection.simd_reduction)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid reduction in ecm.ini\n");
            return 1;
        }
    }
    stage1_result_settings(opt,cfg,std::filesystem::path(ini).parent_path().string(),worker);
    opt.stage1_threads = cfg.stage1_threads;
    if (!cfg.save_name_pattern.empty()) opt.save_name_pattern = cfg.save_name_pattern;
    if (!cfg.exp_cache.empty()) ecm_exp_cache_set_dir(cfg.exp_cache);
    /* ---- [edwards] only ------------------------------------------------------ */
    if (cfg.naf_w >= 3 && cfg.naf_w <= 12) edwards_set_naf_w(cfg.naf_w);
    opt.naf_w = cfg.naf_w;
    /* Montgomery scalar construction, shared by CPU and GPU. */
    {
        if (!ecm_stage1::parse_scalar_construction(cfg.scalar_construction, opt.selection.scalar_construction)) {
            ecm_ts_fprintf(stderr, "ERROR: invalid lcm in ecm.ini (expected 1 or 12)\n");
            return 1;
        }
    }
    opt.verbose = cfg.verbose ? 1 : 0;
    /* ---- [gpu] only ---------------------------------------------------------- */
    opt.device_index = cfg.device;
    opt.selection.gpu_param = cfg.gpu_param;
    opt.ckpt_ms = (cfg.ckpt_seconds > 0.0)
                         ? (unsigned long)(cfg.ckpt_seconds * 1000.0)
                         : 0UL;
    opt.ckpt_backups = (unsigned)std::clamp(cfg.ckpt_backups, 0,
                                           (int)ECM_STAGE1_MAX_BACKUPS);
    opt.gpu_mul_path = cfg.kernel_mul;
    opt.gpu_sqr_path = cfg.kernel_sqr;
    opt.gpu_add_path = cfg.kernel_add;
    opt.gpu_sub_path = cfg.kernel_sub;
    opt.gpu_special_mult_path = cfg.kernel_special_mult;
    /* ---- [task] -------------------------------------------------------------- */
    opt.sigma_fixed = (cfg.sigma != 0);
    opt.fixed_sigma64 = cfg.sigma;
    opt.tmp_dir = resolve_rel_local(exe_dir, cfg.tmp_dir);
    // Validate semantic combinations per task, after applying any save prefix.
    /* ---- [handoff] ----------------------------------------------------------- */
    if (!cfg.p95_dir.empty()) {
        ecm_ts_fprintf(stdout,
            "note: p95_dir=%s is informational here -- ecm.exe only writes local saves\n"
            "      to %s; the stage-2 handoff program ecm_p95feeder reads its own\n"
            "      feeder.ini (same key name)\n",
            cfg.p95_dir.c_str(), opt.tmp_dir.c_str());
    }

    ecm_ts_fprintf(stdout,"%s %s (Stage1)\n",ecm_program::name,ecm_program::version);
    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worker : %d (ini section [Worker #%d]; unprefixed keys are the "
                           "global defaults)\n", worker, worker);
    ecm_ts_fprintf(stdout, "worktodo : %s\n", worktodo_path.c_str());
    ecm_ts_fprintf(stdout, "finished : %s\n", finished_path.c_str());
    ecm_ts_fprintf(stdout, "log_file : %s\n", cfg.log_file.c_str());
    ecm_ts_fprintf(stdout, "saves : %s\n", opt.tmp_dir.c_str());
    ecm_ts_fprintf(stdout, "curve model : %s\n",
                   ecm_stage1::curve_model_name(opt.selection.curve));
    ecm_ts_fprintf(stdout, "execution target : %s\n",
                   ecm_stage1::execution_target_name(opt.selection.target));
    if (opt.selection.target == ecm_stage1::ExecutionTarget::Gpu) {
        /* The GPU implementation is chosen at LINK time (ecm.exe = OpenCL glue,
           ecm_cuda.exe = CUDA backend), so ask the backend instead of hardcoding
           "OpenCL" -- the CUDA build used to print "gpu (OpenCL)" here. */
        if(opt.selection.curve==ecm_stage1::CurveModel::Edwards)
            ecm_ts_fprintf(stdout,"gpu backend : %s, Edwards Atkin-Morain, device %d\n",
                           ecm_backend_name(),opt.device_index);
        else ecm_ts_fprintf(stdout, "gpu backend : %s, param%d, device %d\n",
                            ecm_backend_name(), opt.selection.gpu_param, opt.device_index);
    }

    // Startup full sync (matches the old work_manager.ps1 behaviour).
    ecm_sync_save_files(exe_dir, sync1, sync2, /*full=*/true, 0);
    if (!opt.tmp_dir.empty() && opt.tmp_dir != exe_dir)
        ecm_sync_save_files(opt.tmp_dir, sync1, sync2, /*full=*/true, 0);

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
        const WorktodoReadResult read =
            ecm_worktodo_read_first_line(worktodo_path, worker, line, read_error);
        if (read == WorktodoReadResult::Error) {
            ecm_ts_fprintf(stderr, "ERROR: %s\n", read_error.c_str());
            if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
            return 1;
        }
        if (read == WorktodoReadResult::Empty) {
            if(!saw_task)p95_for_task(opt); // Preserve idle-start recovery for an eligible INI.
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
                opt.target_kbnc = true;
                opt.handoff_k_str = task.k;
                opt.handoff_k = std::strtod(task.k.c_str(), nullptr);
                opt.handoff_b = (uint32_t)std::strtoul(task.b.c_str(), nullptr, 10);
                opt.handoff_n = (uint32_t)task.n;
                opt.handoff_c = (int32_t)std::strtol(task.c.c_str(), nullptr, 10);
                opt.handoff_factors.clear();
                for (size_t fi = 0; fi < task.factors.size(); ++fi) {
                    if (fi) opt.handoff_factors += ",";
                    opt.handoff_factors += task.factors[fi];
                }
            }
            // ECM=/ECM2= has no save filename in the assignment. Give the GPU path a
            // deterministic destination so a completed queue task publishes its Stage1
            // point for Stage2 and save synchronization. CPU Montgomery already writes
            // its own shared save under tmp_dir; both Edwards targets use text saves.
            std::string queue_save_name;
            if (opt.selection.target == ecm_stage1::ExecutionTarget::Gpu || opt.selection.curve==ecm_stage1::CurveModel::Edwards) {
                if (opt.tmp_dir.empty() || !ensure_dir(opt.tmp_dir)) {
                    ecm_ts_fprintf(stderr, "FATAL: cannot create tmp_dir for GPU save: %s\n",
                                   opt.tmp_dir.c_str());
                    mpz_clear(N);
                    if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
                    return 1;
                }
                queue_save_name = opt.tmp_dir + "/" +
                    mont_format_save_name(opt.save_name_pattern,N,task.B1,&opt);
            }
            const bool cont = queue_run_one(
                N, task.B1, task.B2, task.curves_to_run,
                task.has_sigma ? task.sigma : 0, task.has_sigma,
                queue_save_name, n_expr, task.aid, line,
                worktodo_path, finished_path, worker, opt,
                exe_dir, sync1, sync2, full_sync, marker, &processed, p95_for_task(opt));
            mpz_clear(N);
            if (!cont) {
                if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
                return 1;
            }
            continue;
        }

        // ECMSTAGE2= (existing CUDA-oriented format, unchanged semantics).
        EcmStage2Task task;
        if (!ecm_parse_stage2_line(line, task, err)) {
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
            opt.target_kbnc = true;
                opt.handoff_k_str = task.k;
            opt.handoff_k = std::strtod(task.k.c_str(), nullptr);
            opt.handoff_b = (uint32_t)std::strtoul(task.b.c_str(), nullptr, 10);
            opt.handoff_n = (uint32_t)task.n;
            opt.handoff_c = (int32_t)std::strtol(task.c.c_str(), nullptr, 10);
            opt.handoff_factors.clear();
            for (size_t fi = 0; fi < task.factors.size(); ++fi) {
                if (fi) opt.handoff_factors += ",";
                opt.handoff_factors += task.factors[fi];
            }
        }
        // The typed save owns curve/param and B1; untyped legacy names use the INI.
        // Clone options so the next task always starts from the configured defaults.
        auto task_opt=opt;
        if(!stage1_save_construction(task.save_name,task_opt,err) ||
           stage1_selection_error(task_opt.selection,err) || stage1_task_error(task_opt,task.curves_to_run,err)) {
            mpz_clear(N);ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());
            if(logf){ecm_log_set_mirror(nullptr);fclose(logf);}return 1;
        }
        ecm_ts_fprintf(stdout,"queue_save: curve=%s param=%s B1=%.0f\n",
            ecm_stage1::curve_model_name(task_opt.selection.curve),
            task_opt.selection.curve==ecm_stage1::CurveModel::Edwards?"-":stage1_save_prefix(task_opt).substr(1).c_str(),B1);
        const auto queue_save_name=stage1_prefixed_save(task.save_name,stage1_save_prefix(task_opt));
        std::string published_line=line;
        if(queue_save_name!=task.save_name && !ecm_rewrite_stage2_save(line,queue_save_name,published_line,err)) {
            mpz_clear(N);ecm_ts_fprintf(stderr,"FATAL: %s; worktodo unchanged\n",err.c_str());return 1;
        }
        if(queue_save_name!=task.save_name)task_opt.legacy_save_source=task.save_name;
        const bool cont = queue_run_one(
            N, B1, /*B2=*/0.0, task.curves_to_run,
            /*sigma=*/0, /*sigma_fixed=*/false,
            queue_save_name, n_expr, task.aid, line,
            worktodo_path, finished_path, worker, task_opt,
            exe_dir, sync1, sync2, full_sync, marker, &processed, p95_for_task(task_opt),published_line);
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
        if(a == "-ini" && i+1<argc) {
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
           queue path uses, so `-ini ecm.ini --gpu-info` reports what the workers will
           actually run. The ini is read, never created (a missing one leaves the CLI
           defaults alone and creates nothing). */
        const std::string raw_exe_dir = get_exe_dir_local();
        const std::string exe_dir = raw_exe_dir.empty() ? "." : raw_exe_dir;
        std::string ini = ini_path.empty() ? (exe_dir + "/ecm.ini")
                                           : resolve_rel_local(exe_dir, ini_path);
        int eff_device = gpu_device_index;
        int eff_param = gpu_param_set ? gpu_param_cli : ecm_config::defaults::stage1_gpu_param;
        const char *ini_used = nullptr;
        EcmQueueConfig info_cfg;
        if (ecm_queue_config_load(ini, worker_index, info_cfg)) {
            ini_used = ini.c_str();
            if (!device_set) eff_device = info_cfg.device;
            if (!gpu_param_set) eff_param = info_cfg.gpu_param;
        } else if (access(ini.c_str(), 0) == 0) {
            std::cerr << "Invalid or unreadable config: " << ini << std::endl;
            return 1;
        }
        return print_gpu_info(ini_used, worker_index, eff_device, eff_param, gpu_info_bits);
    }

    if (stop_handle && !ecm_stage1_set_stop_handle(stop_handle)) {
        std::cerr << "Invalid inherited Stage1 stop handle" << std::endl;
        return 1;
    }
    ecm_stage1_install_stop_handler();

    if (pos.empty()) {
        // No positional B1/B2 → queue-manager mode (reads ecm.ini + worktodo).
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
    opt.selection = selection;
    /* The two CPU curve models share the thread count. */
    opt.stage1_threads = stage1_threads;
    opt.naf_w = naf_w;                       /* [edwards] */
    if (!affinity_spec.empty()) opt.affinity_cpus = parse_affinity_spec(affinity_spec);

    opt.verbose = verbose ? 1 : 0;
    opt.device_index = gpu_device_index;
    /* Default home of the stage-1 exponent cache for single runs -- run_queue_manager()
       sets the same default for queue mode, and an explicit --exp-cache / ecm.ini
       exp_cache always wins (the getter is only consulted when nothing was set). */
    if (ecm_exp_cache_get_dir().empty()) {
        const std::string d = get_exe_dir_local();
        ecm_exp_cache_set_dir(d.empty() ? "." : d);
    }
    opt.selection.gpu_param = gpu_param_set ? gpu_param_cli
                                            : ecm_config::defaults::stage1_gpu_param;
    {
        std::string selection_error;
        if (stage1_selection_error(opt.selection, selection_error)) {
            std::cerr << "ERROR: " << selection_error << std::endl;
            mpz_clear(N);
            return 1;
        }
    }
    opt.ckpt_ms = ckpt_ms;
    opt.gpu_mul_path = gpu_mul_path;
    opt.gpu_sqr_path = gpu_sqr_path;
    opt.gpu_add_path = gpu_add_path;
    opt.gpu_sub_path = gpu_sub_path;
    opt.gpu_special_mult_path = gpu_special_mult_path;
    opt.sigma_fixed = sigma_fixed;
    opt.fixed_sigma64 = fixed_sigma64;
    {
        std::string input_error;
        if (stage1_task_error(opt, gpucurves, input_error)) {
            std::cerr << "ERROR: " << input_error << std::endl;
            mpz_clear(N);
            return 1;
        }
    }
    if (tmp_dir.empty()) tmp_dir = ".";      // 默认 = 当前目录 (本地)
    opt.tmp_dir = normalize_path_local(tmp_dir);
    {
        const auto ini=ini_path.empty()?std::filesystem::path(get_exe_dir_local())/"ecm.ini":std::filesystem::path(ini_path);
        EcmQueueConfig cfg;
        if(ecm_queue_config_load(ini.string(),worker_index,cfg))stage1_result_settings(opt,cfg,ini.parent_path().string(),worker_index);
        else if(!ini_path.empty()) {std::cerr<<"Cannot read Stage1 result settings from INI"<<std::endl;mpz_clear(N);return 1;}
    }
    if (!p95_dir_ignored.empty()) {
        std::cout << "Prime95 transfer path ignored: --p95-dir is obsolete; Stage1 saves "
                     "remain local. Supported Montgomery handoff uses INI p95_worktodo_path."
                  << std::endl;
    }
    {
        // 存档命名: 假设 N = 2^n - 1 (Mersenne), n = 位长
        opt.handoff_k = 1.0;
        opt.handoff_k_str = "1";
        opt.handoff_b = 2;
        opt.handoff_n = (uint32_t)mpz_sizeinbase(N, 2);
        opt.handoff_c = -1;
    }

    Stage1RunResult result;
    const int rc1 = run_stage1_once(N, B1, B2, gpucurves, savefilename, saveappend, nline, opt, &result);

    std::vector<uint32_t> go_primes;
    if (print_group_order) {
        build_primes_up_to_B1(B1, go_primes);
    }

    /* 中止/失败时 run_stage1_once 已经释放并把 factors/array_found 置空 —— 直接索引
       就是空指针解引用 (0xC0000005, Windows 还会弹一个模态崩溃框, 把管道卡死)。
       队列路径早就用 has_factors 判过了, 这里漏了; 触发条件包括 SIMD 批量报错、
       用户中止 (SIGINT/checkpoint abort) 等任何 stage-1 非正常结束。 */
    const bool cli_has_factors = (result.ret != ECM_ERROR && result.factors != nullptr && result.array_found != nullptr);

    // Numeric Montgomery parameter used by the group-order helper. CPU Suyama
    // uses PARAM0; GPU Montgomery uses its selected family. Edwards hit metadata
    // identifies the curve model and carries no Montgomery PARAM.
    const int go_param = opt.selection.target == ecm_stage1::ExecutionTarget::Gpu
                             ? opt.selection.gpu_param : 0;

    if (cli_has_factors) {
        /* Same D3 fields as the queue path (docs/usage/GUI.md): a consumer must not
           have to guess which of the three back-end hit lines it is looking at. */
        const char *curve_token = ecm_stage1::curve_model_name(opt.selection.curve);
        const char *target_token = ecm_stage1::execution_target_name(opt.selection.target);
        for (uint32_t i = 0; i < result.curves; i++) {
            if (result.array_found[i] != ECM_NO_FACTOR_FOUND) {
                char *s = mpz_get_str(NULL, 10, result.factors[i]);
                std::cout << "factor[" << i << "]=" << (s ? s : "?") << " curve=" << i;
                if (result.sigmas != nullptr) {
                    std::cout << " sigma=" << result.sigmas[i];
                }
                if(opt.selection.curve==ecm_stage1::CurveModel::Montgomery)
                    std::cout << " param=" << go_param;
                std::cout << " curve_model=" << curve_token
                          << " execution_target=" << target_token
                          << " save=" << savefilename << "\n";
                free(s);
                if (print_group_order) {
                    uint32_t sigma_curve = result.firstsigma + i;
                    if (mpz_probab_prime_p(result.factors[i], 25) <= 0) {
                        std::cout << "  go_factor[" << i << "]=[ ] (factor is not prime, skip #E(F_p))\n";
                        continue;
                    }
                    mpz_t go;
                    mpz_init(go);
                    std::string err;
                    std::string go_factor_lines;
                    if (!compute_group_order_pari(go, &go_factor_lines, result.factors[i],
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

        for (uint32_t i = 0; i < result.curves; i++) mpz_clear(result.factors[i]);
        free(result.factors);
        free(result.array_found);
        free(result.sigmas);
    }
    mpz_clear(N);
    /* 让调用方 (脚本/Prime95 前端) 能看到失败, 而不是把 0 当成"跑完了没因子"。 */
    return (rc1 == ECM_ERROR || result.ret == ECM_ERROR) ? 1 : 0;
}
