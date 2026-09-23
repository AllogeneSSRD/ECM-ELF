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
#include "opencl_ecm_runtime_config.h"
#include "ecm.h"
#include "cgbn_stage1.h"            /* gpu_pick_random_sigma / gpu_compute_batch_d */
#include "opencl_ecm_log.h"
#include "ecm_queue_config.h"
#include "ecm_worktodo.h"
#include "ecm_edwards_cpu.h"        /* Edwards (Atkin-Morain) stage-1 CPU path */
#include "ecm_edwards_save.h"       /* Prime95 ECM 二进制存档读写 */

static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
}

class ExprParser {
public:
    explicit ExprParser(const std::string &input) : text(input), pos(0), error(false) {}

    bool parse(mpz_t out) {
        skip_ws();
        parse_expr(out);
        skip_ws();
        if (!error && pos != text.size()) {
            set_error("unexpected trailing characters");
        }
        return !error;
    }

    const std::string &message() const { return message_text; }

private:
    const std::string &text;
    size_t pos;
    bool error;
    std::string message_text;

    void set_error(const std::string &msg) {
        if (!error) {
            error = true;
            message_text = msg;
        }
    }

    void skip_ws() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
    }

    bool match(char ch) {
        skip_ws();
        if (pos < text.size() && text[pos] == ch) {
            ++pos;
            return true;
        }
        return false;
    }

    bool peek(char ch) {
        skip_ws();
        return pos < text.size() && text[pos] == ch;
    }

    void parse_expr(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_term(lhs);
        while (!error) {
            if (match('+')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_add(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('-')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_term(rhs);
                mpz_sub(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_term(mpz_t out) {
        mpz_t lhs;
        mpz_init(lhs);
        parse_power(lhs);
        while (!error) {
            if (match('*')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                mpz_mul(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else if (match('/')) {
                mpz_t rhs;
                mpz_init(rhs);
                parse_power(rhs);
                if (mpz_sgn(rhs) == 0) {
                    set_error("division by zero");
                    mpz_clear(rhs);
                    break;
                }
                if (!mpz_divisible_p(lhs, rhs)) {
                    set_error("division is not exact (result would not be an integer)");
                    mpz_clear(rhs);
                    break;
                }
                mpz_divexact(lhs, lhs, rhs);
                mpz_clear(rhs);
            } else {
                break;
            }
        }
        mpz_set(out, lhs);
        mpz_clear(lhs);
    }

    void parse_power(mpz_t out) {
        mpz_t base;
        mpz_init(base);
        parse_unary(base);
        if (error) {
            mpz_clear(base);
            return;
        }

        if (match('^')) {
            mpz_t exponent;
            mpz_init(exponent);
            parse_power(exponent);
            if (error) {
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            if (mpz_sgn(exponent) < 0 || !mpz_fits_ulong_p(exponent)) {
                set_error("exponent must be a non-negative integer that fits in unsigned long");
                mpz_clear(base);
                mpz_clear(exponent);
                return;
            }
            unsigned long exp = mpz_get_ui(exponent);
            mpz_pow_ui(out, base, exp);
            mpz_clear(base);
            mpz_clear(exponent);
            return;
        }

        mpz_set(out, base);
        mpz_clear(base);
    }

    void parse_unary(mpz_t out) {
        skip_ws();
        if (match('+')) {
            parse_unary(out);
            return;
        }
        if (match('-')) {
            parse_unary(out);
            mpz_neg(out, out);
            return;
        }
        parse_primary(out);
    }

    void parse_primary(mpz_t out) {
        skip_ws();
        if (match('(')) {
            parse_expr(out);
            if (!match(')')) {
                set_error("missing closing parenthesis");
            }
            return;
        }

        size_t start = pos;
        bool saw_digit = false;
        while (pos < text.size()) {
            unsigned char ch = static_cast<unsigned char>(text[pos]);
            if (std::isalnum(ch) || ch == 'x' || ch == 'X') {
                saw_digit = true;
                ++pos;
            } else {
                break;
            }
        }
        if (!saw_digit) {
            set_error("expected number or parenthesized expression");
            return;
        }

        std::string token = text.substr(start, pos - start);
        if (mpz_set_str(out, token.c_str(), 0) != 0) {
            set_error(std::string("invalid integer token: ") + token);
        }
    }
};

// Compute batch product s = prod_{p<=B1} p^{floor(log_p(B1))}
static bool compute_batch_s(mpz_t s, double B1){
    static const unsigned MAX_HEIGHT = 32;

    if(B1 < 2.0) {
        mpz_set_ui(s, 1);
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
        if (!sieve[p]) {
            continue;
        }
        for (uint64_t q = (uint64_t)p * (uint64_t)p; q <= limit; q += p) {
            sieve[(size_t)q] = 0;
        }
    }

    mpz_t acc[MAX_HEIGHT];
    mpz_t ppz;
    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_init(acc[j]);
    }
    mpz_init(ppz);

    unsigned i = 0;
    for (uint32_t pi = 2; pi <= limit; ++pi) {
        if (!sieve[pi]) {
            continue;
        }

        uint64_t pp = pi;
        const uint64_t maxpp = limit / pi;
        while (pp <= maxpp) {
            pp *= pi;
        }

        mpz_import(ppz, 1, -1, sizeof(pp), 0, 0, &pp);

        if ((i & 1u) == 0u) {
            mpz_set(acc[0], ppz);
        } else {
            mpz_mul(acc[0], acc[0], ppz);
        }

        unsigned j = 0;
        while ((i & (1u << j)) != 0u) {
            if (j + 1 >= MAX_HEIGHT - 1) {
                for (unsigned k = 0; k < MAX_HEIGHT; ++k) {
                    mpz_clear(acc[k]);
                }
                mpz_clear(ppz);
                return false;
            }

            if ((i & (1u << (j + 1))) == 0u) {
                mpz_swap(acc[j + 1], acc[j]);
            } else {
                mpz_mul(acc[j + 1], acc[j + 1], acc[j]);
            }
            mpz_set_ui(acc[j], 1);
            ++j;
        }

        ++i;
    }

    if (i == 0) {
        mpz_set_ui(s, 1);
    } else {
        mpz_set(s, acc[0]);
        for (unsigned j = 1; j < MAX_HEIGHT && mpz_cmp_ui(acc[j], 0) != 0; ++j) {
            mpz_mul(s, s, acc[j]);
        }
    }

    for (unsigned j = 0; j < MAX_HEIGHT; ++j) {
        mpz_clear(acc[j]);
    }
    mpz_clear(ppz);
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
static bool parse_sigma64_arg(const std::string &arg, uint64_t *sigma_out) {
    std::string s = arg;
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
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

static bool compute_group_order_pari_for_sigma3(mpz_t order_out, const mpz_t p,
                                                uint32_t sigma, const std::string &gp_path,
                                                std::string *err) {
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
    gpfile << "A = Mod(4*s, p) / Mod(2^32, p) - 2;\n";
    gpfile << "b = 4*A + 10;\n";
    gpfile << "E = ellinit([0, b*A, 0, b^2, 0]);\n";
    gpfile << "print(lift(ellcard(E)));\n";
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
    std::string last_int;
    while (std::getline(iss, line)) {
        trim(line);
        if (line.empty()) continue;
        bool ok = true;
        size_t start = (line[0] == '-') ? 1 : 0;
        if (start == line.size()) ok = false;
        for (size_t i = start; ok && i < line.size(); ++i) {
            if (!std::isdigit((unsigned char)line[i])) ok = false;
        }
        if (ok) last_int = line;
    }
    if (last_int.empty()) {
        if (err) *err = "gp returned no integer ellcard output";
        return false;
    }
    if (mpz_set_str(order_out, last_int.c_str(), 10) != 0) {
        if (err) *err = "failed to parse gp ellcard integer";
        return false;
    }
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

    std::cout << "OpenCL ECM stage-1 driver\n\n"
              << "Usage:\n"
              << "  echo '<N>' | " << name << " [options] B1 [B2]\n\n"
              << "Input:\n"
              << "  N is read from stdin as a decimal integer or expression\n"
              << "  (e.g. '(2^991-1)', '0xdeadbeef'). Whitespace is ignored.\n\n"
              << "Positional:\n"
              << "  B1              Stage-1 bound (required for meaningful runs)\n"
              << "  B2              Stage-2 bound (optional; 0 disables stage 2)\n\n"
              << "Options:\n"
              << "  -gpu                 Enable GPU stage-1 (requires -gpucurves)\n"
              << "  --edwards            Enable CPU Edwards stage-1 (Atkin-Morain, a=1)\n"
              << "  --edwards-threads <n>  Edwards stage-1 worker threads (0=auto, 1=serial)\n"
            << "  --edwards-backend <m>  auto|simd|gmp (batch 8 curves via AVX512-IFMA;\n"
            << "                         simd forces it and errors out if the CPU lacks it)\n"
              << "  --edwards-naf-w <w>  NAF window (default 12); dictionary = 2^(w-2)\n"
              << "  --tmp-dir <dir>      Local dir for stage-1 saves e{n:07d}_c{k}[.tmp]\n"
              << "                       (default: current dir; ecm.exe never writes to p95)\n"
              << "  -gpucurves <n>       Number of ECM curves per launch (Edwards: total curves)\n"
              << "  -gpuckpt <sec>       GPU checkpoint interval in seconds (default: 600)\n"
              << "  -d <index>           OpenCL device index (default: 0)\n"
              << "  -sigma <value>       Fixed curve sigma (1..2^32-1; optional param:3: prefix)\n"
              << "  -v                   Verbose output\n"
              << "  -save <file>         Append factorization lines to file\n"
              << "  -savea <file>        Same as -save (append mode)\n"
              << "  --go                 Print group order diagnostics (requires gp/PARI)\n"
              << "  --gp <path>          Path to gp executable (default: gp on PATH)\n"
              << "  --mul <path>         Montgomery mul kernel path (4096-bit)\n"
              << "  --sqr <path>         Montgomery sqr kernel path\n"
              << "  --add <path>         Modular add kernel path\n"
              << "  --sub <path>         Modular sub kernel path\n"
              << "  --special-mult <path>  special_mult (R=2^32) kernel path\n"
              << "  --showkernel         List available OpenCL kernel paths and exit\n"
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
              << "  --kernel-cache-dir <dir>   OpenCL binary cache directory\n"
              << "  --no-kernel-cache          Disable kernel binary cache\n"
              << "  --kernel-cache-verbose     Verbose cache hit/miss logging\n"
              << "  --compile-verbose          Verbose compile timing\n"
              << " Logging / debug / verify:\n"
              << "  --no-log-timestamp         Disable log timestamps (default on)\n"
              << "  --gpu-dump [--gpu-dump-file <f>]      Dump GPU state to CSV\n"
              << "  --profile-ops [--profile-ops-file <f>]  Operator-count profiling\n"
              << "  --sync-each-batch          Synchronize after each batch\n"
              << "  --verify-gpu [--verify-gpu-strict]   CPU cross-check GPU results\n\n"
              << "Examples:\n"
              << "  echo '(2^991-1)' | " << name << " -v --go -gpu -gpucurves 384 1e6 0\n"
              << "  echo '(2^421-1)' | " << name << " -gpu -gpucurves 256 -d 1 1e5 0\n"
              << "  echo '(2^4003-1)' | " << name << " -gpu -gpucurves 384 --add asm_b32 1e6 0\n"
              << "  " << name << " --showkernel\n\n"
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

static std::string resolve_rel_local(const std::string &base, const std::string &p) {
    if (p.empty() || is_absolute_path_local(p)) return p;
    return base + "/" + p;
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
    bool use_gpu = true;
    bool use_edwards = false;        // CPU Edwards (Atkin-Morain) stage-1 path
    uint32_t edwards_threads = 0;    // 0 = auto (min(curves, #cores)); 1 = 顺序
    int      backend = 0;            // 0=auto 1=simd(强制,无 ISA 则报错) 2=gmp(标量)
    int      backend_auto_pick = -1; // 实际选中的: 1=simd 2=gmp (打印用)
    long long stage1_t0_ms = 0;      // 进度条计时起点
    uint32_t  stage1_curves = 0;     // 进度条总数
    std::vector<unsigned> affinity_cpus;   // 亲核性 (空 = 不绑定)
    int edwards_naf_w = 0;           // 0 = 用 ecm_edwards_cpu 的默认窗口
    int verbose = 0;
    int device_index = 0;
    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    std::string gpu_mul_path, gpu_sqr_path, gpu_add_path, gpu_sub_path, gpu_special_mult_path;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;        // GPU batch sigma (32-bit)
    uint64_t fixed_sigma64 = 0;      // Edwards sigma (64-bit)
    // 本地 stage-1 落盘目录 (空 = 不落盘/不 checkpoint)。
    // 结果写成 <tmp_dir>/e{n:07d}_c{curve:06d}.tmp (MIDSTAGE 供 stage-2) 与
    // <tmp_dir>/e{n:07d}_c{curve:06d} (STAGE1 自检查点); p95 交接由 ecm_p95feeder 负责。
    std::string tmp_dir;
    double handoff_k = 1.0;
    std::string handoff_k_str = "1"; // 原始 k 字符串 (worktodo.add 用)
    uint32_t handoff_b = 2;
    uint32_t handoff_n = 0;
    int32_t handoff_c = -1;
    std::string handoff_factors;     // 透传给 worktodo.add 的已知因子 (逗号分隔)
};

struct Stage1RunResult {
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    uint64_t firstsigma64 = 0;       // full 64-bit sigma (Edwards path)
    mpz_t *factors = nullptr;
    int *array_found = nullptr;
};

// ---- Prime95 交接 + 自我 checkpoint (Edwards stage-1) ----

static volatile sig_atomic_t g_edwards_stop = 0;
static void edwards_sigint_handler(int) { g_edwards_stop = 1; }

// 多曲线并行时串行化输出 (避免 stdout 行交错).
static std::mutex g_edwards_out_mutex;

static long long edwards_now_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- 本地 stage-1 结果落盘 ----
//
// ecm.exe 不再与 Prime95 交互。每条曲线把 stage-1 结果写到**本地**目录:
//     <tmp_dir>/e{n:07d}_c{curve:06d}.tmp       (MIDSTAGE, state=2, 供 stage-2)
//     <tmp_dir>/e{n:07d}_c{curve:06d}           (STAGE1,  state=1, 分块自检查点)
// 把结果送进 Prime95 (在 p95 目录写 e{n:07d} + 往 worktodo.add 的 [Worker #N]
// 段追加 ECM= 行) 由独立程序 ecm_p95feeder 负责。

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

// 曲线本地路径主干: <tmp_dir>/e{n:07d}_B{B1}_c{curve:06d}
//
// B1 必须进名字: 同一个 n 配不同 B1 的不同任务(例如 p95 参考 worktodo 里
// Worker #2/#3/#4 都是 n=12323 而 B1 分别是 55e6/130e6/400e6)否则会互相覆盖。
// p95 侧的存档名仍由 ecm_p95feeder 从存档头部推出 (e{n:07d}), 与本名字无关。
static std::string edwards_local_stem(const Stage1RunOptions &opt, uint32_t curve_idx,
                                     uint64_t B1) {
    std::string dir = opt.tmp_dir;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
    char name[80];
    snprintf(name, sizeof(name), "e%07u_B%llu_c%06u", opt.handoff_n,
             (unsigned long long)B1, curve_idx + 1);
    return dir + name;
}

// 目标文件已存在且头部与本次写入不一致 -> 拒绝覆盖 (返回 false).
// 一致时允许覆盖: 那是正常的 resume / 周期刷新.
static bool local_save_overwrite_ok(const std::string &path, const ecm_save_common &cm) {
    if (access(path.c_str(), 0) != 0) return true;   // 不存在, 直接写
    ecm_save_common old;
    uint32_t st = 0;
    if (!ecm_save_read_header(path, old, &st)) return true;  // 损坏文件允许覆盖
    const bool same = (old.k == cm.k) && (old.b == cm.b) && (old.n == cm.n) &&
                      (old.c == cm.c) && (old.B1 == cm.B1) && (old.sigma == cm.sigma);
    if (!same) {
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
        ecm_ts_fprintf(stderr,
            "ERROR: refusing to overwrite %s: existing save is "
            "(k=%.0f b=%u n=%u c=%d B1=%llu sigma=%llu) but this run is "
            "(k=%.0f b=%u n=%u c=%d B1=%llu sigma=%llu)\n",
            path.c_str(), old.k, old.b, old.n, old.c,
            (unsigned long long)old.B1, (unsigned long long)old.sigma,
            cm.k, cm.b, cm.n, cm.c, (unsigned long long)cm.B1,
            (unsigned long long)cm.sigma);
        return false;
    }
    return true;
}

// 交接行 (ECM2=...); 仅用于日志/追踪 —— ecm_p95feeder 会从存档头部重建同样的行.
static std::string edwards_ecm2_line(const Stage1RunOptions &opt, uint64_t sigma,
                                     uint64_t B1, uint64_t B2) {
    std::string line = "ECM2=" + (opt.handoff_k_str.empty() ? "1" : opt.handoff_k_str);
    line += "," + std::to_string(opt.handoff_b);
    line += "," + std::to_string(opt.handoff_n);
    line += "," + std::to_string(opt.handoff_c);
    line += "," + std::to_string(B1);
    line += "," + std::to_string(B2);
    line += ",1," + std::to_string(sigma);
    if (!opt.handoff_factors.empty()) line += ",\"" + opt.handoff_factors + "\"";
    return line;
}

// 把一条曲线的 stage-1 结果写成本地 MIDSTAGE 存档 (<tmp_dir>/e{n:07d}_c{k}.tmp).
static bool edwards_write_local_midstage(const Stage1RunOptions &opt, uint64_t sigma,
                                         uint32_t curve, uint64_t B1, uint64_t B2,
                                         const mpz_t Qx, const mpz_t Qz,
                                         uint32_t curve_idx) {
    if (opt.tmp_dir.empty()) return true;

    ecm_save_common cm;
    cm.k = opt.handoff_k;
    cm.b = opt.handoff_b;
    cm.n = opt.handoff_n;
    cm.c = opt.handoff_c;
    cm.curve = curve;
    cm.B1 = B1;
    cm.B2 = B2;
    cm.sigma = sigma;

    const std::string path = edwards_local_stem(opt, curve_idx, B1) + ".tmp";
    if (!local_save_overwrite_ok(path, cm)) return false;
    if (!ecm_edwards_write_midstage(path, cm, Qx, Qz)) {
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
        ecm_ts_fprintf(stderr, "ERROR: cannot write stage-1 save %s\n", path.c_str());
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
        ecm_ts_fprintf(stdout, "stage1 save -> %s : %s\n", path.c_str(),
                       edwards_ecm2_line(opt, sigma, B1, B2).c_str());
    }
    return true;
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
};

static int edwards_checkpoint_progress(void *ctx, const edwards_checkpoint_t *cur) {
    EdwardsCheckpointCtx *c = (EdwardsCheckpointCtx *)ctx;
    const long long now = edwards_now_ms();
    const bool due = (now - c->last_ms >= c->interval_ms);
    if (!due && !g_edwards_stop) return 1;   // 继续

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
    if (local_save_overwrite_ok(c->save_path, cm)) {
        ecm_edwards_write_stage1(c->save_path, cm, 2, expbuf, cur->bitnum, dict_size,
                                 Px, Py, cur->Rx, cur->Ry, cur->Rz);
    }
    c->last_ms = now;
    mpz_clears(d, Px, Py, NULL);

    if (g_edwards_stop) {
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
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

/* 定义在后面 (需要 g_stage1_bar / edwards_progress_set), 这里先声明给调用点用 */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur);

static int edwards_run_one_curve(mpz_srcptr N, mpz_srcptr s,
                                 const Stage1RunOptions &opt, double B1, double B2,
                                 uint32_t idx, uint64_t sigma, uint32_t curves,
                                 mpz_t factor_out, const std::string &ckpt_path) {
    mpz_t Qx, Qz;
    mpz_inits(Qx, Qz, NULL);

    int rc;
    if (!ckpt_path.empty()) {
        EdwardsCheckpointCtx ck;
        ck.opt = &opt;
        ck.sigma = sigma;
        ck.B1 = (uint64_t)B1;
        ck.B2 = (uint64_t)B2;
        ck.save_path = ckpt_path;
        ck.interval_ms = (long long)opt.gpuckpt_ms;
        ck.last_ms = edwards_now_ms();
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
                    std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                    ecm_ts_fprintf(stdout, "curve %u: resume STAGE1 checkpoint @ bitnum=%u\n",
                                   idx + 1, rbn);
                }
            }
            mpz_clears(rdx, rdy, rex, rey, rez, NULL);
        }

        const uint32_t chunk_bits = 16384;
        rc = edwards_stage1_curve_progress(factor_out, Qx, Qz, N, sigma, s, chunk_bits,
                                           resume.bitnum > 0 ? &resume : nullptr,
                                           edwards_checkpoint_progress_bar, &ck);
        edwards_checkpoint_clear(&resume);

        if (rc == 2) {
            mpz_clears(Qx, Qz, NULL);
            return 2;
        }
    } else {
        rc = edwards_stage1_curve(factor_out, Qx, Qz, N, sigma, s);
    }

    // 落盘: 把 stage-1 结果写成本地 MIDSTAGE 存档 (总是写, 即使命中因子).
    // 送进 Prime95 由独立程序 ecm_p95feeder 负责。
    if (rc >= 0) {
        edwards_write_local_midstage(opt, sigma, idx + 1, (uint64_t)B1, (uint64_t)B2,
                                     Qx, Qz, idx);
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
    std::atomic<uint32_t> *next;
    std::atomic<int> *aborted;
    /* SIMD 批量 (8 曲线/批) 相关 */
    int simd_ok = 0;
    std::atomic<uint32_t> *done_ctr = nullptr;
};

// ---------------------------------------------------------------------------
// SIMD 批量 stage-1 (8 曲线/批, AVX512-IFMA)
//
// 每批建一个 ifma/ed_soa 上下文 (字典本来就要按 sigma 重建, 见 §13.8 的 w=8 选型),
// 跑完整阶梯后逐 lane 落盘成与标量路径**完全相同**的本地 MIDSTAGE 存档, 因此
// ecm_p95feeder 不需要任何改动。
//
// 与标量路径的已知差别 (按 §13 第 6 条"内部实现可不同但必须记录"):
//   * 没有阶梯中途 checkpoint, 只在批边界响应 SIGINT/写存档 (标量路径每 gpuckpt_ms 写一次);
//   * 因此也没有 resume: 重跑该批从 s 的第一个 digit 开始。
// ---------------------------------------------------------------------------
#include "simd_edwards.h"   /* SoA 批量 Edwards 层 (AVX512-IFMA TU, 见 §13.8) */

/* ---------------------------------------------------------------------------
 * stage-1 进度条 (照 opencl_ecm_stage1.cpp 的 CUDA host 写法)
 *
 * 批量模式下一批 8 条曲线是不可分割的工作单元, 所以进度以"曲线"为粒度、每批跳一格;
 * 更新点都在 g_edwards_out_mutex 保护下, 避免多条 worker 线程同时动光标。
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

static void edwards_progress_set(double done, uint32_t total, double /*unused*/) {
    if (total == 0) return;
    /* 进度/速率估计必须单调: 多线程下各线程上报的 (已完成曲线 + 本曲线比例) 会互相穿插
       (A 报 3.5, B 接着报 5.2, A 下一次又报 3.6), 不单调会产生"大增量/小时间"的虚高
       速度样本, 把显示速率放大十几倍。取运行最大值即可。 */
    static double s_max_done = 0.0;
    if (done <= 0.0) { g_speed.reset(); s_max_done = 0.0; }
    if (done < s_max_done) done = s_max_done; else s_max_done = done;
    const double now = (double)edwards_now_ms() / 1000.0;
    g_speed.sample(now, done);

    double pct = 100.0 * done / (double)total;
    if (pct > 100.0) pct = 100.0;
    const double avg = g_speed.avg_speed();
    const double per_curve_s = avg > 0.0 ? 1.0 / avg : 0.0;              /* 由均值反推 */
    const double elapsed_s = (g_speed.t0 > 0.0) ? now - g_speed.t0 : 0.0;
    double remaining_s = 0.0;
    if (avg > 0.0 && done < (double)total) remaining_s = ((double)total - done) / avg;

    if (!stdout_is_tty_local()) {
        /* 日志模式: 不打 \r, 按衰减节奏打整行 (带时间戳, 会被 mirror 到 log 文件) */
        static unsigned lines = 0;
        static std::atomic<uint32_t> last_emitted(0xFFFFFFFFu);
        const uint32_t done_i = (uint32_t)done;
        if (done >= (double)total || emit_progress_line(++lines)) {
            last_emitted.store(done_i, std::memory_order_relaxed);
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

/* 批内进度: bits 粒度, 把"已完成曲线 + 本批已跑比例 × 批大小"折算成总进度。
   同时承担 checkpoint: 每 gpuckpt_ms(或收到 SIGINT) 就按 lane 把当前点落盘, 存档格式
   与标量路径**完全相同**, 所以后续用标量或 SIMD 续跑都能读。 */
struct EdSimdProg {
    std::atomic<uint32_t> *done_ctr;
    uint32_t curves, batch_first, batch_count;
    size_t   s_bits;
    long long t0_ms;
    const Stage1RunOptions *opt;
    double   B1, B2;
    const uint64_t *sigmas;
    long long last_ms;
    long long interval_ms;
};

static const ifma_ctx_t *g_simd_prog_mc = nullptr;   /* 当前批次的 modulus 上下文 */
static int edwards_simd_progress(void *p, size_t bits_done, size_t bits_total,
                                const uint64_t *Rx, const uint64_t *Ry, const uint64_t *Rz) {
    EdSimdProg *pr = (EdSimdProg *)p;
    const ifma_ctx_t *mc = g_simd_prog_mc;

    /* --- checkpoint (与标量路径同一套写入器/路径/字段) --- */
    const long long now = edwards_now_ms();
    const bool due = (now - pr->last_ms >= pr->interval_ms) || g_edwards_stop;
    if (due && bits_done > 0) {
        if (mc) {
            mpz_t d, Px, Py, rx, ry, rz;
            mpz_inits(d, Px, Py, rx, ry, rz, NULL);
            size_t written = 0;
            for (uint32_t k = 0; k < pr->batch_count; k++) {
                const uint32_t i = pr->batch_first + k;
                const std::string path = edwards_local_stem(*pr->opt, i, (uint64_t)pr->B1) + ".ckpt";
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
                if (local_save_overwrite_ok(path, cm)) {
                    ecm_edwards_write_stage1(path, cm, 2, expbuf, (uint32_t)bits_done,
                                             dict_size, Px, Py, rx, ry, rz);
                    written++;
                }
            }
            pr->last_ms = now;
            mpz_clears(d, Px, Py, rx, ry, rz, NULL);
            if (written && (g_edwards_stop || getenv("ED_SOA_DEBUG"))) {
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                ecm_ts_fprintf(stdout, "checkpoint written (simd batch %u): %zu lane(s) @ bitnum=%zu\n",
                               pr->batch_first, written, bits_done);
            }
        }
    }

    /* --- 进度条 --- */
    if (g_stage1_bar) {
        const uint32_t base = pr->done_ctr ? pr->done_ctr->load(std::memory_order_relaxed) : 0;
        const double frac = (bits_total > 0) ? (double)bits_done / (double)bits_total : 0.0;
        const double done_f = (double)base + frac * (double)pr->batch_count;
        const double lapsed = (double)(edwards_now_ms() - pr->t0_ms);
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
        edwards_progress_set(done_f, pr->curves, done_f > 0.0 ? lapsed / done_f : 0.0);
    }

    return g_edwards_stop ? 1 : 0;                 /* SIGINT: 批内也能停 */
}

/* 标量路径的"曲线内"进度: 包一层 checkpoint 回调, 每 chunk_bits(16384) 位跳一格。
   没有它的话 B1 很大时一条曲线要跑几分钟, 进度条看着就是卡死的。
   (定义在这里而不是函数旁边, 因为要用到上面的 g_stage1_bar/edwards_progress_set。) */
static int edwards_checkpoint_progress_bar(void *ctx, const edwards_checkpoint_t *cur) {
    if (g_stage1_bar && ctx) {
        EdwardsCheckpointCtx *ck = (EdwardsCheckpointCtx *)ctx;
        if (ck->opt && ck->s && ck->opt->stage1_curves > 0) {
            const size_t s_bits = (size_t)mpz_sizeinbase(ck->s, 2);
            if (s_bits > 0) {
                const uint32_t base = g_stage1_done.load(std::memory_order_relaxed);
                double frac = (double)cur->bitnum / (double)s_bits;
                if (frac > 1.0) frac = 1.0;
                const double done_f = (double)base + frac;
                const double lapsed = (double)(edwards_now_ms() - ck->opt->stage1_t0_ms);
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                edwards_progress_set(done_f, ck->opt->stage1_curves,
                                     done_f > 0.0 ? lapsed / done_f : 0.0);
            }
        }
    }
    return edwards_checkpoint_progress(ctx, cur);
}

static int edwards_run_batch_simd_impl(const Stage1RunOptions &opt, mpz_srcptr N, mpz_srcptr s,
                                       double B1, double B2, uint32_t first, uint32_t count,
                                       const uint64_t *sigmas, mpz_t *factors, int *array_found,
                                       std::atomic<uint32_t> *done_ctr) {
    ed_soa_ctx_t ctx;
    /* 字典窗口跟随配置 (edwards_naf_w / ini), 不再硬编码 w=8:
       这样 SIMD 与标量用同一套 digit/字典, 存档逐字节可比, 也便于混合与续跑。
       代价是字典内存 = 3*2^(w-2)*8n 字节 (w=12/n=155 约 29 MB/批), 见启动信息。 */
    int w = edwards_get_naf_w();
    if (w < 3 || w > 12) w = 8;
    if (ed_soa_init(&ctx, N, w) != 0) return -1;

    /* 不足 8 条时把最后一个 sigma 补满 (被补的 lane 结果丢弃) */
    uint64_t sg[8];
    for (uint32_t k = 0; k < 8; k++) sg[k] = sigmas[first + (k < count ? k : count - 1)];

    if (ed_soa_set_curves(&ctx, sg, 8) != 0) { ed_soa_clear(&ctx); return -1; }

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
    pr.last_ms = edwards_now_ms();
    pr.interval_ms = (long long)opt.gpuckpt_ms;
    g_simd_prog_mc = &ctx.mc;
    /* ---- resume: 一批 8 条 lane 必须共用恢复点, 所以仅当**所有** lane 都有有效检查点
       且 bitnum 完全相同时才续跑 (这正是"整批一起被中止、一起落盘"的常态); 否则从头跑。 ---- */
    {
        std::vector<mpz_t> rxs(count), rys(count), rzs(count);
        for (uint32_t k = 0; k < count; k++) mpz_inits(rxs[k], rys[k], rzs[k], NULL);
        uint32_t common_bitnum = 0;
        int all_ok = (count > 0);
        for (uint32_t k = 0; k < count && all_ok; k++) {
            const uint32_t i = first + k;
            const std::string path = edwards_local_stem(opt, i, (uint64_t)B1) + ".ckpt";
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
            std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
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
        return 2;
    }

    int rc = 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = first + k;
        if (mpz_cmp_ui(fac[k], 1) > 0) {
            mpz_set(factors[i], fac[k]);
            array_found[i] = ECM_FACTOR_FOUND_STEP1;
            std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
            ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                           i, (unsigned long long)sigmas[i]);
        }
        edwards_write_local_midstage(opt, sigmas[i], i + 1, (uint64_t)B1, (uint64_t)B2,
                                     Qx[k], Qz[k], i);
    }
    for (int k = 0; k < 8; k++) mpz_clears(Qx[k], Qz[k], fac[k], NULL);
    ed_soa_clear(&ctx);

    if (done_ctr) {
        const uint32_t done = done_ctr->fetch_add(count, std::memory_order_relaxed) + count;
        const double lapsed = (double)(edwards_now_ms() - opt.stage1_t0_ms);
        std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
        edwards_progress_set(done, opt.stage1_curves, done > 0 ? lapsed / done : 0.0);
    }
    (void)s;
    return rc;
}

static void edwards_worker(EdwardsWorkerCtx *ctx) {
    const int use_simd = (ctx->opt->backend != 2) && ctx->simd_ok;
    for (;;) {
        if (g_edwards_stop) { ctx->aborted->store(1, std::memory_order_relaxed); break; }
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
                                                       ctx->array_found, ctx->done_ctr);
            if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
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
                                                 ctx->factors[i], ctx->ckpt_paths[i]);
            if (rc > 0) {
                ctx->array_found[i] = ECM_FACTOR_FOUND_STEP1;
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                if (ctx->opt->verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigma
                              << " -> factor found" << std::endl;
                } else {
                    ecm_ts_fprintf(stdout, "  curve %u sigma=%llu -> factor found\n",
                                   i, (unsigned long long)sigma);
                }
            } else if (rc < 0) {
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                std::cerr << "  curve " << i << " sigma=" << sigma
                          << " -> Edwards stage-1 internal error" << std::endl;
            } else if (rc == 2) {
                ctx->aborted->store(1, std::memory_order_relaxed);
                break;
            }
            if (ctx->done_ctr) {
                const uint32_t done = ctx->done_ctr->fetch_add(1, std::memory_order_relaxed) + 1;
                const double lapsed = (double)(edwards_now_ms() - ctx->opt->stage1_t0_ms);
                std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                edwards_progress_set(done, ctx->opt->stage1_curves, done > 0 ? lapsed / done : 0.0);
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
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            try {
                const long v = std::stol(tok);
                if (v < 0 || v > 1023) {
                    std::cerr << "Affinity: CPU index out of range: " << tok << std::endl;
                } else {
                    out.push_back((unsigned)v);
                }
            } catch (...) {
                std::cerr << "Affinity: ignoring non-numeric entry: " << tok << std::endl;
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
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

// curves == 1 or opt.edwards_threads == 1 -> 顺序执行 (与历史行为一致);
// 否则按曲线并行 (每线程独立曲线, 动态取号), 吞吐随核数线性提升.
static int run_edwards_stage1(const mpz_t N, double B1, double B2, uint32_t curves,
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

    if (curves == 0) {
        std::cerr << "curves must be > 0" << std::endl;
        return ECM_ERROR;
    }

    // NAF 窗口 (字典大小 2^(w-2)); 0 = 保持 ecm_edwards_cpu 的默认值.
    if (opt.edwards_naf_w >= 2) {
        edwards_set_naf_w(opt.edwards_naf_w);
    } else if (opt.edwards_naf_w != 0) {
        std::cerr << "edwards_naf_w must be 0 (default) or >= 2" << std::endl;
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
    for (uint32_t i = 0; i < curves; i++) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    const bool use_ckpt = !opt.tmp_dir.empty();
    if (use_ckpt) {
        g_edwards_stop = 0;
        signal(SIGINT, edwards_sigint_handler);
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
    std::vector<uint64_t> sigmas(curves);
    for (uint32_t i = 0; i < curves; i++) {
        // Fixed sigma → batch start + i (matches -sigma/-gpucurves semantics);
        // otherwise a fresh Prime95-style random sigma per curve -- 但若该曲线已有合格的
        // .ckpt 就**采用存档里的 sigma**（存档覆盖参数）。否则每轮都随机出新的 sigma,
        // 与存档永久不匹配, 自动续跑永远不会发生（这正是"不会自动从 saves/ 续跑"的根因）。
        if (opt.sigma_fixed) {
            sigmas[i] = opt.fixed_sigma64 + i;
        } else {
            sigmas[i] = random_sigma_u64();
            if (!opt.tmp_dir.empty()) {
                const std::string cp = edwards_local_stem(opt, i, (uint64_t)B1) + ".ckpt";
                ecm_save_common rcm;
                uint64_t rsp; uint32_t rebs, rbn, rds;
                mpz_t dx, dy, rx, ry, rz;
                mpz_inits(dx, dy, rx, ry, rz, NULL);
                if (ecm_edwards_read_stage1(cp, rcm, &rsp, &rebs, &rbn, &rds,
                                           dx, dy, rx, ry, rz) &&
                    rbn > 0 && rcm.sigma != 0 && rcm.B1 == (uint64_t)B1) {
                    sigmas[i] = (uint64_t)rcm.sigma;
                    std::lock_guard<std::mutex> lk(g_edwards_out_mutex);
                    ecm_ts_fprintf(stdout,
                                   "  curve %u: found .ckpt -> adopting sigma=%llu (bitnum=%u)\n",
                                   i + 1, (unsigned long long)rcm.sigma, rbn);
                }
                mpz_clears(dx, dy, rx, ry, rz, NULL);
            }
        }
    }
    const uint64_t firstsigma = sigmas[0];

    // checkpoint 路径: 统一每条曲线独立 <tmp_dir>/e{n:07d}_c{6-digit},
    // 与同名的 .tmp (MIDSTAGE) 区分开, 且并行写不冲突.
    std::vector<std::string> ckpt_paths(curves);
    if (use_ckpt) {
        for (uint32_t i = 0; i < curves; i++) {
            ckpt_paths[i] = edwards_local_stem(opt, i, (uint64_t)B1) + ".ckpt";
        }
    }

    uint32_t nthreads = opt.edwards_threads;
    if (nthreads == 0) nthreads = edwards_default_threads(curves);
    if (nthreads > curves) nthreads = curves;
    if (nthreads < 1) nthreads = 1;

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " Edwards curves, CPU";
    if (nthreads > 1) std::cout << ", " << nthreads << " threads";
    std::cout << ")";
    if (use_ckpt) std::cout << " [saves -> " << opt.tmp_dir << "]";
    std::cout << std::endl;

    bool aborted = false;
    if (nthreads <= 1 && curves == 1) {
        for (uint32_t i = 0; i < curves; i++) {
            const int rc = edwards_run_one_curve(N, s, opt, B1, B2, i, sigmas[i], curves,
                                                 factors[i], ckpt_paths[i]);
            if (rc > 0) {
                array_found[i] = ECM_FACTOR_FOUND_STEP1;
                if (opt.verbose) {
                    std::cout << "  curve " << i << " sigma=" << sigmas[i]
                              << " -> factor found" << std::endl;
                }
            } else if (rc < 0) {
                std::cerr << "  curve " << i << " sigma=" << sigmas[i]
                          << " -> Edwards stage-1 internal error" << std::endl;
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
            if (opt.backend == 1 && !isa) {
                ecm_ts_fprintf(stderr,
                               "ERROR: --edwards-backend simd requested but this CPU lacks "
                               "AVX512-F/DQ/IFMA. Refusing to fall back silently; use "
                               "--edwards-backend auto or gmp.\n");
                return -1;
            }
            wctx.simd_ok = (opt.backend == 1) ? 1
                         : (opt.backend == 0) ? (isa && curves >= 8) : 0;
            wctx.done_ctr = &done_ctr;
            char sbuf[160];
            snprintf(sbuf, sizeof(sbuf),
                     "simd (AVX512-IFMA: 8 curves per batch, 1 thread/batch, dict w=%d)",
                     edwards_get_naf_w() >= 3 ? edwards_get_naf_w() : 8);
            ecm_ts_fprintf(stdout, "stage-1 backend : %s\n",
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
        indicators::ProgressBar bar{
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
        g_stage1_bar = &bar;
        ((Stage1RunOptions &)opt).stage1_t0_ms = edwards_now_ms();
        ((Stage1RunOptions &)opt).stage1_curves = curves;
        edwards_progress_set(0, curves, 0.0);
        wctx.B1 = B1;
        wctx.B2 = B2;
        wctx.curves = curves;
        wctx.sigmas = sigmas.data();
        wctx.ckpt_paths = ckpt_paths.data();
        wctx.factors = factors;
        wctx.array_found = array_found;
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

    if (aborted) {
        ecm_ts_fprintf(stdout, "stage-1 aborted (checkpoint saved)\n");
        mpz_clear(s);
        for (uint32_t j = 0; j < curves; j++) mpz_clear(factors[j]);
        free(factors);
        free(array_found);
        out->ret = ECM_ERROR;
        return ECM_ERROR;
    }

    // Binary Prime95 save format is a later task; the OpenCL text save is not
    // valid for the Edwards parametrization, so skip it for now.
    (void)savefilename; (void)saveappend; (void)n_expr;

    mpz_clear(s);

    out->ret = ECM_NO_FACTOR_FOUND;
    out->firstsigma = (uint32_t)(firstsigma & 0xFFFFFFFFu);
    out->firstsigma64 = firstsigma;
    out->factors = factors;
    out->array_found = array_found;
    return out->ret;
}

// Run one stage-1 batch. N and the caller's n_expr are borrowed (not cleared);
// factors / array_found are allocated here and returned via `out` (caller frees).
static int run_stage1_once(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           const Stage1RunOptions &opt, Stage1RunResult *out) {
    if (opt.use_edwards) {
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
    params->gpu = opt.use_gpu ? 1 : 0;
    params->gpu_number_of_curves = curves;
    params->gpu_checkpoint_interval_ms = opt.gpuckpt_ms;
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
    if (!compute_batch_s(batch_s, B1)) {
        std::cerr << "Failed to compute batch_s" << std::endl;
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = B1;

    if (opt.use_gpu) {
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

    uint32_t firstsigma =
        opt.sigma_fixed ? opt.fixed_sigma : gpu_pick_random_sigma(curves);
    if ((uint64_t)firstsigma + curves > 0x100000000ull) {
        std::cerr << "sigma range overflows uint32 (sigma + curves > 2^32)" << std::endl;
        for (uint32_t i = 0; i < curves; i++) mpz_clear(factors[i]);
        free(factors);
        free(array_found);
        mpz_clear(batch_s);
        ecm_clear(params);
        return ECM_ERROR;
    }
    mpz_t batch_d;
    mpz_init(batch_d);
    gpu_compute_batch_d(batch_d, firstsigma, N);

    // B1/B2 are fixed for this run. The *actual* sigma is printed by the backend
    // after it applies any checkpoint resume (the checkpoint may override the
    // freshly-computed sigma), so it is not printed here.
    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << " (" << curves << " curves)" << std::endl;

    float gputime = 0.0f;
    const int ret = ecm_backend_stage1(factors, array_found, N, params->batch_s, curves,
                                       (uint32_t *)&firstsigma, params->gpu_checkpoint_interval_ms,
                                       &gputime, params->verbose,
                                       params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
                                       params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
                                       params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
                                       params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
                                       params->gpu_special_mult_path[0] ? params->gpu_special_mult_path
                                                                       : nullptr);

    std::cout << "GPU stage1 returned: " << ret << " gputime=" << gputime << " ms\n";

    if (ret != ECM_ERROR && !resolved_save.empty()) {
        const std::string n_expr_save =
            opencl_ecm_build_saved_n_expr(n_expr, N, curves, factors, array_found);
        if (!opencl_ecm_append_save_lines(resolved_save, N, B1, firstsigma, curves, factors,
                                          n_expr_save)) {
            std::cerr << "Failed to append OpenCL save lines into " << resolved_save << std::endl;
        }
    }

    mpz_clear(batch_d);
    mpz_clear(batch_s);
    ecm_clear(params);

    out->ret = ret;
    out->firstsigma = firstsigma;
    out->factors = factors;
    out->array_found = array_found;
    return ret;
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

// Run one already-parsed queue task (ECM2= or ECMSTAGE2=), report factors, and
// advance the worktodo file. Returns true to keep processing, false to abort
// the queue (backend prepare failed).
static bool queue_run_one(const mpz_t N, double B1, double B2, uint32_t curves,
                          uint64_t sigma, bool sigma_fixed,
                          const std::string &save_name, const std::string &n_expr,
                          const std::string &aid, const std::string &line,
                          const std::string &worktodo_path, const std::string &finished_path,
                          const Stage1RunOptions &base_opt,
                          const std::string &exe_dir, const std::string &sync1,
                          const std::string &sync2, bool full_sync, long long marker,
                          int *processed) {
    Stage1RunOptions opt = base_opt;
    if (sigma_fixed) {
        opt.sigma_fixed = true;
        opt.fixed_sigma64 = sigma;
        opt.fixed_sigma = (uint32_t)(sigma & 0xFFFFFFFFu);
    }

    Stage1RunResult result;
    run_stage1_once(N, B1, B2, curves, save_name, /*saveappend=*/true, n_expr, opt, &result);

    const bool has_factors = (result.factors != nullptr);

    if (result.prepare_failed) {
        if (has_factors) {
            for (uint32_t i = 0; i < result.curves; ++i) mpz_clear(result.factors[i]);
            free(result.factors);
            free(result.array_found);
        }
        ecm_ts_fprintf(stderr, "FATAL: backend prepare failed; aborting queue.\n");
        return false;
    }

    bool found_factor = false;
    if (has_factors) {
        for (uint32_t i = 0; i < result.curves; ++i) {
            if (result.array_found[i] != ECM_NO_FACTOR_FOUND) {
                char *fs = mpz_get_str(nullptr, 10, result.factors[i]);
                std::cout << "factor[" << i << "]=" << (fs ? fs : "?") << "\n";
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
    }

    if (result.ret == ECM_ERROR) {
        ecm_ts_fprintf(stderr, "ERROR: stage1 failed for task: %s\n", line.c_str());
        ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
        return true;
    }

    if (!ecm_append_text_line(finished_path, line)) {
        ecm_ts_fprintf(stderr, "ERROR: cannot append to %s\n", finished_path.c_str());
        ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
        return true;
    }
    ecm_worktodo_advance(worktodo_path, line, WorktodoAction::Remove);

    ecm_sync_save_files(exe_dir, sync1, sync2, full_sync, marker);
    if (processed) {
        (*processed)++;
    }
    return true;
}

static int run_queue_manager(const std::string &ini_path) {
    const std::string raw_exe_dir = get_exe_dir_local();
    const std::string exe_dir = raw_exe_dir.empty() ? "." : raw_exe_dir;

    // Resolve ini path (default: exe_dir/ecm.ini).
    std::string ini = ini_path;
    if (ini.empty()) {
        ini = exe_dir + "/ecm.ini";
    } else {
        ini = resolve_rel_local(exe_dir, ini);
    }

    // Load config; on first run (missing ini), write a default template.
    EcmQueueConfig cfg;
    if (!ecm_queue_config_load(ini, cfg)) {
        if (!ecm_queue_config_write_default(ini)) {
            ecm_ts_fprintf(stderr, "FATAL: cannot create default config %s\n", ini.c_str());
            return 1;
        }
        ecm_ts_fprintf(stdout, "Created default config: %s\n", ini.c_str());
    }

    opencl_ecm_set_work_dir(exe_dir.c_str());

    ecm_log_set_progress_color(cfg.progress_color.c_str());

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

    Stage1RunOptions opt;
    opt.use_gpu = true;
    opt.use_edwards = (cfg.edwards != 0);
    opt.edwards_threads = (cfg.edwards_threads > 0) ? (uint32_t)cfg.edwards_threads : 0u;
    opt.affinity_cpus = parse_affinity_spec(cfg.affinity);
    /* edwards_backend: auto | simd | gmp  (队列模式下此前只能走 auto) */
    {
        std::string b = cfg.edwards_backend;
        for (size_t i = 0; i < b.size(); i++) {
            const char ch = b[i];
            b[i] = (ch >= 'A' && ch <= 'Z') ? (char)(ch - 'A' + 'a') : ch;
        }
        if (b.empty() || b == "auto") opt.backend = 0;
        else if (b == "simd" || b == "avx512") opt.backend = 1;
        else if (b == "gmp" || b == "scalar" || b == "mpn") opt.backend = 2;
        else {
            ecm_ts_fprintf(stderr,
                           "WARNING: edwards_backend='%s' not recognised (auto|simd|gmp); using auto\n",
                           cfg.edwards_backend.c_str());
            opt.backend = 0;
        }
    }
    /* edwards_naf_w: 之前同样只在命令行生效, 队列模式会忽略 ini 里的设置 */
    if (cfg.edwards_naf_w >= 3 && cfg.edwards_naf_w <= 12) edwards_set_naf_w(cfg.edwards_naf_w);
    opt.edwards_naf_w = cfg.edwards_naf_w;
    opt.verbose = cfg.verbose;
    opt.device_index = cfg.device;
    opt.gpuckpt_ms = (cfg.gpuckpt_seconds > 0.0)
                         ? (unsigned long)(cfg.gpuckpt_seconds * 1000.0)
                         : 0UL;
    opt.gpu_mul_path = cfg.kernel_mul;
    opt.gpu_sqr_path = cfg.kernel_sqr;
    opt.gpu_add_path = cfg.kernel_add;
    opt.gpu_sub_path = cfg.kernel_sub;
    opt.gpu_special_mult_path = cfg.kernel_special_mult;
    opt.sigma_fixed = (cfg.sigma != 0);
    opt.fixed_sigma = cfg.sigma;
    /* Edwards 路径用的是 64 位 sigma 与 sigma_fixed 开关; 队列模式此前只设了 32 位字段,
       导致 ini 里的 sigma 被忽略、每轮都跑随机曲线。 */
    opt.fixed_sigma64 = (uint64_t)cfg.sigma;
    opt.sigma_fixed = (cfg.sigma != 0);
    opt.tmp_dir = resolve_rel_local(exe_dir, cfg.tmp_dir);
    if (!cfg.p95_dir.empty()) {
        ecm_ts_fprintf(stdout,
            "note: p95_dir is now handled by the separate ecm_p95feeder program; "
            "ecm.exe only writes local saves to %s\n", opt.tmp_dir.c_str());
    }

    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worktodo : %s\n", worktodo_path.c_str());
    ecm_ts_fprintf(stdout, "finished : %s\n", finished_path.c_str());
    ecm_ts_fprintf(stdout, "log_file : %s\n", cfg.log_file.c_str());
    ecm_ts_fprintf(stdout, "saves : %s\n", opt.tmp_dir.c_str());
    ecm_ts_fprintf(stdout, "backend : %s\n", opt.use_edwards ? "edwards Z2xZ8" : "gpu");

    // Startup full sync (matches the old work_manager.ps1 behaviour).
    ecm_sync_save_files(exe_dir, sync1, sync2, /*full=*/true, 0);

    int processed = 0;
    const bool full_sync = (cfg.sync_mode == "full");
    while (true) {
        std::string line;
        if (!ecm_worktodo_first_line(worktodo_path, line)) {
            break;
        }

        ecm_ts_fprintf(stdout, "START: %s\n", line.c_str());

        const long long marker = current_epoch_seconds();
        std::string err;

        if (line.compare(0, 5, "ECM2=") == 0 || line.compare(0, 4, "ECM=") == 0) {
            // Prime95 ECM= / ECM2= (等价) 格式: k,b,n,c,B1[,B2][,curves][,sigma][,"factors"].
            Ecm2Task task;
            if (!ecm_parse_ecm2_line(line, task, err)) {
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
                continue;
            }
            mpz_t N;
            mpz_init(N);
            if (!ecm_compute_ecm2_n(task, N, err)) {
                mpz_clear(N);
                ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
                ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
                continue;
            }
            const std::string n_expr = build_n_expr(task.k, task.b, task.n, task.c, task.factors);
            // 交接字段 (从 ECM= 任务读取)
            {
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
            const bool cont = queue_run_one(
                N, task.B1, task.B2, task.curves_to_run,
                task.has_sigma ? task.sigma : 0, task.has_sigma,
                /*save_name=*/"", n_expr, task.aid, line,
                worktodo_path, finished_path, opt,
                exe_dir, sync1, sync2, full_sync, marker, &processed);
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
            ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
            continue;
        }

        double B1 = 0.0;
        if (!ecm_extract_b1_from_save_name(task.save_name, &B1, err)) {
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
            continue;
        }

        mpz_t N;
        mpz_init(N);
        if (!ecm_compute_stage2_n(task, N, err)) {
            mpz_clear(N);
            ecm_ts_fprintf(stderr, "ERROR: %s (line: %s)\n", err.c_str(), line.c_str());
            ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
            continue;
        }

        const std::string n_expr = build_n_expr(task.k, task.b, task.n, task.c, task.factors);
        // 交接字段 (从 ECMSTAGE2 任务读取)
        {
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
        // ECMSTAGE2 lines carry no per-line sigma: use cfg.sigma (already in opt).
        const bool cont = queue_run_one(
            N, B1, /*B2=*/0.0, task.curves_to_run,
            /*sigma=*/0, /*sigma_fixed=*/false,
            task.save_name, n_expr, task.aid, line,
            worktodo_path, finished_path, opt,
            exe_dir, sync1, sync2, full_sync, marker, &processed);
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
    bool use_gpu = false;
    bool use_edwards = false;
    uint32_t edwards_threads = 0;
    int edwards_naf_w = 0;
    int edwards_backend = 0;      /* 0=auto 1=simd 2=gmp */
    uint32_t gpucurves = 0;
    double gpuckpt_seconds = -1.0;
    bool gpuckpt_set = false;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;
    uint64_t fixed_sigma64 = 0;
    int gpu_device_index = 0;
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
    std::string ini_path;
    std::string tmp_dir;                  // 本地 stage-1 落盘目录
    std::string p95_dir_ignored;          // 兼容旧脚本: 现在由 ecm_p95feeder 处理
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "-v") { verbose = true; continue; }
        if(a == "-gpu") { use_gpu = true; continue; }
        if(a == "--edwards") { use_edwards = true; continue; }
        if((a == "--edwards-backend" || a == "--edbackend") && i+1<argc){
            const std::string m = argv[++i];
            if (m == "auto") edwards_backend = 0;
            else if (m == "simd" || m == "avx512") edwards_backend = 1;
            else if (m == "gmp" || m == "scalar") edwards_backend = 2;
            else { std::cerr << "Invalid --edwards-backend, expected auto|simd|gmp" << std::endl; return 1; }
            continue;
        }
        if((a == "--edwards-threads" || a == "--edthreads") && i+1<argc){
            try { edwards_threads = (uint32_t)std::stoul(argv[++i]); }
            catch (...) { std::cerr << "Invalid --edwards-threads value, expected >= 0" << std::endl; return 1; }
            continue;
        }
        if((a == "--edwards-naf-w" || a == "--ednafw") && i+1<argc){
            try { edwards_naf_w = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Invalid --edwards-naf-w value, expected integer" << std::endl; return 1; }            continue;
        }
        if(a == "-gpucurves" && i+1<argc){ gpucurves = (uint32_t)std::stoul(argv[++i]); continue; }
        if(a == "-gpuckpt" && i+1<argc){
            try {
                gpuckpt_seconds = std::stod(argv[++i]);
                gpuckpt_set = true;
            } catch (...) {
                std::cerr << "Invalid -gpuckpt value, expected number of seconds" << std::endl;
                return 1;
            }
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
            continue;
        }
        if((a == "-sigma" || a == "--sigma") && i+1<argc){
            if(!parse_sigma64_arg(argv[++i], &fixed_sigma64)){
                std::cerr << "Invalid -sigma value (need 1..2^64-1, optional param: prefix)" << std::endl;
                return 1;
            }
            sigma_fixed = true;
            if (fixed_sigma64 <= 0xFFFFFFFFull) {
                fixed_sigma = (uint32_t)fixed_sigma64;
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
        if(a == "-ini" && i+1<argc) {
            ini_path = argv[++i];
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

    if (pos.empty()) {
        // No positional B1/B2 → queue-manager mode (reads ecm.ini + worktodo).
        return run_queue_manager(ini_path);
    }

    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    if (gpuckpt_set) {
        const double val_ms = gpuckpt_seconds * 1000.0;
        if (val_ms != val_ms) {
            std::cerr << "Error, invalid -gpuckpt value (NaN)" << std::endl;
            return 1;
        }
        if (val_ms <= 0.0) {
            gpuckpt_ms = 0;
        } else if (val_ms >= (double)ULONG_MAX) {
            gpuckpt_ms = ULONG_MAX;
        } else {
            gpuckpt_ms = (unsigned long)std::llround(val_ms);
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
    std::cout << "  mode: " << (use_edwards ? "edwards" : (use_gpu ? "gpu" : "cpu-stub"))
              << ", gpucurves=" << gpucurves
              << ", gpuckpt=" << (gpuckpt_ms == 0 ? 0.0 : gpuckpt_ms / 1000.0) << "s"
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
    ExprParser parser(nline);
    if(!parser.parse(N)){
        std::cerr << "Failed to parse N: '"<< nline <<"'" << std::endl;
        std::cerr << "Parse error: " << parser.message() << std::endl;
        return 1;
    }

    // std::cout << "Parsed N bit-size: " << mpz_sizeinbase(N, 2) << std::endl;
    // if (verbose) {
    //     std::cout << "Parsed N = ";
    //     mpz_out_str(stdout, 10, N);
    //     std::cout << std::endl;
    // }

    // Execute stage 1 through the shared single-run path.
    if (!use_edwards && sigma_fixed && fixed_sigma64 > 0xFFFFFFFFull) {
        std::cerr << "Error: -sigma value exceeds 2^32-1; the GPU path needs a "
                  << "32-bit sigma (use --edwards for 64-bit Edwards sigma)" << std::endl;
        mpz_clear(N);
        return 1;
    }

    Stage1RunOptions opt;
    opt.use_gpu = use_gpu;
    opt.use_edwards = use_edwards;
    opt.edwards_threads = edwards_threads;
    opt.backend = edwards_backend;
    opt.edwards_naf_w = edwards_naf_w;
    opt.verbose = verbose ? 1 : 0;
    opt.device_index = gpu_device_index;
    opt.gpuckpt_ms = gpuckpt_ms;
    opt.gpu_mul_path = gpu_mul_path;
    opt.gpu_sqr_path = gpu_sqr_path;
    opt.gpu_add_path = gpu_add_path;
    opt.gpu_sub_path = gpu_sub_path;
    opt.gpu_special_mult_path = gpu_special_mult_path;
    opt.sigma_fixed = sigma_fixed;
    opt.fixed_sigma = fixed_sigma;
    opt.fixed_sigma64 = fixed_sigma64;
    if (tmp_dir.empty()) tmp_dir = ".";      // 默认 = 当前目录 (本地)
    opt.tmp_dir = tmp_dir;
    if (!p95_dir_ignored.empty()) {
        std::cerr << "note: --p95-dir is obsolete for ecm.exe (stage-1 saves are written "
                     "locally); use --tmp-dir, and run ecm_p95feeder for the p95 handoff"
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
    run_stage1_once(N, B1, B2, gpucurves, savefilename, saveappend, nline, opt, &result);

    std::vector<uint32_t> go_primes;
    if (print_group_order) {
        build_primes_up_to_B1(B1, go_primes);
    }

    for (uint32_t i = 0; i < result.curves; i++) {
        if (result.array_found[i] != ECM_NO_FACTOR_FOUND) {
            char *s = mpz_get_str(NULL, 10, result.factors[i]);
            std::cout << "factor[" << i << "]=" << (s ? s : "?") << "\n";
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
                if (!compute_group_order_pari_for_sigma3(go, result.factors[i], sigma_curve,
                                                         go_gp_exe, &err)) {
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
                mpz_clear(go);
            }
        }
    }

    for (uint32_t i = 0; i < result.curves; i++) mpz_clear(result.factors[i]);
    free(result.factors);
    free(result.array_found);
    mpz_clear(N);
    return 0;
}
