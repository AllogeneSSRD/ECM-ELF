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
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
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
              << "  -gpucurves <n>       Number of ECM curves per GPU launch\n"
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
    int verbose = 0;
    int device_index = 0;
    unsigned long gpuckpt_ms = ECM_DEFAULT_GPU_CHECKPOINT_INTERVAL_MS;
    std::string gpu_mul_path, gpu_sqr_path, gpu_add_path, gpu_sub_path, gpu_special_mult_path;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;
};

struct Stage1RunResult {
    int ret = ECM_ERROR;
    bool prepare_failed = false;
    uint32_t curves = 0;
    uint32_t firstsigma = 0;
    mpz_t *factors = nullptr;
    int *array_found = nullptr;
};

// Run one stage-1 batch. N and the caller's n_expr are borrowed (not cleared);
// factors / array_found are allocated here and returned via `out` (caller frees).
static int run_stage1_once(const mpz_t N, double B1, double B2, uint32_t curves,
                           const std::string &savefilename, bool saveappend,
                           const std::string &n_expr,
                           const Stage1RunOptions &opt, Stage1RunResult *out) {
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
    const uint32_t lastsigma = firstsigma + curves - 1;

    mpz_t batch_d;
    mpz_init(batch_d);
    gpu_compute_batch_d(batch_d, firstsigma, N);

    std::cout << "Using B1=" << B1 << ", B2=" << B2
              << ", sigma=" << ECM_PARAM_BATCH_32BITS_D << ":" << firstsigma
              << "-" << lastsigma << " (" << curves << " curves)" << std::endl;

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

    ecm_ts_fprintf(stdout, "===== ECM queue manager =====\n");
    ecm_ts_fprintf(stdout, "config : %s\n", ini.c_str());
    ecm_ts_fprintf(stdout, "worktodo : %s\n", worktodo_path.c_str());
    ecm_ts_fprintf(stdout, "finished : %s\n", finished_path.c_str());
    ecm_ts_fprintf(stdout, "log_file : %s\n", cfg.log_file.c_str());

    // Startup full sync (matches the old work_manager.ps1 behaviour).
    ecm_sync_save_files(exe_dir, sync1, sync2, /*full=*/true, 0);

    int processed = 0;
    while (true) {
        std::string line;
        if (!ecm_worktodo_first_line(worktodo_path, line)) {
            break;
        }

        ecm_ts_fprintf(stdout, "START: %s\n", line.c_str());

        EcmStage2Task task;
        std::string err;
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

        // Rebuild the original N expression for the save-file N= field.
        std::string cstr = task.c;
        std::string sign = "+";
        if (!cstr.empty() && cstr[0] == '-') {
            sign = "-";
            cstr = cstr.substr(1);
        }
        std::string n_expr = "(" + task.k + "*" + task.b + "^" + std::to_string(task.n) + sign + cstr + ")";
        if (!task.factors.empty()) {
            n_expr += "/(";
            for (std::size_t i = 0; i < task.factors.size(); ++i) {
                if (i != 0) n_expr += "*";
                n_expr += task.factors[i];
            }
            n_expr += ")";
        }

        const long long marker = current_epoch_seconds();

        Stage1RunResult result;
        run_stage1_once(N, B1, /*B2=*/0.0, task.curves_to_run, task.save_name,
                        /*saveappend=*/true, n_expr, opt, &result);

        const bool has_factors = (result.factors != nullptr);

        if (result.prepare_failed) {
            if (has_factors) {
                for (uint32_t i = 0; i < result.curves; ++i) mpz_clear(result.factors[i]);
                free(result.factors);
                free(result.array_found);
            }
            mpz_clear(N);
            ecm_ts_fprintf(stderr, "FATAL: GPU backend prepare failed; aborting queue.\n");
            if (logf) { ecm_log_set_mirror(nullptr); fclose(logf); }
            return 1;
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
                           task.aid.empty() ? "N/A" : task.aid.c_str(), line.c_str());
        }

        if (has_factors) {
            for (uint32_t i = 0; i < result.curves; ++i) mpz_clear(result.factors[i]);
            free(result.factors);
            free(result.array_found);
        }
        mpz_clear(N);

        if (result.ret == ECM_ERROR) {
            ecm_ts_fprintf(stderr, "ERROR: stage1 failed for task: %s\n", line.c_str());
            ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
            continue;
        }

        // Success: move the raw line to the finished file, drop it from worktodo.
        if (!ecm_append_text_line(finished_path, line)) {
            ecm_ts_fprintf(stderr, "ERROR: cannot append to %s\n", finished_path.c_str());
            ecm_worktodo_advance(worktodo_path, line, WorktodoAction::MarkError);
            continue;
        }
        ecm_worktodo_advance(worktodo_path, line, WorktodoAction::Remove);

        // Incremental sync of *.save touched by this task.
        const bool full = (cfg.sync_mode == "full");
        ecm_sync_save_files(exe_dir, sync1, sync2, full, marker);

        ++processed;
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
    uint32_t gpucurves = 0;
    double gpuckpt_seconds = -1.0;
    bool gpuckpt_set = false;
    bool sigma_fixed = false;
    uint32_t fixed_sigma = 0;
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
    // parse args simple
    std::vector<std::string> pos;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "-v") { verbose = true; continue; }
        if(a == "-gpu") { use_gpu = true; continue; }
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
        if(a == "-sigma" && i+1<argc){
            if(!parse_sigma_arg(argv[++i], &fixed_sigma)){
                std::cerr << "Invalid -sigma value (need 1..2^32-1, optional param:3: prefix)" << std::endl;
                return 1;
            }
            sigma_fixed = true;
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
    std::cout << "  mode: " << (use_gpu ? "gpu" : "cpu-stub")
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
    Stage1RunOptions opt;
    opt.use_gpu = use_gpu;
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
