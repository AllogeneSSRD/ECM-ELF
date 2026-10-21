#include "group_order.h"
#include "ecm/application/configuration/runtime.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <process.h>
#define access _access
#define getpid _getpid
#else
#include <unistd.h>
#endif
namespace ecm_stage1_app {
static void trim(std::string &s){
    while(!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while(!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
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

bool build_primes_up_to_B1(double B1, std::vector<uint32_t> &primes) {
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

bool gp_executable_exists(const std::string &exe_path) {
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

std::string resolve_gp_path(const std::string &exe_path) {
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

std::string get_gp_executable(const std::string &explicit_path) {
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


bool print_group_order_factor(const ecm_stage1::TaskResult &result,uint32_t i,int go_param,const std::string &go_gp_exe,const std::vector<uint32_t> &go_primes) {

                    uint32_t sigma_curve = result.first_sigma() + i;
                    if (mpz_probab_prime_p(result.factor(i), 25) <= 0) {
                        std::cout << "  go_factor[" << i << "]=[ ] (factor is not prime, skip #E(F_p))\n";
                        return true;
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
                        return false;
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
                
return true;
}
}
