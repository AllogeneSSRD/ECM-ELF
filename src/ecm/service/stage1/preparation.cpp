#include "preparation.h"
#include "ecm/core/stage1/selection.h"
#include "../../storage/cache/ecm_stage1_exp_cache.h"
#include "../../storage/save/ecm_edwards_io.h"
#include "ecm/platform/logging.h"
#include <cmath>
#include <chrono>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <utility>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <direct.h>
#define access _access
#else
#include <unistd.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace ecm_stage1 {
using Stage1Options=Stage1Options;
bool compute_batch_s(mpz_t s, double B1, unsigned torsion){
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

uint64_t random_sigma_u64() {
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

std::string stage1_save_prefix(const Stage1Options &opt) {
    if(opt.execution.selection.curve==ecm_stage1::CurveModel::Edwards)return "e";
    return "m"+std::to_string(opt.execution.selection.target==ecm_stage1::ExecutionTarget::Gpu?opt.execution.selection.gpu_param:0);
}

std::string stage1_prefixed_save(const std::string &name,const std::string &prefix) {
    const auto file=std::filesystem::path(name);auto leaf=file.filename().string();
    if(leaf.rfind(prefix+"_",0)==0)return name;
    const auto lower=ecm_stage1::lower_ascii(leaf);
    for(const auto *old:{"ed_","e_","m0_","m2_","m3_"})
        if(lower.rfind(old,0)==0){leaf.erase(0,std::strlen(old));break;}
    return (file.parent_path()/(prefix+"_"+leaf)).string();
}

std::string mont_format_save_name(const std::string &pattern, const mpz_t N, double B1,
                                         const Stage1Options *options) {
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
    if(options && options->identity.target_kbnc) {
        n_str=std::to_string(options->identity.handoff_n);
        mersenne=options->identity.handoff_k_str=="1" && options->identity.handoff_b==2 && options->identity.handoff_c==-1;
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

bool fixed_sigma_batch_exceeds_prime95_range(uint64_t sigma, uint32_t curves) {
    return curves != 0 &&
           (sigma > static_cast<uint64_t>(INT64_MAX) ||
            static_cast<uint64_t>(curves - 1) > static_cast<uint64_t>(INT64_MAX) - sigma);
}

bool ensure_dir(const std::string &dir) {
    if (dir.empty()) return true;
#ifdef _WIN32
    if (_mkdir(dir.c_str()) == 0) return true;
#else
    if (mkdir(dir.c_str(), 0777) == 0) return true;
#endif
    return access(dir.c_str(), 0) == 0;
}

long long stage1_now_ms() {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool stdout_is_tty_local(void) {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

} // namespace ecm_stage1
